//===- OpsVector.cpp - vector dialect handlers ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Vectorized (VF) function bodies in memref-form HIVM are written with the
// upstream `vector` dialect: transfer_read / transfer_write against UB
// subviews, plus broadcasts, shape casts and multi_reduction. Vector values
// live in registers, so they are ordinary SSA values here; only the transfers
// touch memory, and those run on the vector pipe.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace bishengir {
namespace interp {

int64_t getVectorElementCount(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

RuntimeValue makeSplatVector(ArrayRef<int64_t> shape,
                             const RuntimeValue &value) {
  int64_t count = getVectorElementCount(shape);
  std::vector<RuntimeValue> elements(static_cast<size_t>(std::max<int64_t>(
                                         count, 0)),
                                     value);
  return RuntimeValue::getVector(
      SmallVector<int64_t, 4>(shape.begin(), shape.end()),
      std::move(elements));
}

namespace {

/// Evaluate a permutation map at `indices`. Only the projected-permutation
/// forms that vectorization emits are supported: each result is either a
/// dimension of the enclosing iteration space or the constant zero (a
/// broadcast).
bool applyPermutation(AffineMap map, ArrayRef<int64_t> vectorIndex,
                      SmallVectorImpl<int64_t> &memIndex, int64_t memRank) {
  memIndex.assign(memRank, 0);
  if (!map || map.isIdentity()) {
    // Align trailing dimensions.
    for (int64_t d = 0; d < memRank; ++d) {
      int64_t vd = static_cast<int64_t>(vectorIndex.size()) - memRank + d;
      memIndex[d] = vd >= 0 ? vectorIndex[vd] : 0;
    }
    return true;
  }
  // The permutation map takes memref dims to vector dims: result i of the map
  // says which memref dim vector dim i varies along.
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    AffineExpr expr = map.getResult(i);
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dim.getPosition();
      if (pos < memIndex.size() && i < vectorIndex.size())
        memIndex[pos] = vectorIndex[i];
      continue;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      // A zero result is a broadcast along that vector dimension.
      if (constant.getValue() != 0)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

/// Base indices of a transfer op, resolved against the runtime environment.
bool getTransferIndices(Interpreter &interp, CoreState &core, Operation *op,
                        OperandRange indices, SmallVectorImpl<int64_t> &out) {
  out.clear();
  for (Value index : indices) {
    int64_t value = 0;
    if (!getIndex(interp, core, index, op, value))
      return false;
    out.push_back(value);
  }
  return true;
}

ExecResult execTransferRead(Interpreter &interp, CoreState &core,
                            Operation *op) {
  auto readOp = cast<vector::TransferReadOp>(op);
  MemRefValue src;
  if (!isa<MemRefType>(readOp.getSource().getType())) {
    interp.emitError(op) << "vector.transfer_read on a tensor: the "
                            "interpreter only accepts bufferized IR";
    return ExecResult::Error;
  }
  if (!interp.getMemRefOperand(core, readOp.getSource(), src, op))
    return ExecResult::Error;

  SmallVector<int64_t, 4> base;
  if (!getTransferIndices(interp, core, op, readOp.getIndices(), base))
    return ExecResult::Error;

  auto vectorType = readOp.getVectorType();
  SmallVector<int64_t, 4> shape(vectorType.getShape().begin(),
                                vectorType.getShape().end());
  Type elemType = vectorType.getElementType();

  RuntimeValue padding = interp.getValue(core, readOp.getPadding());

  // The vector pipe retires in order, so a read on PIPE_V observes earlier
  // PIPE_V writes. Draining V here models that without hiding hazards
  // against the other pipes, which prepareDirectAccess still reports.
  ByteRange range{src.arena, 0, 0};
  auto [lo, hi] = src.getSpannedBytes();
  range.lo = lo;
  range.hi = hi;
  interp.prepareDirectAccess(core, op, Pipe::V, range, /*isWrite=*/false);

  std::vector<RuntimeValue> elements;
  elements.reserve(static_cast<size_t>(getVectorElementCount(shape)));
  SmallVector<int64_t, 4> memIndex;
  bool failed = false;
  forEachIndex(shape, [&](ArrayRef<int64_t> vectorIndex) {
    if (failed)
      return;
    if (!applyPermutation(readOp.getPermutationMap(), vectorIndex, memIndex,
                          src.getRank())) {
      interp.emitError(op) << "unsupported permutation_map";
      failed = true;
      return;
    }
    bool inBounds = true;
    for (int64_t d = 0; d < src.getRank(); ++d) {
      memIndex[d] += d < static_cast<int64_t>(base.size()) ? base[d] : 0;
      if (memIndex[d] < 0 || memIndex[d] >= src.sizes[d])
        inBounds = false;
    }
    RuntimeValue value;
    if (!inBounds) {
      value = padding;
    } else if (!rawLoadAt(interp, src, memIndex, op, value)) {
      failed = true;
      return;
    }
    elements.push_back(value);
  });
  if (failed)
    return ExecResult::Error;
  (void)elemType;

  interp.setValue(core, readOp.getResult(),
                  RuntimeValue::getVector(shape, std::move(elements)));
  return ExecResult::Advance;
}

ExecResult execTransferWrite(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto writeOp = cast<vector::TransferWriteOp>(op);
  if (!isa<MemRefType>(writeOp.getSource().getType())) {
    interp.emitError(op) << "vector.transfer_write on a tensor: the "
                            "interpreter only accepts bufferized IR";
    return ExecResult::Error;
  }
  MemRefValue dst;
  if (!interp.getMemRefOperand(core, writeOp.getSource(), dst, op))
    return ExecResult::Error;

  RuntimeValue vec = interp.getValue(core, writeOp.getVector());
  if (!vec.isVector()) {
    interp.emitError(op) << "vector.transfer_write source is unbound";
    return ExecResult::Error;
  }

  SmallVector<int64_t, 4> base;
  if (!getTransferIndices(interp, core, op, writeOp.getIndices(), base))
    return ExecResult::Error;

  SmallVector<int64_t, 4> shape(vec.getVectorShape().begin(),
                                vec.getVectorShape().end());
  AffineMap permutation = writeOp.getPermutationMap();

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  std::vector<RuntimeValue> elements = vec.getVectorElements();
  interp.issueEffect(
      core, Pipe::V, op, reads, writes,
      [interpPtr, op, dst, base, shape, permutation, elements]() {
        SmallVector<int64_t, 4> memIndex;
        size_t linear = 0;
        bool failed = false;
        forEachIndex(shape, [&](ArrayRef<int64_t> vectorIndex) {
          if (failed || linear >= elements.size())
            return;
          const RuntimeValue &value = elements[linear++];
          if (!applyPermutation(permutation, vectorIndex, memIndex,
                                dst.getRank())) {
            interpPtr->emitError(op) << "unsupported permutation_map";
            failed = true;
            return;
          }
          bool inBounds = true;
          for (int64_t d = 0; d < dst.getRank(); ++d) {
            memIndex[d] += d < static_cast<int64_t>(base.size()) ? base[d] : 0;
            if (memIndex[d] < 0 || memIndex[d] >= dst.sizes[d])
              inBounds = false;
          }
          if (!inBounds)
            return; // Masked-off lane.
          if (!rawStoreAt(*interpPtr, dst, memIndex, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

ExecResult execBroadcast(Interpreter &interp, CoreState &core, Operation *op) {
  auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!resultType) {
    interp.emitError(op) << "vector.broadcast with a non-vector result";
    return ExecResult::Error;
  }
  RuntimeValue source = interp.getValue(core, op->getOperand(0));
  SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                resultType.getShape().end());
  if (!source.isVector()) {
    interp.setValue(core, op->getResult(0), makeSplatVector(shape, source));
    return ExecResult::Advance;
  }

  // Vector source: stretch the leading and unit dimensions.
  ArrayRef<int64_t> srcShape = source.getVectorShape();
  const std::vector<RuntimeValue> &srcElements = source.getVectorElements();
  std::vector<RuntimeValue> elements;
  elements.reserve(static_cast<size_t>(getVectorElementCount(shape)));
  SmallVector<int64_t, 4> srcIndex;
  forEachIndex(shape, [&](ArrayRef<int64_t> index) {
    int64_t rank = static_cast<int64_t>(srcShape.size());
    srcIndex.assign(rank, 0);
    for (int64_t d = 0; d < rank; ++d) {
      int64_t vd = static_cast<int64_t>(index.size()) - rank + d;
      int64_t value = vd >= 0 ? index[vd] : 0;
      srcIndex[d] = srcShape[d] == 1 ? 0 : value;
    }
    int64_t linear = 0;
    for (int64_t d = 0; d < rank; ++d)
      linear = linear * srcShape[d] + srcIndex[d];
    if (linear >= 0 && static_cast<size_t>(linear) < srcElements.size())
      elements.push_back(srcElements[linear]);
    else
      elements.push_back(RuntimeValue());
  });
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getVector(shape, std::move(elements)));
  return ExecResult::Advance;
}

/// shape_cast keeps the element order and only relabels the shape.
ExecResult execShapeCast(Interpreter &interp, CoreState &core, Operation *op) {
  RuntimeValue source = interp.getValue(core, op->getOperand(0));
  auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!source.isVector() || !resultType) {
    interp.emitError(op) << "vector.shape_cast operand is unbound";
    return ExecResult::Error;
  }
  SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                resultType.getShape().end());
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getVector(shape, source.getVectorElements()));
  return ExecResult::Advance;
}

ExecResult execMultiReduction(Interpreter &interp, CoreState &core,
                              Operation *op) {
  auto reduceOp = cast<vector::MultiDimReductionOp>(op);
  RuntimeValue source = interp.getValue(core, reduceOp.getSource());
  RuntimeValue acc = interp.getValue(core, reduceOp.getAcc());
  if (!source.isVector()) {
    interp.emitError(op) << "vector.multi_reduction source is unbound";
    return ExecResult::Error;
  }

  ArrayRef<int64_t> srcShape = source.getVectorShape();
  // getReductionDims() is an ArrayAttr of IntegerAttrs in this MLIR version.
  SmallVector<int64_t, 2> reduceDims;
  for (Attribute dim : reduceOp.getReductionDims())
    reduceDims.push_back(cast<IntegerAttr>(dim).getInt());

  // Result shape drops the reduced dimensions.
  SmallVector<int64_t, 4> resultShape;
  for (int64_t d = 0; d < static_cast<int64_t>(srcShape.size()); ++d)
    if (!llvm::is_contained(reduceDims, d))
      resultShape.push_back(srcShape[d]);

  std::vector<RuntimeValue> result;
  // Vectorization marks reductions whose accumulator is merely an output
  // buffer placeholder with `withoutInitMergeOp`.  Those reductions start
  // from their first source lane; reading the placeholder would fold UB
  // poison into otherwise valid max/sum results.
  if (op->hasAttr("withoutInitMergeOp")) {
    result.resize(static_cast<size_t>(std::max<int64_t>(
        getVectorElementCount(resultShape), 0)));
  } else if (acc.isVector()) {
    result = acc.getVectorElements();
  } else {
    result.assign(
        static_cast<size_t>(std::max<int64_t>(
            getVectorElementCount(resultShape), 0)),
        acc);
  }

  vector::CombiningKind kind = reduceOp.getKind();
  const std::vector<RuntimeValue> &srcElements = source.getVectorElements();
  size_t linear = 0;
  bool failed = false;
  forEachIndex(srcShape, [&](ArrayRef<int64_t> index) {
    if (failed || linear >= srcElements.size())
      return;
    const RuntimeValue &value = srcElements[linear++];
    int64_t slot = 0;
    size_t rd = 0;
    for (int64_t d = 0; d < static_cast<int64_t>(srcShape.size()); ++d) {
      if (llvm::is_contained(reduceDims, d))
        continue;
      slot = slot * resultShape[rd] + index[d];
      ++rd;
    }
    if (slot < 0 || static_cast<size_t>(slot) >= result.size())
      return;
    RuntimeValue &accumulator = result[slot];
    if (accumulator.isFloat() && value.isFloat()) {
      llvm::APFloat a = accumulator.getFloatValue();
      const llvm::APFloat &b = value.getFloatValue();
      switch (kind) {
      case vector::CombiningKind::ADD:
        a.add(b, llvm::APFloat::rmNearestTiesToEven);
        break;
      case vector::CombiningKind::MUL:
        a.multiply(b, llvm::APFloat::rmNearestTiesToEven);
        break;
      case vector::CombiningKind::MAXIMUMF:
      case vector::CombiningKind::MAXNUMF:
        a = a.isNaN() ? a : (b.isNaN() ? b : (a > b ? a : b));
        break;
      case vector::CombiningKind::MINIMUMF:
      case vector::CombiningKind::MINNUMF:
        a = a.isNaN() ? a : (b.isNaN() ? b : (a < b ? a : b));
        break;
      default:
        interp.emitError(op) << "unsupported float combining kind";
        failed = true;
        return;
      }
      accumulator = RuntimeValue::getFloat(a);
      return;
    }
    if (accumulator.isInt() && value.isInt()) {
      llvm::APInt a = accumulator.getIntValue();
      llvm::APInt b = value.getIntValue();
      if (a.getBitWidth() != b.getBitWidth())
        b = b.sextOrTrunc(a.getBitWidth());
      switch (kind) {
      case vector::CombiningKind::ADD:
        a = a + b;
        break;
      case vector::CombiningKind::MUL:
        a = a * b;
        break;
      case vector::CombiningKind::MAXSI:
        a = a.sgt(b) ? a : b;
        break;
      case vector::CombiningKind::MINSI:
        a = a.slt(b) ? a : b;
        break;
      case vector::CombiningKind::MAXUI:
        a = a.ugt(b) ? a : b;
        break;
      case vector::CombiningKind::MINUI:
        a = a.ult(b) ? a : b;
        break;
      case vector::CombiningKind::AND:
        a = a & b;
        break;
      case vector::CombiningKind::OR:
        a = a | b;
        break;
      case vector::CombiningKind::XOR:
        a = a ^ b;
        break;
      default:
        interp.emitError(op) << "unsupported integer combining kind";
        failed = true;
        return;
      }
      accumulator = RuntimeValue::getInt(a);
      return;
    }
    accumulator = value;
  });
  if (failed)
    return ExecResult::Error;

  if (resultShape.empty() && !result.empty())
    interp.setValue(core, reduceOp.getResult(), result.front());
  else
    interp.setValue(core, reduceOp.getResult(),
                    RuntimeValue::getVector(resultShape, std::move(result)));
  return ExecResult::Advance;
}

ExecResult execSplat(Interpreter &interp, CoreState &core, Operation *op) {
  auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!resultType) {
    interp.emitError(op) << "vector.splat with a non-vector result";
    return ExecResult::Error;
  }
  RuntimeValue source = interp.getValue(core, op->getOperand(0));
  SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                resultType.getShape().end());
  interp.setValue(core, op->getResult(0), makeSplatVector(shape, source));
  return ExecResult::Advance;
}

ExecResult execConstantMask(Interpreter &interp, CoreState &core,
                            Operation *op) {
  auto maskOp = cast<vector::ConstantMaskOp>(op);
  VectorType type = maskOp.getResult().getType();
  SmallVector<int64_t, 4> shape(type.getShape().begin(),
                                type.getShape().end());
  SmallVector<int64_t, 4> active;
  for (Attribute attr : maskOp.getMaskDimSizes())
    active.push_back(cast<IntegerAttr>(attr).getInt());

  std::vector<RuntimeValue> elements;
  elements.reserve(static_cast<size_t>(getVectorElementCount(shape)));
  forEachIndex(shape, [&](ArrayRef<int64_t> index) {
    bool enabled = true;
    for (size_t d = 0; d < index.size(); ++d)
      enabled &= index[d] < active[d];
    elements.push_back(RuntimeValue::getInt(llvm::APInt(1, enabled)));
  });
  interp.setValue(core, maskOp.getResult(),
                  RuntimeValue::getVector(shape, std::move(elements)));
  return ExecResult::Advance;
}

ExecResult execExtract(Interpreter &interp, CoreState &core, Operation *op) {
  auto extractOp = cast<vector::ExtractOp>(op);
  RuntimeValue source = interp.getValue(core, extractOp.getVector());
  if (!source.isVector()) {
    interp.emitError(op) << "vector.extract source is unbound";
    return ExecResult::Error;
  }
  ArrayRef<int64_t> shape = source.getVectorShape();
  SmallVector<int64_t, 4> position;
  for (OpFoldResult ofr : extractOp.getMixedPosition()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    position.push_back(value);
  }

  // The leading `position.size()` dims are indexed; the rest form the result.
  int64_t stride = 1;
  for (size_t d = position.size(); d < shape.size(); ++d)
    stride *= shape[d];
  int64_t base = 0;
  for (size_t d = 0; d < position.size(); ++d)
    base = base * shape[d] + position[d];
  base *= stride;

  const std::vector<RuntimeValue> &elements = source.getVectorElements();
  if (base < 0 || static_cast<size_t>(base + stride) > elements.size()) {
    interp.emitError(op) << "vector.extract position out of range";
    return ExecResult::Error;
  }
  if (stride == 1 && position.size() == shape.size()) {
    interp.setValue(core, extractOp.getResult(), elements[base]);
    return ExecResult::Advance;
  }
  SmallVector<int64_t, 4> resultShape(shape.begin() + position.size(),
                                      shape.end());
  std::vector<RuntimeValue> slice(elements.begin() + base,
                                  elements.begin() + base + stride);
  interp.setValue(core, extractOp.getResult(),
                  RuntimeValue::getVector(resultShape, std::move(slice)));
  return ExecResult::Advance;
}

} // namespace

void registerVectorOps(OpRegistry &registry) {
  registry.add("vector.transfer_read", execTransferRead);
  registry.add("vector.transfer_write", execTransferWrite);
  registry.add("vector.broadcast", execBroadcast);
  registry.add("vector.splat", execSplat);
  registry.add("vector.constant_mask", execConstantMask);
  registry.add("vector.shape_cast", execShapeCast);
  registry.add("vector.multi_reduction", execMultiReduction);
  registry.add("vector.extract", execExtract);
}

} // namespace interp
} // namespace bishengir
