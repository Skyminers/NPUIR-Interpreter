//===- OpsMisc.cpp - HIVM query / shuffle / reduce / mmad -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <cstring>

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

//===----------------------------------------------------------------------===//
// Query ops
//===----------------------------------------------------------------------===//

ExecResult setI64Result(Interpreter &interp, CoreState &core, Operation *op,
                        int64_t value) {
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getInt(llvm::APInt(
                      64, static_cast<uint64_t>(value), /*isSigned=*/true)));
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Broadcast
//===----------------------------------------------------------------------===//

ExecResult execVBrc(Interpreter &interp, CoreState &core, Operation *op) {
  auto brcOp = cast<hivm::VBrcOp>(op);
  MemRefValue dst;
  if (!interp.getMemRefOperand(core, brcOp.getDst(), dst, op))
    return ExecResult::Error;

  Value srcValue = brcOp.getSrc();
  bool scalarSource = !isa<ShapedType>(srcValue.getType());
  MemRefValue src;
  RuntimeValue scalar;
  if (scalarSource) {
    scalar = interp.getValue(core, srcValue);
    if (scalar.isNone()) {
      interp.emitError(op) << "vbrc scalar source is unbound";
      return ExecResult::Error;
    }
  } else if (!interp.getMemRefOperand(core, srcValue, src, op)) {
    return ExecResult::Error;
  }

  SmallVector<ByteRange, 4> reads, writes;
  if (!scalarSource)
    interp.collectRanges(src, reads);
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::V), op, reads, writes,
      [interpPtr, op, dst, src, scalar, scalarSource]() {
        SmallVector<int64_t, 4> srcIndex;
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          RuntimeValue value = scalar;
          if (!scalarSource) {
            // Broadcast dims have extent 1 in the source; mapSourceIndex
            // already collapses them.
            mapSourceIndex(index, src, {}, srcIndex);
            if (!rawLoadAt(*interpPtr, src, srcIndex, op, value)) {
              failed = true;
              return;
            }
          }
          if (!rawStoreAt(*interpPtr, dst, index, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

/// Combine `acc` with `value` under the reduction `kind`.
RuntimeValue reduceCombine(hivm::ReduceOperation kind, const RuntimeValue &acc,
                           const RuntimeValue &value) {
  if (acc.isFloat() && value.isFloat()) {
    llvm::APFloat a = acc.getFloatValue();
    const llvm::APFloat &b = value.getFloatValue();
    switch (kind) {
    case hivm::ReduceOperation::sum:
      a.add(b, llvm::APFloat::rmNearestTiesToEven);
      break;
    case hivm::ReduceOperation::prod:
      a.multiply(b, llvm::APFloat::rmNearestTiesToEven);
      break;
    case hivm::ReduceOperation::max:
    case hivm::ReduceOperation::max_with_index:
      // min/max propagate NaN, per the VReduceOp description, and order the
      // signed zeros - which is IEEE 754-2019 maximum/minimum exactly.
      a = llvm::maximum(a, b);
      break;
    case hivm::ReduceOperation::min:
    case hivm::ReduceOperation::min_with_index:
      a = llvm::minimum(a, b);
      break;
    default:
      a = b;
      break;
    }
    return RuntimeValue::getFloat(a);
  }
  llvm::APInt a = acc.getIntValue();
  const llvm::APInt &b = value.getIntValue();
  switch (kind) {
  case hivm::ReduceOperation::sum:
    a = a + b;
    break;
  case hivm::ReduceOperation::prod:
    a = a * b;
    break;
  case hivm::ReduceOperation::max:
  case hivm::ReduceOperation::max_with_index:
    a = a.sgt(b) ? a : b;
    break;
  case hivm::ReduceOperation::min:
  case hivm::ReduceOperation::min_with_index:
    a = a.slt(b) ? a : b;
    break;
  case hivm::ReduceOperation::any:
  case hivm::ReduceOperation::ori:
    a = a | b;
    break;
  case hivm::ReduceOperation::all:
  case hivm::ReduceOperation::andi:
    a = a & b;
    break;
  case hivm::ReduceOperation::xori:
    a = a ^ b;
    break;
  default:
    a = b;
    break;
  }
  return RuntimeValue::getInt(a);
}

ExecResult execVReduce(Interpreter &interp, CoreState &core, Operation *op) {
  auto reduceOp = cast<hivm::VReduceOp>(op);
  MemRefValue src, dst;
  if (!interp.getMemRefOperand(core, reduceOp.getSrc(), src, op))
    return ExecResult::Error;
  if (reduceOp.getDst().empty()) {
    interp.emitError(op) << "vreduce without a destination";
    return ExecResult::Error;
  }
  if (!interp.getMemRefOperand(core, reduceOp.getDst().front(), dst, op))
    return ExecResult::Error;

  hivm::ReduceOperation kind = reduceOp.getArith().getReduceOp();
  SmallVector<int64_t, 2> reduceDims(reduceOp.getReduceDims().begin(),
                                     reduceOp.getReduceDims().end());

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  // The destination also seeds the accumulator, so it is read as well.
  interp.collectRanges(dst, reads);
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::V), op, reads, writes,
      [interpPtr, op, src, dst, kind, reduceDims]() {
        // Walk the full source; every element folds into the destination slot
        // obtained by zeroing the reduced dimensions. The destination's
        // current contents are the reduction's init value, per the ODS.
        SmallVector<int64_t, 4> dstIndex;
        bool failed = false;
        forEachIndex(src.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          dstIndex.assign(index.begin(), index.end());
          for (int64_t d : reduceDims)
            if (d >= 0 && d < static_cast<int64_t>(dstIndex.size()))
              dstIndex[d] = 0;
          // A destination of lower rank drops the reduced dims entirely.
          SmallVector<int64_t, 4> mapped;
          if (dst.getRank() == static_cast<int64_t>(dstIndex.size())) {
            mapped.assign(dstIndex.begin(), dstIndex.end());
            for (size_t d = 0; d < mapped.size(); ++d)
              if (dst.sizes[d] == 1)
                mapped[d] = 0;
          } else {
            for (size_t d = 0; d < dstIndex.size(); ++d)
              if (!llvm::is_contained(reduceDims, static_cast<int64_t>(d)))
                mapped.push_back(dstIndex[d]);
          }

          RuntimeValue value, acc;
          if (!rawLoadAt(*interpPtr, src, index, op, value) ||
              !rawLoadAt(*interpPtr, dst, mapped, op, acc)) {
            failed = true;
            return;
          }
          RuntimeValue result = reduceCombine(kind, acc, value);
          if (!rawStoreAt(*interpPtr, dst, mapped, op, result))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

ExecResult execVTranspose(Interpreter &interp, CoreState &core, Operation *op) {
  auto transposeOp = cast<hivm::VTransposeOp>(op);
  MemRefValue src, dst;
  if (!interp.getMemRefOperand(core, transposeOp.getSrc(), src, op) ||
      !interp.getMemRefOperand(core, transposeOp.getDst(), dst, op))
    return ExecResult::Error;
  if (dst.getRank() != 2 || src.getRank() != 2) {
    interp.emitError(op) << "vtranspose expects rank-2 operands";
    return ExecResult::Error;
  }

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  interp.issueEffect(core, getOpPipe(op, Pipe::V), op, reads, writes,
                     [interpPtr, op, src, dst]() {
                       bool failed = false;
                       forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
                         if (failed)
                           return;
                         int64_t swapped[2] = {index[1], index[0]};
                         RuntimeValue value;
                         if (!rawLoadAt(*interpPtr, src, swapped, op, value) ||
                             !rawStoreAt(*interpPtr, dst, index, op, value))
                           failed = true;
                       });
                     });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Matrix multiply
//===----------------------------------------------------------------------===//

/// The two halves of a local matmul. `mmad_l1` carries
/// `MacroOpPipeTrait<MTE1, M>`: MTE1 stages the L1 tiles into L0A/L0B, then
/// the cube multiplies them into L0C. Each half retires on its own pipe, so
/// once `set_flag[PIPE_MTE1, ...]` has released the A/B tiles the IR is
/// entitled to refill them while the cube is still working - and the cube must
/// still see the values MTE1 captured.
struct MmadStaging {
  Interpreter *interp = nullptr;
  Operation *op = nullptr;
  MemRefValue a, b, c;
  int64_t m = 0, k = 0, n = 0;
  bool transposeA = false, transposeB = false, clearFirst = true;

  /// A as m x k and B as k x n, packed row-major with the transposes already
  /// resolved. Raw bytes rather than RuntimeValues: a 256x256 f16 tile is
  /// 128 KB this way and several megabytes the other.
  std::vector<uint8_t> aTile, bTile;
  bool staged = false;

  /// MTE1 half. Idempotent, because a flush rule may drain the cube's queue
  /// without having drained MTE1 first.
  void stage() {
    if (staged)
      return;
    staged = true;
    unsigned aBytes = a.elemBytes, bBytes = b.elemBytes;
    aTile.assign(static_cast<size_t>(m) * k * aBytes, 0);
    bTile.assign(static_cast<size_t>(k) * n * bBytes, 0);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t p = 0; p < k; ++p) {
        int64_t idx[2] = {transposeA ? p : i, transposeA ? i : p};
        if (const uint8_t *src =
                elementPtr(*interp, a, a.getByteAddr(idx), op))
          std::memcpy(&aTile[(static_cast<size_t>(i) * k + p) * aBytes], src,
                      aBytes);
      }
    for (int64_t p = 0; p < k; ++p)
      for (int64_t j = 0; j < n; ++j) {
        int64_t idx[2] = {transposeB ? j : p, transposeB ? p : j};
        if (const uint8_t *src =
                elementPtr(*interp, b, b.getByteAddr(idx), op))
          std::memcpy(&bTile[(static_cast<size_t>(p) * n + j) * bBytes], src,
                      bBytes);
      }
  }

  /// PIPE_M half.
  void multiply() {
    stage();
    const llvm::fltSemantics *sem = getFloatSemantics(c.elemType);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        int64_t cIdx[2] = {i, j};
        RuntimeValue acc;
        if (clearFirst) {
          acc = sem ? RuntimeValue::getFloat(llvm::APFloat::getZero(*sem))
                    : RuntimeValue::getInt(
                          llvm::APInt(getStorageSize(c.elemType) * 8, 0));
        } else if (!rawLoadAt(*interp, c, cIdx, op, acc)) {
          return;
        }
        for (int64_t p = 0; p < k; ++p) {
          RuntimeValue av = loadElement(
              &aTile[(static_cast<size_t>(i) * k + p) * a.elemBytes],
              a.elemType);
          RuntimeValue bv = loadElement(
              &bTile[(static_cast<size_t>(p) * n + j) * b.elemBytes],
              b.elemType);
          if (sem) {
            // Accumulate in the (wider) destination format, matching the f32
            // L0C accumulator behind f16/bf16 inputs.
            llvm::APFloat x =
                av.isFloat() ? av.getFloatValue() : llvm::APFloat(*sem);
            llvm::APFloat y =
                bv.isFloat() ? bv.getFloatValue() : llvm::APFloat(*sem);
            bool losesInfo = false;
            x.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            y.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            x.multiply(y, llvm::APFloat::rmNearestTiesToEven);
            llvm::APFloat sum = acc.getFloatValue();
            sum.add(x, llvm::APFloat::rmNearestTiesToEven);
            acc = RuntimeValue::getFloat(sum);
          } else {
            unsigned width = acc.getIntValue().getBitWidth();
            llvm::APInt x = av.getIntValue().sextOrTrunc(width);
            llvm::APInt y = bv.getIntValue().sextOrTrunc(width);
            acc = RuntimeValue::getInt(acc.getIntValue() + x * y);
          }
        }
        if (!rawStoreAt(*interp, c, cIdx, op, acc))
          return;
      }
    }
  }
};

/// C += A * B, accumulating in the destination element type. `init_condition`
/// selects whether the L0C accumulator is cleared first.
ExecResult execMmad(Interpreter &interp, CoreState &core, Operation *op,
                    Value aValue, Value bValue, Value cValue,
                    Value initCondition, bool transposeA, bool transposeB,
                    Value realMValue, Value realKValue, Value realNValue) {
  MemRefValue a, b, c;
  if (isa<RankedTensorType>(aValue.getType())) {
    interp.emitError(op) << "tensor-form mmad: the interpreter only accepts "
                            "fully bufferized (memref) HIVM IR";
    return ExecResult::Error;
  }
  if (!interp.getMemRefOperand(core, aValue, a, op) ||
      !interp.getMemRefOperand(core, bValue, b, op) ||
      !interp.getMemRefOperand(core, cValue, c, op))
    return ExecResult::Error;
  if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2) {
    interp.emitError(op) << "mmad expects rank-2 A, B and C";
    return ExecResult::Error;
  }

  // L1 tiles are padded up to whole fractal blocks, so the buffer extents
  // are an upper bound on the real problem. Multiplying the padding in would
  // fold uninitialised bytes into the result.
  int64_t bufM = c.sizes[0];
  int64_t bufN = c.sizes[1];
  int64_t bufK = transposeA ? a.sizes[0] : a.sizes[1];
  int64_t realM = bufM, realK = bufK, realN = bufN;
  if ((realMValue && !getIndex(interp, core, realMValue, op, realM)) ||
      (realKValue && !getIndex(interp, core, realKValue, op, realK)) ||
      (realNValue && !getIndex(interp, core, realNValue, op, realN)))
    return ExecResult::Error;
  if (realM > bufM || realN > bufN || realK > bufK) {
    interp.emitError(op) << "mmad real dimensions (" << realM << "x" << realK
                         << "x" << realN << ") exceed the buffer extents ("
                         << bufM << "x" << bufK << "x" << bufN << ")";
    return ExecResult::Error;
  }

  bool clearFirst = true;
  if (initCondition) {
    RuntimeValue init = interp.getValue(core, initCondition);
    clearFirst = init.isInt() && !init.getIntValue().isZero();
  }

  auto state = std::make_shared<MmadStaging>();
  state->interp = &interp;
  state->op = op;
  state->a = a;
  state->b = b;
  state->c = c;
  state->m = realM;
  state->k = realK;
  state->n = realN;
  state->transposeA = transposeA;
  state->transposeB = transposeB;
  state->clearFirst = clearFirst;

  // First half: MTE1 reads the L1 tiles. Nothing the address model can see is
  // written - L0A/L0B are internal to the macro op - so the effect carries
  // reads only, and its point is to keep A and B pinned until a flag on the
  // input pipe releases them.
  SmallVector<ByteRange, 4> stageReads;
  interp.collectRanges(a, stageReads);
  interp.collectRanges(b, stageReads);
  interp.issueEffect(core, getOpInPipe(op, Pipe::MTE1), op, stageReads,
                     /*writes=*/{}, [state]() { state->stage(); });

  // Second half: the cube multiplies into L0C. The result becomes visible to
  // fixpipe only after an M->FIX flag.
  SmallVector<ByteRange, 4> reads, writes;
  if (!clearFirst)
    interp.collectRanges(c, reads);
  interp.collectRanges(c, writes);
  interp.issueEffect(core, getOpPipe(op, Pipe::M), op, reads, writes,
                     [state]() { state->multiply(); });
  return ExecResult::Advance;
}

} // namespace

void registerHIVMMiscOps(OpRegistry &registry) {
  // --- query ops ---
  registry.add(hivm::GetBlockIdxOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 return setI64Result(interp, core, op, core.id.blockIdx);
               });
  registry.add(hivm::GetBlockNumOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 return setI64Result(interp, core, op,
                                     interp.getOptions().blockDim);
               });
  registry.add(hivm::GetSubBlockIdxOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 return setI64Result(interp, core, op, core.id.subBlockIdx);
               });
  registry.add(hivm::GetSubBlockNumOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 return setI64Result(interp, core, op,
                                     interp.getOptions().subBlockNum);
               });

  // --- shuffle / reduce ---
  registry.add(hivm::VBrcOp::getOperationName(), execVBrc);
  registry.add(hivm::VReduceOp::getOperationName(), execVReduce);
  registry.add(hivm::VTransposeOp::getOperationName(), execVTranspose);

  // --- matrix multiply ---
  registry.add(hivm::MmadL1Op::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<hivm::MmadL1Op>(op);
                 return execMmad(interp, core, op, typed.getA(), typed.getB(),
                                 typed.getC(), typed.getInitCondition(),
                                 typed.getATranspose().value_or(false),
                                 typed.getBTranspose().value_or(false),
                                 typed.getRealM(), typed.getRealK(),
                                 typed.getRealN());
               });
  registry.add(hivm::BatchMmadL1Op::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<hivm::BatchMmadL1Op>(op);
                 return execMmad(interp, core, op, typed.getA(), typed.getB(),
                                 typed.getC(), typed.getInitCondition(),
                                 typed.getATranspose().value_or(false),
                                 typed.getBTranspose().value_or(false),
                                 typed.getRealM(), typed.getRealK(),
                                 typed.getRealN());
               });
  // hivm.hir.matmul / mix_matmul / mix_group_matmul are whole-problem macro
  // ops carrying tiling and epilogue parameters, not a single MAC over L1
  // tiles. They are left unregistered so the driver reports them clearly
  // rather than silently computing something else.
}

} // namespace interp
} // namespace bishengir
