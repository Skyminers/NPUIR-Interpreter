//===- OpsIndirect.cpp - Sparse, strided and atomic accesses ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory accesses whose addresses are data rather than shape: the indirect
// (gather/scatter) family, the strided family, atomics, and the scalar-unit
// load. What they share is that the interpreter cannot know at issue time
// which bytes they will touch, so each one has to be explicit about the range
// it declares to the race detector.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

/// Resolve a memref operand, rejecting the pre-bufferization tensor form.
bool getBuffer(Interpreter &interp, CoreState &core, Operation *op, Value value,
               MemRefValue &out) {
  if (isa<RankedTensorType>(value.getType())) {
    interp.emitError(op) << "tensor operand: the interpreter only accepts "
                            "fully bufferized (memref) HIVM IR";
    return false;
  }
  return interp.getMemRefOperand(core, value, out, op);
}

/// True when `mask` is absent or reads as nonzero at `index`.
bool maskAllows(Interpreter &interp, const std::optional<MemRefValue> &mask,
                ArrayRef<int64_t> index, Operation *op, bool &failed) {
  if (!mask)
    return true;
  RuntimeValue value;
  if (!rawLoadAt(interp, *mask, index, op, value)) {
    failed = true;
    return false;
  }
  return value.isInt() ? !value.getIntValue().isZero() : value.toDouble() != 0;
}

//===----------------------------------------------------------------------===//
// load_scalar
//===----------------------------------------------------------------------===//

/// `hivm.hir.load_scalar` reads one element through an `llvm.ptr`. It is a
/// scalar-unit access, so it goes on PIPE_S like `memref.load` does.
ExecResult execLoadScalar(Interpreter &interp, CoreState &core, Operation *op) {
  auto loadOp = cast<hivm::LoadScalarOp>(op);
  MemRefValue ptr;
  if (!interp.getMemRefOperand(core, loadOp.getAddr(), ptr, op))
    return ExecResult::Error;
  ptr.elemType = loadOp.getResult().getType();
  ptr.elemBytes = getStorageSize(ptr.elemType);

  interp.issueResidentAccess(
      core, Pipe::S, op,
      {ptr.arena, ptr.byteOffset, ptr.byteOffset + ptr.elemBytes},
      /*isWrite=*/false, /*isRawPointer=*/true);
  RuntimeValue value;
  if (!rawLoad(interp, ptr, 0, op, value))
    return ExecResult::Error;
  interp.setValue(core, loadOp.getResult(), value);
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// indirect_load / indirect_store
//===----------------------------------------------------------------------===//

/// Element offsets are data, so the range the effect declares is the whole
/// source buffer: at issue time the interpreter does not know which elements
/// the op will reach. Widening here can over-report sharing, which is the safe
/// direction - the alternative is committing the effect early just to read the
/// offsets, which would defeat the deferred model.
void collectWholeBuffer(Interpreter &interp, const MemRefValue &mem,
                        SmallVectorImpl<ByteRange> &out) {
  if (!mem.isValid() || mem.getNumElements() <= 0)
    return;
  auto [lo, hi] = mem.getSpannedBytes();
  out.push_back({mem.arena, lo, hi});
}

ExecResult execIndirectLoad(Interpreter &interp, CoreState &core,
                            Operation *op) {
  auto loadOp = cast<hivm::IndirectLoadOp>(op);
  MemRefValue src, offsets, dst;
  if (!getBuffer(interp, core, op, loadOp.getSrc(), src) ||
      !getBuffer(interp, core, op, loadOp.getOffsets(), offsets) ||
      !getBuffer(interp, core, op, loadOp.getDst(), dst))
    return ExecResult::Error;

  std::optional<MemRefValue> mask, other;
  if (loadOp.getMask()) {
    MemRefValue m;
    if (!getBuffer(interp, core, op, loadOp.getMask(), m))
      return ExecResult::Error;
    mask = m;
  }
  if (loadOp.getOther()) {
    MemRefValue o;
    if (!getBuffer(interp, core, op, loadOp.getOther(), o))
      return ExecResult::Error;
    other = o;
  }

  SmallVector<ByteRange, 4> reads, writes;
  collectWholeBuffer(interp, src, reads);
  interp.collectRanges(offsets, reads);
  if (mask)
    interp.collectRanges(*mask, reads);
  if (other)
    interp.collectRanges(*other, reads);
  interp.collectRanges(dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::V), op, reads, writes,
      [ip, op, src, offsets, dst, mask, other]() {
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          RuntimeValue value;
          if (!maskAllows(*ip, mask, index, op, failed)) {
            if (failed)
              return;
            // Masked out: fall back to `other`, or leave the element alone
            // when the op did not supply one.
            if (!other)
              return;
            if (!rawLoadAt(*ip, *other, index, op, value)) {
              failed = true;
              return;
            }
          } else {
            RuntimeValue at;
            if (!rawLoadAt(*ip, offsets, index, op, at)) {
              failed = true;
              return;
            }
            // The offset is a linear element index into `src`, not a
            // multi-dimensional one.
            if (!rawLoad(*ip, src, at.getIndexValue(), op, value)) {
              failed = true;
              return;
            }
          }
          if (!rawStoreAt(*ip, dst, index, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

ExecResult execIndirectStore(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto storeOp = cast<hivm::IndirectStoreOp>(op);
  MemRefValue dst, offsets, src;
  if (!getBuffer(interp, core, op, storeOp.getDst(), dst) ||
      !getBuffer(interp, core, op, storeOp.getOffsets(), offsets) ||
      !getBuffer(interp, core, op, storeOp.getSrc(), src))
    return ExecResult::Error;

  std::optional<MemRefValue> mask;
  if (storeOp.getMask()) {
    MemRefValue m;
    if (!getBuffer(interp, core, op, storeOp.getMask(), m))
      return ExecResult::Error;
    mask = m;
  }

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  interp.collectRanges(offsets, reads);
  if (mask)
    interp.collectRanges(*mask, reads);
  collectWholeBuffer(interp, dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(core, getOpPipe(op, Pipe::V), op, reads, writes,
                     [ip, op, src, offsets, dst, mask]() {
                       bool failed = false;
                       forEachIndex(src.sizes, [&](ArrayRef<int64_t> index) {
                         if (failed)
                           return;
                         if (!maskAllows(*ip, mask, index, op, failed))
                           return;
                         RuntimeValue at, value;
                         if (!rawLoadAt(*ip, offsets, index, op, at) ||
                             !rawLoadAt(*ip, src, index, op, value) ||
                             !rawStore(*ip, dst, at.getIndexValue(), op, value))
                           failed = true;
                       });
                     });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// stride_load / stride_store
//===----------------------------------------------------------------------===//

/// Resolve the `offset`, `strides[]` and `numels[]` operands shared by the
/// strided pair.
bool getStrideParams(Interpreter &interp, CoreState &core, Operation *op,
                     Value offsetValue, OperandRange strideValues,
                     OperandRange numelValues, int64_t &offset,
                     SmallVectorImpl<int64_t> &strides,
                     SmallVectorImpl<int64_t> &numels) {
  if (!getIndex(interp, core, offsetValue, op, offset))
    return false;
  for (Value v : strideValues) {
    int64_t s = 0;
    if (!getIndex(interp, core, v, op, s))
      return false;
    strides.push_back(s);
  }
  for (Value v : numelValues) {
    int64_t n = 0;
    if (!getIndex(interp, core, v, op, n))
      return false;
    numels.push_back(n);
  }
  return true;
}

ExecResult execStrideLoad(Interpreter &interp, CoreState &core, Operation *op) {
  auto loadOp = cast<hivm::StrideLoadOp>(op);
  MemRefValue src, dst;
  if (!getBuffer(interp, core, op, loadOp.getSrc(), src) ||
      !getBuffer(interp, core, op, loadOp.getDst(), dst))
    return ExecResult::Error;

  int64_t offset = 0;
  SmallVector<int64_t, 3> strides, numels;
  if (!getStrideParams(interp, core, op, loadOp.getOffset(),
                       loadOp.getStride(), loadOp.getNumel(), offset, strides,
                       numels))
    return ExecResult::Error;
  if (static_cast<int64_t>(strides.size()) != dst.getRank() ||
      static_cast<int64_t>(numels.size()) != dst.getRank()) {
    interp.emitError(op) << "stride_load needs one stride and one numel per "
                            "destination dimension";
    return ExecResult::Error;
  }

  RuntimeValue pad = interp.getValue(core, loadOp.getOther());
  if (pad.isNone()) {
    interp.emitError(op) << "stride_load `other` value is unbound";
    return ExecResult::Error;
  }

  SmallVector<ByteRange, 4> reads, writes;
  collectWholeBuffer(interp, src, reads);
  interp.collectRanges(dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::V), op, reads, writes,
      [ip, op, src, dst, offset, strides, numels, pad]() {
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          // `numels` is the logical extent; anything past it is padding, and
          // reading it would be an out-of-bounds access on a buffer the IR
          // deliberately over-allocated.
          bool inside = true;
          int64_t linear = offset;
          for (size_t d = 0; d < index.size(); ++d) {
            if (index[d] >= numels[d])
              inside = false;
            linear += index[d] * strides[d];
          }
          RuntimeValue value = pad;
          if (inside && !rawLoad(*ip, src, linear, op, value)) {
            failed = true;
            return;
          }
          if (!rawStoreAt(*ip, dst, index, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

ExecResult execStrideStore(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto storeOp = cast<hivm::StrideStoreOp>(op);
  MemRefValue src, dst;
  if (!getBuffer(interp, core, op, storeOp.getSrc(), src) ||
      !getBuffer(interp, core, op, storeOp.getDst(), dst))
    return ExecResult::Error;

  int64_t offset = 0;
  SmallVector<int64_t, 3> strides, numels;
  if (!getStrideParams(interp, core, op, storeOp.getOffset(),
                       storeOp.getStride(), storeOp.getNumel(), offset,
                       strides, numels))
    return ExecResult::Error;
  if (static_cast<int64_t>(strides.size()) != src.getRank() ||
      static_cast<int64_t>(numels.size()) != src.getRank()) {
    interp.emitError(op) << "stride_store needs one stride and one numel per "
                            "source dimension";
    return ExecResult::Error;
  }

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  collectWholeBuffer(interp, dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::V), op, reads, writes,
      [ip, op, src, dst, offset, strides, numels]() {
        bool failed = false;
        forEachIndex(src.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          int64_t linear = offset;
          for (size_t d = 0; d < index.size(); ++d) {
            if (index[d] >= numels[d])
              return;
            linear += index[d] * strides[d];
          }
          RuntimeValue value;
          if (!rawLoadAt(*ip, src, index, op, value) ||
              !rawStore(*ip, dst, linear, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// atomic_cas / atomic_xchg
//===----------------------------------------------------------------------===//

/// True when two runtime values have the same bit pattern in `type`. CAS
/// compares representations, not numbers, so a NaN never matches and the two
/// zeros are distinct - which is what makes a CAS loop terminate predictably.
bool sameBits(const RuntimeValue &a, const RuntimeValue &b) {
  if (a.isFloat() && b.isFloat())
    return a.getFloatValue().bitcastToAPInt() ==
           b.getFloatValue().bitcastToAPInt();
  if (a.isInt() && b.isInt())
    return a.getIntValue() == b.getIntValue();
  return false;
}

ExecResult execAtomicCas(Interpreter &interp, CoreState &core, Operation *op) {
  auto casOp = cast<hivm::AtomicCasOp>(op);
  if (casOp.getSrc().size() != 2) {
    interp.emitError(op) << "atomic_cas expects an expected-value and a "
                            "new-value operand";
    return ExecResult::Error;
  }
  MemRefValue expected, desired, dst;
  if (!getBuffer(interp, core, op, casOp.getSrc()[0], expected) ||
      !getBuffer(interp, core, op, casOp.getSrc()[1], desired) ||
      !getBuffer(interp, core, op, casOp.getDst(), dst))
    return ExecResult::Error;

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(expected, reads);
  interp.collectRanges(desired, reads);
  interp.collectRanges(dst, reads);
  interp.collectRanges(dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::MTE3), op, reads, writes,
      [ip, op, expected, desired, dst]() {
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          RuntimeValue current, want, next;
          if (!rawLoadAt(*ip, dst, index, op, current) ||
              !rawLoadAt(*ip, expected, index, op, want) ||
              !rawLoadAt(*ip, desired, index, op, next)) {
            failed = true;
            return;
          }
          if (sameBits(current, want) &&
              !rawStoreAt(*ip, dst, index, op, next))
            failed = true;
        });
      },
      /*isAtomic=*/true);
  return ExecResult::Advance;
}

ExecResult execAtomicXchg(Interpreter &interp, CoreState &core, Operation *op) {
  auto xchgOp = cast<hivm::AtomicXchgOp>(op);
  MemRefValue src, dst;
  if (!getBuffer(interp, core, op, xchgOp.getSrc(), src) ||
      !getBuffer(interp, core, op, xchgOp.getDst(), dst))
    return ExecResult::Error;

  std::optional<MemRefValue> mask;
  if (xchgOp.getMask()) {
    MemRefValue m;
    if (!getBuffer(interp, core, op, xchgOp.getMask(), m))
      return ExecResult::Error;
    mask = m;
  }

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  if (mask)
    interp.collectRanges(*mask, reads);
  interp.collectRanges(dst, writes);

  Interpreter *ip = &interp;
  interp.issueEffect(
      core, getOpPipe(op, Pipe::MTE3), op, reads, writes,
      [ip, op, src, dst, mask]() {
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          if (!maskAllows(*ip, mask, index, op, failed))
            return;
          RuntimeValue value;
          if (!rawLoadAt(*ip, src, index, op, value) ||
              !rawStoreAt(*ip, dst, index, op, value))
            failed = true;
        });
      },
      /*isAtomic=*/true);
  return ExecResult::Advance;
}

} // namespace

void registerHIVMIndirectOps(OpRegistry &registry) {
  registry.add(hivm::LoadScalarOp::getOperationName(), execLoadScalar);
  registry.add(hivm::IndirectLoadOp::getOperationName(), execIndirectLoad);
  registry.add(hivm::IndirectStoreOp::getOperationName(), execIndirectStore);
  registry.add(hivm::StrideLoadOp::getOperationName(), execStrideLoad);
  registry.add(hivm::StrideStoreOp::getOperationName(), execStrideStore);
  registry.add(hivm::AtomicCasOp::getOperationName(), execAtomicCas);
  registry.add(hivm::AtomicXchgOp::getOperationName(), execAtomicXchg);
}

} // namespace interp
} // namespace bishengir
