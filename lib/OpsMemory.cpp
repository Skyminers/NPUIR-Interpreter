//===- OpsMemory.cpp - HIVM DMA / view op handlers --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The copy family (load / store / copy / fixpipe / nd2nz / nz2nd / l12ub) and
// the view family (convert_layout / pointer_cast / bitcast). Every member of
// the copy family carries a pipe, so all of them defer.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

/// Elementwise conversion applied while a DMA moves data between buffers of
/// different element types (fixpipe f32 accumulator -> f16 output, say).
RuntimeValue convertForCopy(const RuntimeValue &value, Type from, Type to) {
  if (from == to)
    return value;
  return convertValue(value, from, to, InterpRoundMode::RINT,
                      /*isUnsigned=*/false);
}

/// Queue a rectangular copy from `src` to `dst` on `pipe`.
///
/// The destination shape drives the iteration. Positions outside the source
/// are filled with `padValue` when one is supplied, and skipped otherwise -
/// this covers `load`'s pad modes without a separate code path.
ExecResult issueCopy(Interpreter &interp, CoreState &core, Operation *op,
                     Pipe pipe, const MemRefValue &src, const MemRefValue &dst,
                     std::optional<RuntimeValue> padValue,
                     int64_t leftPadding) {
  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  interp.issueEffect(
      core, pipe, op, reads, writes,
      [interpPtr, op, src, dst, padValue, leftPadding]() {
        SmallVector<int64_t, 4> srcIndex;
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          // Align trailing dimensions so a rank-reduced source still maps.
          int64_t srcRank = src.getRank();
          int64_t dstRank = static_cast<int64_t>(index.size());
          srcIndex.assign(srcRank, 0);
          bool inRange = true;
          for (int64_t d = 0; d < srcRank; ++d) {
            int64_t dstDim = dstRank - srcRank + d;
            int64_t value = dstDim >= 0 ? index[dstDim] : 0;
            // Left padding shifts the destination window right.
            if (d == srcRank - 1)
              value -= leftPadding;
            if (src.sizes[d] == 1 && value != 0)
              value = 0;
            if (value < 0 || value >= src.sizes[d]) {
              inRange = false;
              break;
            }
            srcIndex[d] = value;
          }

          RuntimeValue value;
          if (inRange) {
            if (!rawLoadAt(*interpPtr, src, srcIndex, op, value)) {
              failed = true;
              return;
            }
            value = convertForCopy(value, src.elemType, dst.elemType);
          } else if (padValue) {
            value = convertForCopy(*padValue, dst.elemType, dst.elemType);
          } else {
            return; // Nothing to write outside the source extent.
          }
          if (!rawStoreAt(*interpPtr, dst, index, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

/// Shared prologue: resolve src/dst, reject tensor operands with a clear
/// message rather than a confusing "unbound operand".
bool resolveCopyOperands(Interpreter &interp, CoreState &core, Operation *op,
                         Value srcValue, Value dstValue, MemRefValue &src,
                         MemRefValue &dst) {
  if (isa<RankedTensorType>(srcValue.getType()) ||
      isa<RankedTensorType>(dstValue.getType())) {
    interp.emitError(op)
        << "tensor-form operand: the interpreter only accepts fully "
           "bufferized (memref) HIVM IR; run the remaining bufferization "
           "passes first";
    return false;
  }
  return interp.getMemRefOperand(core, srcValue, src, op) &&
         interp.getMemRefOperand(core, dstValue, dst, op);
}

//===----------------------------------------------------------------------===//
// load / store / copy
//===----------------------------------------------------------------------===//

ExecResult execLoad(Interpreter &interp, CoreState &core, Operation *op) {
  auto loadOp = cast<hivm::LoadOp>(op);
  MemRefValue src, dst;
  if (!resolveCopyOperands(interp, core, op, loadOp.getSrc(), loadOp.getDst(),
                           src, dst))
    return ExecResult::Error;

  std::optional<RuntimeValue> padValue;
  if (loadOp.getPadMode()) {
    switch (loadOp.getPadMode()->getPadmode()) {
    case hivm::PadMode::PadValue:
      if (loadOp.getPadValue())
        padValue = interp.getValue(core, loadOp.getPadValue());
      break;
    case hivm::PadMode::PadFirstElem: {
      SmallVector<int64_t, 4> zero(src.getRank(), 0);
      RuntimeValue first;
      if (rawLoadAt(interp, src, zero, op, first))
        padValue = first;
      break;
    }
    case hivm::PadMode::PadNull:
      break;
    }
  }

  int64_t leftPadding = 0;
  if (loadOp.getLeftPaddingNum() &&
      !getIndex(interp, core, loadOp.getLeftPaddingNum(), op, leftPadding))
    return ExecResult::Error;

  return issueCopy(interp, core, op, Pipe::MTE2, src, dst, padValue,
                   leftPadding);
}

ExecResult execStore(Interpreter &interp, CoreState &core, Operation *op) {
  auto storeOp = cast<hivm::StoreOp>(op);
  MemRefValue src, dst;
  if (!resolveCopyOperands(interp, core, op, storeOp.getSrc(),
                           storeOp.getDst(), src, dst))
    return ExecResult::Error;

  auto atomicKind = storeOp.getAtomicKind();
  if (!atomicKind || *atomicKind == hivm::AtomicKind::NONE)
    return issueCopy(interp, core, op, Pipe::MTE3, src, dst, std::nullopt, 0);

  // Atomic store: read-modify-write against whatever is already in GM. The
  // combine happens at commit time, so two cores racing on the same range are
  // still reported by the shadow memory.
  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  interp.collectRanges(dst, writes);
  // An atomic store also reads its destination.
  interp.collectRanges(dst, reads);

  hivm::AtomicKind kind = *atomicKind;
  Interpreter *interpPtr = &interp;
  interp.issueEffect(
      core, Pipe::MTE3, op, reads, writes,
      [interpPtr, op, src, dst, kind]() {
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          RuntimeValue incoming, existing;
          if (!rawLoadAt(*interpPtr, src, index, op, incoming) ||
              !rawLoadAt(*interpPtr, dst, index, op, existing)) {
            failed = true;
            return;
          }
          incoming = convertForCopy(incoming, src.elemType, dst.elemType);
          RuntimeValue result = incoming;
          if (existing.isFloat() && incoming.isFloat()) {
            llvm::APFloat a = existing.getFloatValue();
            const llvm::APFloat &b = incoming.getFloatValue();
            switch (kind) {
            case hivm::AtomicKind::ADD:
              a.add(b, llvm::APFloat::rmNearestTiesToEven);
              break;
            case hivm::AtomicKind::MAX:
              a = a > b ? a : b;
              break;
            case hivm::AtomicKind::MIN:
              a = a < b ? a : b;
              break;
            default:
              a = b;
              break;
            }
            result = RuntimeValue::getFloat(a);
          } else if (existing.isInt() && incoming.isInt()) {
            llvm::APInt a = existing.getIntValue();
            const llvm::APInt &b = incoming.getIntValue();
            switch (kind) {
            case hivm::AtomicKind::ADD:
              a = a + b;
              break;
            case hivm::AtomicKind::MAX:
              a = a.sgt(b) ? a : b;
              break;
            case hivm::AtomicKind::MIN:
              a = a.slt(b) ? a : b;
              break;
            case hivm::AtomicKind::UMAX:
              a = a.ugt(b) ? a : b;
              break;
            case hivm::AtomicKind::UMIN:
              a = a.ult(b) ? a : b;
              break;
            case hivm::AtomicKind::AND:
              a = a & b;
              break;
            case hivm::AtomicKind::OR:
              a = a | b;
              break;
            case hivm::AtomicKind::XOR:
              a = a ^ b;
              break;
            default:
              a = b;
              break;
            }
            result = RuntimeValue::getInt(a);
          }
          if (!rawStoreAt(*interpPtr, dst, index, op, result))
            failed = true;
        });
      },
      /*isAtomic=*/true);
  return ExecResult::Advance;
}

ExecResult execCopy(Interpreter &interp, CoreState &core, Operation *op) {
  auto copyOp = cast<hivm::CopyOp>(op);
  MemRefValue src, dst;
  if (!resolveCopyOperands(interp, core, op, copyOp.getSrc(), copyOp.getDst(),
                           src, dst))
    return ExecResult::Error;
  if (!interp.checkLayout(op, src, dst))
    return ExecResult::Error;

  std::optional<RuntimeValue> padValue;
  if (copyOp.getPadMode() &&
      copyOp.getPadMode()->getPadmode() == hivm::PadMode::PadValue &&
      copyOp.getPadValue())
    padValue = interp.getValue(core, copyOp.getPadValue());

  // The pipe depends on the source/destination pair, so ask the op.
  Pipe pipe = getOpPipe(op, Pipe::MTE3);
  return issueCopy(interp, core, op, pipe, src, dst, padValue, 0);
}

//===----------------------------------------------------------------------===//
// Layout-changing DMA
//===----------------------------------------------------------------------===//

/// nd2nz / nz2nd / fixpipe(nz2nd) / l12ub. Stage 1 of the plan's layout
/// strategy: data stays in logical ND order and only the tag moves, which
/// catches mismatched producers and consumers at negligible cost. Byte-exact
/// fractal addressing is behind --exact-layout.
ExecResult issueLayoutCopy(Interpreter &interp, CoreState &core, Operation *op,
                           Pipe pipe, Value srcValue, Value dstValue,
                           LayoutTag resultTag) {
  MemRefValue src, dst;
  if (!resolveCopyOperands(interp, core, op, srcValue, dstValue, src, dst))
    return ExecResult::Error;

  if (interp.getOptions().exactLayout) {
    interp.emitError(op) << "--exact-layout is not implemented yet; run "
                            "without it to use logical-ND layout checking";
    return ExecResult::Error;
  }

  // Record the destination's new tag so a later consumer can be checked.
  RuntimeValue dstValueRt = interp.getValue(core, dstValue);
  if (dstValueRt.isMemRef()) {
    MemRefValue tagged = dstValueRt.getMemRefValue();
    tagged.layout = resultTag;
    interp.setValue(core, dstValue, RuntimeValue::getMemRef(tagged));
    dst.layout = resultTag;
  }

  SmallVector<ByteRange, 4> reads, writes;
  interp.collectRanges(src, reads);
  interp.collectRanges(dst, writes);

  Interpreter *interpPtr = &interp;
  interp.issueEffect(core, pipe, op, reads, writes,
                     [interpPtr, op, src, dst]() {
                       // Rank-4 fractal views describe physical strides. In
                       // stage-1 layout mode their bytes instead stay in
                       // dense logical order; only the layout tag changes.
                       auto logicalStorage = [](MemRefValue mem) {
                         if (mem.getRank() != 4)
                           return mem;
                         int64_t stride = 1;
                         for (int64_t d = mem.getRank() - 1; d >= 0; --d) {
                           mem.strides[d] = stride;
                           stride *= mem.sizes[d];
                         }
                         return mem;
                       };
                       MemRefValue logicalSrc = logicalStorage(src);
                       MemRefValue logicalDst = logicalStorage(dst);
                       int64_t srcCount = logicalSrc.getNumElements();
                       int64_t dstCount = logicalDst.getNumElements();
                       const llvm::fltSemantics *sem =
                           getFloatSemantics(logicalDst.elemType);
                       RuntimeValue zero = sem
                                               ? RuntimeValue::getFloat(
                                                     llvm::APFloat::getZero(*sem))
                                               : RuntimeValue::getInt(llvm::APInt(
                                                     logicalDst.elemBytes * 8,
                                                     0));
                       for (int64_t n = 0; n < dstCount; ++n) {
                         RuntimeValue value;
                         if (n < srcCount) {
                           if (!rawLoad(*interpPtr, logicalSrc, n, op, value))
                             return;
                           value = convertForCopy(value, logicalSrc.elemType,
                                                  logicalDst.elemType);
                         } else {
                           // A tail ND tile occupies a whole fractal block.
                           // Stage-1 logical layout represents that padding as
                           // trailing zeroes rather than physical NZ holes.
                           value = zero;
                         }
                         if (!rawStore(*interpPtr, logicalDst, n, op, value))
                           return;
                       }
                     });
  return ExecResult::Advance;
}

ExecResult execFixpipe(Interpreter &interp, CoreState &core, Operation *op) {
  auto fixpipeOp = cast<hivm::FixpipeOp>(op);
  // NZ2ND / NZ2DN land in ND; NZ2NZ (normal) keeps the fractal layout.
  LayoutTag tag = fixpipeOp.getDmaMode() == hivm::FixpipeDMAMode::NZ2NZ
                      ? LayoutTag::NZ
                      : LayoutTag::ND;

  auto dual = fixpipeOp.getDualDstMode();
  if (dual && dual.getDualDstMode() == hivm::FixpipeDualDstMode::ROW_SPLIT) {
    if (interp.getOptions().subBlockNum != 2) {
      interp.emitError(op) << "ROW_SPLIT fixpipe requires --sub-block-num=2";
      return ExecResult::Error;
    }

    MemRefValue src, dst;
    if (!resolveCopyOperands(interp, core, op, fixpipeOp.getSrc(),
                             fixpipeOp.getDst(), src, dst))
      return ExecResult::Error;
    if (dst.space != AddrSpace::UB) {
      interp.emitError(op) << "ROW_SPLIT fixpipe destination must be UB";
      return ExecResult::Error;
    }

    RuntimeValue dstRuntime = interp.getValue(core, fixpipeOp.getDst());
    if (dstRuntime.isMemRef()) {
      MemRefValue tagged = dstRuntime.getMemRefValue();
      tagged.layout = tag;
      interp.setValue(core, fixpipeOp.getDst(), RuntimeValue::getMemRef(tagged));
      dst.layout = tag;
    }

    SmallVector<MemRefValue, 2> destinations;
    SmallVector<ByteRange, 4> reads, writes;
    interp.collectRanges(src, reads);
    for (unsigned sub = 0; sub < 2; ++sub) {
      MemRefValue laneDst = dst;
      CoreId lane = core.id;
      lane.kind = CoreKind::AIV;
      lane.subBlockIdx = sub;
      laneDst.arena = interp.getArenaId(AddrSpace::UB, lane);
      destinations.push_back(laneDst);
      interp.collectRanges(laneDst, writes);
    }

    int64_t srcCount = src.getNumElements();
    int64_t laneCount = dst.getNumElements();
    if (srcCount < 2 * laneCount) {
      interp.emitError(op) << "ROW_SPLIT source has " << srcCount
                           << " elements but two destinations require "
                           << 2 * laneCount;
      return ExecResult::Error;
    }

    Interpreter *interpPtr = &interp;
    interp.issueEffect(
        core, Pipe::FIX, op, reads, writes,
        [interpPtr, op, src, destinations, laneCount]() {
          auto logicalStorage = [](MemRefValue mem) {
            if (mem.getRank() != 4)
              return mem;
            int64_t stride = 1;
            for (int64_t d = mem.getRank() - 1; d >= 0; --d) {
              mem.strides[d] = stride;
              stride *= mem.sizes[d];
            }
            return mem;
          };
          MemRefValue logicalSrc = logicalStorage(src);
          for (auto [sub, laneDst] : llvm::enumerate(destinations)) {
            for (int64_t n = 0; n < laneCount; ++n) {
              RuntimeValue value;
              if (!rawLoad(*interpPtr, logicalSrc,
                           static_cast<int64_t>(sub) * laneCount + n, op,
                           value))
                return;
              value = convertForCopy(value, logicalSrc.elemType,
                                     laneDst.elemType);
              if (!rawStore(*interpPtr, laneDst, n, op, value))
                return;
            }
          }
        });
    return ExecResult::Advance;
  }
  return issueLayoutCopy(interp, core, op, Pipe::FIX, fixpipeOp.getSrc(),
                         fixpipeOp.getDst(), tag);
}

//===----------------------------------------------------------------------===//
// View family
//===----------------------------------------------------------------------===//

/// Map a `hivm::DataLayout` onto the interpreter's coarser tag.
LayoutTag toLayoutTag(hivm::DataLayout layout) {
  switch (layout) {
  case hivm::DataLayout::ND:
    return LayoutTag::ND;
  case hivm::DataLayout::nZ:
    return LayoutTag::NZ;
  case hivm::DataLayout::zN:
  case hivm::DataLayout::Fractal:
    return LayoutTag::ZN;
  case hivm::DataLayout::DOTA_ND:
  case hivm::DataLayout::SCALEA_ND:
  case hivm::DataLayout::SCALEA_zZ:
    return LayoutTag::DOTA_ND;
  case hivm::DataLayout::DOTB_ND:
  case hivm::DataLayout::SCALEB_DN:
  case hivm::DataLayout::SCALEB_nN:
    return LayoutTag::DOTB_ND;
  case hivm::DataLayout::DOTC_ND:
    return LayoutTag::DOTC_ND;
  }
  return LayoutTag::ND;
}

/// `convert_layout` moves no data; it relabels the view. Stage-1 layout
/// checking therefore just verifies the declared source layout against what
/// the producer actually left behind, and retags the result.
ExecResult execConvertLayout(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto convertOp = cast<hivm::ConvertLayoutOp>(op);
  MemRefValue source;
  if (!interp.getMemRefOperand(core, convertOp.getSource(), source, op))
    return ExecResult::Error;

  LayoutTag declared = toLayoutTag(convertOp.getSrcLayout().getDataLayout());
  if (source.layout != declared) {
    interp.emitError(op) << "layout mismatch: convert_layout declares its "
                            "source is "
                         << getLayoutTagName(declared)
                         << " but the producer left it as "
                         << getLayoutTagName(source.layout);
    return ExecResult::Error;
  }

  MemRefValue result = source;
  result.layout = toLayoutTag(convertOp.getDstLayout().getDataLayout());
  auto resultType = dyn_cast<MemRefType>(convertOp.getResult().getType());
  if (resultType && resultType.hasStaticShape()) {
    result.sizes.assign(resultType.getShape().begin(),
                        resultType.getShape().end());
    result.strides.resize(result.sizes.size());
    int64_t acc = 1;
    for (int64_t d = static_cast<int64_t>(result.sizes.size()) - 1; d >= 0;
         --d) {
      result.strides[d] = acc;
      acc *= result.sizes[d];
    }
  }
  interp.setValue(core, convertOp.getResult(), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

ExecResult execPointerCast(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto castOp = cast<hivm::PointerCastOp>(op);
  if (castOp.getAddrs().empty()) {
    interp.emitError(op) << "pointer_cast without an address operand";
    return ExecResult::Error;
  }
  if (castOp.getAddrs().size() > 1) {
    // Several addresses describe a tightly-coupled multi-bank buffer; the
    // interpreter has no bank model, so refuse rather than silently using
    // only the first bank.
    interp.emitError(op)
        << "multi-address pointer_cast (tightly coupled buffer) is not "
           "modelled";
    return ExecResult::Error;
  }

  RuntimeValue addr = interp.getValue(core, castOp.getAddrs().front());
  if (!addr.isInt()) {
    interp.emitError(op) << "pointer_cast address is unbound";
    return ExecResult::Error;
  }

  auto resultType = cast<MemRefType>(castOp.getResult().getType());
  MemRefValue result;
  result.elemType = resultType.getElementType();
  result.elemBytes = getStorageSize(result.elemType);
  result.space = getAddrSpaceOf(resultType);
  result.arena = interp.getArenaId(result.space, core.id);
  result.byteOffset = addr.getIntValue().getZExtValue();
  result.baseOffset = result.byteOffset;

  unsigned dynIdx = 0;
  for (int64_t dim : resultType.getShape()) {
    if (!ShapedType::isDynamic(dim)) {
      result.sizes.push_back(dim);
      continue;
    }
    int64_t value = 0;
    if (dynIdx >= castOp.getDynamicSizes().size() ||
        !getIndex(interp, core, castOp.getDynamicSizes()[dynIdx++], op, value))
      return ExecResult::Error;
    result.sizes.push_back(value);
  }
  result.strides.resize(result.sizes.size());
  int64_t acc = 1;
  for (int64_t d = static_cast<int64_t>(result.sizes.size()) - 1; d >= 0; --d) {
    result.strides[d] = acc;
    acc *= result.sizes[d];
  }

  // A baked address must still land inside its pool.
  uint64_t bytes = static_cast<uint64_t>(result.getNumElements()) *
                   result.elemBytes;
  Arena &arena = interp.getArena(result.arena);
  if (!arena.inBounds(result.byteOffset, bytes)) {
    interp.emitError(op) << "pointer_cast address " << result.byteOffset
                         << " + " << bytes << " bytes runs past "
                         << getAddrSpaceName(result.space) << " (capacity "
                         << arena.getCapacity() << ")";
    return ExecResult::Error;
  }

  interp.setValue(core, castOp.getResult(), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

/// `hivm.hir.bitcast` reinterprets the element type in place.
ExecResult execBitcast(Interpreter &interp, CoreState &core, Operation *op) {
  MemRefValue source;
  if (!interp.getMemRefOperand(core, op->getOperand(0), source, op))
    return ExecResult::Error;
  auto resultType = dyn_cast<MemRefType>(op->getResult(0).getType());
  if (resultType) {
    unsigned oldBytes = source.elemBytes;
    source.elemType = resultType.getElementType();
    source.elemBytes = getStorageSize(source.elemType);
    if (source.elemBytes != oldBytes) {
      interp.emitError(op)
          << "bitcast between element types of different sizes is not "
             "modelled";
      return ExecResult::Error;
    }
  }
  interp.setValue(core, op->getResult(0), RuntimeValue::getMemRef(source));
  return ExecResult::Advance;
}

} // namespace

void registerHIVMMemoryOps(OpRegistry &registry) {
  registry.add(hivm::LoadOp::getOperationName(), execLoad);
  registry.add(hivm::StoreOp::getOperationName(), execStore);
  registry.add(hivm::CopyOp::getOperationName(), execCopy);
  registry.add(hivm::FixpipeOp::getOperationName(), execFixpipe);

  registry.add(hivm::ND2NZOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<hivm::ND2NZOp>(op);
                 return issueLayoutCopy(interp, core, op, Pipe::MTE2,
                                        typed.getSrc(), typed.getDst(),
                                        LayoutTag::NZ);
               });
  registry.add(hivm::NZ2NDOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<hivm::NZ2NDOp>(op);
                 return issueLayoutCopy(interp, core, op, Pipe::MTE3,
                                        typed.getSrc(), typed.getDst(),
                                        LayoutTag::ND);
               });
  registry.add(hivm::L12UBOp::getOperationName(),
               [](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<hivm::L12UBOp>(op);
                 return issueLayoutCopy(interp, core, op, Pipe::MTE1,
                                        typed.getSrc(), typed.getDst(),
                                        LayoutTag::ND);
               });

  registry.add(hivm::ConvertLayoutOp::getOperationName(), execConvertLayout);
  registry.add(hivm::PointerCastOp::getOperationName(), execPointerCast);
  registry.add(hivm::BitcastOp::getOperationName(), execBitcast);
}

} // namespace interp
} // namespace bishengir
