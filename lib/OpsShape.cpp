//===- OpsShape.cpp - HIVM data-movement and scan op handlers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ops that rearrange elements rather than compute on them (flip, concat, pad,
// gather, interleave, sort) plus the prefix-scan family. They share a shape:
// resolve the operands, queue one PIPE_V effect, and do the shuffling inside
// the commit closure.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <algorithm>
#include <numeric>

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

/// Normalise a possibly-negative axis against `rank`.
int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

/// Resolve every memref in `values`, returning false after the handler has
/// already emitted a diagnostic.
bool getMemRefs(Interpreter &interp, CoreState &core, Operation *op,
                ValueRange values, SmallVectorImpl<MemRefValue> &out) {
  for (Value v : values) {
    MemRefValue mem;
    if (isa<RankedTensorType>(v.getType())) {
      interp.emitError(op) << "tensor operand: the interpreter only accepts "
                              "fully bufferized (memref) HIVM IR";
      return false;
    }
    if (!interp.getMemRefOperand(core, v, mem, op))
      return false;
    out.push_back(std::move(mem));
  }
  return true;
}

/// Issue one PIPE_V effect reading `srcs` and writing `dsts`.
void issueShapeEffect(Interpreter &interp, CoreState &core, Operation *op,
                      ArrayRef<MemRefValue> srcs, ArrayRef<MemRefValue> dsts,
                      std::function<void()> body) {
  SmallVector<ByteRange, 4> reads, writes;
  for (const MemRefValue &m : srcs)
    interp.collectRanges(m, reads);
  for (const MemRefValue &m : dsts)
    interp.collectRanges(m, writes);
  interp.issueEffect(core, getOpPipe(op, Pipe::V), op, reads, writes,
                     std::move(body));
}

//===----------------------------------------------------------------------===//
// vflip
//===----------------------------------------------------------------------===//

ExecResult execVFlip(Interpreter &interp, CoreState &core, Operation *op) {
  auto flipOp = cast<hivm::VFlipOp>(op);
  SmallVector<MemRefValue, 2> mems;
  if (!getMemRefs(interp, core, op, {flipOp.getSrc(), flipOp.getDst()}, mems))
    return ExecResult::Error;
  MemRefValue src = mems[0], dst = mems[1];

  int64_t axis = normalizeAxis(flipOp.getFlipAxis(), dst.getRank());
  if (axis < 0 || axis >= dst.getRank()) {
    interp.emitError(op) << "vflip axis " << flipOp.getFlipAxis()
                         << " is out of range for a rank-" << dst.getRank()
                         << " destination";
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(interp, core, op, {src}, {dst}, [ip, op, src, dst, axis]() {
    SmallVector<int64_t, 4> srcIndex;
    bool failed = false;
    forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
      if (failed)
        return;
      srcIndex.assign(index.begin(), index.end());
      srcIndex[axis] = src.sizes[axis] - 1 - index[axis];
      RuntimeValue value;
      if (!rawLoadAt(*ip, src, srcIndex, op, value) ||
          !rawStoreAt(*ip, dst, index, op, value))
        failed = true;
    });
  });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// vconcat
//===----------------------------------------------------------------------===//

ExecResult execVConcat(Interpreter &interp, CoreState &core, Operation *op) {
  auto concatOp = cast<hivm::VConcatOp>(op);
  SmallVector<MemRefValue, 4> srcs;
  if (!getMemRefs(interp, core, op, concatOp.getSrc(), srcs))
    return ExecResult::Error;
  SmallVector<MemRefValue, 1> dsts;
  if (!getMemRefs(interp, core, op, {concatOp.getDst()}, dsts))
    return ExecResult::Error;
  MemRefValue dst = dsts[0];

  int64_t dim = normalizeAxis(concatOp.getDim(), dst.getRank());
  if (dim < 0 || dim >= dst.getRank()) {
    interp.emitError(op) << "vconcat dim " << concatOp.getDim()
                         << " is out of range for a rank-" << dst.getRank()
                         << " destination";
    return ExecResult::Error;
  }
  int64_t total = 0;
  for (const MemRefValue &s : srcs) {
    if (s.getRank() != dst.getRank()) {
      interp.emitError(op) << "vconcat operands must all have the "
                              "destination's rank";
      return ExecResult::Error;
    }
    total += s.sizes[dim];
  }
  if (total != dst.sizes[dim]) {
    interp.emitError(op) << "vconcat sources span " << total
                         << " along dim " << dim << " but the destination is "
                         << dst.sizes[dim];
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(interp, core, op, srcs, {dst},
                   [ip, op, srcs, dst, dim]() {
                     int64_t base = 0;
                     bool failed = false;
                     for (const MemRefValue &src : srcs) {
                       SmallVector<int64_t, 4> dstIndex;
                       forEachIndex(src.sizes, [&](ArrayRef<int64_t> index) {
                         if (failed)
                           return;
                         dstIndex.assign(index.begin(), index.end());
                         dstIndex[dim] += base;
                         RuntimeValue value;
                         if (!rawLoadAt(*ip, src, index, op, value) ||
                             !rawStoreAt(*ip, dst, dstIndex, op, value))
                           failed = true;
                       });
                       base += src.sizes[dim];
                     }
                   });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// vpad
//===----------------------------------------------------------------------===//

ExecResult execVPad(Interpreter &interp, CoreState &core, Operation *op) {
  auto padOp = cast<hivm::VPadOp>(op);
  SmallVector<MemRefValue, 2> mems;
  if (!getMemRefs(interp, core, op, {padOp.getSrc(), padOp.getDst()}, mems))
    return ExecResult::Error;
  MemRefValue src = mems[0], dst = mems[1];

  SmallVector<int64_t, 4> low, high;
  for (OpFoldResult ofr : padOp.getMixedLowPad()) {
    int64_t v = 0;
    if (!getFoldedIndex(interp, core, ofr, op, v))
      return ExecResult::Error;
    low.push_back(v);
  }
  for (OpFoldResult ofr : padOp.getMixedHighPad()) {
    int64_t v = 0;
    if (!getFoldedIndex(interp, core, ofr, op, v))
      return ExecResult::Error;
    high.push_back(v);
  }
  if (static_cast<int64_t>(low.size()) != dst.getRank() ||
      static_cast<int64_t>(high.size()) != dst.getRank()) {
    interp.emitError(op) << "vpad needs one low/high pair per destination "
                            "dimension";
    return ExecResult::Error;
  }

  RuntimeValue pad = interp.getValue(core, padOp.getPadValue());
  if (pad.isNone()) {
    interp.emitError(op) << "vpad value is unbound";
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(interp, core, op, {src}, {dst},
                   [ip, op, src, dst, low, high, pad]() {
                     SmallVector<int64_t, 4> srcIndex;
                     bool failed = false;
                     forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
                       if (failed)
                         return;
                       bool inside = true;
                       srcIndex.assign(index.begin(), index.end());
                       for (int64_t d = 0; d < dst.getRank(); ++d) {
                         srcIndex[d] = index[d] - low[d];
                         if (srcIndex[d] < 0 || srcIndex[d] >= src.sizes[d])
                           inside = false;
                       }
                       RuntimeValue value = pad;
                       if (inside && !rawLoadAt(*ip, src, srcIndex, op, value))
                         failed = true;
                       else if (!rawStoreAt(*ip, dst, index, op, value))
                         failed = true;
                     });
                   });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// vgather
//===----------------------------------------------------------------------===//

ExecResult execVGather(Interpreter &interp, CoreState &core, Operation *op) {
  auto gatherOp = cast<hivm::VGatherOp>(op);
  SmallVector<MemRefValue, 3> mems;
  if (!getMemRefs(interp, core, op,
                  {gatherOp.getSrc(), gatherOp.getIndices(), gatherOp.getDst()},
                  mems))
    return ExecResult::Error;
  MemRefValue src = mems[0], indices = mems[1], dst = mems[2];

  // `gather_axis` defaults to the last dimension, which -1 also names.
  int64_t axis = normalizeAxis(
      gatherOp.getGatherAxis().value_or(-1), dst.getRank());
  if (axis < 0 || axis >= dst.getRank()) {
    interp.emitError(op) << "vgather axis is out of range for a rank-"
                         << dst.getRank() << " destination";
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(
      interp, core, op, {src, indices}, {dst},
      [ip, op, src, indices, dst, axis]() {
        SmallVector<int64_t, 4> srcIndex, idxIndex;
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          // The index buffer is indexed the same way as the destination,
          // collapsing any dimension it does not carry.
          idxIndex.assign(index.begin(), index.end());
          for (int64_t d = 0; d < indices.getRank() &&
                              d < static_cast<int64_t>(idxIndex.size());
               ++d)
            if (indices.sizes[d] == 1)
              idxIndex[d] = 0;
          idxIndex.resize(indices.getRank(), 0);

          RuntimeValue position;
          if (!rawLoadAt(*ip, indices, idxIndex, op, position)) {
            failed = true;
            return;
          }
          int64_t at = position.getIndexValue();
          if (at < 0 || at >= src.sizes[axis]) {
            ip->emitError(op) << "vgather index " << at
                              << " is outside the source extent "
                              << src.sizes[axis] << " on axis " << axis;
            failed = true;
            return;
          }
          srcIndex.assign(index.begin(), index.end());
          srcIndex[axis] = at;
          RuntimeValue value;
          if (!rawLoadAt(*ip, src, srcIndex, op, value) ||
              !rawStoreAt(*ip, dst, index, op, value))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// vinterleave / vdeinterleave
//===----------------------------------------------------------------------===//

ExecResult execVInterleave(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto ilvOp = cast<hivm::VInterleaveOp>(op);
  SmallVector<MemRefValue, 4> srcs;
  if (!getMemRefs(interp, core, op, ilvOp.getSrc(), srcs))
    return ExecResult::Error;
  SmallVector<MemRefValue, 1> dsts;
  if (!getMemRefs(interp, core, op, {ilvOp.getDst()}, dsts))
    return ExecResult::Error;
  MemRefValue dst = dsts[0];
  if (srcs.empty()) {
    interp.emitError(op) << "vinterleave without sources";
    return ExecResult::Error;
  }

  int64_t last = dst.getRank() - 1;
  int64_t channels = static_cast<int64_t>(srcs.size());
  if (dst.sizes[last] != srcs[0].sizes[last] * channels) {
    interp.emitError(op) << "vinterleave destination has " << dst.sizes[last]
                         << " elements on the last dimension, expected "
                         << srcs[0].sizes[last] * channels;
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(interp, core, op, srcs, {dst},
                   [ip, op, srcs, dst, last, channels]() {
                     SmallVector<int64_t, 4> dstIndex;
                     bool failed = false;
                     for (int64_t c = 0; c < channels; ++c) {
                       const MemRefValue &src = srcs[c];
                       forEachIndex(src.sizes, [&](ArrayRef<int64_t> index) {
                         if (failed)
                           return;
                         dstIndex.assign(index.begin(), index.end());
                         dstIndex[last] = index[last] * channels + c;
                         RuntimeValue value;
                         if (!rawLoadAt(*ip, src, index, op, value) ||
                             !rawStoreAt(*ip, dst, dstIndex, op, value))
                           failed = true;
                       });
                     }
                   });
  return ExecResult::Advance;
}

ExecResult execVDeinterleave(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto deOp = cast<hivm::VDeinterleaveOp>(op);
  SmallVector<MemRefValue, 1> srcs;
  if (!getMemRefs(interp, core, op, {deOp.getSrc()}, srcs))
    return ExecResult::Error;
  SmallVector<MemRefValue, 4> dsts;
  if (!getMemRefs(interp, core, op, deOp.getDst(), dsts))
    return ExecResult::Error;
  MemRefValue src = srcs[0];
  if (dsts.empty()) {
    interp.emitError(op) << "vdeinterleave without destinations";
    return ExecResult::Error;
  }

  // `channel_num` is the interleaving factor of the source; `index_mode`
  // selects which channel a single destination receives. With ALL_CHANNELS
  // there is one destination per channel.
  int64_t channels = deOp.getChannelNum();
  int64_t firstChannel = 0;
  switch (deOp.getIndexMode()) {
  case hivm::DeinterleaveMode::CHANNEL_0:
    firstChannel = 0;
    break;
  case hivm::DeinterleaveMode::CHANNEL_1:
    firstChannel = 1;
    break;
  case hivm::DeinterleaveMode::ALL_CHANNELS:
    break;
  }
  if (deOp.getIndexMode() != hivm::DeinterleaveMode::ALL_CHANNELS &&
      dsts.size() != 1) {
    interp.emitError(op) << "vdeinterleave with a single-channel index_mode "
                            "needs exactly one destination";
    return ExecResult::Error;
  }
  if (deOp.getIndexMode() == hivm::DeinterleaveMode::ALL_CHANNELS &&
      static_cast<int64_t>(dsts.size()) != channels) {
    interp.emitError(op) << "vdeinterleave has " << dsts.size()
                         << " destinations but channel_num is " << channels;
    return ExecResult::Error;
  }

  int64_t last = src.getRank() - 1;
  Interpreter *ip = &interp;
  issueShapeEffect(
      interp, core, op, {src}, dsts,
      [ip, op, src, dsts, last, channels, firstChannel]() {
        SmallVector<int64_t, 4> srcIndex;
        bool failed = false;
        for (size_t d = 0; d < dsts.size(); ++d) {
          const MemRefValue &dst = dsts[d];
          int64_t channel = firstChannel + static_cast<int64_t>(d);
          forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
            if (failed)
              return;
            srcIndex.assign(index.begin(), index.end());
            srcIndex[last] = index[last] * channels + channel;
            RuntimeValue value;
            if (!rawLoadAt(*ip, src, srcIndex, op, value) ||
                !rawStoreAt(*ip, dst, index, op, value))
              failed = true;
          });
        }
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// vsort
//===----------------------------------------------------------------------===//

/// Strict weak ordering over runtime values, which `std::stable_sort` needs
/// and a `<` on floats does not provide: every comparison against a NaN is
/// false, so a NaN operand makes the comparator inconsistent and the sort
/// undefined. Ordering by the IEEE total order instead puts NaNs at the ends
/// and leaves everything else where `<` would. Integers compare as APInt
/// rather than through `double`, which loses bits above 2^53.
bool sortsBefore(const RuntimeValue &a, const RuntimeValue &b) {
  if (a.isFloat() && b.isFloat()) {
    const llvm::APFloat &x = a.getFloatValue();
    const llvm::APFloat &y = b.getFloatValue();
    if (x.isNaN() || y.isNaN())
      return !x.isNaN() && y.isNaN();
    return x < y;
  }
  if (a.isInt() && b.isInt())
    return a.getIntValue().slt(b.getIntValue());
  return false;
}

ExecResult execVSort(Interpreter &interp, CoreState &core, Operation *op) {
  auto sortOp = cast<hivm::VSortOp>(op);
  SmallVector<MemRefValue, 1> srcs;
  if (!getMemRefs(interp, core, op, {sortOp.getSrc()}, srcs))
    return ExecResult::Error;
  SmallVector<MemRefValue, 2> dsts;
  if (!getMemRefs(interp, core, op, sortOp.getDst(), dsts))
    return ExecResult::Error;
  MemRefValue src = srcs[0];
  if (dsts.empty()) {
    interp.emitError(op) << "vsort without a destination";
    return ExecResult::Error;
  }

  int64_t axis = normalizeAxis(sortOp.getSortAxis(), src.getRank());
  if (axis != src.getRank() - 1) {
    // The op documents "currently only tail axis sorting is supported"; say so
    // rather than quietly sorting something else.
    interp.emitError(op) << "vsort only supports the last axis, got axis "
                         << sortOp.getSortAxis();
    return ExecResult::Error;
  }
  bool descending = sortOp.getDescending();

  Interpreter *ip = &interp;
  issueShapeEffect(
      interp, core, op, {src}, dsts,
      [ip, op, src, dsts, axis, descending]() {
        // Iterate the outer dimensions; each innermost run is one sort.
        SmallVector<int64_t, 4> outer(src.sizes.begin(), src.sizes.end());
        int64_t n = outer[axis];
        outer[axis] = 1;
        bool failed = false;
        SmallVector<int64_t, 4> index;
        forEachIndex(outer, [&](ArrayRef<int64_t> base) {
          if (failed)
            return;
          SmallVector<RuntimeValue, 8> values(n);
          index.assign(base.begin(), base.end());
          for (int64_t i = 0; i < n; ++i) {
            index[axis] = i;
            if (!rawLoadAt(*ip, src, index, op, values[i])) {
              failed = true;
              return;
            }
          }
          SmallVector<int64_t, 8> order(n);
          std::iota(order.begin(), order.end(), 0);
          // Stable, so equal keys keep their input order and the index
          // output is reproducible.
          std::stable_sort(order.begin(), order.end(),
                           [&](int64_t a, int64_t b) {
                             bool less = sortsBefore(values[a], values[b]);
                             return descending
                                        ? sortsBefore(values[b], values[a])
                                        : less;
                           });
          for (int64_t i = 0; i < n; ++i) {
            index[axis] = i;
            if (!rawStoreAt(*ip, dsts[0], index, op, values[order[i]])) {
              failed = true;
              return;
            }
            if (dsts.size() > 1) {
              RuntimeValue pos = RuntimeValue::getInt(llvm::APInt(
                  getStorageSize(dsts[1].elemType) * 8,
                  static_cast<uint64_t>(order[i]), /*isSigned=*/true));
              if (!rawStoreAt(*ip, dsts[1], index, op, pos)) {
                failed = true;
                return;
              }
            }
          }
        });
      });
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Prefix scans: vcumsum / vcumprod / vcummax / vcummin
//===----------------------------------------------------------------------===//

enum class ScanKind { Sum, Prod, Max, Min };

RuntimeValue scanCombine(ScanKind kind, const RuntimeValue &acc,
                         const RuntimeValue &value) {
  if (acc.isFloat() && value.isFloat()) {
    llvm::APFloat a = acc.getFloatValue();
    const llvm::APFloat &b = value.getFloatValue();
    switch (kind) {
    case ScanKind::Sum:
      a.add(b, llvm::APFloat::rmNearestTiesToEven);
      break;
    case ScanKind::Prod:
      a.multiply(b, llvm::APFloat::rmNearestTiesToEven);
      break;
    case ScanKind::Max:
      a = llvm::maximum(a, b);
      break;
    case ScanKind::Min:
      a = llvm::minimum(a, b);
      break;
    }
    return RuntimeValue::getFloat(a);
  }
  llvm::APInt a = acc.getIntValue();
  const llvm::APInt &b = value.getIntValue();
  switch (kind) {
  case ScanKind::Sum:
    a = a + b;
    break;
  case ScanKind::Prod:
    a = a * b;
    break;
  case ScanKind::Max:
    a = a.sgt(b) ? a : b;
    break;
  case ScanKind::Min:
    a = a.slt(b) ? a : b;
    break;
  }
  return RuntimeValue::getInt(a);
}

/// Shared body of the four scans. `cum_dims` is a list, so the axes are
/// applied one after another - the first reading the source, the rest running
/// in place over the destination. The op's verifier currently accepts only one
/// entry, but the attribute says list and following it costs a loop.
ExecResult runScan(Interpreter &interp, CoreState &core, Operation *op,
                   ScanKind kind, Value srcValue, Value dstValue,
                   ArrayRef<int64_t> cumDims, bool reverse) {
  SmallVector<MemRefValue, 2> mems;
  if (!getMemRefs(interp, core, op, {srcValue, dstValue}, mems))
    return ExecResult::Error;
  MemRefValue src = mems[0], dst = mems[1];

  SmallVector<int64_t, 2> axes;
  for (int64_t d : cumDims) {
    int64_t axis = normalizeAxis(d, src.getRank());
    if (axis < 0 || axis >= src.getRank()) {
      interp.emitError(op) << "cum_dims entry " << d
                           << " is out of range for a rank-" << src.getRank()
                           << " source";
      return ExecResult::Error;
    }
    axes.push_back(axis);
  }
  if (axes.empty()) {
    interp.emitError(op) << "scan without cum_dims";
    return ExecResult::Error;
  }

  Interpreter *ip = &interp;
  issueShapeEffect(
      interp, core, op, {src}, {dst},
      [ip, op, src, dst, axes, kind, reverse]() {
        bool failed = false;
        for (size_t pass = 0; pass < axes.size(); ++pass) {
          // The first pass reads the source; any later one continues over
          // what the previous pass left in the destination.
          const MemRefValue &from = pass == 0 ? src : dst;
          int64_t axis = axes[pass];
          SmallVector<int64_t, 4> outer(dst.sizes.begin(), dst.sizes.end());
          int64_t n = outer[axis];
          outer[axis] = 1;
          SmallVector<int64_t, 4> index;
          forEachIndex(outer, [&](ArrayRef<int64_t> base) {
            if (failed)
              return;
            index.assign(base.begin(), base.end());
            RuntimeValue acc;
            for (int64_t step = 0; step < n; ++step) {
              // `reverse` scans from the far end, so element i accumulates
              // everything at or after it instead of at or before it.
              index[axis] = reverse ? n - 1 - step : step;
              RuntimeValue value;
              if (!rawLoadAt(*ip, from, index, op, value)) {
                failed = true;
                return;
              }
              acc = step == 0 ? value : scanCombine(kind, acc, value);
              if (!rawStoreAt(*ip, dst, index, op, acc)) {
                failed = true;
                return;
              }
            }
          });
        }
      });
  return ExecResult::Advance;
}

template <typename OpT>
void addScan(OpRegistry &registry, ScanKind kind) {
  registry.add(OpT::getOperationName(),
               [kind](Interpreter &interp, CoreState &core, Operation *op) {
                 auto typed = cast<OpT>(op);
                 return runScan(interp, core, op, kind, typed.getSrc(),
                                typed.getDst(), typed.getCumDims(),
                                typed.getReverse());
               });
}

} // namespace

void registerHIVMShapeOps(OpRegistry &registry) {
  registry.add(hivm::VFlipOp::getOperationName(), execVFlip);
  registry.add(hivm::VConcatOp::getOperationName(), execVConcat);
  registry.add(hivm::VPadOp::getOperationName(), execVPad);
  registry.add(hivm::VGatherOp::getOperationName(), execVGather);
  registry.add(hivm::VInterleaveOp::getOperationName(), execVInterleave);
  registry.add(hivm::VDeinterleaveOp::getOperationName(), execVDeinterleave);
  registry.add(hivm::VSortOp::getOperationName(), execVSort);

  addScan<hivm::VCumsumOp>(registry, ScanKind::Sum);
  addScan<hivm::VCumprodOp>(registry, ScanKind::Prod);
  addScan<hivm::VCummaxOp>(registry, ScanKind::Max);
  addScan<hivm::VCumminOp>(registry, ScanKind::Min);
}

} // namespace interp
} // namespace bishengir
