//===- OpUtils.cpp - Shared helpers for interpreter op handlers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"


using namespace mlir;

namespace bishengir {
namespace interp {

Type getScalarType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

uint8_t *elementPtr(Interpreter &interp, const MemRefValue &mem,
                    uint64_t byteAddr, Operation *op) {
  if (!mem.isValid()) {
    interp.emitError(op) << "access through an unbound memref";
    return nullptr;
  }
  Arena &arena = interp.getArena(mem.arena);
  uint8_t *ptr = arena.at(byteAddr, mem.elemBytes);
  if (!ptr) {
    interp.emitError(op) << "out of bounds access to "
                         << getAddrSpaceName(mem.space) << " at byte "
                         << byteAddr << " (arena capacity "
                         << arena.getCapacity() << ", element size "
                         << mem.elemBytes << ")";
  }
  return ptr;
}

bool rawLoad(Interpreter &interp, const MemRefValue &mem, int64_t linearPos,
             Operation *op, RuntimeValue &out) {
  uint8_t *ptr = elementPtr(interp, mem, mem.getByteAddrLinear(linearPos), op);
  if (!ptr)
    return false;
  out = loadElement(ptr, mem.elemType);
  return true;
}

bool rawStore(Interpreter &interp, const MemRefValue &mem, int64_t linearPos,
              Operation *op, const RuntimeValue &value) {
  uint8_t *ptr = elementPtr(interp, mem, mem.getByteAddrLinear(linearPos), op);
  if (!ptr)
    return false;
  storeElement(ptr, mem.elemType, value);
  return true;
}

bool rawLoadAt(Interpreter &interp, const MemRefValue &mem,
               ArrayRef<int64_t> indices, Operation *op, RuntimeValue &out) {
  uint8_t *ptr = elementPtr(interp, mem, mem.getByteAddr(indices), op);
  if (!ptr)
    return false;
  out = loadElement(ptr, mem.elemType);
  return true;
}

bool rawStoreAt(Interpreter &interp, const MemRefValue &mem,
                ArrayRef<int64_t> indices, Operation *op,
                const RuntimeValue &value) {
  uint8_t *ptr = elementPtr(interp, mem, mem.getByteAddr(indices), op);
  if (!ptr)
    return false;
  storeElement(ptr, mem.elemType, value);
  return true;
}

bool getIndex(Interpreter &interp, CoreState &core, Value value, Operation *op,
              int64_t &out) {
  RuntimeValue rv = interp.getValue(core, value);
  if (!rv.isInt()) {
    interp.emitError(op) << "expected a bound integer/index operand";
    return false;
  }
  out = rv.getIndexValue();
  return true;
}

bool getFoldedIndex(Interpreter &interp, CoreState &core, OpFoldResult ofr,
                    Operation *op, int64_t &out) {
  if (auto attr = dyn_cast<Attribute>(ofr)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      interp.emitError(op) << "expected an integer attribute";
      return false;
    }
    out = intAttr.getInt();
    return true;
  }
  return getIndex(interp, core, cast<Value>(ofr), op, out);
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

static llvm::APFloat::roundingMode toAPFloatRounding(InterpRoundMode mode) {
  switch (mode) {
  case InterpRoundMode::RINT:
    return llvm::APFloat::rmNearestTiesToEven;
  case InterpRoundMode::ROUND:
    return llvm::APFloat::rmNearestTiesToAway;
  case InterpRoundMode::FLOOR:
    return llvm::APFloat::rmTowardNegative;
  case InterpRoundMode::CEIL:
    return llvm::APFloat::rmTowardPositive;
  case InterpRoundMode::TRUNC:
  case InterpRoundMode::TRUNC_OVF:
    return llvm::APFloat::rmTowardZero;
  case InterpRoundMode::ODD:
    // APFloat has no round-to-odd mode; handled separately below.
    return llvm::APFloat::rmNearestTiesToEven;
  }
  return llvm::APFloat::rmNearestTiesToEven;
}

/// Round-to-odd (Von Neumann): round toward zero, then force the least
/// significant mantissa bit to 1 whenever the conversion was inexact. This is
/// the double-rounding-safe mode HIVM exposes as `odd`: narrowing through it
/// and then rounding to nearest gives the same answer as narrowing in one
/// correctly-rounded step, which is what makes it usable as an intermediate.
static llvm::APFloat convertRoundToOdd(const llvm::APFloat &in,
                                       const llvm::fltSemantics &sem) {
  llvm::APFloat truncated = in;
  bool losesInfo = false;
  llvm::APFloat::opStatus status =
      truncated.convert(sem, llvm::APFloat::rmTowardZero, &losesInfo);
  if (!losesInfo && status == llvm::APFloat::opOK)
    return truncated;
  if (truncated.isInfinity() || truncated.isNaN())
    return truncated;
  if (truncated.isZero())
    // The input underflowed. Answering zero would throw away the very fact
    // round-to-odd exists to preserve - a later rounding step could no longer
    // tell "exactly zero" from "too small to represent" - so the result is
    // the smallest subnormal, whose significand is already odd.
    return llvm::APFloat::getSmallest(sem, in.isNegative());
  llvm::APInt bits = truncated.bitcastToAPInt();
  bits.setBit(0);
  return llvm::APFloat(sem, bits);
}

RuntimeValue convertValue(const RuntimeValue &in, Type fromType, Type toType,
                          InterpRoundMode mode, bool isUnsigned) {
  const llvm::fltSemantics *toSem = getFloatSemantics(toType);
  const llvm::fltSemantics *fromSem = getFloatSemantics(fromType);

  if (fromSem && toSem) {
    llvm::APFloat value = in.getFloatValue();
    if (mode == InterpRoundMode::ODD)
      return RuntimeValue::getFloat(convertRoundToOdd(value, *toSem));
    // FLOOR/CEIL/TRUNC/RINT on a same-or-wider float mean "round to an
    // integral value", not "change format". Apply the integral rounding first
    // when the target is not narrower.
    if (mode != InterpRoundMode::RINT || fromSem == toSem) {
      llvm::APFloat integral = value;
      switch (mode) {
      case InterpRoundMode::FLOOR:
      case InterpRoundMode::CEIL:
      case InterpRoundMode::TRUNC:
      case InterpRoundMode::ROUND:
        integral.roundToIntegral(toAPFloatRounding(mode));
        value = integral;
        break;
      default:
        break;
      }
    }
    bool losesInfo = false;
    value.convert(*toSem, toAPFloatRounding(mode), &losesInfo);
    return RuntimeValue::getFloat(value);
  }

  if (fromSem && !toSem) {
    // Float -> integer.
    unsigned width = 64;
    if (auto intType = dyn_cast<IntegerType>(toType))
      width = intType.getWidth();
    llvm::APFloat value = in.getFloatValue();
    if (value.isNaN())
      return RuntimeValue::getInt(llvm::APInt(width, 0));
    llvm::APFloat rounded = value;
    if (mode != InterpRoundMode::TRUNC && mode != InterpRoundMode::TRUNC_OVF)
      rounded.roundToIntegral(toAPFloatRounding(mode));
    llvm::APSInt result(width, isUnsigned);
    bool exact = false;
    rounded.convertToInteger(result, llvm::APFloat::rmTowardZero, &exact);
    return RuntimeValue::getInt(llvm::APInt(result));
  }

  if (!fromSem && toSem) {
    // Integer -> float.
    llvm::APFloat value(*toSem);
    value.convertFromAPInt(in.getIntValue(), !isUnsigned,
                           toAPFloatRounding(mode));
    return RuntimeValue::getFloat(value);
  }

  // Integer -> integer.
  unsigned width = 64;
  if (auto intType = dyn_cast<IntegerType>(toType))
    width = intType.getWidth();
  llvm::APInt value = in.getIntValue();
  if (value.getBitWidth() < width)
    value = isUnsigned ? value.zext(width) : value.sext(width);
  else if (value.getBitWidth() > width)
    value = value.trunc(width);
  return RuntimeValue::getInt(value);
}

//===----------------------------------------------------------------------===//
// Iteration
//===----------------------------------------------------------------------===//

void forEachIndex(ArrayRef<int64_t> sizes,
                  llvm::function_ref<void(ArrayRef<int64_t>)> body) {
  int64_t rank = static_cast<int64_t>(sizes.size());
  int64_t total = 1;
  for (int64_t s : sizes) {
    if (s <= 0)
      return;
    total *= s;
  }
  SmallVector<int64_t, 4> index(rank, 0);
  for (int64_t n = 0; n < total; ++n) {
    body(index);
    for (int64_t d = rank - 1; d >= 0; --d) {
      if (++index[d] < sizes[d])
        break;
      index[d] = 0;
    }
  }
}

void mapSourceIndex(ArrayRef<int64_t> dstIndex, const MemRefValue &src,
                    ArrayRef<int64_t> transpose,
                    SmallVectorImpl<int64_t> &out) {
  int64_t srcRank = src.getRank();
  out.assign(srcRank, 0);
  int64_t dstRank = static_cast<int64_t>(dstIndex.size());
  // Align trailing dimensions, NumPy-style, so a lower-rank source still
  // indexes correctly.
  for (int64_t d = 0; d < srcRank; ++d) {
    int64_t dstDim = dstRank - srcRank + d;
    int64_t value = dstDim >= 0 ? dstIndex[dstDim] : 0;
    if (!transpose.empty() && d < static_cast<int64_t>(transpose.size())) {
      int64_t permuted = transpose[d];
      if (permuted >= 0 && permuted < dstRank)
        value = dstIndex[permuted];
    }
    // An extent of 1 against a larger destination extent is a broadcast.
    out[d] = src.sizes[d] == 1 ? 0 : value;
  }
}

Pipe getOpPipe(Operation *op, Pipe fallback) {
  if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
    Pipe pipe = fallback;
    if (pipeOp.isSinglePipeOp() &&
        convertPipe(static_cast<int32_t>(pipeOp.getPipe()), pipe))
      return pipe;
    if (pipeOp.isMacroOp() &&
        convertPipe(static_cast<int32_t>(pipeOp.getOutPipe()), pipe))
      return pipe;
  }
  return fallback;
}

Pipe getOpInPipe(Operation *op, Pipe fallback) {
  auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op);
  if (!pipeOp || !pipeOp.isMacroOp())
    return fallback;
  Pipe pipe = fallback;
  if (convertPipe(static_cast<int32_t>(pipeOp.getInPipe()), pipe))
    return pipe;
  return fallback;
}

} // namespace interp
} // namespace bishengir
