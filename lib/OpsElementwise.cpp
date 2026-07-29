//===- OpsElementwise.cpp - HIVM elementwise op handlers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HIVM has ~60 elementwise vector ops with identical structure, so they share
// one driver and differ only by a scalar lambda. Arithmetic goes through
// APInt/APFloat: emulating f16/bf16 with `float` gets the tie behaviour and
// the round-to-odd mode wrong, and a 1-ulp interpreter bug is far harder to
// track down than a missing op.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include <cmath>

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

/// Scalar kernel of an elementwise op: `srcs` are already read in the
/// destination's element type domain.
using EwFn = std::function<RuntimeValue(ArrayRef<RuntimeValue>, Type,
                                        Operation *)>;

/// One resolved elementwise source: either a memref view or a splat scalar.
struct EwSource {
  bool isScalar = false;
  MemRefValue mem;
  RuntimeValue scalar;
  Type elemType;
};

/// Resolve the operands of an elementwise op and queue its effect on PIPE_V.
ExecResult runElementwise(Interpreter &interp, CoreState &core, Operation *op,
                          OperandRange srcs, OperandRange dsts,
                          ArrayRef<int64_t> transpose, const EwFn &fn) {
  if (dsts.empty()) {
    interp.emitError(op) << "elementwise op without a destination";
    return ExecResult::Error;
  }

  MemRefValue dst;
  if (!interp.getMemRefOperand(core, dsts.front(), dst, op))
    return ExecResult::Error;

  SmallVector<EwSource, 3> sources;
  for (Value src : srcs) {
    EwSource entry;
    if (isa<MemRefType>(src.getType())) {
      if (!interp.getMemRefOperand(core, src, entry.mem, op))
        return ExecResult::Error;
      entry.elemType = entry.mem.elemType;
    } else {
      entry.isScalar = true;
      entry.scalar = interp.getValue(core, src);
      entry.elemType = src.getType();
      if (entry.scalar.isNone()) {
        interp.emitError(op) << "scalar operand is unbound";
        return ExecResult::Error;
      }
    }
    sources.push_back(std::move(entry));
  }

  SmallVector<ByteRange, 4> reads, writes;
  for (const EwSource &src : sources)
    if (!src.isScalar)
      interp.collectRanges(src.mem, reads);
  interp.collectRanges(dst, writes);

  SmallVector<int64_t, 4> transposeVec(transpose.begin(), transpose.end());
  Interpreter *interpPtr = &interp;

  // Vector ops run on PIPE_V, so their result is only observable once V has
  // been drained by a flag, barrier or pipe_barrier.
  Pipe pipe = getOpPipe(op, Pipe::V);
  interp.issueEffect(
      core, pipe, op, reads, writes,
      [interpPtr, op, dst, sources, transposeVec, fn]() {
        SmallVector<RuntimeValue, 3> args(sources.size());
        SmallVector<int64_t, 4> srcIndex;
        bool failed = false;
        forEachIndex(dst.sizes, [&](ArrayRef<int64_t> index) {
          if (failed)
            return;
          for (size_t i = 0; i < sources.size(); ++i) {
            const EwSource &src = sources[i];
            if (src.isScalar) {
              args[i] = src.scalar;
              continue;
            }
            mapSourceIndex(index, src.mem, transposeVec, srcIndex);
            if (!rawLoadAt(*interpPtr, src.mem, srcIndex, op, args[i])) {
              failed = true;
              return;
            }
          }
          RuntimeValue result = fn(args, dst.elemType, op);
          if (!rawStoreAt(*interpPtr, dst, index, op, result))
            failed = true;
        });
      });
  return ExecResult::Advance;
}

/// Dispatch on the destination element type, coercing sources as needed.
template <typename OpT>
ExecResult runTyped(Interpreter &interp, CoreState &core, Operation *op,
                    const EwFn &fn) {
  auto typed = cast<OpT>(op);
  return runElementwise(interp, core, op, typed.getSrc(), typed.getDst(),
                        typed.getTranspose(), fn);
}

//===----------------------------------------------------------------------===//
// Scalar kernels
//===----------------------------------------------------------------------===//

/// Coerce a source value into the arithmetic domain of `type`.
RuntimeValue coerce(const RuntimeValue &value, Type type) {
  const llvm::fltSemantics *sem = getFloatSemantics(type);
  if (sem && value.isFloat()) {
    if (&value.getFloatValue().getSemantics() == sem)
      return value;
    llvm::APFloat converted = value.getFloatValue();
    bool losesInfo = false;
    converted.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return RuntimeValue::getFloat(converted);
  }
  if (sem && value.isInt()) {
    llvm::APFloat converted(*sem);
    converted.convertFromAPInt(value.getIntValue(), /*IsSigned=*/true,
                               llvm::APFloat::rmNearestTiesToEven);
    return RuntimeValue::getFloat(converted);
  }
  if (!sem && value.isFloat()) {
    unsigned width = 64;
    if (auto intType = dyn_cast<IntegerType>(type))
      width = intType.getWidth();
    llvm::APSInt result(width, /*isUnsigned=*/false);
    bool exact = false;
    value.getFloatValue().convertToInteger(
        result, llvm::APFloat::rmTowardZero, &exact);
    return RuntimeValue::getInt(llvm::APInt(result));
  }
  if (!sem && value.isInt()) {
    unsigned width = 64;
    if (auto intType = dyn_cast<IntegerType>(type))
      width = intType.getWidth();
    llvm::APInt v = value.getIntValue();
    if (v.getBitWidth() < width)
      v = v.sext(width);
    else if (v.getBitWidth() > width)
      v = v.trunc(width);
    return RuntimeValue::getInt(v);
  }
  return value;
}

using IntUnary = llvm::APInt (*)(const llvm::APInt &);
using IntBinary = llvm::APInt (*)(const llvm::APInt &, const llvm::APInt &);
using FltUnary = llvm::APFloat (*)(const llvm::APFloat &);
using FltBinary = llvm::APFloat (*)(const llvm::APFloat &,
                                    const llvm::APFloat &);

/// Build a kernel that works on both integer and float element types.
EwFn makeBinary(IntBinary intFn, FltBinary fltFn, StringRef name) {
  return [intFn, fltFn, name](ArrayRef<RuntimeValue> args, Type type,
                              Operation *op) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], type);
    RuntimeValue b = coerce(args.size() > 1 ? args[1] : args[0], type);
    if (a.isFloat()) {
      if (!fltFn) {
        op->emitError() << name << " is not defined for float element types";
        return a;
      }
      return RuntimeValue::getFloat(fltFn(a.getFloatValue(),
                                          b.getFloatValue()));
    }
    if (!intFn) {
      op->emitError() << name << " is not defined for integer element types";
      return a;
    }
    return RuntimeValue::getInt(intFn(a.getIntValue(), b.getIntValue()));
  };
}

EwFn makeUnary(IntUnary intFn, FltUnary fltFn, StringRef name) {
  return [intFn, fltFn, name](ArrayRef<RuntimeValue> args, Type type,
                              Operation *op) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], type);
    if (a.isFloat()) {
      if (!fltFn) {
        op->emitError() << name << " is not defined for float element types";
        return a;
      }
      return RuntimeValue::getFloat(fltFn(a.getFloatValue()));
    }
    if (!intFn) {
      op->emitError() << name << " is not defined for integer element types";
      return a;
    }
    return RuntimeValue::getInt(intFn(a.getIntValue()));
  };
}

/// Float kernel evaluated in double precision then rounded back to the
/// element type. Used for the transcendentals, where the hardware's own
/// approximation differs anyway.
EwFn makeTranscendental(double (*fn)(double)) {
  return [fn](ArrayRef<RuntimeValue> args, Type type,
              Operation *) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], type);
    const llvm::fltSemantics *sem = getFloatSemantics(type);
    if (!sem || !a.isFloat())
      return a;
    llvm::APFloat wide = a.getFloatValue();
    bool losesInfo = false;
    wide.convert(llvm::APFloat::IEEEdouble(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    llvm::APFloat result(fn(wide.convertToDouble()));
    result.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return RuntimeValue::getFloat(result);
  };
}

// --- concrete scalar operations ---

llvm::APInt iAdd(const llvm::APInt &a, const llvm::APInt &b) { return a + b; }
llvm::APInt iSub(const llvm::APInt &a, const llvm::APInt &b) { return a - b; }
llvm::APInt iMul(const llvm::APInt &a, const llvm::APInt &b) { return a * b; }
llvm::APInt iDiv(const llvm::APInt &a, const llvm::APInt &b) {
  return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.sdiv(b);
}
llvm::APInt iMod(const llvm::APInt &a, const llvm::APInt &b) {
  return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.srem(b);
}
llvm::APInt iModU(const llvm::APInt &a, const llvm::APInt &b) {
  return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.urem(b);
}
llvm::APInt iMax(const llvm::APInt &a, const llvm::APInt &b) {
  return a.sgt(b) ? a : b;
}
llvm::APInt iMin(const llvm::APInt &a, const llvm::APInt &b) {
  return a.slt(b) ? a : b;
}
llvm::APInt iAnd(const llvm::APInt &a, const llvm::APInt &b) { return a & b; }
llvm::APInt iOr(const llvm::APInt &a, const llvm::APInt &b) { return a | b; }
llvm::APInt iXor(const llvm::APInt &a, const llvm::APInt &b) { return a ^ b; }
llvm::APInt iShl(const llvm::APInt &a, const llvm::APInt &b) {
  return a.shl(b);
}
llvm::APInt iShr(const llvm::APInt &a, const llvm::APInt &b) {
  return a.ashr(b);
}
llvm::APInt iNot(const llvm::APInt &a) { return ~a; }
llvm::APInt iAbs(const llvm::APInt &a) { return a.isNegative() ? -a : a; }
llvm::APInt iRelu(const llvm::APInt &a) {
  return a.isNegative() ? llvm::APInt(a.getBitWidth(), 0) : a;
}

llvm::APFloat fAdd(const llvm::APFloat &a, const llvm::APFloat &b) {
  llvm::APFloat r = a;
  r.add(b, llvm::APFloat::rmNearestTiesToEven);
  return r;
}
llvm::APFloat fSub(const llvm::APFloat &a, const llvm::APFloat &b) {
  llvm::APFloat r = a;
  r.subtract(b, llvm::APFloat::rmNearestTiesToEven);
  return r;
}
llvm::APFloat fMul(const llvm::APFloat &a, const llvm::APFloat &b) {
  llvm::APFloat r = a;
  r.multiply(b, llvm::APFloat::rmNearestTiesToEven);
  return r;
}
llvm::APFloat fDiv(const llvm::APFloat &a, const llvm::APFloat &b) {
  llvm::APFloat r = a;
  r.divide(b, llvm::APFloat::rmNearestTiesToEven);
  return r;
}
llvm::APFloat fMod(const llvm::APFloat &a, const llvm::APFloat &b) {
  llvm::APFloat r = a;
  r.mod(b);
  return r;
}
/// HIVM min/max propagate NaN: `LowerToLoops` turns a float `vmax` into
/// `arith.maximumf`, which is IEEE 754-2019 `maximum`. That also orders the
/// signed zeros - `maximum(+0, -0)` is `+0` - which a plain `a > b ? a : b`
/// gets wrong, since neither zero compares greater.
llvm::APFloat fMax(const llvm::APFloat &a, const llvm::APFloat &b) {
  return llvm::maximum(a, b);
}
llvm::APFloat fMin(const llvm::APFloat &a, const llvm::APFloat &b) {
  return llvm::minimum(a, b);
}
llvm::APFloat fAbs(const llvm::APFloat &a) {
  llvm::APFloat r = a;
  r.clearSign();
  return r;
}
/// `HIVMToArith` rewrites a float `vrelu` to `arith.maximumf(0, x)`, so relu
/// inherits IEEE maximum: NaN propagates, and relu(-0) is +0. Testing the
/// sign bit instead would answer +0 for a negative NaN and NaN for a positive
/// one, which is not a property of the value at all.
llvm::APFloat fRelu(const llvm::APFloat &a) {
  return llvm::maximum(llvm::APFloat::getZero(a.getSemantics()), a);
}
llvm::APFloat fRec(const llvm::APFloat &a) {
  llvm::APFloat one(a.getSemantics(), 1);
  one.divide(a, llvm::APFloat::rmNearestTiesToEven);
  return one;
}

/// Register an op whose only per-op difference is the scalar kernel.
template <typename OpT>
void addEw(OpRegistry &registry, EwFn fn) {
  registry.add(OpT::getOperationName(),
               [fn](Interpreter &interp, CoreState &core, Operation *op) {
                 return runTyped<OpT>(interp, core, op, fn);
               });
}

//===----------------------------------------------------------------------===//
// Ops needing more than a scalar lambda
//===----------------------------------------------------------------------===//

InterpRoundMode toInterpRoundMode(hivm::RoundMode mode) {
  switch (mode) {
  case hivm::RoundMode::RINT:
    return InterpRoundMode::RINT;
  case hivm::RoundMode::ROUND:
    return InterpRoundMode::ROUND;
  case hivm::RoundMode::FLOOR:
    return InterpRoundMode::FLOOR;
  case hivm::RoundMode::CEIL:
    return InterpRoundMode::CEIL;
  case hivm::RoundMode::TRUNC:
    return InterpRoundMode::TRUNC;
  case hivm::RoundMode::ODD:
    return InterpRoundMode::ODD;
  case hivm::RoundMode::TRUNCWITHOVERFLOW:
    return InterpRoundMode::TRUNC_OVF;
  }
  return InterpRoundMode::RINT;
}

ExecResult execVCast(Interpreter &interp, CoreState &core, Operation *op) {
  auto castOp = cast<hivm::VCastOp>(op);
  InterpRoundMode mode = toInterpRoundMode(castOp.getRoundMode());
  bool isUnsigned = castOp.getCast() == hivm::TypeFn::cast_unsigned;
  Type srcType = cast<ShapedType>(castOp.getSrc().front().getType())
                     .getElementType();
  EwFn fn = [mode, isUnsigned, srcType](ArrayRef<RuntimeValue> args, Type type,
                                        Operation *) -> RuntimeValue {
    return convertValue(args[0], srcType, type, mode, isUnsigned);
  };
  return runTyped<hivm::VCastOp>(interp, core, op, fn);
}

ExecResult execVCmp(Interpreter &interp, CoreState &core, Operation *op) {
  auto cmpOp = cast<hivm::VCmpOp>(op);
  hivm::CompareMode mode = cmpOp.getCompareMode();
  Type srcType =
      cast<ShapedType>(cmpOp.getSrc().front().getType()).getElementType();
  EwFn fn = [mode, srcType](ArrayRef<RuntimeValue> args, Type dstType,
                            Operation *) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], srcType);
    RuntimeValue b = coerce(args.size() > 1 ? args[1] : args[0], srcType);
    bool result = false;
    if (a.isFloat()) {
      llvm::APFloat::cmpResult cmp = a.getFloatValue().compare(
          b.getFloatValue());
      bool eq = cmp == llvm::APFloat::cmpEqual;
      bool lt = cmp == llvm::APFloat::cmpLessThan;
      bool gt = cmp == llvm::APFloat::cmpGreaterThan;
      switch (mode) {
      case hivm::CompareMode::EQ:
        result = eq;
        break;
      case hivm::CompareMode::NE:
        result = !eq;
        break;
      case hivm::CompareMode::LT:
        result = lt;
        break;
      case hivm::CompareMode::GT:
        result = gt;
        break;
      case hivm::CompareMode::GE:
        result = gt || eq;
        break;
      case hivm::CompareMode::LE:
        result = lt || eq;
        break;
      }
    } else {
      const llvm::APInt &x = a.getIntValue();
      const llvm::APInt &y = b.getIntValue();
      switch (mode) {
      case hivm::CompareMode::EQ:
        result = x == y;
        break;
      case hivm::CompareMode::NE:
        result = x != y;
        break;
      case hivm::CompareMode::LT:
        result = x.slt(y);
        break;
      case hivm::CompareMode::GT:
        result = x.sgt(y);
        break;
      case hivm::CompareMode::GE:
        result = x.sge(y);
        break;
      case hivm::CompareMode::LE:
        result = x.sle(y);
        break;
      }
    }
    unsigned width = 8;
    if (auto intType = dyn_cast<IntegerType>(dstType))
      width = intType.getWidth();
    return RuntimeValue::getInt(llvm::APInt(width, result ? 1 : 0));
  };
  return runTyped<hivm::VCmpOp>(interp, core, op, fn);
}

ExecResult execVSel(Interpreter &interp, CoreState &core, Operation *op) {
  // Operand order follows the ODS: (mask, trueValue, falseValue).
  EwFn fn = [](ArrayRef<RuntimeValue> args, Type type,
               Operation *) -> RuntimeValue {
    if (args.size() < 3)
      return coerce(args[0], type);
    bool taken = args[0].isFloat() ? !args[0].getFloatValue().isZero()
                                   : !args[0].getIntValue().isZero();
    return coerce(taken ? args[1] : args[2], type);
  };
  return runTyped<hivm::VSelOp>(interp, core, op, fn);
}

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerHIVMElementwiseOps(OpRegistry &registry) {
  // --- binary arithmetic ---
  addEw<hivm::VAddOp>(registry, makeBinary(iAdd, fAdd, "vadd"));
  addEw<hivm::VSubOp>(registry, makeBinary(iSub, fSub, "vsub"));
  addEw<hivm::VMulOp>(registry, makeBinary(iMul, fMul, "vmul"));
  addEw<hivm::VMulExtOp>(registry, makeBinary(iMul, fMul, "vmulext"));
  addEw<hivm::VDivOp>(registry, makeBinary(iDiv, fDiv, "vdiv"));
  addEw<hivm::VMaxOp>(registry, makeBinary(iMax, fMax, "vmax"));
  addEw<hivm::VMinOp>(registry, makeBinary(iMin, fMin, "vmin"));
  addEw<hivm::VModOp>(registry, makeBinary(iMod, fMod, "vmod"));
  addEw<hivm::VModUIOp>(registry, makeBinary(iModU, fMod, "vmodui"));
  addEw<hivm::VAndOp>(registry, makeBinary(iAnd, nullptr, "vand"));
  addEw<hivm::VOrOp>(registry, makeBinary(iOr, nullptr, "vor"));
  addEw<hivm::VXorOp>(registry, makeBinary(iXor, nullptr, "vxor"));
  addEw<hivm::VShLOp>(registry, makeBinary(iShl, nullptr, "vshl"));
  addEw<hivm::VShROp>(registry, makeBinary(iShr, nullptr, "vshr"));
  addEw<hivm::VPowOp>(registry, [](ArrayRef<RuntimeValue> args, Type type,
                                   Operation *) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], type);
    RuntimeValue b = coerce(args.size() > 1 ? args[1] : args[0], type);
    const llvm::fltSemantics *sem = getFloatSemantics(type);
    if (!sem || !a.isFloat())
      return a;
    llvm::APFloat result(std::pow(a.getFloatValue().convertToDouble(),
                                  b.getFloatValue().convertToDouble()));
    bool losesInfo = false;
    result.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return RuntimeValue::getFloat(result);
  });
  addEw<hivm::VLdexpOp>(registry, [](ArrayRef<RuntimeValue> args, Type type,
                                     Operation *) -> RuntimeValue {
    RuntimeValue a = coerce(args[0], type);
    if (!a.isFloat() || args.size() < 2)
      return a;
    int exp = static_cast<int>(args[1].isFloat()
                                   ? args[1].toDouble()
                                   : args[1].getIntValue().getSExtValue());
    return RuntimeValue::getFloat(
        llvm::scalbn(a.getFloatValue(), exp,
                     llvm::APFloat::rmNearestTiesToEven));
  });

  // --- unary arithmetic ---
  addEw<hivm::VAbsOp>(registry, makeUnary(iAbs, fAbs, "vabs"));
  addEw<hivm::VNotOp>(registry, makeUnary(iNot, nullptr, "vnot"));
  addEw<hivm::VReluOp>(registry, makeUnary(iRelu, fRelu, "vrelu"));
  addEw<hivm::VRecOp>(registry, makeUnary(nullptr, fRec, "vrec"));

  // --- transcendentals ---
  addEw<hivm::VExpOp>(registry,
                      makeTranscendental([](double x) { return std::exp(x); }));
  addEw<hivm::VExp2Op>(
      registry, makeTranscendental([](double x) { return std::exp2(x); }));
  addEw<hivm::VExpM1Op>(
      registry, makeTranscendental([](double x) { return std::expm1(x); }));
  addEw<hivm::VLnOp>(registry,
                     makeTranscendental([](double x) { return std::log(x); }));
  addEw<hivm::VLog2Op>(
      registry, makeTranscendental([](double x) { return std::log2(x); }));
  addEw<hivm::VLog10Op>(
      registry, makeTranscendental([](double x) { return std::log10(x); }));
  addEw<hivm::VLog1pOp>(
      registry, makeTranscendental([](double x) { return std::log1p(x); }));
  addEw<hivm::VSqrtOp>(
      registry, makeTranscendental([](double x) { return std::sqrt(x); }));
  addEw<hivm::VRsqrtOp>(registry, makeTranscendental([](double x) {
                          return 1.0 / std::sqrt(x);
                        }));
  addEw<hivm::VSinOp>(registry,
                      makeTranscendental([](double x) { return std::sin(x); }));
  addEw<hivm::VCosOp>(registry,
                      makeTranscendental([](double x) { return std::cos(x); }));
  addEw<hivm::VTanOp>(registry,
                      makeTranscendental([](double x) { return std::tan(x); }));
  addEw<hivm::VTanhOp>(
      registry, makeTranscendental([](double x) { return std::tanh(x); }));
  addEw<hivm::VAtanOp>(
      registry, makeTranscendental([](double x) { return std::atan(x); }));
  addEw<hivm::VErfOp>(registry,
                      makeTranscendental([](double x) { return std::erf(x); }));
  addEw<hivm::VIlogbOp>(registry, [](ArrayRef<RuntimeValue> args, Type type,
                                     Operation *) -> RuntimeValue {
    unsigned width = 32;
    if (auto intType = dyn_cast<IntegerType>(type))
      width = intType.getWidth();
    if (!args[0].isFloat())
      return RuntimeValue::getInt(llvm::APInt(width, 0));
    // Unbiased base-2 exponent, matching C's ilogb. Every format the NPU
    // supports fits in double, so the widening is exact.
    llvm::APFloat wide = args[0].getFloatValue();
    bool losesInfo = false;
    wide.convert(llvm::APFloat::IEEEdouble(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    int exp = std::ilogb(wide.convertToDouble());
    return RuntimeValue::getInt(
        llvm::APInt(width, static_cast<uint64_t>(exp), /*isSigned=*/true));
  });

  // --- predicates ---
  addEw<hivm::VIsNanOp>(registry, [](ArrayRef<RuntimeValue> args, Type type,
                                     Operation *) -> RuntimeValue {
    unsigned width = 8;
    if (auto intType = dyn_cast<IntegerType>(type))
      width = intType.getWidth();
    bool nan = args[0].isFloat() && args[0].getFloatValue().isNaN();
    return RuntimeValue::getInt(llvm::APInt(width, nan ? 1 : 0));
  });
  addEw<hivm::VIsInfOp>(registry, [](ArrayRef<RuntimeValue> args, Type type,
                                     Operation *) -> RuntimeValue {
    unsigned width = 8;
    if (auto intType = dyn_cast<IntegerType>(type))
      width = intType.getWidth();
    bool inf = args[0].isFloat() && args[0].getFloatValue().isInfinity();
    return RuntimeValue::getInt(llvm::APInt(width, inf ? 1 : 0));
  });

  // --- ops with their own attributes ---
  registry.add(hivm::VCastOp::getOperationName(), execVCast);
  registry.add(hivm::VCmpOp::getOperationName(), execVCmp);
  registry.add(hivm::VSelOp::getOperationName(), execVSel);
}

} // namespace interp
} // namespace bishengir
