//===- OpsCommunity.cpp - func/scf/arith/math/memref handlers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The upstream dialects that survive into memref-form HIVM IR. Everything
// here is plain synchronous execution: none of these ops carry a pipe.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <cmath>

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Bind all results of `op` from `values`.
void bindResults(Interpreter &interp, CoreState &core, Operation *op,
                 ArrayRef<RuntimeValue> values) {
  for (auto [result, value] : llvm::zip(op->getResults(), values))
    interp.setValue(core, result, value);
}

/// Collect the operands of `op` as runtime values.
SmallVector<RuntimeValue, 4> gatherOperands(Interpreter &interp,
                                            CoreState &core, Operation *op) {
  SmallVector<RuntimeValue, 4> values;
  for (Value operand : op->getOperands())
    values.push_back(interp.getValue(core, operand));
  return values;
}

/// Turn a constant attribute into a runtime value.
bool materializeConstant(Interpreter &interp, Attribute attr, Type type,
                         Operation *op, RuntimeValue &out) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (isa<IndexType>(type)) {
      out = RuntimeValue::getIndex(intAttr.getInt());
      return true;
    }
    out = RuntimeValue::getInt(intAttr.getValue());
    return true;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    out = RuntimeValue::getFloat(floatAttr.getValue());
    return true;
  }
  if (auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    out = RuntimeValue::getBool(boolAttr.getValue());
    return true;
  }
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    auto shapedType = cast<ShapedType>(denseAttr.getType());
    SmallVector<int64_t, 4> shape(shapedType.getShape().begin(),
                                  shapedType.getShape().end());
    Type elemType = shapedType.getElementType();
    if (denseAttr.isSplat()) {
      RuntimeValue splat;
      if (!materializeConstant(interp, denseAttr.getSplatValue<Attribute>(),
                               elemType, op, splat))
        return false;
      out = makeSplatVector(shape, splat);
      return true;
    }
    std::vector<RuntimeValue> elements;
    elements.reserve(static_cast<size_t>(denseAttr.getNumElements()));
    for (Attribute element : denseAttr.getValues<Attribute>()) {
      RuntimeValue value;
      if (!materializeConstant(interp, element, elemType, op, value))
        return false;
      elements.push_back(value);
    }
    out = RuntimeValue::getVector(shape, std::move(elements));
    return true;
  }
  interp.emitError(op) << "unsupported constant attribute";
  return false;
}

//===----------------------------------------------------------------------===//
// Vector lifting
//===----------------------------------------------------------------------===//
//
// Vectorized function bodies apply plain arith/math ops to `vector<...>`
// values. Rather than duplicating every handler, the scalar handlers are
// wrapped: if any operand is a vector, the op is mapped over its lanes.

/// Element `lane` of `value`, or `value` itself when it is a scalar (an
/// implicitly splatted operand).
const RuntimeValue &laneOf(const RuntimeValue &value, size_t lane) {
  if (!value.isVector())
    return value;
  const std::vector<RuntimeValue> &elements = value.getVectorElements();
  return lane < elements.size() ? elements[lane] : value;
}

/// Run `scalarFn` once per lane when any operand of `op` is a vector.
/// Returns std::nullopt when there is nothing to vectorize.
std::optional<ExecResult> mapOverLanes(
    Interpreter &interp, CoreState &core, Operation *op,
    llvm::function_ref<ExecResult(Interpreter &, CoreState &, Operation *)>
        scalarFn) {
  SmallVector<int64_t, 4> shape;
  size_t laneCount = 0;
  for (Value operand : op->getOperands()) {
    RuntimeValue value = interp.getValue(core, operand);
    if (!value.isVector())
      continue;
    shape.assign(value.getVectorShape().begin(), value.getVectorShape().end());
    laneCount = std::max(laneCount, value.getVectorElements().size());
  }
  if (shape.empty() && laneCount == 0)
    return std::nullopt;

  // Save the vector operands, then rebind each lane's scalars in place and
  // reuse the scalar handler. The environment is restored afterwards so the
  // vector values stay available to later ops.
  SmallVector<std::pair<Value, RuntimeValue>, 4> saved;
  for (Value operand : op->getOperands())
    saved.emplace_back(operand, interp.getValue(core, operand));

  SmallVector<std::vector<RuntimeValue>, 2> results(op->getNumResults());
  ExecResult status = ExecResult::Advance;
  for (size_t lane = 0; lane < laneCount; ++lane) {
    for (auto &[operand, value] : saved)
      interp.setValue(core, operand, laneOf(value, lane));
    status = scalarFn(interp, core, op);
    if (status == ExecResult::Error)
      break;
    for (unsigned r = 0; r < op->getNumResults(); ++r)
      results[r].push_back(interp.getValue(core, op->getResult(r)));
  }
  for (auto &[operand, value] : saved)
    interp.setValue(core, operand, value);
  if (status == ExecResult::Error)
    return ExecResult::Error;

  for (unsigned r = 0; r < op->getNumResults(); ++r)
    interp.setValue(core, op->getResult(r),
                    RuntimeValue::getVector(shape, std::move(results[r])));
  return ExecResult::Advance;
}

/// Wrap a scalar handler so it also accepts vector operands.
OpHandlerFn vectorize(OpHandlerFn scalarFn) {
  return [scalarFn](Interpreter &interp, CoreState &core, Operation *op) {
    if (auto result = mapOverLanes(interp, core, op, scalarFn))
      return *result;
    return scalarFn(interp, core, op);
  };
}

//===----------------------------------------------------------------------===//
// func / control flow
//===----------------------------------------------------------------------===//

ExecResult execReturn(Interpreter &interp, CoreState &core, Operation *op) {
  SmallVector<RuntimeValue, 4> results = gatherOperands(interp, core, op);
  // Everything still sitting in a pipe must land before the kernel is
  // considered finished, or the last batch of writes silently disappears.
  if (core.callStack.size() == 1) {
    interp.flushAllPipes(core);
    core.returnedAt = op;
    core.status = CoreStatus::Done;
    interp.popCall(core, results);
    return ExecResult::Handled;
  }
  interp.popCall(core, results);
  return ExecResult::Handled;
}

ExecResult execCall(Interpreter &interp, CoreState &core, Operation *op) {
  auto callOp = cast<func::CallOp>(op);
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, callOp.getCalleeAttr()));
  if (!callee) {
    interp.emitError(op) << "cannot resolve callee '" << callOp.getCallee()
                         << "'";
    return ExecResult::Error;
  }
  SmallVector<RuntimeValue, 4> args = gatherOperands(interp, core, op);
  if (failed(interp.pushCall(core, callee, op, args)))
    return ExecResult::Error;
  return ExecResult::Handled;
}

//===----------------------------------------------------------------------===//
// scf
//===----------------------------------------------------------------------===//

ExecResult execFor(Interpreter &interp, CoreState &core, Operation *op) {
  auto forOp = cast<scf::ForOp>(op);
  int64_t lb = 0, ub = 0, stepValue = 0;
  if (!getIndex(interp, core, forOp.getLowerBound(), op, lb) ||
      !getIndex(interp, core, forOp.getUpperBound(), op, ub) ||
      !getIndex(interp, core, forOp.getStep(), op, stepValue))
    return ExecResult::Error;
  if (stepValue == 0) {
    interp.emitError(op) << "scf.for with a zero step would never terminate";
    return ExecResult::Error;
  }

  SmallVector<RuntimeValue, 2> iterArgs;
  for (Value init : forOp.getInitArgs())
    iterArgs.push_back(interp.getValue(core, init));

  // Zero-trip loop: results are the initial values.
  if ((stepValue > 0 && lb >= ub) || (stepValue < 0 && lb <= ub)) {
    bindResults(interp, core, op, iterArgs);
    return ExecResult::Advance;
  }

  Block *body = forOp.getBody();
  interp.pushRegion(core, op, body);
  RegionFrame &frame = interp.currentRegion(core);
  frame.isForLoop = true;
  frame.iv = lb;
  frame.ub = ub;
  frame.step = stepValue;
  frame.iterArgs = iterArgs;

  interp.setValue(core, body->getArgument(0), RuntimeValue::getIndex(lb));
  for (unsigned i = 0; i < iterArgs.size(); ++i)
    interp.setValue(core, body->getArgument(i + 1), iterArgs[i]);
  return ExecResult::Handled;
}

ExecResult execIf(Interpreter &interp, CoreState &core, Operation *op) {
  auto ifOp = cast<scf::IfOp>(op);
  RuntimeValue cond = interp.getValue(core, ifOp.getCondition());
  if (!cond.isInt()) {
    interp.emitError(op) << "scf.if condition is unbound";
    return ExecResult::Error;
  }
  bool taken = !cond.getIntValue().isZero();
  Region &region = taken ? ifOp.getThenRegion() : ifOp.getElseRegion();
  if (region.empty()) {
    // No else region and the condition is false: nothing to do. An scf.if
    // without an else cannot have results.
    return ExecResult::Advance;
  }
  interp.pushRegion(core, op, &region.front());
  return ExecResult::Handled;
}

ExecResult execWhile(Interpreter &interp, CoreState &core, Operation *op) {
  auto whileOp = cast<scf::WhileOp>(op);
  SmallVector<RuntimeValue, 2> args;
  for (Value init : whileOp.getInits())
    args.push_back(interp.getValue(core, init));

  Block *before = whileOp.getBeforeBody();
  interp.pushRegion(core, op, before);
  RegionFrame &frame = interp.currentRegion(core);
  frame.isWhileBefore = true;
  for (auto [arg, value] : llvm::zip(before->getArguments(), args))
    interp.setValue(core, arg, value);
  return ExecResult::Handled;
}

ExecResult execCondition(Interpreter &interp, CoreState &core, Operation *op) {
  auto condOp = cast<scf::ConditionOp>(op);
  auto whileOp = cast<scf::WhileOp>(condOp->getParentOp());
  RuntimeValue cond = interp.getValue(core, condOp.getCondition());
  if (!cond.isInt()) {
    interp.emitError(op) << "scf.condition value is unbound";
    return ExecResult::Error;
  }
  SmallVector<RuntimeValue, 2> forwarded;
  for (Value arg : condOp.getArgs())
    forwarded.push_back(interp.getValue(core, arg));

  interp.popRegion(core);
  if (cond.getIntValue().isZero()) {
    bindResults(interp, core, whileOp.getOperation(), forwarded);
    // The parent instruction pointer is still on the scf.while.
    advanceIp(core);
    return ExecResult::Handled;
  }

  Block *after = whileOp.getAfterBody();
  interp.pushRegion(core, whileOp.getOperation(), after);
  RegionFrame &frame = interp.currentRegion(core);
  frame.isWhileAfter = true;
  for (auto [arg, value] : llvm::zip(after->getArguments(), forwarded))
    interp.setValue(core, arg, value);
  return ExecResult::Handled;
}

ExecResult execYield(Interpreter &interp, CoreState &core, Operation *op) {
  if (!Interpreter::hasRegion(core)) {
    interp.emitError(op) << "region terminator outside any region";
    return ExecResult::Error;
  }
  SmallVector<RuntimeValue, 4> values = gatherOperands(interp, core, op);
  RegionFrame &frame = interp.currentRegion(core);
  Operation *owner = frame.owner;

  if (frame.isForLoop) {
    frame.iterArgs.assign(values.begin(), values.end());
    frame.iv += frame.step;
    bool more = frame.step > 0 ? frame.iv < frame.ub : frame.iv > frame.ub;
    if (more) {
      Block *body = frame.block;
      frame.ip = body->begin();
      interp.setValue(core, body->getArgument(0),
                      RuntimeValue::getIndex(frame.iv));
      for (unsigned i = 0; i < values.size(); ++i)
        interp.setValue(core, body->getArgument(i + 1), values[i]);
      return ExecResult::Handled;
    }
    SmallVector<RuntimeValue, 2> results = frame.iterArgs;
    interp.popRegion(core);
    bindResults(interp, core, owner, results);
    advanceIp(core);
    return ExecResult::Handled;
  }

  if (frame.isWhileAfter) {
    auto whileOp = cast<scf::WhileOp>(owner);
    interp.popRegion(core);
    Block *before = whileOp.getBeforeBody();
    interp.pushRegion(core, owner, before);
    RegionFrame &next = interp.currentRegion(core);
    next.isWhileBefore = true;
    for (auto [arg, value] : llvm::zip(before->getArguments(), values))
      interp.setValue(core, arg, value);
    return ExecResult::Handled;
  }

  // scf.if / scope.scope / generic single-pass region.
  interp.popRegion(core);
  if (owner) {
    bindResults(interp, core, owner, values);
    advanceIp(core);
  }
  return ExecResult::Handled;
}

ExecResult execExecuteRegion(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto regionOp = cast<scf::ExecuteRegionOp>(op);
  interp.pushRegion(core, op, &regionOp.getRegion().front());
  return ExecResult::Handled;
}

ExecResult execIndexSwitch(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto switchOp = cast<scf::IndexSwitchOp>(op);
  int64_t arg = 0;
  if (!getIndex(interp, core, switchOp.getArg(), op, arg))
    return ExecResult::Error;
  Region *target = &switchOp.getDefaultRegion();
  ArrayRef<int64_t> cases = switchOp.getCases();
  for (auto [i, value] : llvm::enumerate(cases)) {
    if (value == arg) {
      target = &switchOp.getCaseRegions()[i];
      break;
    }
  }
  interp.pushRegion(core, op, &target->front());
  return ExecResult::Handled;
}

//===----------------------------------------------------------------------===//
// cf
//===----------------------------------------------------------------------===//

ExecResult jumpTo(Interpreter &interp, CoreState &core, Block *dest,
                  ArrayRef<RuntimeValue> args) {
  RegionFrame &frame = interp.currentRegion(core);
  frame.block = dest;
  frame.ip = dest->begin();
  for (auto [arg, value] : llvm::zip(dest->getArguments(), args))
    interp.setValue(core, arg, value);
  return ExecResult::Handled;
}

ExecResult execBranch(Interpreter &interp, CoreState &core, Operation *op) {
  auto branchOp = cast<cf::BranchOp>(op);
  SmallVector<RuntimeValue, 2> args;
  for (Value operand : branchOp.getDestOperands())
    args.push_back(interp.getValue(core, operand));
  return jumpTo(interp, core, branchOp.getDest(), args);
}

ExecResult execCondBranch(Interpreter &interp, CoreState &core, Operation *op) {
  auto branchOp = cast<cf::CondBranchOp>(op);
  RuntimeValue cond = interp.getValue(core, branchOp.getCondition());
  if (!cond.isInt()) {
    interp.emitError(op) << "cf.cond_br condition is unbound";
    return ExecResult::Error;
  }
  bool taken = !cond.getIntValue().isZero();
  SmallVector<RuntimeValue, 2> args;
  for (Value operand : taken ? branchOp.getTrueDestOperands()
                             : branchOp.getFalseDestOperands())
    args.push_back(interp.getValue(core, operand));
  return jumpTo(interp, core,
                taken ? branchOp.getTrueDest() : branchOp.getFalseDest(), args);
}

//===----------------------------------------------------------------------===//
// arith
//===----------------------------------------------------------------------===//

ExecResult execArithConstant(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto constOp = cast<arith::ConstantOp>(op);
  RuntimeValue value;
  if (!materializeConstant(interp, constOp.getValue(),
                           constOp.getType(), op, value))
    return ExecResult::Error;
  interp.setValue(core, constOp.getResult(), value);
  return ExecResult::Advance;
}

/// Binary integer op described by a lambda over APInts.
template <typename Fn>
ExecResult intBinary(Interpreter &interp, CoreState &core, Operation *op,
                     Fn fn) {
  RuntimeValue lhs = interp.getValue(core, op->getOperand(0));
  RuntimeValue rhs = interp.getValue(core, op->getOperand(1));
  if (!lhs.isInt() || !rhs.isInt()) {
    interp.emitError(op) << "integer operands are unbound";
    return ExecResult::Error;
  }
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getInt(fn(lhs.getIntValue(),
                                          rhs.getIntValue())));
  return ExecResult::Advance;
}

template <typename Fn>
ExecResult floatBinary(Interpreter &interp, CoreState &core, Operation *op,
                       Fn fn) {
  RuntimeValue lhs = interp.getValue(core, op->getOperand(0));
  RuntimeValue rhs = interp.getValue(core, op->getOperand(1));
  if (!lhs.isFloat() || !rhs.isFloat()) {
    interp.emitError(op) << "float operands are unbound";
    return ExecResult::Error;
  }
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getFloat(fn(lhs.getFloatValue(),
                                            rhs.getFloatValue())));
  return ExecResult::Advance;
}

template <typename Fn>
ExecResult floatUnary(Interpreter &interp, CoreState &core, Operation *op,
                      Fn fn) {
  RuntimeValue operand = interp.getValue(core, op->getOperand(0));
  if (!operand.isFloat()) {
    interp.emitError(op) << "float operand is unbound";
    return ExecResult::Error;
  }
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getFloat(fn(operand.getFloatValue())));
  return ExecResult::Advance;
}

/// Evaluate `fn` on the operand as a double and convert back. Used for the
/// transcendental math ops, where bit-exactness against the NPU's own
/// approximations is not achievable anyway.
template <typename Fn>
ExecResult floatUnaryViaDouble(Interpreter &interp, CoreState &core,
                               Operation *op, Fn fn) {
  return floatUnary(interp, core, op, [&](const llvm::APFloat &value) {
    llvm::APFloat wide = value;
    bool losesInfo = false;
    wide.convert(llvm::APFloat::IEEEdouble(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    llvm::APFloat result(fn(wide.convertToDouble()));
    result.convert(value.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                   &losesInfo);
    return result;
  });
}

ExecResult execCmpI(Interpreter &interp, CoreState &core, Operation *op) {
  auto cmpOp = cast<arith::CmpIOp>(op);
  RuntimeValue lhs = interp.getValue(core, cmpOp.getLhs());
  RuntimeValue rhs = interp.getValue(core, cmpOp.getRhs());
  if (!lhs.isInt() || !rhs.isInt()) {
    interp.emitError(op) << "arith.cmpi operands are unbound";
    return ExecResult::Error;
  }
  const llvm::APInt &a = lhs.getIntValue();
  const llvm::APInt &b = rhs.getIntValue();
  bool result = false;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    result = a == b;
    break;
  case arith::CmpIPredicate::ne:
    result = a != b;
    break;
  case arith::CmpIPredicate::slt:
    result = a.slt(b);
    break;
  case arith::CmpIPredicate::sle:
    result = a.sle(b);
    break;
  case arith::CmpIPredicate::sgt:
    result = a.sgt(b);
    break;
  case arith::CmpIPredicate::sge:
    result = a.sge(b);
    break;
  case arith::CmpIPredicate::ult:
    result = a.ult(b);
    break;
  case arith::CmpIPredicate::ule:
    result = a.ule(b);
    break;
  case arith::CmpIPredicate::ugt:
    result = a.ugt(b);
    break;
  case arith::CmpIPredicate::uge:
    result = a.uge(b);
    break;
  }
  interp.setValue(core, cmpOp.getResult(), RuntimeValue::getBool(result));
  return ExecResult::Advance;
}

ExecResult execCmpF(Interpreter &interp, CoreState &core, Operation *op) {
  auto cmpOp = cast<arith::CmpFOp>(op);
  RuntimeValue lhs = interp.getValue(core, cmpOp.getLhs());
  RuntimeValue rhs = interp.getValue(core, cmpOp.getRhs());
  if (!lhs.isFloat() || !rhs.isFloat()) {
    interp.emitError(op) << "arith.cmpf operands are unbound";
    return ExecResult::Error;
  }
  llvm::APFloat::cmpResult cmp =
      lhs.getFloatValue().compare(rhs.getFloatValue());
  bool unordered = cmp == llvm::APFloat::cmpUnordered;
  bool eq = cmp == llvm::APFloat::cmpEqual;
  bool lt = cmp == llvm::APFloat::cmpLessThan;
  bool gt = cmp == llvm::APFloat::cmpGreaterThan;
  bool result = false;
  switch (cmpOp.getPredicate()) {
  case arith::CmpFPredicate::AlwaysFalse:
    result = false;
    break;
  case arith::CmpFPredicate::OEQ:
    result = eq;
    break;
  case arith::CmpFPredicate::OGT:
    result = gt;
    break;
  case arith::CmpFPredicate::OGE:
    result = gt || eq;
    break;
  case arith::CmpFPredicate::OLT:
    result = lt;
    break;
  case arith::CmpFPredicate::OLE:
    result = lt || eq;
    break;
  case arith::CmpFPredicate::ONE:
    result = lt || gt;
    break;
  case arith::CmpFPredicate::ORD:
    result = !unordered;
    break;
  case arith::CmpFPredicate::UEQ:
    result = eq || unordered;
    break;
  case arith::CmpFPredicate::UGT:
    result = gt || unordered;
    break;
  case arith::CmpFPredicate::UGE:
    result = gt || eq || unordered;
    break;
  case arith::CmpFPredicate::ULT:
    result = lt || unordered;
    break;
  case arith::CmpFPredicate::ULE:
    result = lt || eq || unordered;
    break;
  case arith::CmpFPredicate::UNE:
    result = !eq;
    break;
  case arith::CmpFPredicate::UNO:
    result = unordered;
    break;
  case arith::CmpFPredicate::AlwaysTrue:
    result = true;
    break;
  }
  interp.setValue(core, cmpOp.getResult(), RuntimeValue::getBool(result));
  return ExecResult::Advance;
}

ExecResult execSelect(Interpreter &interp, CoreState &core, Operation *op) {
  auto selectOp = cast<arith::SelectOp>(op);
  RuntimeValue cond = interp.getValue(core, selectOp.getCondition());
  if (!cond.isInt()) {
    interp.emitError(op) << "arith.select condition is unbound";
    return ExecResult::Error;
  }
  Value picked = cond.getIntValue().isZero() ? selectOp.getFalseValue()
                                             : selectOp.getTrueValue();
  interp.setValue(core, selectOp.getResult(), interp.getValue(core, picked));
  return ExecResult::Advance;
}

/// Handlers for the width- and signedness-changing casts.
ExecResult execCast(Interpreter &interp, CoreState &core, Operation *op,
                    bool isUnsigned, InterpRoundMode mode) {
  RuntimeValue in = interp.getValue(core, op->getOperand(0));
  if (in.isNone()) {
    interp.emitError(op) << "cast operand is unbound";
    return ExecResult::Error;
  }
  // Unwrap shaped types: vector lifting hands us one lane at a time, but the
  // operand and result types are still `vector<...>`.
  Type fromType = getScalarType(op->getOperand(0).getType());
  Type toType = getScalarType(op->getResult(0).getType());
  interp.setValue(core, op->getResult(0),
                  convertValue(in, fromType, toType, mode, isUnsigned));
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// memref
//===----------------------------------------------------------------------===//

ExecResult execAlloc(Interpreter &interp, CoreState &core, Operation *op) {
  auto memrefType = cast<MemRefType>(op->getResult(0).getType());
  // Dynamic dimensions come from operands.
  MemRefValue mem;
  if (!memrefType.hasStaticShape()) {
    SmallVector<int64_t, 4> sizes;
    unsigned dynIdx = 0;
    for (int64_t dim : memrefType.getShape()) {
      if (!ShapedType::isDynamic(dim)) {
        sizes.push_back(dim);
        continue;
      }
      int64_t value = 0;
      if (dynIdx >= op->getNumOperands() ||
          !getIndex(interp, core, op->getOperand(dynIdx++), op, value))
        return ExecResult::Error;
      sizes.push_back(value);
    }
    auto staticType =
        MemRefType::get(sizes, memrefType.getElementType(),
                        MemRefLayoutAttrInterface(),
                        memrefType.getMemorySpace());
    if (!interp.allocateMemRef(core, staticType, op, mem))
      return ExecResult::Error;
  } else if (!interp.allocateMemRef(core, memrefType, op, mem)) {
    return ExecResult::Error;
  }
  interp.setValue(core, op->getResult(0), RuntimeValue::getMemRef(mem));
  return ExecResult::Advance;
}

ExecResult execDealloc(Interpreter &interp, CoreState &core, Operation *op) {
  MemRefValue mem;
  if (!interp.getMemRefOperand(core, op->getOperand(0), mem, op))
    return ExecResult::Error;
  interp.getArena(mem.arena).deallocate(mem.byteOffset);
  return ExecResult::Advance;
}

ExecResult execMemRefLoad(Interpreter &interp, CoreState &core, Operation *op) {
  auto loadOp = cast<memref::LoadOp>(op);
  MemRefValue mem;
  if (!interp.getMemRefOperand(core, loadOp.getMemRef(), mem, op))
    return ExecResult::Error;
  SmallVector<int64_t, 4> indices;
  for (Value index : loadOp.getIndices()) {
    int64_t value = 0;
    if (!getIndex(interp, core, index, op, value))
      return ExecResult::Error;
    indices.push_back(value);
  }
  // The scalar unit performs the read itself, so the value is available at
  // once - but it is on PIPE_S, and the other pipes only get to see the
  // buffer it touched once a set_flag[PIPE_S, ...] retires the access.
  uint64_t addr = mem.getByteAddr(indices);
  interp.issueResidentAccess(core, Pipe::S, op,
                             {mem.arena, addr, addr + mem.elemBytes},
                             /*isWrite=*/false);
  RuntimeValue value;
  if (!rawLoadAt(interp, mem, indices, op, value))
    return ExecResult::Error;
  interp.setValue(core, loadOp.getResult(), value);
  return ExecResult::Advance;
}

ExecResult execMemRefStore(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto storeOp = cast<memref::StoreOp>(op);
  MemRefValue mem;
  if (!interp.getMemRefOperand(core, storeOp.getMemRef(), mem, op))
    return ExecResult::Error;
  SmallVector<int64_t, 4> indices;
  for (Value index : storeOp.getIndices()) {
    int64_t value = 0;
    if (!getIndex(interp, core, index, op, value))
      return ExecResult::Error;
    indices.push_back(value);
  }
  uint64_t addr = mem.getByteAddr(indices);
  interp.issueResidentAccess(core, Pipe::S, op,
                             {mem.arena, addr, addr + mem.elemBytes},
                             /*isWrite=*/true);
  if (!rawStoreAt(interp, mem, indices, op,
                  interp.getValue(core, storeOp.getValueToStore())))
    return ExecResult::Error;
  return ExecResult::Advance;
}

ExecResult execSubView(Interpreter &interp, CoreState &core, Operation *op) {
  auto subViewOp = cast<memref::SubViewOp>(op);
  MemRefValue source;
  if (!interp.getMemRefOperand(core, subViewOp.getSource(), source, op))
    return ExecResult::Error;

  SmallVector<int64_t, 4> offsets, sizes, strides;
  for (OpFoldResult ofr : subViewOp.getMixedOffsets()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    offsets.push_back(value);
  }
  for (OpFoldResult ofr : subViewOp.getMixedSizes()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    sizes.push_back(value);
  }
  for (OpFoldResult ofr : subViewOp.getMixedStrides()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    strides.push_back(value);
  }

  MemRefValue result = source;
  int64_t elemDelta = 0;
  for (size_t d = 0; d < offsets.size() && d < source.strides.size(); ++d)
    elemDelta += offsets[d] * source.strides[d];
  result.byteOffset =
      source.byteOffset + static_cast<uint64_t>(elemDelta) * source.elemBytes;

  SmallVector<int64_t, 4> newSizes, newStrides;
  llvm::SmallBitVector dropped = subViewOp.getDroppedDims();
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (d < dropped.size() && dropped[d])
      continue;
    newSizes.push_back(sizes[d]);
    newStrides.push_back(strides[d] * (d < source.strides.size()
                                           ? source.strides[d]
                                           : 1));
  }
  result.sizes = std::move(newSizes);
  result.strides = std::move(newStrides);
  interp.setValue(core, subViewOp.getResult(), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

ExecResult execView(Interpreter &interp, CoreState &core, Operation *op) {
  auto viewOp = cast<memref::ViewOp>(op);
  MemRefValue source;
  if (!interp.getMemRefOperand(core, viewOp.getSource(), source, op))
    return ExecResult::Error;
  int64_t byteShift = 0;
  if (!getIndex(interp, core, viewOp.getByteShift(), op, byteShift))
    return ExecResult::Error;

  auto resultType = cast<MemRefType>(viewOp.getResult().getType());
  MemRefValue result;
  result.arena = source.arena;
  // A view rebases: its result's offset field is zero relative to the shifted
  // pointer, so the shifted address becomes the new allocation base.
  result.byteOffset = source.byteOffset + static_cast<uint64_t>(byteShift);
  result.baseOffset = result.byteOffset;
  result.elemType = resultType.getElementType();
  result.elemBytes = getStorageSize(result.elemType);
  result.space = source.space;
  result.layout = source.layout;

  unsigned dynIdx = 0;
  for (int64_t dim : resultType.getShape()) {
    if (!ShapedType::isDynamic(dim)) {
      result.sizes.push_back(dim);
      continue;
    }
    int64_t value = 0;
    if (dynIdx >= viewOp.getSizes().size() ||
        !getIndex(interp, core, viewOp.getSizes()[dynIdx++], op, value))
      return ExecResult::Error;
    result.sizes.push_back(value);
  }
  result.strides.resize(result.sizes.size());
  int64_t acc = 1;
  for (int64_t d = static_cast<int64_t>(result.sizes.size()) - 1; d >= 0; --d) {
    result.strides[d] = acc;
    acc *= result.sizes[d];
  }
  interp.setValue(core, viewOp.getResult(), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

ExecResult execReinterpretCast(Interpreter &interp, CoreState &core,
                               Operation *op) {
  auto castOp = cast<memref::ReinterpretCastOp>(op);
  MemRefValue source;
  if (!interp.getMemRefOperand(core, castOp.getSource(), source, op))
    return ExecResult::Error;

  int64_t offset = 0;
  if (!getFoldedIndex(interp, core, castOp.getMixedOffsets().front(), op,
                      offset))
    return ExecResult::Error;

  MemRefValue result = source;
  auto resultType = cast<MemRefType>(castOp.getResult().getType());
  result.elemType = resultType.getElementType();
  result.elemBytes = getStorageSize(result.elemType);
  // The offset is counted in elements of the *result* type and is absolute
  // with respect to the source's base pointer: reinterpret_cast keeps the
  // source's allocated/aligned pointer and *sets* the offset field rather
  // than adding to it. Chaining two casts must therefore not accumulate.
  result.byteOffset =
      source.baseOffset + static_cast<uint64_t>(offset) * result.elemBytes;

  result.sizes.clear();
  for (OpFoldResult ofr : castOp.getMixedSizes()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    result.sizes.push_back(value);
  }
  result.strides.clear();
  for (OpFoldResult ofr : castOp.getMixedStrides()) {
    int64_t value = 0;
    if (!getFoldedIndex(interp, core, ofr, op, value))
      return ExecResult::Error;
    result.strides.push_back(value);
  }
  interp.setValue(core, castOp.getResult(), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

/// `memref.cast` / `memref.memory_space_cast` only change the static type.
ExecResult execMemRefRetype(Interpreter &interp, CoreState &core,
                            Operation *op) {
  MemRefValue source;
  if (!interp.getMemRefOperand(core, op->getOperand(0), source, op))
    return ExecResult::Error;
  auto resultType = dyn_cast<MemRefType>(op->getResult(0).getType());
  if (resultType) {
    source.space = getAddrSpaceOf(resultType);
    // A cast that materialises static dims must not lose them.
    for (auto [d, dim] : llvm::enumerate(resultType.getShape()))
      if (!ShapedType::isDynamic(dim) && d < source.sizes.size())
        source.sizes[d] = dim;
  }
  interp.setValue(core, op->getResult(0), RuntimeValue::getMemRef(source));
  return ExecResult::Advance;
}

/// Reshapes keep the byte offset and recompute a dense stride vector. The
/// interpreter only supports reshapes of contiguous views, which is what
/// bufferization produces.
ExecResult execReshape(Interpreter &interp, CoreState &core, Operation *op,
                       bool expanding) {
  MemRefValue source;
  if (!interp.getMemRefOperand(core, op->getOperand(0), source, op))
    return ExecResult::Error;
  auto resultType = cast<MemRefType>(op->getResult(0).getType());
  if (!source.isContiguous()) {
    interp.emitError(op)
        << "reshape of a non-contiguous memref is not supported";
    return ExecResult::Error;
  }

  MemRefValue result = source;
  result.sizes.assign(resultType.getShape().begin(),
                      resultType.getShape().end());

  // Dynamic result dims. expand_shape carries them as operands; take those
  // when present, otherwise derive a single unknown from the element count.
  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    // static_output_shape holds every dimension, with kDynamic where an
    // operand supplies the value; the operands appear in that same order.
    ArrayRef<int64_t> staticShape = expandOp.getStaticOutputShape();
    OperandRange dynamic = expandOp.getOutputShape();
    if (staticShape.size() == result.sizes.size()) {
      unsigned dynIdx = 0;
      for (auto [d, declared] : llvm::enumerate(staticShape)) {
        if (!ShapedType::isDynamic(declared)) {
          result.sizes[d] = declared;
          continue;
        }
        if (dynIdx >= dynamic.size())
          break;
        int64_t value = 0;
        if (!getIndex(interp, core, dynamic[dynIdx++], op, value))
          return ExecResult::Error;
        result.sizes[d] = value;
      }
    }
  }
  int64_t known = 1;
  SmallVector<int, 2> dynDims;
  for (auto [d, size] : llvm::enumerate(result.sizes)) {
    if (ShapedType::isDynamic(size))
      dynDims.push_back(static_cast<int>(d));
    else
      known *= size;
  }
  if (dynDims.size() > 1) {
    // More than one unknown cannot be recovered from the element count, and
    // guessing would silently produce a view of the wrong extent.
    interp.emitError(op) << "reshape with " << dynDims.size()
                         << " dynamic result dimensions is ambiguous";
    return ExecResult::Error;
  }
  if (dynDims.size() == 1 && known > 0)
    result.sizes[dynDims.front()] = source.getNumElements() / known;

  result.strides.resize(result.sizes.size());
  int64_t acc = 1;
  for (int64_t d = static_cast<int64_t>(result.sizes.size()) - 1; d >= 0; --d) {
    result.strides[d] = acc;
    acc *= result.sizes[d];
  }
  (void)expanding;
  interp.setValue(core, op->getResult(0), RuntimeValue::getMemRef(result));
  return ExecResult::Advance;
}

ExecResult execDim(Interpreter &interp, CoreState &core, Operation *op) {
  auto dimOp = cast<memref::DimOp>(op);
  MemRefValue mem;
  if (!interp.getMemRefOperand(core, dimOp.getSource(), mem, op))
    return ExecResult::Error;
  int64_t index = 0;
  if (!getIndex(interp, core, dimOp.getIndex(), op, index))
    return ExecResult::Error;
  if (index < 0 || index >= mem.getRank()) {
    interp.emitError(op) << "memref.dim index " << index << " out of range";
    return ExecResult::Error;
  }
  interp.setValue(core, dimOp.getResult(),
                  RuntimeValue::getIndex(mem.sizes[index]));
  return ExecResult::Advance;
}

ExecResult execNoOp(Interpreter &, CoreState &, Operation *) {
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// llvm
//===----------------------------------------------------------------------===//
//
// Fully lowered HIVM still contains a little raw-pointer scalar code: the
// cross-core flag scratchpad is addressed with `llvm.inttoptr` to a constant
// address plus volatile loads and stores. Those are modelled as one-element
// views into the arena named by the pointer's address space, which is the
// same numbering memref memory spaces use.

/// Arena for a raw pointer's address space.
AddrSpace getPointerSpace(Type type) {
  auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type);
  if (!ptrType)
    return AddrSpace::GM;
  switch (ptrType.getAddressSpace()) {
  case 1:
    return AddrSpace::GM;
  case 2:
    return AddrSpace::L1;
  case 3:
    return AddrSpace::L0A;
  case 4:
    return AddrSpace::L0B;
  case 5:
    return AddrSpace::L0C;
  case 6:
    return AddrSpace::UB;
  case 11:
    return AddrSpace::SSBUF;
  default:
    return AddrSpace::GM;
  }
}

ExecResult execIntToPtr(Interpreter &interp, CoreState &core, Operation *op) {
  RuntimeValue addr = interp.getValue(core, op->getOperand(0));
  if (!addr.isInt()) {
    interp.emitError(op) << "llvm.inttoptr operand is unbound";
    return ExecResult::Error;
  }
  MemRefValue ptr;
  ptr.space = getPointerSpace(op->getResult(0).getType());
  ptr.arena = interp.getArenaId(ptr.space, core.id);
  ptr.byteOffset = addr.getIntValue().getZExtValue();
  ptr.baseOffset = ptr.byteOffset;
  ptr.sizes.assign({1});
  ptr.strides.assign({1});
  // The element type is decided by the load or store that uses the pointer;
  // until then treat it as raw bytes rather than leaving it null.
  ptr.elemType = IntegerType::get(op->getContext(), 8);
  ptr.elemBytes = 1;
  interp.setValue(core, op->getResult(0), RuntimeValue::getMemRef(ptr));
  return ExecResult::Advance;
}

ExecResult execPtrToInt(Interpreter &interp, CoreState &core, Operation *op) {
  MemRefValue ptr;
  if (!interp.getMemRefOperand(core, op->getOperand(0), ptr, op))
    return ExecResult::Error;
  unsigned width = 64;
  if (auto intType = dyn_cast<IntegerType>(op->getResult(0).getType()))
    width = intType.getWidth();
  interp.setValue(core, op->getResult(0),
                  RuntimeValue::getInt(llvm::APInt(width, ptr.byteOffset)));
  return ExecResult::Advance;
}

ExecResult execLLVMLoad(Interpreter &interp, CoreState &core, Operation *op) {
  MemRefValue ptr;
  if (!interp.getMemRefOperand(core, op->getOperand(0), ptr, op))
    return ExecResult::Error;
  ptr.elemType = op->getResult(0).getType();
  ptr.elemBytes = getStorageSize(ptr.elemType);
  interp.issueResidentAccess(
      core, Pipe::S, op,
      {ptr.arena, ptr.byteOffset, ptr.byteOffset + ptr.elemBytes},
      /*isWrite=*/false, /*isRawPointer=*/true);
  RuntimeValue value;
  if (!rawLoad(interp, ptr, 0, op, value))
    return ExecResult::Error;
  interp.setValue(core, op->getResult(0), value);
  return ExecResult::Advance;
}

ExecResult execLLVMStore(Interpreter &interp, CoreState &core, Operation *op) {
  RuntimeValue value = interp.getValue(core, op->getOperand(0));
  MemRefValue ptr;
  if (!interp.getMemRefOperand(core, op->getOperand(1), ptr, op))
    return ExecResult::Error;
  ptr.elemType = op->getOperand(0).getType();
  ptr.elemBytes = getStorageSize(ptr.elemType);
  interp.issueResidentAccess(
      core, Pipe::S, op,
      {ptr.arena, ptr.byteOffset, ptr.byteOffset + ptr.elemBytes},
      /*isWrite=*/true, /*isRawPointer=*/true);
  if (!rawStore(interp, ptr, 0, op, value))
    return ExecResult::Error;
  return ExecResult::Advance;
}

ExecResult execGEP(Interpreter &interp, CoreState &core, Operation *op) {
  auto gepOp = cast<LLVM::GEPOp>(op);
  MemRefValue ptr;
  if (!interp.getMemRefOperand(core, gepOp.getBase(), ptr, op))
    return ExecResult::Error;
  unsigned elemBytes = getStorageSize(gepOp.getElemType());
  int64_t total = 0;
  for (auto index : gepOp.getIndices()) {
    int64_t value = 0;
    if (auto attr = dyn_cast<IntegerAttr>(index)) {
      value = attr.getInt();
    } else {
      RuntimeValue rv = interp.getValue(core, cast<Value>(index));
      if (!rv.isInt()) {
        interp.emitError(op) << "llvm.getelementptr index is unbound";
        return ExecResult::Error;
      }
      value = rv.getIndexValue();
    }
    total += value;
  }
  ptr.byteOffset += static_cast<uint64_t>(total) * elemBytes;
  interp.setValue(core, gepOp.getResult(), RuntimeValue::getMemRef(ptr));
  return ExecResult::Advance;
}

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerCommunityOps(OpRegistry &registry) {
  // --- func / control flow ---
  registry.add("func.return", execReturn);
  registry.add("func.call", execCall);
  registry.add("cf.br", execBranch);
  registry.add("cf.cond_br", execCondBranch);

  // --- scf ---
  registry.add("scf.for", execFor);
  registry.add("scf.if", execIf);
  registry.add("scf.while", execWhile);
  registry.add("scf.condition", execCondition);
  registry.add("scf.yield", execYield);
  registry.add("scf.execute_region", execExecuteRegion);
  registry.add("scf.index_switch", execIndexSwitch);

  // --- bishengir scope: a plain single-entry region ---
  registry.add("scope.scope", [](Interpreter &interp, CoreState &core,
                                 Operation *op) {
    interp.pushRegion(core, op, &op->getRegion(0).front());
    return ExecResult::Handled;
  });
  registry.add("scope.return", execYield);

  // --- annotation markers carry no runtime effect ---
  registry.add("annotation.mark", execNoOp);

  // --- arith ---
  registry.add("arith.constant", execArithConstant);
  registry.add("arith.addi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a + b; });
  });
  registry.add("arith.subi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a - b; });
  });
  registry.add("arith.muli", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a * b; });
  });
  registry.add("arith.divsi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.sdiv(b);
    });
  });
  registry.add("arith.divui", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.udiv(b);
    });
  });
  registry.add("arith.remsi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.srem(b);
    });
  });
  registry.add("arith.remui", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.urem(b);
    });
  });
  registry.add("arith.andi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a & b; });
  });
  registry.add("arith.ori", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a | b; });
  });
  registry.add("arith.xori", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a ^ b; });
  });
  registry.add("arith.shli", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.shl(b); });
  });
  registry.add("arith.shrsi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.ashr(b); });
  });
  registry.add("arith.shrui", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.lshr(b); });
  });
  registry.add("arith.maxsi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.sgt(b) ? a : b; });
  });
  registry.add("arith.minsi", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.slt(b) ? a : b; });
  });
  registry.add("arith.maxui", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.ugt(b) ? a : b; });
  });
  registry.add("arith.minui", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.ult(b) ? a : b; });
  });

  registry.add("arith.addf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](llvm::APFloat a, const llvm::APFloat &b) {
      a.add(b, llvm::APFloat::rmNearestTiesToEven);
      return a;
    });
  });
  registry.add("arith.subf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](llvm::APFloat a, const llvm::APFloat &b) {
      a.subtract(b, llvm::APFloat::rmNearestTiesToEven);
      return a;
    });
  });
  registry.add("arith.mulf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](llvm::APFloat a, const llvm::APFloat &b) {
      a.multiply(b, llvm::APFloat::rmNearestTiesToEven);
      return a;
    });
  });
  registry.add("arith.divf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](llvm::APFloat a, const llvm::APFloat &b) {
      a.divide(b, llvm::APFloat::rmNearestTiesToEven);
      return a;
    });
  });
  registry.add("arith.remf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](llvm::APFloat a, const llvm::APFloat &b) {
      a.mod(b);
      return a;
    });
  });
  registry.add("arith.negf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatUnary(i, c, o, [](const llvm::APFloat &a) { return -a; });
  });
  // `maximumf`/`minimumf` are IEEE 754-2019 maximum/minimum: NaN propagates,
  // and -0 sorts below +0. The latter needs an explicit zero case, which is
  // why these go through APFloat's helpers rather than a bare comparison.
  registry.add("arith.maximumf", [](Interpreter &i, CoreState &c,
                                    Operation *o) {
    return floatBinary(i, c, o, [](const llvm::APFloat &a,
                                   const llvm::APFloat &b) {
      return llvm::maximum(a, b);
    });
  });
  registry.add("arith.minimumf", [](Interpreter &i, CoreState &c,
                                    Operation *o) {
    return floatBinary(i, c, o, [](const llvm::APFloat &a,
                                   const llvm::APFloat &b) {
      return llvm::minimum(a, b);
    });
  });
  registry.add("arith.maxnumf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](const llvm::APFloat &a,
                                   const llvm::APFloat &b) {
      return llvm::maxnum(a, b);
    });
  });
  registry.add("arith.minnumf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](const llvm::APFloat &a,
                                   const llvm::APFloat &b) {
      return llvm::minnum(a, b);
    });
  });

  registry.add("arith.cmpi", execCmpI);
  registry.add("arith.cmpf", execCmpF);
  registry.add("arith.select", execSelect);

  auto addCast = [&](StringRef name, bool isUnsigned, InterpRoundMode mode) {
    registry.add(name, [isUnsigned, mode](Interpreter &i, CoreState &c,
                                          Operation *o) {
      return execCast(i, c, o, isUnsigned, mode);
    });
  };
  addCast("arith.extsi", false, InterpRoundMode::RINT);
  addCast("arith.extui", true, InterpRoundMode::RINT);
  addCast("arith.trunci", false, InterpRoundMode::TRUNC);
  addCast("arith.index_cast", false, InterpRoundMode::RINT);
  addCast("arith.index_castui", true, InterpRoundMode::RINT);
  addCast("arith.sitofp", false, InterpRoundMode::RINT);
  addCast("arith.uitofp", true, InterpRoundMode::RINT);
  addCast("arith.fptosi", false, InterpRoundMode::TRUNC);
  addCast("arith.fptoui", true, InterpRoundMode::TRUNC);
  addCast("arith.extf", false, InterpRoundMode::RINT);

  // `arith.truncf` may carry an explicit HIVM/HFusion rounding mode.
  registry.add("arith.truncf", [](Interpreter &interp, CoreState &core,
                                  Operation *op) {
    InterpRoundMode mode = InterpRoundMode::RINT;
    if (auto attr = op->getAttr("round_mode")) {
      // The attribute prints as #hfusion.round_mode<rint>; match on its text
      // so the interpreter does not depend on the HFusion enum's numbering.
      std::string text;
      llvm::raw_string_ostream os(text);
      attr.print(os);
      StringRef s = os.str();
      if (s.contains("floor"))
        mode = InterpRoundMode::FLOOR;
      else if (s.contains("ceil"))
        mode = InterpRoundMode::CEIL;
      else if (s.contains("odd"))
        mode = InterpRoundMode::ODD;
      else if (s.contains("round"))
        mode = InterpRoundMode::ROUND;
      else if (s.contains("trunc"))
        mode = InterpRoundMode::TRUNC;
    }
    return execCast(interp, core, op, /*isUnsigned=*/false, mode);
  });

  registry.add("arith.bitcast", [](Interpreter &interp, CoreState &core,
                                   Operation *op) {
    RuntimeValue in = interp.getValue(core, op->getOperand(0));
    Type toType = getScalarType(op->getResult(0).getType());
    llvm::APInt bits = in.isFloat() ? in.getFloatValue().bitcastToAPInt()
                                    : in.getIntValue();
    if (const llvm::fltSemantics *sem = getFloatSemantics(toType))
      interp.setValue(core, op->getResult(0),
                      RuntimeValue::getFloat(llvm::APFloat(*sem, bits)));
    else
      interp.setValue(core, op->getResult(0), RuntimeValue::getInt(bits));
    return ExecResult::Advance;
  });

  // --- math ---
  auto addMath = [&](StringRef name, double (*fn)(double)) {
    registry.add(name, [fn](Interpreter &i, CoreState &c, Operation *o) {
      return floatUnaryViaDouble(i, c, o, fn);
    });
  };
  addMath("math.exp", [](double x) { return std::exp(x); });
  addMath("math.exp2", [](double x) { return std::exp2(x); });
  addMath("math.expm1", [](double x) { return std::expm1(x); });
  addMath("math.log", [](double x) { return std::log(x); });
  addMath("math.log2", [](double x) { return std::log2(x); });
  addMath("math.log10", [](double x) { return std::log10(x); });
  addMath("math.log1p", [](double x) { return std::log1p(x); });
  addMath("math.sqrt", [](double x) { return std::sqrt(x); });
  addMath("math.rsqrt", [](double x) { return 1.0 / std::sqrt(x); });
  addMath("math.sin", [](double x) { return std::sin(x); });
  addMath("math.cos", [](double x) { return std::cos(x); });
  addMath("math.tan", [](double x) { return std::tan(x); });
  addMath("math.tanh", [](double x) { return std::tanh(x); });
  addMath("math.atan", [](double x) { return std::atan(x); });
  addMath("math.erf", [](double x) { return std::erf(x); });
  addMath("math.ceil", [](double x) { return std::ceil(x); });
  addMath("math.floor", [](double x) { return std::floor(x); });
  addMath("math.round", [](double x) { return std::round(x); });
  addMath("math.roundeven", [](double x) { return std::nearbyint(x); });
  addMath("math.trunc", [](double x) { return std::trunc(x); });
  registry.add("math.absf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatUnary(i, c, o, [](llvm::APFloat a) {
      a.clearSign();
      return a;
    });
  });
  registry.add("math.absi", [](Interpreter &interp, CoreState &core,
                               Operation *op) {
    RuntimeValue in = interp.getValue(core, op->getOperand(0));
    if (!in.isInt()) {
      interp.emitError(op) << "math.absi operand is unbound";
      return ExecResult::Error;
    }
    llvm::APInt v = in.getIntValue();
    interp.setValue(core, op->getResult(0),
                    RuntimeValue::getInt(v.isNegative() ? -v : v));
    return ExecResult::Advance;
  });
  registry.add("math.powf", [](Interpreter &i, CoreState &c, Operation *o) {
    return floatBinary(i, c, o, [](const llvm::APFloat &a,
                                   const llvm::APFloat &b) {
      llvm::APFloat result(std::pow(a.convertToDouble(), b.convertToDouble()));
      bool losesInfo = false;
      result.convert(a.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                     &losesInfo);
      return result;
    });
  });
  registry.add("math.fma", [](Interpreter &interp, CoreState &core,
                              Operation *op) {
    RuntimeValue a = interp.getValue(core, op->getOperand(0));
    RuntimeValue b = interp.getValue(core, op->getOperand(1));
    RuntimeValue c = interp.getValue(core, op->getOperand(2));
    if (!a.isFloat() || !b.isFloat() || !c.isFloat()) {
      interp.emitError(op) << "math.fma operands are unbound";
      return ExecResult::Error;
    }
    llvm::APFloat result = a.getFloatValue();
    result.fusedMultiplyAdd(b.getFloatValue(), c.getFloatValue(),
                            llvm::APFloat::rmNearestTiesToEven);
    interp.setValue(core, op->getResult(0), RuntimeValue::getFloat(result));
    return ExecResult::Advance;
  });

  // --- index (same semantics as arith on 64-bit values) ---
  registry.add("index.constant", execArithConstant);
  registry.add("index.add", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a + b; });
  });
  registry.add("index.sub", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a - b; });
  });
  registry.add("index.mul", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a * b; });
  });
  registry.add("index.divs", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.sdiv(b);
    });
  });
  registry.add("index.rems", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) {
      return b.isZero() ? llvm::APInt(a.getBitWidth(), 0) : a.srem(b);
    });
  });
  registry.add("index.maxs", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.sgt(b) ? a : b; });
  });
  registry.add("index.mins", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a.slt(b) ? a : b; });
  });
  registry.add("index.casts", [](Interpreter &i, CoreState &c, Operation *o) {
    return execCast(i, c, o, false, InterpRoundMode::RINT);
  });
  registry.add("index.castu", [](Interpreter &i, CoreState &c, Operation *o) {
    return execCast(i, c, o, true, InterpRoundMode::RINT);
  });

  // --- memref ---
  registry.add("memref.alloc", execAlloc);
  registry.add("memref.alloca", execAlloc);
  registry.add("memref.dealloc", execDealloc);
  registry.add("memref.load", execMemRefLoad);
  registry.add("memref.store", execMemRefStore);
  registry.add("memref.subview", execSubView);
  registry.add("memref.view", execView);
  registry.add("memref.reinterpret_cast", execReinterpretCast);
  registry.add("memref.cast", execMemRefRetype);
  registry.add("memref.memory_space_cast", execMemRefRetype);
  registry.add("memref.dim", execDim);
  registry.add("memref.collapse_shape", [](Interpreter &i, CoreState &c,
                                           Operation *o) {
    return execReshape(i, c, o, /*expanding=*/false);
  });
  registry.add("memref.expand_shape", [](Interpreter &i, CoreState &c,
                                         Operation *o) {
    return execReshape(i, c, o, /*expanding=*/true);
  });
  registry.add("memref.assume_alignment", execNoOp);
  registry.add("memref.prefetch", execNoOp);

  // --- llvm (raw-pointer scalar code left behind by lowering) ---
  registry.add("llvm.mlir.constant", [](Interpreter &interp, CoreState &core,
                                        Operation *op) {
    RuntimeValue value;
    if (!materializeConstant(interp, cast<LLVM::ConstantOp>(op).getValue(),
                             op->getResult(0).getType(), op, value))
      return ExecResult::Error;
    interp.setValue(core, op->getResult(0), value);
    return ExecResult::Advance;
  });
  registry.add("llvm.inttoptr", execIntToPtr);
  registry.add("llvm.ptrtoint", execPtrToInt);
  registry.add("llvm.load", execLLVMLoad);
  registry.add("llvm.store", execLLVMStore);
  registry.add("llvm.getelementptr", execGEP);
  registry.add("llvm.add", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a + b; });
  });
  registry.add("llvm.sub", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a - b; });
  });
  registry.add("llvm.mul", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a * b; });
  });
  registry.add("llvm.and", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a & b; });
  });
  registry.add("llvm.or", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a | b; });
  });
  registry.add("llvm.xor", [](Interpreter &i, CoreState &c, Operation *o) {
    return intBinary(i, c, o, [](auto a, auto b) { return a ^ b; });
  });
  registry.add("llvm.zext", [](Interpreter &i, CoreState &c, Operation *o) {
    return execCast(i, c, o, true, InterpRoundMode::RINT);
  });
  registry.add("llvm.sext", [](Interpreter &i, CoreState &c, Operation *o) {
    return execCast(i, c, o, false, InterpRoundMode::RINT);
  });
  registry.add("llvm.trunc", [](Interpreter &i, CoreState &c, Operation *o) {
    return execCast(i, c, o, false, InterpRoundMode::TRUNC);
  });
  registry.add("llvm.fence", execNoOp);

  // Vectorized function bodies apply these same ops to vector<...> values.
  registry.transformMatching({"arith.", "math."}, vectorize);
}

} // namespace interp
} // namespace bishengir
