//===- DialectCompat.cpp - Minimal AscendNPU-IR link shims ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The HIVM/HFusion IR libraries reference three predicates implemented by
// their much larger compiler utility libraries.  The interpreter needs the
// predicates for parsing/verifying ops, but none of the transformation stack.
// Keep the definitions here equivalent to the pinned AscendNPU-IR revision.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::hivm {

bool isVFCall(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op))
    return callOp->hasAttr(VectorFunctionAttr::name);
  return false;
}

namespace util {

bool isSIMTVF(Operation *op) {
  auto attr = op->getAttrOfType<VFModeAttr>(VFModeAttr::name);
  return attr && attr.getValue() == VFMode::SIMT;
}

} // namespace util
} // namespace mlir::hivm

namespace mlir::hfusion {

bool isFP8(Type type, Builder builder) {
  return type == builder.getFloat8E5M2Type() ||
         type == builder.getFloat8E4M3Type() ||
         type == builder.getFloat8E4M3FNType() ||
         type == builder.getFloat8E5M2FNUZType() ||
         type == builder.getFloat8E4M3FNUZType() ||
         type == builder.getFloat8E4M3B11FNUZType();
}

} // namespace mlir::hfusion
