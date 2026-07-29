//===- OpUtils.h - Shared helpers for interpreter op handlers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_OPUTILS_H
#define BISHENGIR_TOOLS_INTERP_OPUTILS_H

#include "bishengir/Tools/Interp/Interpreter.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace bishengir {
namespace interp {

/// Raw element read with bounds checking but no shadow recording. Used from
/// inside commit closures, where the access was already recorded at issue.
bool rawLoad(Interpreter &interp, const MemRefValue &mem, int64_t linearPos,
             mlir::Operation *op, RuntimeValue &out);

/// Raw element write, counterpart of rawLoad.
bool rawStore(Interpreter &interp, const MemRefValue &mem, int64_t linearPos,
              mlir::Operation *op, const RuntimeValue &value);

/// Element read at an explicit multi-dimensional index.
bool rawLoadAt(Interpreter &interp, const MemRefValue &mem,
               llvm::ArrayRef<int64_t> indices, mlir::Operation *op,
               RuntimeValue &out);

bool rawStoreAt(Interpreter &interp, const MemRefValue &mem,
                llvm::ArrayRef<int64_t> indices, mlir::Operation *op,
                const RuntimeValue &value);

/// Bounds-checked pointer to one element, or null (after emitting an OOB
/// diagnostic) when the address escapes its arena.
uint8_t *elementPtr(Interpreter &interp, const MemRefValue &mem,
                    uint64_t byteAddr, mlir::Operation *op);

/// Resolve an index-typed operand.
bool getIndex(Interpreter &interp, CoreState &core, mlir::Value value,
              mlir::Operation *op, int64_t &out);

/// Resolve an `OpFoldResult` that is either a static attribute or an operand.
bool getFoldedIndex(Interpreter &interp, CoreState &core,
                    mlir::OpFoldResult ofr, mlir::Operation *op, int64_t &out);

/// Convert a RuntimeValue to the representation of `type`, applying the given
/// HIVM rounding mode for float narrowing / float-to-int conversions.
enum class InterpRoundMode { RINT, ROUND, FLOOR, CEIL, TRUNC, ODD, TRUNC_OVF };

RuntimeValue convertValue(const RuntimeValue &in, mlir::Type fromType,
                          mlir::Type toType, InterpRoundMode mode,
                          bool isUnsigned);

/// Iterate the destination shape in row-major order, calling `body` with the
/// current multi-dimensional index.
void forEachIndex(llvm::ArrayRef<int64_t> sizes,
                  llvm::function_ref<void(llvm::ArrayRef<int64_t>)> body);

/// Map a source index from a destination index, collapsing broadcast
/// dimensions and applying `transpose` when non-empty.
void mapSourceIndex(llvm::ArrayRef<int64_t> dstIndex,
                    const MemRefValue &src,
                    llvm::ArrayRef<int64_t> transpose,
                    llvm::SmallVectorImpl<int64_t> &out);

/// Element type of `type`, unwrapping vectors and tensors. Handlers that
/// inspect operand or result *types* must go through this: under vector
/// lifting the values are single lanes while the types are still shaped, and
/// reading `vector<8xf32>` as a float type silently yields nothing.
mlir::Type getScalarType(mlir::Type type);

/// Number of lanes implied by a vector shape.
int64_t getVectorElementCount(llvm::ArrayRef<int64_t> shape);

/// A vector of `shape` with every lane set to `value`.
RuntimeValue makeSplatVector(llvm::ArrayRef<int64_t> shape,
                             const RuntimeValue &value);

/// Pipe an op runs on, from `hivm::OpPipeInterface`. For a macro op this is
/// its *output* pipe. Falls back to PIPE_V for unpiped vector ops and to
/// `fallback` otherwise.
Pipe getOpPipe(mlir::Operation *op, Pipe fallback);

/// Input pipe of a macro op: the one that stages its operands before the
/// output pipe consumes them (MTE1 ahead of the cube, for `mmad_l1`). Returns
/// `fallback` for anything that is not a macro op.
Pipe getOpInPipe(mlir::Operation *op, Pipe fallback);

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_OPUTILS_H
