//===- Value.h - NPUIR interpreter runtime values ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime value representation for the NPUIR (HIVM) interpreter: scalars are
// kept in APInt/APFloat so that f16/bf16/f8 arithmetic and the six HIVM
// rounding modes stay bit-exact, and memrefs are strided views into a byte
// arena.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_VALUE_H
#define BISHENGIR_TOOLS_INTERP_VALUE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace bishengir {
namespace interp {

//===----------------------------------------------------------------------===//
// Address spaces
//===----------------------------------------------------------------------===//

/// Physical memory pools of an Ascend core. Mirrors
/// `hivm::AddressSpace` but adds `Host` for interpreter-private scratch
/// (`memref.alloca` in scalar code, lock variables synthesised by the driver).
enum class AddrSpace : uint8_t {
  GM = 0,
  UB,
  L1,
  L0A,
  L0B,
  L0C,
  SSBUF,
  Host,
  NumSpaces
};

llvm::StringRef getAddrSpaceName(AddrSpace space);

/// True for pools that are private to a single AI core. GM and SSBUF are
/// shared by every core in the kernel; UB is shared by the sub-blocks of one
/// AIV (which is exactly why UB still needs race checking).
bool isCoreLocal(AddrSpace space);

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

/// Storage size of `type` in bytes as the interpreter lays it out. `i1` is
/// stored as one byte, everything else is `bitwidth / 8` rounded up.
unsigned getStorageSize(mlir::Type type);

/// The APFloat semantics for a float type, or null for non-float types.
const llvm::fltSemantics *getFloatSemantics(mlir::Type type);

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

/// Layout tag tracked alongside a memref view. Stage-1 layout checking keeps
/// data in logical ND order and only validates that producer and consumer
/// agree on the tag (see the plan's §12 two-stage strategy).
enum class LayoutTag : uint8_t { ND = 0, ZN, NZ, DOTA_ND, DOTB_ND, DOTC_ND };

llvm::StringRef getLayoutTagName(LayoutTag tag);

/// A strided view into an arena. `byteOffset` is absolute within the arena;
/// `strides` are in elements, matching MLIR's strided layout convention.
struct MemRefValue {
  int arena = -1;
  uint64_t byteOffset = 0;
  /// Byte offset of the *underlying allocation* this view sits in.
  /// `memref.reinterpret_cast` sets its result's offset relative to the
  /// source's base pointer rather than adding to the source's current
  /// offset, so the base has to be tracked separately from `byteOffset`.
  /// `memref.view` and `hivm.hir.pointer_cast` rebase; subviews do not.
  uint64_t baseOffset = 0;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<int64_t, 4> strides;
  mlir::Type elemType;
  unsigned elemBytes = 0;
  AddrSpace space = AddrSpace::GM;
  LayoutTag layout = LayoutTag::ND;

  bool isValid() const { return arena >= 0; }
  int64_t getRank() const { return static_cast<int64_t>(sizes.size()); }

  /// Number of logical elements covered by the view.
  int64_t getNumElements() const {
    int64_t n = 1;
    for (int64_t s : sizes)
      n *= s;
    return n;
  }

  /// Byte address of the element at `indices` (which must have rank entries).
  uint64_t getByteAddr(llvm::ArrayRef<int64_t> indices) const {
    int64_t elemDelta = 0;
    for (size_t i = 0, e = indices.size(); i < e; ++i)
      elemDelta += indices[i] * strides[i];
    return byteOffset + static_cast<uint64_t>(elemDelta) * elemBytes;
  }

  /// Byte address of the linear element `pos` in row-major iteration order.
  uint64_t getByteAddrLinear(int64_t pos) const {
    int64_t elemDelta = 0;
    for (int64_t d = getRank() - 1; d >= 0; --d) {
      int64_t sz = sizes[d];
      if (sz <= 0)
        continue;
      elemDelta += (pos % sz) * strides[d];
      pos /= sz;
    }
    return byteOffset + static_cast<uint64_t>(elemDelta) * elemBytes;
  }

  /// True when the view covers a single contiguous byte range, which lets the
  /// shadow memory record one interval instead of one per row.
  bool isContiguous() const;

  /// Half-open byte range [lo, hi) spanned by the view, ignoring holes.
  std::pair<uint64_t, uint64_t> getSpannedBytes() const;
};

//===----------------------------------------------------------------------===//
// RuntimeValue
//===----------------------------------------------------------------------===//

/// A value bound to an SSA `Value` during execution. Index values are held as
/// 64-bit signed APInts.
class RuntimeValue {
public:
  enum class Kind : uint8_t { None, Int, Float, MemRef, Vector };

  RuntimeValue() = default;

  static RuntimeValue getInt(llvm::APInt v) {
    RuntimeValue rv;
    rv.kind = Kind::Int;
    rv.ival = std::move(v);
    return rv;
  }
  static RuntimeValue getIndex(int64_t v) {
    return getInt(llvm::APInt(64, static_cast<uint64_t>(v), /*isSigned=*/true));
  }
  static RuntimeValue getBool(bool v) { return getInt(llvm::APInt(1, v)); }
  static RuntimeValue getFloat(llvm::APFloat v) {
    RuntimeValue rv;
    rv.kind = Kind::Float;
    rv.fval = std::move(v);
    return rv;
  }
  static RuntimeValue getMemRef(MemRefValue v) {
    RuntimeValue rv;
    rv.kind = Kind::MemRef;
    rv.mval = std::move(v);
    return rv;
  }
  /// An SSA vector value: elements in row-major order plus its shape. Used by
  /// the `vector` dialect ops in vectorized (VF) function bodies.
  static RuntimeValue getVector(llvm::SmallVector<int64_t, 4> shape,
                                std::vector<RuntimeValue> elements) {
    RuntimeValue rv;
    rv.kind = Kind::Vector;
    rv.vshape = std::move(shape);
    rv.velems = std::move(elements);
    return rv;
  }

  Kind getKind() const { return kind; }
  bool isNone() const { return kind == Kind::None; }
  bool isInt() const { return kind == Kind::Int; }
  bool isFloat() const { return kind == Kind::Float; }
  bool isMemRef() const { return kind == Kind::MemRef; }
  bool isVector() const { return kind == Kind::Vector; }

  const llvm::APInt &getIntValue() const { return ival; }
  const llvm::APFloat &getFloatValue() const { return *fval; }
  const MemRefValue &getMemRefValue() const { return mval; }
  MemRefValue &getMemRefValue() { return mval; }
  llvm::ArrayRef<int64_t> getVectorShape() const { return vshape; }
  const std::vector<RuntimeValue> &getVectorElements() const { return velems; }
  std::vector<RuntimeValue> &getVectorElements() { return velems; }

  /// Interpret an integer value as a signed 64-bit index.
  int64_t getIndexValue() const { return ival.getSExtValue(); }

  /// Numeric value as a double, for diagnostics and index-ish scalars.
  double toDouble() const;

  void print(llvm::raw_ostream &os) const;

private:
  Kind kind = Kind::None;
  llvm::APInt ival;
  std::optional<llvm::APFloat> fval;
  MemRefValue mval;
  llvm::SmallVector<int64_t, 4> vshape;
  std::vector<RuntimeValue> velems;
};

//===----------------------------------------------------------------------===//
// Raw element access
//===----------------------------------------------------------------------===//

/// Decode `bytes` (at least getStorageSize(type) long) as a value of `type`.
RuntimeValue loadElement(const uint8_t *bytes, mlir::Type type);

/// Encode `value` into `bytes` using the storage layout of `type`.
void storeElement(uint8_t *bytes, mlir::Type type, const RuntimeValue &value);

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_VALUE_H
