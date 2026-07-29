//===- Value.cpp - NPUIR interpreter runtime values -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/Value.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>

using namespace mlir;

namespace bishengir {
namespace interp {

StringRef getAddrSpaceName(AddrSpace space) {
  switch (space) {
  case AddrSpace::GM:
    return "gm";
  case AddrSpace::UB:
    return "ub";
  case AddrSpace::L1:
    return "cbuf";
  case AddrSpace::L0A:
    return "ca";
  case AddrSpace::L0B:
    return "cb";
  case AddrSpace::L0C:
    return "cc";
  case AddrSpace::SSBUF:
    return "ssbuf";
  case AddrSpace::Host:
    return "host";
  case AddrSpace::NumSpaces:
    break;
  }
  return "<invalid>";
}

bool isCoreLocal(AddrSpace space) {
  switch (space) {
  case AddrSpace::GM:
  case AddrSpace::SSBUF:
  case AddrSpace::Host:
    return false;
  case AddrSpace::UB:
  case AddrSpace::L1:
  case AddrSpace::L0A:
  case AddrSpace::L0B:
  case AddrSpace::L0C:
    return true;
  case AddrSpace::NumSpaces:
    break;
  }
  return false;
}

StringRef getLayoutTagName(LayoutTag tag) {
  switch (tag) {
  case LayoutTag::ND:
    return "ND";
  case LayoutTag::ZN:
    return "zN";
  case LayoutTag::NZ:
    return "nZ";
  case LayoutTag::DOTA_ND:
    return "DOTA_ND";
  case LayoutTag::DOTB_ND:
    return "DOTB_ND";
  case LayoutTag::DOTC_ND:
    return "DOTC_ND";
  }
  return "?";
}

unsigned getStorageSize(Type type) {
  // A null type reaches here only from a MemRefValue that was built without
  // an element type; answer defensively rather than dereferencing null.
  if (!type)
    return 1;
  if (isa<IndexType>(type))
    return 8;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    // i1 gets a whole byte: HIVM mask buffers are addressed bytewise and a
    // bit-packed representation would make every byte range approximate.
    unsigned width = intType.getWidth();
    return width <= 8 ? 1 : (width + 7) / 8;
  }
  if (auto floatType = dyn_cast<FloatType>(type))
    return (floatType.getWidth() + 7) / 8;
  // Pointer-like or opaque types are addressed as 8 bytes.
  return 8;
}

const llvm::fltSemantics *getFloatSemantics(Type type) {
  if (!type || !isa<FloatType>(type))
    return nullptr;
  if (isa<Float16Type>(type))
    return &llvm::APFloat::IEEEhalf();
  if (isa<BFloat16Type>(type))
    return &llvm::APFloat::BFloat();
  if (isa<Float32Type>(type))
    return &llvm::APFloat::IEEEsingle();
  if (isa<Float64Type>(type))
    return &llvm::APFloat::IEEEdouble();
  if (isa<Float8E4M3FNType>(type))
    return &llvm::APFloat::Float8E4M3FN();
  if (isa<Float8E5M2Type>(type))
    return &llvm::APFloat::Float8E5M2();
  return &cast<FloatType>(type).getFloatSemantics();
}

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

bool MemRefValue::isContiguous() const {
  // Walk dimensions from the fastest-varying one and check that each stride
  // equals the product of the trailing sizes. Unit-extent dims are skipped
  // because their stride is unconstrained.
  int64_t expected = 1;
  for (int64_t d = getRank() - 1; d >= 0; --d) {
    if (sizes[d] == 1)
      continue;
    if (sizes[d] <= 0)
      return false;
    if (strides[d] != expected)
      return false;
    expected *= sizes[d];
  }
  return true;
}

std::pair<uint64_t, uint64_t> MemRefValue::getSpannedBytes() const {
  int64_t minDelta = 0, maxDelta = 0;
  for (int64_t d = 0, e = getRank(); d < e; ++d) {
    if (sizes[d] <= 0)
      continue;
    int64_t extent = (sizes[d] - 1) * strides[d];
    if (extent < 0)
      minDelta += extent;
    else
      maxDelta += extent;
  }
  uint64_t lo = byteOffset + static_cast<uint64_t>(minDelta) * elemBytes;
  uint64_t hi = byteOffset + static_cast<uint64_t>(maxDelta) * elemBytes +
                elemBytes;
  return {lo, hi};
}

//===----------------------------------------------------------------------===//
// RuntimeValue
//===----------------------------------------------------------------------===//

double RuntimeValue::toDouble() const {
  switch (kind) {
  case Kind::Int:
    return static_cast<double>(ival.getSExtValue());
  case Kind::Float: {
    llvm::APFloat copy = *fval;
    bool losesInfo = false;
    copy.convert(llvm::APFloat::IEEEdouble(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return copy.convertToDouble();
  }
  case Kind::None:
  case Kind::MemRef:
  case Kind::Vector:
    // Not scalars; callers that reach here are asking the wrong question.
    break;
  }
  return 0.0;
}

void RuntimeValue::print(llvm::raw_ostream &os) const {
  switch (kind) {
  case Kind::None:
    os << "<unset>";
    return;
  case Kind::Int:
    os << ival.getSExtValue();
    return;
  case Kind::Float: {
    llvm::SmallString<32> buf;
    fval->toString(buf);
    os << buf;
    return;
  }
  case Kind::MemRef:
    os << "memref<";
    for (size_t i = 0; i < mval.sizes.size(); ++i)
      os << (i ? "x" : "") << mval.sizes[i];
    os << ", " << getAddrSpaceName(mval.space) << " @0x";
    os.write_hex(mval.byteOffset);
    os << ">";
    return;
  case Kind::Vector: {
    os << "vector<";
    for (size_t i = 0; i < vshape.size(); ++i)
      os << (i ? "x" : "") << vshape[i];
    os << ">[";
    // A few lanes are enough to tell values apart in a trace.
    size_t shown = std::min<size_t>(velems.size(), 4);
    for (size_t i = 0; i < shown; ++i) {
      if (i)
        os << ", ";
      velems[i].print(os);
    }
    if (velems.size() > shown)
      os << ", ...";
    os << ']';
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Raw element access
//===----------------------------------------------------------------------===//

RuntimeValue loadElement(const uint8_t *bytes, Type type) {
  unsigned size = getStorageSize(type);
  if (const llvm::fltSemantics *sem = getFloatSemantics(type)) {
    unsigned bits = llvm::APFloat::semanticsSizeInBits(*sem);
    llvm::APInt raw(bits, 0);
    uint64_t word = 0;
    std::memcpy(&word, bytes, std::min<unsigned>(size, sizeof(uint64_t)));
    raw = llvm::APInt(bits, word);
    return RuntimeValue::getFloat(llvm::APFloat(*sem, raw));
  }

  unsigned width = 64;
  if (auto intType = dyn_cast<IntegerType>(type))
    width = intType.getWidth();
  else if (isa<IndexType>(type))
    width = 64;

  uint64_t word = 0;
  std::memcpy(&word, bytes, std::min<unsigned>(size, sizeof(uint64_t)));
  if (width < 64)
    word &= (width == 64) ? ~0ull : ((1ull << width) - 1);
  return RuntimeValue::getInt(llvm::APInt(width, word));
}

void storeElement(uint8_t *bytes, Type type, const RuntimeValue &value) {
  unsigned size = getStorageSize(type);
  uint64_t word = 0;
  if (value.isFloat()) {
    llvm::APFloat fv = value.getFloatValue();
    if (const llvm::fltSemantics *sem = getFloatSemantics(type)) {
      if (&fv.getSemantics() != sem) {
        bool losesInfo = false;
        fv.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      }
    }
    word = fv.bitcastToAPInt().getZExtValue();
  } else if (value.isInt()) {
    word = value.getIntValue().getZExtValue();
  }
  std::memcpy(bytes, &word, std::min<unsigned>(size, sizeof(uint64_t)));
}

} // namespace interp
} // namespace bishengir
