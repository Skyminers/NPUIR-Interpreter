//===- ValueTest.cpp - MemRefValue addressing and element codec tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Address arithmetic decides which bytes every access touches, and the
// contiguity test decides whether the shadow memory records one interval or
// many. Both are easy to get subtly wrong and hard to see through IR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/Memory.h"
#include "bishengir/Tools/Interp/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace bishengir::interp;

namespace {

MemRefValue makeView(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                     unsigned elemBytes = 4, uint64_t byteOffset = 0) {
  MemRefValue view;
  view.sizes.assign(sizes.begin(), sizes.end());
  view.strides.assign(strides.begin(), strides.end());
  view.elemBytes = elemBytes;
  view.byteOffset = byteOffset;
  view.baseOffset = byteOffset;
  view.arena = 0;
  return view;
}

//===----------------------------------------------------------------------===//
// Contiguity
//===----------------------------------------------------------------------===//

TEST(MemRefValue, DenseRowMajorIsContiguous) {
  EXPECT_TRUE(makeView({4, 8}, {8, 1}).isContiguous());
  EXPECT_TRUE(makeView({16}, {1}).isContiguous());
  EXPECT_TRUE(makeView({2, 3, 4}, {12, 4, 1}).isContiguous());
}

TEST(MemRefValue, GappedRowsAreNotContiguous) {
  // A row slice of a wider buffer: 8 useful elements every 128.
  EXPECT_FALSE(makeView({4, 8}, {128, 1}).isContiguous());
}

TEST(MemRefValue, StridedInnerDimIsNotContiguous) {
  EXPECT_FALSE(makeView({8}, {2}).isContiguous());
}

TEST(MemRefValue, UnitDimsDoNotBreakContiguity) {
  // Unit extents have an unconstrained stride; a rank-4 view with unit outer
  // dims over dense data is still one byte range.
  EXPECT_TRUE(makeView({1, 4, 8}, {999, 8, 1}).isContiguous());
  EXPECT_TRUE(makeView({4, 1, 8}, {8, 999, 1}).isContiguous());
  EXPECT_TRUE(makeView({1, 1, 1}, {7, 7, 7}).isContiguous());
}

TEST(MemRefValue, TransposedViewIsNotContiguous) {
  EXPECT_FALSE(makeView({4, 8}, {1, 4}).isContiguous());
}

//===----------------------------------------------------------------------===//
// Addressing
//===----------------------------------------------------------------------===//

TEST(MemRefValue, LinearIterationMatchesRowMajorIndexing) {
  MemRefValue view = makeView({3, 4}, {10, 2}, /*elemBytes=*/4,
                              /*byteOffset=*/64);
  int64_t pos = 0;
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 4; ++j, ++pos) {
      int64_t index[2] = {i, j};
      EXPECT_EQ(view.getByteAddrLinear(pos), view.getByteAddr(index))
          << "mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(MemRefValue, LinearIterationHandlesUnitAndRankOne) {
  MemRefValue rank1 = makeView({5}, {3}, 2, 8);
  for (int64_t n = 0; n < 5; ++n) {
    int64_t index[1] = {n};
    EXPECT_EQ(rank1.getByteAddrLinear(n), rank1.getByteAddr(index));
  }
  MemRefValue withUnit = makeView({1, 5}, {100, 1}, 2, 8);
  for (int64_t n = 0; n < 5; ++n) {
    int64_t index[2] = {0, n};
    EXPECT_EQ(withUnit.getByteAddrLinear(n), withUnit.getByteAddr(index));
  }
}

TEST(MemRefValue, SpannedBytesCoversTheWholeView) {
  MemRefValue view = makeView({3, 4}, {10, 2}, 4, 64);
  auto [lo, hi] = view.getSpannedBytes();
  // First element at 64, last at 64 + (2*10 + 3*2)*4 = 64 + 104.
  EXPECT_EQ(lo, 64u);
  EXPECT_EQ(hi, 64u + 104u + 4u);
  // Every element must fall inside the reported span.
  for (int64_t n = 0; n < view.getNumElements(); ++n) {
    uint64_t addr = view.getByteAddrLinear(n);
    EXPECT_GE(addr, lo);
    EXPECT_LE(addr + view.elemBytes, hi);
  }
}

TEST(MemRefValue, ContiguousSpanIsExactlyTheElementCount) {
  MemRefValue view = makeView({4, 8}, {8, 1}, 4, 32);
  auto [lo, hi] = view.getSpannedBytes();
  EXPECT_EQ(lo, 32u);
  EXPECT_EQ(hi - lo, 4u * 8u * 4u);
}

//===----------------------------------------------------------------------===//
// Element codec
//===----------------------------------------------------------------------===//

TEST(Element, StorageSizes) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  // i1 occupies a whole byte so byte ranges stay exact.
  EXPECT_EQ(getStorageSize(b.getI1Type()), 1u);
  EXPECT_EQ(getStorageSize(b.getI8Type()), 1u);
  EXPECT_EQ(getStorageSize(b.getI16Type()), 2u);
  EXPECT_EQ(getStorageSize(b.getI32Type()), 4u);
  EXPECT_EQ(getStorageSize(b.getI64Type()), 8u);
  EXPECT_EQ(getStorageSize(b.getF16Type()), 2u);
  EXPECT_EQ(getStorageSize(b.getBF16Type()), 2u);
  EXPECT_EQ(getStorageSize(b.getF32Type()), 4u);
  EXPECT_EQ(getStorageSize(b.getF64Type()), 8u);
  EXPECT_EQ(getStorageSize(b.getIndexType()), 8u);
}

TEST(Element, FloatRoundTripsPreserveBits) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  struct Case {
    Type type;
    double value;
  };
  Type types[] = {b.getF16Type(), b.getBF16Type(), b.getF32Type(),
                  b.getF64Type()};
  for (Type type : types) {
    const llvm::fltSemantics *sem = getFloatSemantics(type);
    ASSERT_NE(sem, nullptr);
    for (double v : {0.0, 1.0, -1.0, 0.5, -2.25, 1024.0}) {
      llvm::APFloat original(v);
      bool losesInfo = false;
      original.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);

      uint8_t bytes[8] = {};
      storeElement(bytes, type, RuntimeValue::getFloat(original));
      RuntimeValue back = loadElement(bytes, type);
      ASSERT_TRUE(back.isFloat());
      EXPECT_TRUE(back.getFloatValue().bitwiseIsEqual(original))
          << "round trip changed the bits for " << v;
    }
  }
}

TEST(Element, NarrowFloatWritesOnlyItsOwnBytes) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  // A 2-byte store must not clobber the neighbouring element.
  uint8_t bytes[4] = {0xAA, 0xAA, 0xAA, 0xAA};
  llvm::APFloat one(llvm::APFloat::IEEEhalf(), "1.0");
  storeElement(bytes, b.getF16Type(), RuntimeValue::getFloat(one));
  EXPECT_EQ(bytes[2], 0xAA);
  EXPECT_EQ(bytes[3], 0xAA);
}

TEST(Element, IntegerRoundTripsIncludingNegatives) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  struct Case {
    Type type;
    unsigned width;
  };
  Case cases[] = {{b.getI8Type(), 8},
                  {b.getI16Type(), 16},
                  {b.getI32Type(), 32},
                  {b.getI64Type(), 64}};
  for (const Case &c : cases) {
    for (int64_t v : {0, 1, -1, 42, -42}) {
      llvm::APInt original(c.width, static_cast<uint64_t>(v),
                           /*isSigned=*/true);
      uint8_t bytes[8] = {};
      storeElement(bytes, c.type, RuntimeValue::getInt(original));
      RuntimeValue back = loadElement(bytes, c.type);
      ASSERT_TRUE(back.isInt());
      EXPECT_EQ(back.getIntValue().getSExtValue(), v)
          << "round trip failed at width " << c.width;
    }
  }
}

TEST(Element, BooleanRoundTrip) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  uint8_t bytes[1] = {0xFF};
  storeElement(bytes, b.getI1Type(), RuntimeValue::getBool(false));
  EXPECT_TRUE(loadElement(bytes, b.getI1Type()).getIntValue().isZero());
  storeElement(bytes, b.getI1Type(), RuntimeValue::getBool(true));
  EXPECT_FALSE(loadElement(bytes, b.getI1Type()).getIntValue().isZero());
}

TEST(Poison, FloatPatternIsQuietNaNOfTheRightFormat) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  for (Type type : {b.getF16Type(), b.getBF16Type(), b.getF32Type()}) {
    unsigned size = getStorageSize(type);
    uint8_t bytes[16] = {};
    fillPoison(bytes, sizeof(bytes), type);
    for (unsigned off = 0; off + size <= sizeof(bytes); off += size) {
      RuntimeValue v = loadElement(bytes + off, type);
      ASSERT_TRUE(v.isFloat());
      EXPECT_TRUE(v.getFloatValue().isNaN())
          << "poison at offset " << off << " is not NaN";
    }
  }
}

} // namespace
