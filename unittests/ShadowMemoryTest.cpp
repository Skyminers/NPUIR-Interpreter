//===- ShadowMemoryTest.cpp - Interval map and vector clock tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The interval splitting in ShadowMemory and the ordering rules in
// VectorClock decide every race verdict, and partial-overlap cases are hard
// to provoke reliably through IR. They get direct tests.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/Memory.h"

#include "gtest/gtest.h"

using namespace bishengir::interp;

namespace {

AccessRecord makeRecord(unsigned core, std::initializer_list<uint64_t> ticks) {
  AccessRecord record;
  record.core = core;
  record.valid = true;
  record.clock = VectorClock(static_cast<unsigned>(ticks.size()));
  unsigned i = 0;
  for (uint64_t t : ticks) {
    for (uint64_t n = 0; n < t; ++n)
      record.clock.tick(i);
    ++i;
  }
  return record;
}

//===----------------------------------------------------------------------===//
// VectorClock
//===----------------------------------------------------------------------===//

TEST(VectorClock, OrderingAndConcurrency) {
  VectorClock a(2), b(2);
  a.tick(0);
  b.tick(1);
  // <1,0> and <0,1> dominate each other in neither direction.
  EXPECT_TRUE(a.isConcurrentWith(b));
  EXPECT_FALSE(a.happensBefore(b));
  EXPECT_FALSE(b.happensBefore(a));

  // After a join, b covers a.
  b.join(a);
  EXPECT_TRUE(a.happensBefore(b));
  EXPECT_FALSE(b.happensBefore(a));
  EXPECT_FALSE(a.isConcurrentWith(b));
}

TEST(VectorClock, EqualClocksAreNotConcurrent) {
  VectorClock a(2), b(2);
  a.tick(0);
  b.tick(0);
  EXPECT_TRUE(a.happensBefore(b));
  EXPECT_TRUE(b.happensBefore(a));
  EXPECT_FALSE(a.isConcurrentWith(b));
}

TEST(VectorClock, MismatchedWidthsTreatMissingComponentsAsZero) {
  // A default-constructed clock has no components at all; it must still
  // compare as "before" a clock that has ticked, not as concurrent.
  VectorClock empty;
  VectorClock ticked(3);
  ticked.tick(2);
  EXPECT_TRUE(empty.happensBefore(ticked));
  EXPECT_FALSE(ticked.happensBefore(empty));
  EXPECT_FALSE(empty.isConcurrentWith(ticked));

  // Joining a wider clock into a narrower one must widen it.
  VectorClock narrow(1);
  narrow.join(ticked);
  EXPECT_EQ(narrow.size(), 3u);
  EXPECT_EQ(narrow[2], 1u);
}

//===----------------------------------------------------------------------===//
// ShadowMemory
//===----------------------------------------------------------------------===//

TEST(ShadowMemory, SameCoreNeverRaces) {
  ShadowMemory shadow;
  AccessRecord first = makeRecord(0, {1, 0});
  AccessRecord second = makeRecord(0, {2, 0});
  EXPECT_FALSE(shadow.access(0, 64, true, first));
  EXPECT_FALSE(shadow.access(0, 64, true, second));
}

TEST(ShadowMemory, OrderedAccessesDoNotRace) {
  ShadowMemory shadow;
  AccessRecord writer = makeRecord(0, {1, 0});
  // The reader has joined the writer's clock, as a flag pair would arrange.
  AccessRecord reader = makeRecord(1, {1, 1});
  EXPECT_FALSE(shadow.access(0, 64, true, writer));
  EXPECT_FALSE(shadow.access(0, 64, false, reader));
}

TEST(ShadowMemory, ConcurrentWriteThenReadRaces) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(0, {1, 0})));
  auto race = shadow.access(0, 64, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_TRUE(race->firstIsWrite);
  EXPECT_FALSE(race->secondIsWrite);
  EXPECT_EQ(race->first.core, 0u);
  EXPECT_EQ(race->second.core, 1u);
}

TEST(ShadowMemory, ConcurrentReadThenWriteRaces) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, false, makeRecord(0, {1, 0})));
  auto race = shadow.access(0, 64, true, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_FALSE(race->firstIsWrite);
  EXPECT_TRUE(race->secondIsWrite);
}

TEST(ShadowMemory, ConcurrentReadsDoNotRace) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, false, makeRecord(0, {1, 0})));
  EXPECT_FALSE(shadow.access(0, 64, false, makeRecord(1, {0, 1})));
}

TEST(ShadowMemory, PartialOverlapOnTheRightRaces) {
  // [0,64) written, then [32,96) read: the halves must be split so the
  // overlapping part still sees the earlier write.
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(0, {1, 0})));
  auto race = shadow.access(32, 96, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->lo, 32u);
  EXPECT_EQ(race->hi, 64u);
}

TEST(ShadowMemory, PartialOverlapOnTheLeftRaces) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(32, 96, true, makeRecord(0, {1, 0})));
  auto race = shadow.access(0, 64, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->lo, 32u);
  EXPECT_EQ(race->hi, 64u);
}

TEST(ShadowMemory, EnclosedRangeRaces) {
  // A wide write, then a narrow read strictly inside it.
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 256, true, makeRecord(0, {1, 0})));
  auto race = shadow.access(64, 128, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->lo, 64u);
  EXPECT_EQ(race->hi, 128u);
}

TEST(ShadowMemory, EnclosingRangeRaces) {
  // A narrow write, then a wide read that covers it plus untouched bytes.
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(64, 128, true, makeRecord(0, {1, 0})));
  auto race = shadow.access(0, 256, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->lo, 64u);
  EXPECT_EQ(race->hi, 128u);
}

TEST(ShadowMemory, DisjointRangesDoNotRace) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(0, {1, 0})));
  EXPECT_FALSE(shadow.access(64, 128, true, makeRecord(1, {0, 1})));
  // Touching but not overlapping, from the other side too.
  EXPECT_FALSE(shadow.access(128, 192, true, makeRecord(0, {2, 0})));
}

TEST(ShadowMemory, EmptyRangeIsIgnored) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(64, 64, true, makeRecord(0, {1, 0})));
  // Nothing was recorded, so a later concurrent access sees no conflict.
  EXPECT_FALSE(shadow.access(0, 128, true, makeRecord(1, {0, 1})));
}

TEST(ShadowMemory, WriteClearsEarlierReaders) {
  // Once a write is ordered after some reads, those reads must not keep
  // producing reports against later accesses.
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, false, makeRecord(0, {1, 0})));
  // A write that happens after the read (its clock dominates).
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(0, {2, 0})));
  // A later concurrent read only conflicts with the write, once.
  auto race = shadow.access(0, 64, false, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_TRUE(race->firstIsWrite);
}

TEST(ShadowMemory, ManyReadersFromOneCoreDoNotAccumulate) {
  // A read-only buffer touched repeatedly by one core must not grow an
  // unbounded history, and must still race against a concurrent write.
  ShadowMemory shadow;
  for (unsigned n = 1; n < 100; ++n)
    EXPECT_FALSE(shadow.access(0, 64, false, makeRecord(0, {n, 0})));
  auto race = shadow.access(0, 64, true, makeRecord(1, {0, 1}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->first.core, 0u);
}

TEST(ShadowMemory, InvalidateForgetsHistory) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(0, {1, 0})));
  shadow.invalidate(0, 64);
  // The storage was recycled, so the old writer must not be reported.
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(1, {0, 1})));
}

TEST(ShadowMemory, InvalidatePartialKeepsTheRest) {
  ShadowMemory shadow;
  EXPECT_FALSE(shadow.access(0, 128, true, makeRecord(0, {1, 0})));
  shadow.invalidate(0, 64);
  EXPECT_FALSE(shadow.access(0, 64, true, makeRecord(1, {0, 1})));
  auto race = shadow.access(64, 128, true, makeRecord(1, {0, 2}));
  ASSERT_TRUE(race.has_value());
  EXPECT_EQ(race->lo, 64u);
}

TEST(ShadowMemory, RepeatedSplitsStayConsistent) {
  // Interleave overlapping accesses so the interval map splits repeatedly,
  // then confirm every byte still remembers a conflicting writer.
  ShadowMemory shadow;
  for (uint64_t lo = 0; lo < 256; lo += 16)
    EXPECT_FALSE(shadow.access(lo, lo + 48, true, makeRecord(0, {1, 0})));
  for (uint64_t lo = 0; lo < 256; lo += 7) {
    auto race = shadow.access(lo, lo + 5, false, makeRecord(1, {0, 1}));
    if (lo + 5 <= 256 + 32)
      EXPECT_TRUE(race.has_value()) << "no race reported at " << lo;
  }
}

//===----------------------------------------------------------------------===//
// Poison
//===----------------------------------------------------------------------===//

TEST(Poison, IntegerPatternFillsEveryByte) {
  uint8_t bytes[7] = {};
  fillPoison(bytes, sizeof(bytes), nullptr);
  for (uint8_t b : bytes)
    EXPECT_EQ(b, 0xCD);
}

} // namespace
