//===- PipeEngineTest.cpp - Pipe queue and resident-marker tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A PIPE_S access completes at once but stays queued as a marker so the other
// pipes still see it. Coalescing those markers is what keeps a scalar loop
// from queueing one per element, and getting it slightly wrong either leaks
// memory or invents a hazard over bytes nobody touched - neither of which a
// .mlir test would show clearly. So the merge rules get a direct test.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/PipeEngine.h"

#include "gtest/gtest.h"

using namespace bishengir::interp;

namespace {

/// A resident marker on PIPE_S covering [lo, hi) of arena 0.
Effect makeMarker(mlir::Operation *op, uint64_t lo, uint64_t hi, bool isWrite,
                  int arena = 0) {
  Effect effect;
  effect.op = op;
  effect.pipe = Pipe::S;
  effect.isResident = true;
  ByteRange range{arena, lo, hi};
  (isWrite ? effect.writes : effect.reads).push_back(range);
  return effect;
}

/// Distinct non-null Operation pointers. Never dereferenced - coalescing only
/// compares identity - so fabricating them beats standing up an MLIRContext.
mlir::Operation *fakeOp(uintptr_t id) {
  return reinterpret_cast<mlir::Operation *>(id * sizeof(void *));
}

const ByteRange &soleRange(const Effect &effect) {
  return effect.writes.empty() ? effect.reads.front() : effect.writes.front();
}

TEST(PipeEngineCoalesce, AbuttingRangesFromTheSameOpMerge) {
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 4, /*isWrite=*/true));
  for (uint64_t lo = 4; lo < 64; lo += 4) {
    Effect next = makeMarker(fakeOp(1), lo, lo + 4, /*isWrite=*/true);
    EXPECT_TRUE(engine.coalesceResident(next));
  }
  ASSERT_EQ(engine.getQueue(Pipe::S).size(), 1u);
  const ByteRange &range = soleRange(engine.getQueue(Pipe::S).front());
  EXPECT_EQ(range.lo, 0u);
  EXPECT_EQ(range.hi, 64u);
}

TEST(PipeEngineCoalesce, OverlappingRangesMerge) {
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 16, 32, /*isWrite=*/false));
  Effect next = makeMarker(fakeOp(1), 24, 48, /*isWrite=*/false);
  EXPECT_TRUE(engine.coalesceResident(next));
  const ByteRange &range = soleRange(engine.getQueue(Pipe::S).front());
  EXPECT_EQ(range.lo, 16u);
  EXPECT_EQ(range.hi, 48u);
}

TEST(PipeEngineCoalesce, GapPreventsMerge) {
  // Merging over the hole would claim bytes 8..15 are in flight when nothing
  // has touched them, which reads back as a hazard that does not exist.
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  Effect next = makeMarker(fakeOp(1), 16, 24, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, DifferentOpsDoNotMerge) {
  // The report names the op that is still in flight; merging two of them
  // would make it name the wrong one.
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  Effect next = makeMarker(fakeOp(2), 8, 16, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, ReadsAndWritesDoNotMerge) {
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/false));
  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, DifferentArenasDoNotMerge) {
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true,
                                  /*arena=*/0));
  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true, /*arena=*/1);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, TokenBlocksMerge) {
  // A `wait_flag` drains up to the token. Merging across it would retire the
  // later access early and hide whatever it conflicts with.
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  Effect token;
  token.pipe = Pipe::S;
  token.isToken = true;
  token.token = FlagKey{Pipe::S, Pipe::MTE3, 0};
  engine.push(Pipe::S, std::move(token));

  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, NonResidentEffectsAreNotMergeTargets) {
  PipeEngine engine;
  Effect deferred = makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true);
  deferred.isResident = false;
  engine.push(Pipe::S, std::move(deferred));

  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, MultiRangeEffectsAreNotCoalesced) {
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true);
  next.writes.push_back(ByteRange{0, 32, 40});
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngineCoalesce, OnlyTheTailIsScanned) {
  // Eight intervening markers push the first one out of reach; merging with
  // something that far back would span everything in between.
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  for (unsigned i = 0; i < 8; ++i)
    engine.push(Pipe::S, makeMarker(fakeOp(2 + i), 1000 + i * 8,
                                    1008 + i * 8, /*isWrite=*/true));
  Effect next = makeMarker(fakeOp(1), 8, 16, /*isWrite=*/true);
  EXPECT_FALSE(engine.coalesceResident(next));
}

TEST(PipeEngine, ResidentMarkersAreNotCountedAsInFlight) {
  // A marker left over at the end of a kernel is not a lost write, so it must
  // not show up in the "finished with N effects still queued" warning.
  PipeEngine engine;
  engine.push(Pipe::S, makeMarker(fakeOp(1), 0, 8, /*isWrite=*/true));
  EXPECT_EQ(engine.pendingCount(), 0u);

  Effect deferred = makeMarker(fakeOp(2), 16, 24, /*isWrite=*/true);
  deferred.isResident = false;
  deferred.pipe = Pipe::MTE2;
  engine.push(Pipe::MTE2, std::move(deferred));
  EXPECT_EQ(engine.pendingCount(), 1u);
}

} // namespace
