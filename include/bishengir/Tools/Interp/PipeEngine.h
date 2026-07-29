//===- PipeEngine.h - Deferred pipe effects ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The heart of the tool. On real hardware `hivm.hir.load` returns as soon as
// it is issued to MTE2; the data only lands in UB when the matching
// set_flag/wait_flag pair says so. If the interpreter committed every op in
// program order it would impose a stronger ordering than the machine and
// could never observe a missing flag. So ops carrying a pipe push a deferred
// Effect onto that pipe's queue, and the queue is only drained at the points
// where the hardware would drain it.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_PIPEENGINE_H
#define BISHENGIR_TOOLS_INTERP_PIPEENGINE_H

#include "bishengir/Tools/Interp/Memory.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <deque>
#include <functional>

namespace bishengir {
namespace interp {

/// Hardware pipes, mirroring `hivm::PIPE`. Kept as a local enum so the
/// interpreter can index arrays densely and so that PIPE_ALL / unassigned are
/// handled explicitly rather than by accident.
enum class Pipe : uint8_t {
  S = 0,
  V,
  M,
  MTE1,
  MTE2,
  MTE3,
  MTE4,
  MTE5,
  V2,
  FIX,
  NumPipes
};

static constexpr unsigned kNumPipes = static_cast<unsigned>(Pipe::NumPipes);

llvm::StringRef getPipeName(Pipe pipe);

/// Identity of an intra-core flag: `set_flag[set, wait, event]`.
struct FlagKey {
  Pipe setPipe = Pipe::S;
  Pipe waitPipe = Pipe::S;
  int64_t eventId = 0;

  bool operator==(const FlagKey &o) const {
    return setPipe == o.setPipe && waitPipe == o.waitPipe &&
           eventId == o.eventId;
  }
  bool operator<(const FlagKey &o) const {
    if (setPipe != o.setPipe)
      return setPipe < o.setPipe;
    if (waitPipe != o.waitPipe)
      return waitPipe < o.waitPipe;
    return eventId < o.eventId;
  }
};

/// A byte range within a specific arena.
struct ByteRange {
  int arena = -1;
  uint64_t lo = 0;
  uint64_t hi = 0;
};

/// A pipe operation that has been issued but not yet completed. `commit`
/// performs the real data movement; `reads`/`writes` are what the shadow
/// memory records, stamped with the clock the op had when it was issued.
struct Effect {
  mlir::Operation *op = nullptr;
  Pipe pipe = Pipe::S;
  llvm::SmallVector<ByteRange, 2> reads;
  llvm::SmallVector<ByteRange, 2> writes;
  std::function<void()> commit;
  VectorClock issueClock;
  /// Set for a hardware atomic read-modify-write, so the race detector does
  /// not flag two atomics against each other.
  bool isAtomic = false;
  /// Set for an access made through a raw pointer, i.e. `llvm.load`/
  /// `llvm.store`. That is how the cross-core flag words are poked, so two
  /// raw accesses to the same address are the synchronisation mechanism
  /// rather than a bug.
  bool isRawPointer = false;

  /// Set for an effect whose data movement and shadow-memory record have
  /// already happened. PIPE_S is the issuing unit and never defers, so its
  /// effects complete the moment they are issued - but the *other* pipes are
  /// not entitled to see the result until a `set_flag` on PIPE_S says so.
  /// Such an effect therefore stays queued purely as a marker, and committing
  /// it a second time when the queue drains would double-count the access.
  bool isResident = false;

  /// Set for the marker pushed by `set_flag`. A `wait_flag` drains the queue
  /// up to and including its matching token.
  bool isToken = false;
  FlagKey token;

  /// The single range this effect covers, or null when it covers none or
  /// several. Only single-range effects can be coalesced.
  ByteRange *getSoleRange() {
    if (reads.size() == 1 && writes.empty())
      return &reads.front();
    if (writes.size() == 1 && reads.empty())
      return &writes.front();
    return nullptr;
  }
  bool isWriteEffect() const { return !writes.empty(); }
};

/// Per-core queues plus the intra-core flag state.
class PipeEngine {
public:
  PipeEngine() = default;

  std::deque<Effect> &getQueue(Pipe pipe) {
    return queues[static_cast<unsigned>(pipe)];
  }
  const std::deque<Effect> &getQueue(Pipe pipe) const {
    return queues[static_cast<unsigned>(pipe)];
  }

  void push(Pipe pipe, Effect effect) {
    queues[static_cast<unsigned>(pipe)].push_back(std::move(effect));
  }

  bool empty() const {
    for (const auto &q : queues)
      if (!q.empty())
        return false;
    return true;
  }

  /// Number of effects still in flight, for the summary report. Resident
  /// markers are excluded: their data has already landed, so leaving one
  /// behind is not a lost write.
  size_t pendingCount() const {
    size_t n = 0;
    for (const auto &q : queues)
      for (const Effect &e : q)
        n += e.isResident ? 0 : 1;
    return n;
  }

  /// True when `pipe` holds a token matching `key` somewhere in its queue.
  bool hasToken(Pipe pipe, const FlagKey &key) const;

  /// Fold `effect` into a resident marker already queued on its pipe that it
  /// continues, and report whether that succeeded. A scalar loop walking a
  /// buffer one element at a time would otherwise queue one marker per
  /// element and the queue would grow without bound.
  bool coalesceResident(Effect &effect);

  // --- intra-core flag semaphores -----------------------------------------

  int64_t getFlagCount(const FlagKey &key) const {
    auto it = flagSem.find(key);
    return it == flagSem.end() ? 0 : it->second;
  }
  void incFlag(const FlagKey &key, const VectorClock &clock);
  void decFlag(const FlagKey &key) { --flagSem[key]; }
  const VectorClock *getFlagClock(const FlagKey &key) const {
    auto it = flagClock.find(key);
    return it == flagClock.end() ? nullptr : &it->second;
  }

  /// Ordered so that leak reports are reproducible.
  const std::map<FlagKey, int64_t> &getFlagCounts() const { return flagSem; }

private:
  std::array<std::deque<Effect>, kNumPipes> queues;
  std::map<FlagKey, int64_t> flagSem;
  std::map<FlagKey, VectorClock> flagClock;
};

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_PIPEENGINE_H
