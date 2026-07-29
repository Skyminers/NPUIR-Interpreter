//===- Memory.h - NPUIR interpreter memory + race shadow --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Byte arenas (one per address space per owning core), poison filling,
// bounds/capacity checking, vector clocks and the interval-based shadow
// memory used for happens-before data-race detection.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_MEMORY_H
#define BISHENGIR_TOOLS_INTERP_MEMORY_H

#include "bishengir/Tools/Interp/Value.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace bishengir {
namespace interp {

//===----------------------------------------------------------------------===//
// VectorClock
//===----------------------------------------------------------------------===//

/// A per-core logical clock vector. Two accesses race when their clocks are
/// unordered (neither `<=` nor `>=`).
class VectorClock {
public:
  VectorClock() = default;
  explicit VectorClock(unsigned numCores) : ticks(numCores, 0) {}

  void resize(unsigned numCores) { ticks.resize(numCores, 0); }
  unsigned size() const { return static_cast<unsigned>(ticks.size()); }

  uint64_t operator[](unsigned core) const { return ticks[core]; }
  void tick(unsigned core) { ++ticks[core]; }

  /// Element-wise max: `this = this join other`.
  void join(const VectorClock &other);

  /// True when every component of `this` is <= the matching one of `other`.
  bool happensBefore(const VectorClock &other) const;

  /// True when neither clock dominates the other.
  bool isConcurrentWith(const VectorClock &other) const {
    return !happensBefore(other) && !other.happensBefore(*this);
  }

  void print(llvm::raw_ostream &os) const;

private:
  llvm::SmallVector<uint64_t, 4> ticks;
};

//===----------------------------------------------------------------------===//
// Shadow memory
//===----------------------------------------------------------------------===//

/// One recorded access, kept so a race report can name both sides.
struct AccessRecord {
  unsigned core = 0;
  VectorClock clock;
  mlir::Operation *op = nullptr;
  bool valid = false;
  /// True for accesses made through a raw `llvm.load`/`llvm.store` on an
  /// `inttoptr` address. That is the compiler's own flag scratchpad, where
  /// concurrent access is the mechanism rather than a bug.
  bool raw = false;
  /// True for a hardware read-modify-write (`hivm.hir.store atomic = <...>`).
  /// Two atomics on the same address are serialised by the hardware and do
  /// not race; an atomic against a plain access still does.
  bool atomic = false;
};

/// Per-interval access history. Reads are capped so that a widely shared
/// read-only buffer does not grow without bound.
struct ShadowEntry {
  AccessRecord lastWrite;
  llvm::SmallVector<AccessRecord, 2> reads;
};

/// A detected data race, ready for formatting.
struct RaceReport {
  AccessRecord first;
  bool firstIsWrite = false;
  AccessRecord second;
  bool secondIsWrite = false;
  AddrSpace space = AddrSpace::GM;
  std::string bufferName;
  uint64_t lo = 0, hi = 0;
};

/// Interval map from byte offset to access history. Intervals are split on
/// partial overlap, so a 8 MB tensor costs one entry, not eight million.
class ShadowMemory {
public:
  /// Record an access to [lo, hi). Returns a race report if the access is
  /// concurrent with a conflicting recorded access.
  std::optional<RaceReport> access(uint64_t lo, uint64_t hi, bool isWrite,
                                   const AccessRecord &record);

  /// Forget everything about [lo, hi) - used when a buffer is freed and its
  /// storage recycled.
  void invalidate(uint64_t lo, uint64_t hi);

  void clear() { entries.clear(); }

private:
  /// Split the interval containing `pos` so that an interval starts at `pos`.
  void splitAt(uint64_t pos);

  struct Interval {
    uint64_t end;
    ShadowEntry entry;
  };
  // Keyed by interval start; values carry the exclusive end. Ordered map so
  // that iteration (and therefore diagnostics) is deterministic.
  std::map<uint64_t, Interval> entries;
};

//===----------------------------------------------------------------------===//
// Arena
//===----------------------------------------------------------------------===//

/// A record of one live allocation, used for out-of-bounds diagnostics.
struct AllocRecord {
  uint64_t lo = 0;
  uint64_t hi = 0;
  std::string name;
  mlir::Operation *op = nullptr;
  bool live = true;
};

/// A flat byte pool backing one address space of one core.
class Arena {
public:
  Arena(AddrSpace space, unsigned ownerCore, uint64_t capacity,
        bool poisonOnAlloc);

  AddrSpace getSpace() const { return space; }
  unsigned getOwnerCore() const { return ownerCore; }
  uint64_t getCapacity() const { return capacity; }
  uint64_t getHighWaterMark() const { return bumpTop; }

  /// Bump-allocate `size` bytes with `align` alignment. Returns the byte
  /// offset, or std::nullopt when the arena would overflow.
  std::optional<uint64_t> allocate(uint64_t size, uint64_t align,
                                   llvm::StringRef name, mlir::Operation *op);

  void deallocate(uint64_t offset);

  /// Bounds-checked raw pointer to `offset`; null when out of range.
  uint8_t *at(uint64_t offset, uint64_t size);
  const uint8_t *at(uint64_t offset, uint64_t size) const;

  bool inBounds(uint64_t offset, uint64_t size) const {
    return offset <= storage.size() && size <= storage.size() - offset;
  }

  /// Name of the allocation covering `offset`, for diagnostics.
  std::string describeAddress(uint64_t offset) const;

  ShadowMemory &getShadow() { return shadow; }

  /// Fill [offset, offset+size) with the poison pattern for `type`.
  void poison(uint64_t offset, uint64_t size, mlir::Type type);

private:
  AddrSpace space;
  unsigned ownerCore;
  uint64_t capacity;
  bool poisonOnAlloc;
  uint64_t bumpTop = 0;
  std::vector<uint8_t> storage;
  std::vector<AllocRecord> allocs;
  ShadowMemory shadow;
};

/// Poison byte pattern for integer storage (0xCD repeated) and the qNaN
/// pattern used for float storage of the given type.
void fillPoison(uint8_t *bytes, uint64_t size, mlir::Type type);

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_MEMORY_H
