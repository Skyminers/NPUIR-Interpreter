//===- Memory.cpp - Arenas, vector clocks, shadow memory --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/Memory.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>

using namespace mlir;

namespace bishengir {
namespace interp {

//===----------------------------------------------------------------------===//
// VectorClock
//===----------------------------------------------------------------------===//

void VectorClock::join(const VectorClock &other) {
  if (other.ticks.size() > ticks.size())
    ticks.resize(other.ticks.size(), 0);
  for (size_t i = 0, e = other.ticks.size(); i < e; ++i)
    ticks[i] = std::max(ticks[i], other.ticks[i]);
}

bool VectorClock::happensBefore(const VectorClock &other) const {
  size_t n = std::max(ticks.size(), other.ticks.size());
  for (size_t i = 0; i < n; ++i) {
    uint64_t a = i < ticks.size() ? ticks[i] : 0;
    uint64_t b = i < other.ticks.size() ? other.ticks[i] : 0;
    if (a > b)
      return false;
  }
  return true;
}

void VectorClock::print(llvm::raw_ostream &os) const {
  os << '<';
  for (size_t i = 0; i < ticks.size(); ++i)
    os << (i ? "," : "") << ticks[i];
  os << '>';
}

//===----------------------------------------------------------------------===//
// ShadowMemory
//===----------------------------------------------------------------------===//

void ShadowMemory::splitAt(uint64_t pos) {
  auto it = entries.upper_bound(pos);
  if (it == entries.begin())
    return;
  --it;
  // `it` is the last interval starting at or before pos.
  if (it->first == pos || it->second.end <= pos)
    return;
  Interval tail{it->second.end, it->second.entry};
  it->second.end = pos;
  entries.emplace(pos, std::move(tail));
}

std::optional<RaceReport> ShadowMemory::access(uint64_t lo, uint64_t hi,
                                               bool isWrite,
                                               const AccessRecord &record) {
  if (lo >= hi)
    return std::nullopt;

  splitAt(lo);
  splitAt(hi);

  std::optional<RaceReport> race;

  // Walk the covered intervals, filling any gaps with fresh entries.
  uint64_t cursor = lo;
  while (cursor < hi) {
    auto it = entries.lower_bound(cursor);
    if (it == entries.end() || it->first > cursor) {
      // Gap: create an interval up to the next existing one (or hi).
      uint64_t end = hi;
      if (it != entries.end())
        end = std::min(hi, it->first);
      it = entries.emplace(cursor, Interval{end, ShadowEntry{}}).first;
    }

    ShadowEntry &entry = it->second.entry;
    uint64_t end = std::min(it->second.end, hi);

    // A conflict is a write vs. any prior access, or a read vs. a prior
    // write, where the two clocks are unordered.
    if (!race && entry.lastWrite.valid &&
        entry.lastWrite.core != record.core &&
        entry.lastWrite.clock.isConcurrentWith(record.clock)) {
      RaceReport r;
      r.first = entry.lastWrite;
      r.firstIsWrite = true;
      r.second = record;
      r.secondIsWrite = isWrite;
      r.lo = it->first;
      r.hi = end;
      race = std::move(r);
    }
    if (!race && isWrite) {
      for (const AccessRecord &prior : entry.reads) {
        if (!prior.valid || prior.core == record.core)
          continue;
        if (!prior.clock.isConcurrentWith(record.clock))
          continue;
        RaceReport r;
        r.first = prior;
        r.firstIsWrite = false;
        r.second = record;
        r.secondIsWrite = true;
        r.lo = it->first;
        r.hi = end;
        race = std::move(r);
        break;
      }
    }

    if (isWrite) {
      entry.lastWrite = record;
      entry.reads.clear();
    } else {
      // At most one record per core, so the history is bounded by the core
      // count without needing eviction. Evicting instead would silently drop
      // one side of a real race once enough cores had read the interval.
      bool replaced = false;
      for (AccessRecord &prior : entry.reads) {
        if (prior.core == record.core) {
          prior = record;
          replaced = true;
          break;
        }
      }
      if (!replaced)
        entry.reads.push_back(record);
    }

    cursor = end;
  }

  return race;
}

void ShadowMemory::invalidate(uint64_t lo, uint64_t hi) {
  if (lo >= hi)
    return;
  splitAt(lo);
  splitAt(hi);
  auto first = entries.lower_bound(lo);
  auto last = entries.lower_bound(hi);
  entries.erase(first, last);
}

//===----------------------------------------------------------------------===//
// Poison
//===----------------------------------------------------------------------===//

void fillPoison(uint8_t *bytes, uint64_t size, Type type) {
  const llvm::fltSemantics *sem = type ? getFloatSemantics(type) : nullptr;
  if (!sem) {
    // 0xCD is the classic "uninitialised" pattern; as an integer it is a
    // large obviously-wrong value in every width.
    std::memset(bytes, 0xCD, size);
    return;
  }

  // A quiet NaN in the target format, so a missing flag surfaces as NaN
  // rather than as a small numeric discrepancy.
  llvm::APFloat nan = llvm::APFloat::getQNaN(*sem);
  llvm::APInt raw = nan.bitcastToAPInt();
  unsigned elemBytes = getStorageSize(type);
  uint64_t word = raw.getZExtValue();
  for (uint64_t off = 0; off + elemBytes <= size; off += elemBytes)
    std::memcpy(bytes + off, &word, elemBytes);
  // Tail bytes that do not form a whole element.
  uint64_t tail = size % elemBytes;
  if (tail)
    std::memset(bytes + size - tail, 0xCD, tail);
}

//===----------------------------------------------------------------------===//
// Arena
//===----------------------------------------------------------------------===//

Arena::Arena(AddrSpace space, unsigned ownerCore, uint64_t capacity,
             bool poisonOnAlloc)
    : space(space), ownerCore(ownerCore), capacity(capacity),
      poisonOnAlloc(poisonOnAlloc) {
  storage.assign(capacity, 0xCD);
}

std::optional<uint64_t> Arena::allocate(uint64_t size, uint64_t align,
                                        StringRef name, Operation *op) {
  if (align == 0)
    align = 1;
  uint64_t offset = (bumpTop + align - 1) / align * align;
  if (offset > capacity || size > capacity - offset)
    return std::nullopt;
  bumpTop = offset + size;
  allocs.push_back({offset, offset + size, name.str(), op, true});
  return offset;
}

void Arena::deallocate(uint64_t offset) {
  for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
    if (it->lo == offset && it->live) {
      it->live = false;
      shadow.invalidate(it->lo, it->hi);
      return;
    }
  }
}

uint8_t *Arena::at(uint64_t offset, uint64_t size) {
  if (!inBounds(offset, size))
    return nullptr;
  return storage.data() + offset;
}

const uint8_t *Arena::at(uint64_t offset, uint64_t size) const {
  if (!inBounds(offset, size))
    return nullptr;
  return storage.data() + offset;
}

std::string Arena::describeAddress(uint64_t offset) const {
  for (const AllocRecord &rec : allocs) {
    if (offset >= rec.lo && offset < rec.hi) {
      std::string out = rec.name.empty() ? std::string("<anon>") : rec.name;
      out += " +" + std::to_string(offset - rec.lo);
      return out;
    }
  }
  return "<unallocated>";
}

void Arena::poison(uint64_t offset, uint64_t size, Type type) {
  if (!poisonOnAlloc)
    return;
  uint8_t *p = at(offset, size);
  if (!p)
    return;
  fillPoison(p, size, type);
}

} // namespace interp
} // namespace bishengir
