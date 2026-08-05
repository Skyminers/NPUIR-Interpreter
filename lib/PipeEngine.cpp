//===- PipeEngine.cpp - Deferred pipe effects -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/PipeEngine.h"

#include <algorithm>

namespace bishengir {
namespace interp {

llvm::StringRef getPipeName(Pipe pipe) {
  switch (pipe) {
  case Pipe::S:
    return "PIPE_S";
  case Pipe::V:
    return "PIPE_V";
  case Pipe::M:
    return "PIPE_M";
  case Pipe::MTE1:
    return "PIPE_MTE1";
  case Pipe::MTE2:
    return "PIPE_MTE2";
  case Pipe::MTE3:
    return "PIPE_MTE3";
  case Pipe::FIX:
    return "PIPE_FIX";
  case Pipe::NumPipes:
    break;
  }
  return "PIPE_?";
}

bool PipeEngine::hasToken(Pipe pipe, const FlagKey &key) const {
  for (const Effect &effect : getQueue(pipe))
    if (effect.isToken && effect.token == key)
      return true;
  return false;
}

bool PipeEngine::coalesceResident(Effect &effect) {
  ByteRange *incoming = effect.getSoleRange();
  if (!incoming)
    return false;

  // Only the tail of the queue is a candidate. Merging across an intervening
  // access would widen the marker over bytes nobody touched, and a widened
  // marker reports hazards that do not exist.
  static constexpr unsigned kScanDepth = 8;
  std::deque<Effect> &queue = getQueue(effect.pipe);
  unsigned scanned = 0;
  for (auto it = queue.rbegin(); it != queue.rend() && scanned < kScanDepth;
       ++it, ++scanned) {
    // A token is a retirement point: everything before it is drained by a
    // different `wait_flag` than everything after, so the two sides must not
    // be merged.
    if (it->isToken)
      return false;
    if (!it->isResident || it->op != effect.op ||
        it->isWriteEffect() != effect.isWriteEffect() ||
        it->isRawPointer != effect.isRawPointer)
      continue;
    ByteRange *kept = it->getSoleRange();
    if (!kept || kept->arena != incoming->arena)
      continue;
    // Overlapping or exactly abutting only - anything else leaves a gap.
    if (incoming->lo > kept->hi || kept->lo > incoming->hi)
      continue;
    kept->lo = std::min(kept->lo, incoming->lo);
    kept->hi = std::max(kept->hi, incoming->hi);
    return true;
  }
  return false;
}

void PipeEngine::incFlag(const FlagKey &key, const VectorClock &clock) {
  ++flagSem[key];
  flagClock[key].join(clock);
}

} // namespace interp
} // namespace bishengir
