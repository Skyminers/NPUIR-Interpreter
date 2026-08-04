//===- OpsSync.cpp - HIVM synchronisation op handlers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These handlers implement the flush rules that make the deferred-commit
// model meaningful, and the happens-before edges the race detector consumes.
//
//===----------------------------------------------------------------------===//

#include "OpUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

/// Read a flag/event id that may be a static attribute or a runtime operand.
bool getFlagId(Interpreter &interp, CoreState &core, Operation *op,
               Attribute staticId, Value dynamicId, int64_t &out) {
  if (staticId) {
    if (auto intAttr = dyn_cast<IntegerAttr>(staticId)) {
      out = intAttr.getInt();
      return true;
    }
    // The event id is an enum attribute whose case value is the id.
    if (auto eventAttr = dyn_cast<hivm::EventAttr>(staticId)) {
      out = static_cast<int64_t>(eventAttr.getEvent());
      return true;
    }
  }
  if (dynamicId) {
    RuntimeValue rv = interp.getValue(core, dynamicId);
    if (!rv.isInt()) {
      interp.emitError(op) << "dynamic flag id is unbound";
      return false;
    }
    out = rv.getIndexValue();
    return true;
  }
  interp.emitError(op) << "op has neither a static nor a dynamic flag id";
  return false;
}

/// Resolve the pipe pair of an intra-core flag op.
bool getFlagKey(Interpreter &interp, CoreState &core, Operation *op,
                hivm::PipeAttr setPipe, hivm::PipeAttr waitPipe,
                Attribute staticId, Value dynamicId, FlagKey &out) {
  if (!convertPipe(static_cast<int32_t>(setPipe.getPipe()), out.setPipe) ||
      !convertPipe(static_cast<int32_t>(waitPipe.getPipe()), out.waitPipe)) {
    interp.emitError(op) << "flag uses a pipe the interpreter does not model";
    return false;
  }
  return getFlagId(interp, core, op, staticId, dynamicId, out.eventId);
}

//===----------------------------------------------------------------------===//
// Intra-core flags
//===----------------------------------------------------------------------===//

ExecResult execSetFlag(Interpreter &interp, CoreState &core, Operation *op) {
  auto setOp = cast<hivm::SetFlagOp>(op);
  FlagKey key;
  if (!getFlagKey(interp, core, op, setOp.getSetPipeAttr(),
                  setOp.getWaitPipeAttr(),
                  setOp.getStaticEventIdAttr(), setOp.getDynamicEventId(), key))
    return ExecResult::Error;

  core.clock.tick(core.index);
  interp.setLastSyncOp(op);

  if (interp.getOptions().sched == SchedMode::InOrder) {
    core.pipes.incFlag(key, core.clock);
    return ExecResult::Advance;
  }

  // The flag is raised by the pipe itself, so it only becomes visible once
  // everything queued ahead of it on that pipe has completed. Modelling it as
  // a token in the queue is what makes "set_flag on the wrong pipe" visible.
  Effect token;
  token.op = op;
  token.pipe = key.setPipe;
  token.isToken = true;
  token.token = key;
  token.issueClock = core.clock;
  core.pipes.push(key.setPipe, std::move(token));
  return ExecResult::Advance;
}

ExecResult execWaitFlag(Interpreter &interp, CoreState &core, Operation *op) {
  auto waitOp = cast<hivm::WaitFlagOp>(op);
  FlagKey key;
  if (!getFlagKey(interp, core, op, waitOp.getSetPipeAttr(),
                  waitOp.getWaitPipeAttr(), waitOp.getStaticEventIdAttr(),
                  waitOp.getDynamicEventId(), key))
    return ExecResult::Error;

  if (core.pipes.getFlagCount(key) <= 0) {
    // Drain the setting pipe up to the matching token, which is what the
    // hardware does when it blocks on the flag.
    if (!interp.flushUntilToken(core, key.setPipe, key)) {
      core.status = CoreStatus::BlockedOnFlag;
      core.blockedOn.op = op;
      core.blockedOn.what =
          ("wait_flag[" + getPipeName(key.setPipe) + ", " +
           getPipeName(key.waitPipe) + ", " + Twine(key.eventId) + "]").str();
      core.blockedOn.flagId = key.eventId;
      return ExecResult::Block;
    }
  }

  core.pipes.decFlag(key);
  if (const VectorClock *published = core.pipes.getFlagClock(key))
    core.clock.join(*published);
  core.clock.tick(core.index);
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

ExecResult execPipeBarrier(Interpreter &interp, CoreState &core,
                           Operation *op) {
  auto barrierOp = cast<hivm::PipeBarrierOp>(op);
  Pipe pipe;
  if (convertPipe(static_cast<int32_t>(barrierOp.getPipe().getPipe()), pipe)) {
    interp.flushPipe(core, pipe);
  } else {
    // PIPE_ALL and friends drain everything.
    interp.flushAllPipes(core);
  }
  core.clock.tick(core.index);
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Cross-core flags
//===----------------------------------------------------------------------===//

/// Scope of a cross-core flag: inter-block flags are global, everything else
/// is scoped to the issuing block.
int64_t getFlagScope(hivm::SyncBlockInstrModeAttr mode, const CoreId &id) {
  if (mode && mode.getSyncInstrMode() ==
                  hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION)
    return -1;
  return static_cast<int64_t>(id.blockIdx);
}

ExecResult execSyncBlockSet(Interpreter &interp, CoreState &core,
                            Operation *op) {
  auto setOp = cast<hivm::SyncBlockSetOp>(op);
  CrossFlagKey key;
  if (!convertPipe(static_cast<int32_t>(setOp.getTpipe().getPipe()),
                   key.tpipe) ||
      !convertPipe(static_cast<int32_t>(setOp.getPipe().getPipe()), key.pipe)) {
    interp.emitError(op) << "sync_block_set uses an unmodelled pipe";
    return ExecResult::Error;
  }
  if (!getFlagId(interp, core, op, setOp.getStaticFlagIdAttr(),
                 setOp.getDynamicFlagId(), key.flagId))
    return ExecResult::Error;
  key.scope = getFlagScope(setOp.getTsyncInstrMode(), core.id);

  // The flag is raised by `tpipe` once that pipe has finished the work the
  // consumer is waiting for, so everything queued on it must land first.
  // Crucially, other pipes are left alone: a producer that wrote on the wrong
  // pipe stays invisible, which is the bug we want to surface.
  interp.flushPipe(core, key.tpipe);
  core.clock.tick(core.index);

  CrossFlagState &state = interp.getCrossFlags()[key];
  CoreKind opposite = core.id.kind == CoreKind::AIC ? CoreKind::AIV
                                                     : CoreKind::AIC;
  bool hasOpposite = llvm::any_of(interp.getCores(), [&](const CoreState &c) {
    return c.id.kind == opposite;
  });
  if (!hasOpposite) {
    unsigned kind = static_cast<unsigned>(core.id.kind);
    ++state.sameKindCount[kind];
    state.sameKindClock[kind].join(core.clock);
  } else if (core.id.kind == CoreKind::AIC) {
    ++state.aicGeneration;
    state.aicClock.join(core.clock);
    state.aicLastSetter = op;
  } else {
    ++state.aivCount[core.index];
    state.aivClock[core.index].join(core.clock);
    state.aivLastSetter[core.index] = op;
  }
  // A core already parked on this flag will not re-test it on its own.
  interp.wakeWaitersOn(key.str());
  interp.noteProgress();
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

ExecResult execSyncBlockWait(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto waitOp = cast<hivm::SyncBlockWaitOp>(op);
  CrossFlagKey key;
  if (!convertPipe(static_cast<int32_t>(waitOp.getTpipe().getPipe()),
                   key.tpipe) ||
      !convertPipe(static_cast<int32_t>(waitOp.getPipe().getPipe()),
                   key.pipe)) {
    interp.emitError(op) << "sync_block_wait uses an unmodelled pipe";
    return ExecResult::Error;
  }
  if (!getFlagId(interp, core, op, waitOp.getStaticFlagIdAttr(),
                 waitOp.getDynamicFlagId(), key.flagId))
    return ExecResult::Error;
  key.scope = getFlagScope(waitOp.getTsyncInstrMode(), core.id);

  // MIX flags are directional. Cube-to-vector is broadcast to every sub-core;
  // vector-to-cube aggregates one contribution from every sub-core in scope.
  CoreKind opposite = core.id.kind == CoreKind::AIC ? CoreKind::AIV
                                                     : CoreKind::AIC;
  bool hasOpposite = llvm::any_of(interp.getCores(), [&](const CoreState &c) {
    return c.id.kind == opposite;
  });

  auto &flags = interp.getCrossFlags();
  auto it = flags.find(key);
  bool ready = false;
  if (it != flags.end()) {
    CrossFlagState &state = it->second;
    if (!hasOpposite) {
      unsigned kind = static_cast<unsigned>(core.id.kind);
      ready = state.sameKindCount[kind] > 0;
    } else if (core.id.kind == CoreKind::AIV) {
      ready = state.aivSeenAicGeneration[core.index] < state.aicGeneration;
    } else {
      ready = true;
      for (const CoreState &candidate : interp.getCores()) {
        if (candidate.id.kind != CoreKind::AIV)
          continue;
        if (key.scope >= 0 && candidate.id.blockIdx != core.id.blockIdx)
          continue;
        if (state.aivCount[candidate.index] <= 0) {
          ready = false;
          break;
        }
      }
    }
  }
  if (!ready) {
    core.status = CoreStatus::BlockedOnFlag;
    core.blockedOn.op = op;
    core.blockedOn.what = "sync_block_wait" + key.str();
    core.blockedOn.flagId = key.flagId;
    core.blockedOn.crossFlagKey = key.str();
    return ExecResult::Block;
  }

  CrossFlagState &state = it->second;
  if (!hasOpposite) {
    unsigned kind = static_cast<unsigned>(core.id.kind);
    --state.sameKindCount[kind];
    core.clock.join(state.sameKindClock[kind]);
  } else if (core.id.kind == CoreKind::AIV) {
    ++state.aivSeenAicGeneration[core.index];
    core.clock.join(state.aicClock);
  } else {
    for (const CoreState &candidate : interp.getCores()) {
      if (candidate.id.kind != CoreKind::AIV)
        continue;
      if (key.scope >= 0 && candidate.id.blockIdx != core.id.blockIdx)
        continue;
      --state.aivCount[candidate.index];
      core.clock.join(state.aivClock[candidate.index]);
    }
  }
  core.clock.tick(core.index);
  core.status = CoreStatus::Runnable;
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Barriers
//===----------------------------------------------------------------------===//

StringRef getSyncBlockModeName(hivm::SyncBlockMode mode) {
  switch (mode) {
  case hivm::SyncBlockMode::ALL_CUBE:
    return "ALL_CUBE";
  case hivm::SyncBlockMode::ALL_VECTOR:
    return "ALL_VECTOR";
  case hivm::SyncBlockMode::ALL_SUB_VECTOR:
    return "ALL_SUB_VECTOR";
  case hivm::SyncBlockMode::BARRIER_CUBE:
    return "BARRIER_CUBE";
  case hivm::SyncBlockMode::BARRIER_VECTOR:
    return "BARRIER_VECTOR";
  case hivm::SyncBlockMode::ALL:
    return "ALL";
  }
  return "ALL";
}

ExecResult execSyncBlock(Interpreter &interp, CoreState &core, Operation *op) {
  auto syncOp = cast<hivm::SyncBlockOp>(op);
  StringRef mode = getSyncBlockModeName(syncOp.getSyncBlockMode().getSyncMode());
  int64_t flagId =
      syncOp.getFlagId() ? syncOp.getFlagId()->getInt() : 0;

  // One barrier object per (mode, flag id). It deliberately does *not*
  // include the op identity: after SplitMixKernel the AIC and AIV halves live
  // in different functions, so the two sync_block ops that must rendezvous
  // are distinct Operations. A barrier that only some participants execute
  // therefore shows up as an arrival mismatch, which is precisely the
  // "barrier cloned into a conditional region" bug.
  std::string key;
  {
    llvm::raw_string_ostream os(key);
    os << mode << ':' << flagId;
    // ALL_SUB_VECTOR only gathers one block's sub-vector cores, so each
    // block needs its own rendezvous.
    if (mode == "ALL_SUB_VECTOR")
      os << ":block" << core.id.blockIdx;
  }

  BarrierState &state = interp.getBarriers()[key];
  if (state.expected == 0) {
    SmallVector<unsigned> participants;
    interp.getBarrierParticipants(core.id, mode, participants);
    state.expected = static_cast<unsigned>(participants.size());
  }

  uint64_t &myGeneration = core.barrierGeneration[key];

  // The site has fired since we last went through it, so this is our release.
  if (myGeneration < state.generation) {
    ++myGeneration;
    core.clock.join(state.mergedClock);
    core.status = CoreStatus::Runnable;
    interp.setLastSyncOp(op);
    return ExecResult::Advance;
  }

  if (!llvm::is_contained(state.arrived, core.index)) {
    // A barrier is a full drain: everything this core issued must complete
    // before the others are allowed past.
    interp.flushAllPipes(core);
    state.arrived.push_back(core.index);
    state.arrivedAt.push_back(op);
    core.clock.tick(core.index);
  }

  if (state.arrived.size() < state.expected) {
    core.status = CoreStatus::BlockedOnBarrier;
    core.blockedOn.op = op;
    core.blockedOn.what = ("sync_block[" + mode + ", " + Twine(flagId) +
                           "] (" + Twine(state.arrived.size()) + "/" +
                           Twine(state.expected) + " arrived)").str();
    core.blockedOn.barrierKey = key;
    return ExecResult::Block;
  }

  // Last one in: merge every participant's clock, bump the generation and
  // wake the rest. They are still parked on their own sync_block op and will
  // re-execute it, taking the release path above exactly once each.
  state.mergedClock = VectorClock();
  for (unsigned idx : state.arrived)
    state.mergedClock.join(interp.getCores()[idx].clock);
  SmallVector<unsigned> participants(state.arrived.begin(),
                                     state.arrived.end());
  state.arrived.clear();
  state.arrivedAt.clear();
  ++state.generation;
  interp.noteProgress();
  for (unsigned idx : participants) {
    CoreState &participant = interp.getCores()[idx];
    if (participant.status == CoreStatus::BlockedOnBarrier &&
        participant.blockedOn.barrierKey == key) {
      participant.status = CoreStatus::Runnable;
      participant.blockedOn = BlockReason();
    }
  }

  // We are through as well.
  ++myGeneration;
  core.clock.join(state.mergedClock);
  core.status = CoreStatus::Runnable;
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Block locks
//===----------------------------------------------------------------------===//

ExecResult execSyncBlockLock(Interpreter &interp, CoreState &core,
                             Operation *op) {
  auto lockOp = cast<hivm::SyncBlockLockOp>(op);
  MemRefValue lock;
  if (!interp.getMemRefOperand(core, lockOp.getLockVar(), lock, op))
    return ExecResult::Error;

  RuntimeValue current;
  if (!rawLoad(interp, lock, 0, op, current))
    return ExecResult::Error;
  if (current.getIndexValue() != static_cast<int64_t>(core.id.blockIdx)) {
    core.status = CoreStatus::BlockedOnLock;
    core.blockedOn.op = op;
    core.blockedOn.what =
        ("sync_block_lock (lock_var=" + Twine(current.getIndexValue()) +
         ", need " + Twine(core.id.blockIdx) + ")").str();
    return ExecResult::Block;
  }
  interp.flushAllPipes(core);
  // Inherit everything the previous holder had done when it released.
  core.clock.join(interp.getLockClock(lock.arena, lock.byteOffset));
  core.clock.tick(core.index);
  core.status = CoreStatus::Runnable;
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

/// Advance the lock so the next block may enter. Wakes every core parked on a
/// lock so they re-test the variable.
ExecResult releaseLock(Interpreter &interp, CoreState &core, Operation *op,
                       Value lockVar) {
  MemRefValue lock;
  if (!interp.getMemRefOperand(core, lockVar, lock, op))
    return ExecResult::Error;
  interp.flushAllPipes(core);

  RuntimeValue current;
  if (!rawLoad(interp, lock, 0, op, current))
    return ExecResult::Error;
  int64_t next = current.getIndexValue() + 1;
  unsigned blockDim = interp.getOptions().blockDim;
  if (blockDim && next >= static_cast<int64_t>(blockDim))
    next = 0;
  if (!rawStore(interp, lock, 0, op,
                RuntimeValue::getInt(llvm::APInt(64,
                                                 static_cast<uint64_t>(next),
                                                 true))))
    return ExecResult::Error;

  core.clock.tick(core.index);
  interp.noteProgress();
  // Publish for whoever takes the lock next, whether or not they are parked
  // on it right now.
  interp.getLockClock(lock.arena, lock.byteOffset).join(core.clock);
  for (CoreState &other : interp.getCores()) {
    if (other.status != CoreStatus::BlockedOnLock)
      continue;
    other.status = CoreStatus::Runnable;
    other.blockedOn = BlockReason();
  }
  interp.setLastSyncOp(op);
  return ExecResult::Advance;
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

ExecResult execNoOp(Interpreter &, CoreState &, Operation *) {
  return ExecResult::Advance;
}

} // namespace

void registerHIVMSyncOps(OpRegistry &registry) {
  registry.add("hivm.hir.set_flag", execSetFlag);
  registry.add("hivm.hir.wait_flag", execWaitFlag);
  registry.add("hivm.hir.pipe_barrier", execPipeBarrier);
  registry.add("hivm.hir.sync_block", execSyncBlock);
  registry.add("hivm.hir.sync_block_set", execSyncBlockSet);
  registry.add("hivm.hir.sync_block_wait", execSyncBlockWait);
  registry.add("hivm.hir.sync_block_lock", execSyncBlockLock);
  registry.add("hivm.hir.sync_block_unlock", [](Interpreter &interp,
                                                CoreState &core,
                                                Operation *op) {
    return releaseLock(interp, core, op,
                       cast<hivm::SyncBlockUnlockOp>(op).getLockVar());
  });
  registry.add("hivm.hir.free_lock_var", [](Interpreter &interp,
                                            CoreState &core, Operation *op) {
    // Runs one lock/unlock pair; with the lock already free this reduces to
    // advancing the variable.
    return releaseLock(interp, core, op,
                       cast<hivm::FreeLockVarOp>(op).getLockVar());
  });
  registry.add("hivm.hir.create_sync_block_lock", [](Interpreter &interp,
                                                     CoreState &core,
                                                     Operation *op) {
    auto createOp = cast<hivm::CreateSyncBlockLockOp>(op);
    MemRefValue lock;
    if (createOp.getLockArg()) {
      // Carve the lock out of the buffer the runtime handed us.
      if (!interp.getMemRefOperand(core, createOp.getLockArg(), lock, op))
        return ExecResult::Error;
      auto resultType = cast<MemRefType>(createOp.getMemref().getType());
      lock.elemType = resultType.getElementType();
      lock.elemBytes = getStorageSize(lock.elemType);
      lock.sizes.assign({1});
      lock.strides.assign({1});
      // The runtime hands this buffer over already zeroed, so block 0 owns
      // the lock first. Writing 0 here instead would reset the lock every
      // time another block reached this op - which is every block, since all
      // of them run the same code.
    } else {
      auto resultType = cast<MemRefType>(createOp.getMemref().getType());
      if (!interp.allocateMemRef(core, resultType, op, lock))
        return ExecResult::Error;
      if (!rawStore(interp, lock, 0, op,
                    RuntimeValue::getInt(llvm::APInt(64, 0))))
        return ExecResult::Error;
    }
    interp.setValue(core, createOp.getMemref(), RuntimeValue::getMemRef(lock));
    return ExecResult::Advance;
  });

  // Markers and cache maintenance carry no interpretable state.
  registry.add("hivm.hir.anchor", execNoOp);
  registry.add("hivm.hir.dcci", execNoOp);
  registry.add("hivm.hir.set_ctrl", execNoOp);
  registry.add("hivm.hir.set_mask_norm", execNoOp);
  registry.add("hivm.hir.set_ffts_base_addr", execNoOp);
  registry.add("hivm.hir.set_atomic", execNoOp);
  registry.add("hivm.hir.init_debug", execNoOp);
  registry.add("hivm.hir.debug", execNoOp);
  registry.add("hivm.hir.finish_debug", execNoOp);
  registry.add("hivm.hir.multi_buffer_counter", execNoOp);
}

} // namespace interp
} // namespace bishengir
