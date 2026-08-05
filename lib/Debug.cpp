//===- Debug.cpp - Replayable interpreter debug sessions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Debug.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

#include <algorithm>

using namespace mlir;

namespace bishengir {
namespace interp {

namespace {

std::string stringifyLocation(Operation *op) {
  if (!op)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  os << op->getLoc();
  return os.str();
}

std::string stringifyClock(const VectorClock &clock,
                           ArrayRef<unsigned> visibleCores) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << '<';
  for (auto [position, core] : llvm::enumerate(visibleCores)) {
    if (position)
      os << ',';
    os << (core < clock.size() ? clock[core] : 0);
  }
  os << '>';
  return os.str();
}

std::string stringifyType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << type;
  return os.str();
}

std::string pipeName(hivm::PipeAttr attr) {
  Pipe pipe;
  if (!attr || !convertPipe(static_cast<int32_t>(attr.getPipe()), pipe))
    return "PIPE_UNSUPPORTED";
  return getPipeName(pipe).str();
}

int64_t staticId(Attribute attr) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return integer.getInt();
  if (auto event = dyn_cast_or_null<hivm::EventAttr>(attr))
    return static_cast<int64_t>(event.getEvent());
  return -1;
}

llvm::json::Object flagSyncValue(StringRef kind, hivm::PipeAttr setPipe,
                                 hivm::PipeAttr waitPipe, Attribute id) {
  llvm::json::Object sync;
  sync["kind"] = kind;
  sync["set_pipe"] = pipeName(setPipe);
  sync["wait_pipe"] = pipeName(waitPipe);
  int64_t eventId = staticId(id);
  sync["event_id"] = eventId >= 0 ? llvm::json::Value(eventId)
                                    : llvm::json::Value(nullptr);
  sync["dynamic_event"] = eventId < 0;
  return sync;
}

llvm::json::Object rangeValue(const ByteRange &range) {
  llvm::json::Object value;
  value["arena"] = static_cast<int64_t>(range.arena);
  value["begin"] = static_cast<int64_t>(range.lo);
  value["end"] = static_cast<int64_t>(range.hi);
  return value;
}

StringRef statusName(CoreStatus status) {
  switch (status) {
  case CoreStatus::Runnable:
    return "runnable";
  case CoreStatus::BlockedOnFlag:
    return "blocked_on_flag";
  case CoreStatus::BlockedOnBarrier:
    return "blocked_on_barrier";
  case CoreStatus::BlockedOnLock:
    return "blocked_on_lock";
  case CoreStatus::Done:
    return "done";
  case CoreStatus::Failed:
    return "failed";
  }
  return "unknown";
}

StringRef resultName(ExecResult result) {
  switch (result) {
  case ExecResult::Advance:
    return "advance";
  case ExecResult::Handled:
    return "handled";
  case ExecResult::Block:
    return "block";
  case ExecResult::Error:
    return "error";
  }
  return "unknown";
}

bool pipeBelongsTo(CoreKind kind, Pipe pipe) {
  switch (pipe) {
  case Pipe::S:
  case Pipe::MTE2:
  case Pipe::MTE3:
    return true;
  case Pipe::V:
    return kind == CoreKind::AIV;
  case Pipe::M:
  case Pipe::MTE1:
  case Pipe::FIX:
    return kind == CoreKind::AIC;
  case Pipe::NumPipes:
    return false;
  }
  return false;
}

Operation *currentOperation(const CoreState &core) {
  if (!Interpreter::hasRegion(core))
    return nullptr;
  const RegionFrame &region = core.callStack.back().regions.back();
  if (region.ip == region.block->end())
    return nullptr;
  return &*region.ip;
}

std::string hexBytes(ArrayRef<uint8_t> bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.resize(bytes.size() * 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    text[i * 2] = digits[bytes[i] >> 4];
    text[i * 2 + 1] = digits[bytes[i] & 0xf];
  }
  return text;
}

} // namespace

DebugRecorder::DebugRecorder(Interpreter &interp, StringRef path)
    : interp(interp) {
  std::error_code ec;
  auto output =
      std::make_unique<llvm::raw_fd_ostream>(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "warning: cannot open debug session '" << path
                 << "': " << ec.message() << '\n';
    return;
  }
  stream = std::move(output);
  previousMemory.resize(interp.arenas.size());

  irAsmState = std::make_unique<AsmState>(interp.module.getOperation(),
                                           OpPrintingFlags(), &irLocations);
  llvm::raw_string_ostream irStream(irText);
  interp.module.print(irStream, *irAsmState);

  OpPrintingFlags operationFlags;
  operationFlags.skipRegions();
  operationAsmState = std::make_unique<AsmState>(interp.module.getOperation(),
                                                  operationFlags);
}

std::vector<DebugRecorder::ValueSnapshot>
DebugRecorder::snapshotValues(const CoreState &core, ValueRange values) {
  std::vector<ValueSnapshot> snapshots;
  snapshots.reserve(values.size());
  for (Value value : values) {
    ValueSnapshot snapshot;
    llvm::raw_string_ostream ssaStream(snapshot.ssa);
    value.printAsOperand(ssaStream, *operationAsmState);
    snapshot.type = stringifyType(value.getType());

    const RuntimeValue *runtime = nullptr;
    for (auto frame = core.callStack.rbegin(); frame != core.callStack.rend();
         ++frame) {
      auto found = frame->env.find(value);
      if (found != frame->env.end()) {
        runtime = &found->second;
        break;
      }
    }
    snapshot.bound = runtime && !runtime->isNone();
    if (runtime) {
      llvm::raw_string_ostream valueStream(snapshot.value);
      runtime->print(valueStream);
    } else {
      snapshot.value = "<unset>";
    }
    snapshots.push_back(std::move(snapshot));
  }
  return snapshots;
}

llvm::json::Value DebugRecorder::operationValue(
    Operation *op, const CoreState *core,
    const std::vector<ValueSnapshot> *inputOverride) {
  if (!op)
    return nullptr;

  llvm::json::Object value;
  value["name"] = op->getName().getStringRef();
  value["location"] = stringifyLocation(op);
  std::string operationText;
  llvm::raw_string_ostream operationStream(operationText);
  op->print(operationStream, *operationAsmState);
  value["text"] = std::move(operationText);
  if (auto found = irLocations.find(op); found != irLocations.end()) {
    value["ir_line"] = static_cast<int64_t>(found->second.first);
    value["ir_column"] = static_cast<int64_t>(found->second.second);
  }

  auto bindingsValue = [](ArrayRef<ValueSnapshot> snapshots) {
    llvm::json::Array bindings;
    for (const ValueSnapshot &snapshot : snapshots) {
      llvm::json::Object binding;
      binding["ssa"] = snapshot.ssa;
      binding["type"] = snapshot.type;
      binding["value"] = snapshot.value;
      binding["bound"] = snapshot.bound;
      bindings.push_back(std::move(binding));
    }
    return bindings;
  };
  if (inputOverride) {
    value["inputs"] = bindingsValue(*inputOverride);
  } else if (core) {
    value["inputs"] = bindingsValue(snapshotValues(*core, op->getOperands()));
  } else {
    value["inputs"] = llvm::json::Array();
  }
  value["outputs"] = core
                         ? llvm::json::Value(bindingsValue(
                               snapshotValues(*core, op->getResults())))
                         : llvm::json::Value(llvm::json::Array());

  if (auto set = dyn_cast<hivm::SetFlagOp>(op)) {
    value["sync"] = flagSyncValue("set_flag", set.getSetPipeAttr(),
                                   set.getWaitPipeAttr(),
                                   set.getStaticEventIdAttr());
  } else if (auto wait = dyn_cast<hivm::WaitFlagOp>(op)) {
    value["sync"] = flagSyncValue("wait_flag", wait.getSetPipeAttr(),
                                   wait.getWaitPipeAttr(),
                                   wait.getStaticEventIdAttr());
  } else if (auto set = dyn_cast<hivm::SyncBlockSetOp>(op)) {
    value["sync"] = flagSyncValue("sync_block_set", set.getTpipe(),
                                   set.getPipe(), set.getStaticFlagIdAttr());
  } else if (auto wait = dyn_cast<hivm::SyncBlockWaitOp>(op)) {
    value["sync"] = flagSyncValue("sync_block_wait", wait.getTpipe(),
                                   wait.getPipe(),
                                   wait.getStaticFlagIdAttr());
  } else if (auto barrier = dyn_cast<hivm::PipeBarrierOp>(op)) {
    llvm::json::Object sync;
    sync["kind"] = "pipe_barrier";
    sync["pipe"] = pipeName(barrier.getPipe());
    value["sync"] = std::move(sync);
  }
  return value;
}

void DebugRecorder::resolveProgramFilter() {
  if (filterResolved)
    return;
  filterResolved = true;
  if (interp.options.debugCore.empty())
    return;
  for (const CoreState &core : interp.cores) {
    if (core.id.str() == interp.options.debugCore) {
      programFilter = core.id.blockIdx;
      return;
    }
  }
  if (!interp.cores.empty()) {
    programFilter = interp.cores.front().id.blockIdx;
    llvm::errs() << "warning: debug core '" << interp.options.debugCore
                 << "' does not exist; recording program " << *programFilter
                 << " instead\n";
  }
}

bool DebugRecorder::recordsCore(const CoreState &core) const {
  return !programFilter || core.id.blockIdx == *programFilter;
}

bool DebugRecorder::recordsArena(const Arena &arena) const {
  return !programFilter || !isCoreLocal(arena.getSpace()) ||
         arena.getOwnerCore() == *programFilter;
}

void DebugRecorder::recordInitial() {
  if (!stream)
    return;
  resolveProgramFilter();

  llvm::json::Object meta;
  meta["event"] = "meta";
  meta["schema"] = "npuir-interp-debug/v1";
  meta["target_model"] = "ascend-single-core/v2";
  meta["ir"] = irText;
  meta["entry"] = interp.options.entry;
  meta["schedule"] = interp.options.sched == SchedMode::InOrder
                         ? "inorder"
                         : interp.options.sched == SchedMode::Lazy ? "lazy"
                                                                   : "fuzz";
  meta["block_dim"] = static_cast<int64_t>(interp.options.blockDim);
  meta["sub_block_num"] = static_cast<int64_t>(interp.options.subBlockNum);
  meta["core_filter"] = interp.options.debugCore;
  meta["program_filter"] = programFilter
                               ? llvm::json::Value(*programFilter)
                               : llvm::json::Value(nullptr);
  llvm::json::Array arenas;
  std::map<unsigned, unsigned> ubLanes;
  for (size_t i = 0; i < interp.arenas.size(); ++i) {
    const Arena &arena = *interp.arenas[i];
    if (!recordsArena(arena))
      continue;
    llvm::json::Object entry;
    entry["id"] = static_cast<int64_t>(i);
    entry["space"] = getAddrSpaceName(arena.getSpace());
    entry["owner"] = static_cast<int64_t>(arena.getOwnerCore());
    if (arena.getSpace() == AddrSpace::UB)
      entry["sub_core_id"] =
          static_cast<int64_t>(ubLanes[arena.getOwnerCore()]++);
    entry["capacity"] = static_cast<int64_t>(arena.getCapacity());
    arenas.push_back(std::move(entry));
  }
  meta["arenas"] = std::move(arenas);
  *stream << llvm::json::Value(std::move(meta)) << '\n';
  recordState("initial", nullptr, nullptr, "ready");
}

void DebugRecorder::recordBeforeStep(const CoreState &core,
                                     Operation *operation) {
  resolveProgramFilter();
  pendingOperation = nullptr;
  pendingInputs.clear();
  if (!stream || !recordsCore(core) || !operation)
    return;
  pendingOperation = operation;
  pendingCoreIndex = core.index;
  pendingInputs = snapshotValues(core, operation->getOperands());
}

void DebugRecorder::recordStep(const CoreState &core, Operation *executed,
                               ExecResult result) {
  resolveProgramFilter();
  if (!recordsCore(core))
    return;
  const std::vector<ValueSnapshot> *inputs =
      pendingOperation == executed && pendingCoreIndex == core.index
          ? &pendingInputs
          : nullptr;
  recordState("step", &core, executed, resultName(result), inputs);
  pendingOperation = nullptr;
  pendingInputs.clear();
}

void DebugRecorder::recordFinal() {
  recordState("final", nullptr, nullptr, "complete");
}

void DebugRecorder::recordFinish(bool succeeded) {
  if (!stream)
    return;
  llvm::json::Object event;
  event["event"] = "finish";
  event["sequence"] = static_cast<int64_t>(sequence++);
  event["result"] = succeeded ? "success" : "failure";
  event["steps"] = static_cast<int64_t>(interp.totalSteps);
  event["errors"] = static_cast<int64_t>(interp.errorCount);
  event["races"] = static_cast<int64_t>(interp.raceCount);
  event["missing_sync"] = static_cast<int64_t>(interp.missingSyncCount);
  *stream << llvm::json::Value(std::move(event)) << '\n';
  stream->flush();
}

void DebugRecorder::memoryWritten(ArrayRef<ByteRange> ranges) {
  pendingWrites.insert(pendingWrites.end(), ranges.begin(), ranges.end());
}

void DebugRecorder::recordState(StringRef reason, const CoreState *selected,
                                Operation *executed, StringRef result,
                                const std::vector<ValueSnapshot> *executedInputs) {
  if (!stream)
    return;

  llvm::json::Object event;
  event["event"] = "state";
  event["sequence"] = static_cast<int64_t>(sequence++);
  event["reason"] = reason;
  event["result"] = result;
  event["selected_core"] = selected ? llvm::json::Value(selected->id.str())
                                    : llvm::json::Value(nullptr);
  event["executed"] = operationValue(executed, selected, executedInputs);

  llvm::SmallVector<unsigned> visibleCores;
  for (const CoreState &core : interp.cores)
    if (recordsCore(core))
      visibleCores.push_back(core.index);

  llvm::json::Array cores;
  for (const CoreState &core : interp.cores) {
    if (!recordsCore(core))
      continue;
    llvm::json::Object coreValue;
    coreValue["id"] = core.id.str();
    // `blockIdx` identifies a Triton program instance, not another physical
    // AIC/AIV.  Keep the old compact id for diagnostics, but expose the
    // coordinates separately so replay UIs do not have to infer hardware
    // topology from a display string such as AIV#17.0.
    coreValue["program_id"] = static_cast<int64_t>(core.id.blockIdx);
    coreValue["kind"] = getCoreKindName(core.id.kind);
    coreValue["sub_core_id"] = static_cast<int64_t>(core.id.subBlockIdx);
    coreValue["index"] = static_cast<int64_t>(core.index);
    coreValue["status"] = statusName(core.status);
    coreValue["steps"] = static_cast<int64_t>(core.stepCount);
    coreValue["clock"] = stringifyClock(core.clock, visibleCores);
    coreValue["current"] = operationValue(currentOperation(core), &core);

    llvm::json::Object blocked;
    blocked["what"] = core.blockedOn.what;
    blocked["operation"] = operationValue(core.blockedOn.op, &core);
    blocked["flag_id"] = core.blockedOn.flagId;
    blocked["cross_flag"] = core.blockedOn.crossFlagKey;
    blocked["barrier"] = core.blockedOn.barrierKey;
    coreValue["blocked"] = std::move(blocked);

    llvm::json::Array pipes;
    for (unsigned p = 0; p < kNumPipes; ++p) {
      Pipe pipe = static_cast<Pipe>(p);
      if (!pipeBelongsTo(core.id.kind, pipe))
        continue;
      llvm::json::Object pipeValue;
      pipeValue["pipe"] = getPipeName(pipe);
      llvm::json::Array tasks;
      for (const Effect &effect : core.pipes.getQueue(pipe)) {
        llvm::json::Object task;
        task["kind"] = effect.isToken   ? "token"
                       : effect.isResident ? "resident"
                                           : "effect";
        task["operation"] = operationValue(effect.op, &core);
        task["clock"] = stringifyClock(effect.issueClock, visibleCores);
        task["atomic"] = effect.isAtomic;
        task["raw_pointer"] = effect.isRawPointer;
        if (effect.isToken) {
          task["set_pipe"] = getPipeName(effect.token.setPipe);
          task["wait_pipe"] = getPipeName(effect.token.waitPipe);
          task["event_id"] = effect.token.eventId;
        }
        llvm::json::Array reads;
        for (const ByteRange &range : effect.reads)
          reads.push_back(rangeValue(range));
        task["reads"] = std::move(reads);
        llvm::json::Array writes;
        for (const ByteRange &range : effect.writes)
          writes.push_back(rangeValue(range));
        task["writes"] = std::move(writes);
        tasks.push_back(std::move(task));
      }
      pipeValue["tasks"] = std::move(tasks);
      pipes.push_back(std::move(pipeValue));
    }
    coreValue["pipes"] = std::move(pipes);

    llvm::json::Array flags;
    for (const auto &[key, count] : core.pipes.getFlagCounts()) {
      if (count == 0)
        continue;
      llvm::json::Object flag;
      flag["set_pipe"] = getPipeName(key.setPipe);
      flag["wait_pipe"] = getPipeName(key.waitPipe);
      flag["event_id"] = key.eventId;
      flag["count"] = count;
      flags.push_back(std::move(flag));
    }
    coreValue["flags"] = std::move(flags);
    cores.push_back(std::move(coreValue));
  }
  event["cores"] = std::move(cores);

  llvm::json::Array crossFlags;
  for (const auto &[key, state] : interp.crossFlags) {
    if (programFilter && key.scope >= 0 &&
        key.scope != static_cast<int64_t>(*programFilter))
      continue;
    llvm::json::Object flag;
    flag["key"] = key.str();
    flag["scope"] = key.scope;
    flag["aic_generation"] = state.aicGeneration;
    llvm::json::Array sameKind;
    for (int64_t count : state.sameKindCount)
      sameKind.push_back(count);
    flag["same_kind_count"] = std::move(sameKind);
    llvm::json::Array aivCounts;
    for (const auto &[core, count] : state.aivCount) {
      llvm::json::Object countValue;
      countValue["core"] = static_cast<int64_t>(core);
      countValue["count"] = count;
      aivCounts.push_back(std::move(countValue));
    }
    flag["aiv_count"] = std::move(aivCounts);
    crossFlags.push_back(std::move(flag));
  }
  event["cross_flags"] = std::move(crossFlags);

  llvm::json::Array barriers;
  for (const auto &[key, state] : interp.barriers) {
    llvm::json::Object barrier;
    barrier["key"] = key;
    barrier["expected"] = static_cast<int64_t>(state.expected);
    barrier["generation"] = static_cast<int64_t>(state.generation);
    llvm::json::Array arrived;
    for (unsigned core : state.arrived)
      arrived.push_back(static_cast<int64_t>(core));
    barrier["arrived"] = std::move(arrived);
    barriers.push_back(std::move(barrier));
  }
  event["barriers"] = std::move(barriers);

  llvm::json::Array arenaStates;
  llvm::json::Array patches;
  static constexpr size_t kPatchBytes = 4096;
  std::map<unsigned, unsigned> ubLanes;
  for (size_t i = 0; i < interp.arenas.size(); ++i) {
    const Arena &arena = *interp.arenas[i];
    if (!recordsArena(arena))
      continue;
    size_t size = static_cast<size_t>(arena.getHighWaterMark());
    const uint8_t *bytes = arena.at(0, size);
    std::vector<uint8_t> &previous = previousMemory[i];
    size_t previousSize = previous.size();
    previous.resize(size);

    std::vector<std::pair<size_t, size_t>> dirty;
    if (previousSize < size)
      dirty.emplace_back(previousSize, size);
    for (const ByteRange &range : pendingWrites) {
      if (range.arena != static_cast<int>(i) || range.lo >= size)
        continue;
      dirty.emplace_back(static_cast<size_t>(range.lo),
                         static_cast<size_t>(std::min<uint64_t>(range.hi,
                                                                size)));
    }
    std::sort(dirty.begin(), dirty.end());
    size_t merged = 0;
    for (const auto &range : dirty) {
      if (range.first >= range.second)
        continue;
      if (merged && range.first <= dirty[merged - 1].second) {
        dirty[merged - 1].second =
            std::max(dirty[merged - 1].second, range.second);
      } else {
        dirty[merged++] = range;
      }
    }
    dirty.resize(merged);

    for (const auto &[dirtyBegin, dirtyEnd] : dirty) {
      for (size_t cursor = dirtyBegin; cursor < dirtyEnd;) {
        bool changed =
            cursor >= previousSize || previous[cursor] != bytes[cursor];
        if (!changed) {
          ++cursor;
          continue;
        }
        size_t begin = cursor;
        while (cursor < dirtyEnd && cursor - begin < kPatchBytes &&
               (cursor >= previousSize || previous[cursor] != bytes[cursor]))
          ++cursor;
        llvm::json::Object patch;
        patch["arena"] = static_cast<int64_t>(i);
        patch["offset"] = static_cast<int64_t>(begin);
        patch["bytes"] = hexBytes(ArrayRef<uint8_t>(bytes + begin,
                                                     cursor - begin));
        patches.push_back(std::move(patch));
        std::copy(bytes + begin, bytes + cursor, previous.begin() + begin);
      }
    }

    llvm::json::Object arenaValue;
    arenaValue["id"] = static_cast<int64_t>(i);
    arenaValue["space"] = getAddrSpaceName(arena.getSpace());
    arenaValue["owner"] = static_cast<int64_t>(arena.getOwnerCore());
    if (arena.getSpace() == AddrSpace::UB)
      arenaValue["sub_core_id"] =
          static_cast<int64_t>(ubLanes[arena.getOwnerCore()]++);
    arenaValue["high_water"] = static_cast<int64_t>(size);
    llvm::json::Array allocations;
    for (const AllocRecord &alloc : arena.getAllocations()) {
      llvm::json::Object allocation;
      allocation["begin"] = static_cast<int64_t>(alloc.lo);
      allocation["end"] = static_cast<int64_t>(alloc.hi);
      allocation["name"] = alloc.name.empty() ? "<anon>" : alloc.name;
      allocation["live"] = alloc.live;
      allocation["operation"] = operationValue(alloc.op);
      const MemRefValue *view = nullptr;
      auto matches = [&](const MemRefValue &candidate) {
        return candidate.arena == static_cast<int>(i) &&
               candidate.baseOffset == alloc.lo;
      };
      for (const auto &entry : interp.globalArgs) {
        const RuntimeValue &value = entry.second;
        if (value.isMemRef() && matches(value.getMemRefValue())) {
          view = &value.getMemRefValue();
          break;
        }
      }
      if (!view) {
        for (const auto &entry : interp.allocCache) {
          const MemRefValue &candidate = entry.second;
          if (matches(candidate)) {
            view = &candidate;
            break;
          }
        }
      }
      if (view) {
        allocation["element_type"] = stringifyType(view->elemType);
        allocation["element_bytes"] = static_cast<int64_t>(view->elemBytes);
        llvm::json::Array shape;
        for (int64_t dim : view->sizes)
          shape.push_back(dim);
        allocation["shape"] = std::move(shape);
        llvm::json::Array strides;
        for (int64_t stride : view->strides)
          strides.push_back(stride);
        allocation["strides"] = std::move(strides);
      }
      allocations.push_back(std::move(allocation));
    }
    arenaValue["allocations"] = std::move(allocations);
    arenaStates.push_back(std::move(arenaValue));
  }
  event["arenas"] = std::move(arenaStates);
  event["memory_patches"] = std::move(patches);
  pendingWrites.clear();

  *stream << llvm::json::Value(std::move(event)) << '\n';
  stream->flush();
}

} // namespace interp
} // namespace bishengir
