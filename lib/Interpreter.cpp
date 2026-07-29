//===- Interpreter.cpp - NPUIR interpreter core -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/Interp/Interpreter.h"
#include "NpyIO.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>

using namespace mlir;

namespace bishengir {
namespace interp {

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

StringRef getCoreKindName(CoreKind kind) {
  return kind == CoreKind::AIC ? "AIC" : "AIV";
}

std::string CoreId::str() const {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << getCoreKindName(kind) << '#' << blockIdx;
  if (kind == CoreKind::AIV)
    os << '.' << subBlockIdx;
  return os.str();
}

std::string CrossFlagKey::str() const {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << '[' << getPipeName(tpipe) << "->" << getPipeName(pipe) << " flag="
     << flagId;
  if (scope < 0)
    os << " inter-block";
  else
    os << " block=" << scope;
  os << ']';
  return os.str();
}

bool convertPipe(int32_t hivmPipe, Pipe &out) {
  // Values mirror HIVM_PipeEnum in HIVMAttrs.td.
  switch (hivmPipe) {
  case 0:
    out = Pipe::S;
    return true;
  case 1:
    out = Pipe::V;
    return true;
  case 2:
    out = Pipe::M;
    return true;
  case 3:
    out = Pipe::MTE1;
    return true;
  case 4:
    out = Pipe::MTE2;
    return true;
  case 5:
    out = Pipe::MTE3;
    return true;
  case 7:
    out = Pipe::MTE4;
    return true;
  case 8:
    out = Pipe::MTE5;
    return true;
  case 9:
    out = Pipe::V2;
    return true;
  case 10:
    out = Pipe::FIX;
    return true;
  // 6 = PIPE_ALL, 11/12 = virtual MTE2 pipes, 13 = PIPE_NUM, 99 = unassigned.
  default:
    return false;
  }
}

AddrSpace getAddrSpaceOf(MemRefType type) {
  Attribute space = type.getMemorySpace();
  if (!space)
    // Post-bufferization HIVM IR routinely drops the tag on vector temporaries;
    // those live in UB.
    return AddrSpace::UB;
  if (auto hivmSpace = dyn_cast<hivm::AddressSpaceAttr>(space)) {
    switch (hivmSpace.getAddressSpace()) {
    case hivm::AddressSpace::GM:
      return AddrSpace::GM;
    case hivm::AddressSpace::L1:
      return AddrSpace::L1;
    case hivm::AddressSpace::L0A:
      return AddrSpace::L0A;
    case hivm::AddressSpace::L0B:
      return AddrSpace::L0B;
    case hivm::AddressSpace::L0C:
      return AddrSpace::L0C;
    case hivm::AddressSpace::UB:
      return AddrSpace::UB;
    case hivm::AddressSpace::SSBUF:
      return AddrSpace::SSBUF;
    case hivm::AddressSpace::Zero:
      return AddrSpace::UB;
    }
  }
  if (auto intSpace = dyn_cast<IntegerAttr>(space)) {
    // Numeric memory spaces follow the same encoding as the HIVM enum.
    switch (intSpace.getInt()) {
    case 1:
      return AddrSpace::GM;
    case 2:
      return AddrSpace::L1;
    case 3:
      return AddrSpace::L0A;
    case 4:
      return AddrSpace::L0B;
    case 5:
      return AddrSpace::L0C;
    case 6:
      return AddrSpace::UB;
    case 11:
      return AddrSpace::SSBUF;
    default:
      return AddrSpace::UB;
    }
  }
  return AddrSpace::UB;
}

void advanceIp(CoreState &core) {
  if (!Interpreter::hasRegion(core))
    return;
  RegionFrame &frame = core.callStack.back().regions.back();
  ++frame.ip;
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

Interpreter::Interpreter(ModuleOp module, InterpOptions options)
    : module(module), options(std::move(options)),
      rng(this->options.seed) {
  registerAllInterpOps(registry);

  auto makeArena = [&](AddrSpace space, unsigned owner, uint64_t capacity) {
    int id = static_cast<int>(arenas.size());
    arenas.push_back(std::make_unique<Arena>(space, owner, capacity,
                                             this->options.poison));
    arenaIndex[{static_cast<unsigned>(space), owner}] = id;
  };

  makeArena(AddrSpace::GM, 0, this->options.gmSize);
  makeArena(AddrSpace::SSBUF, 0, this->options.ssbufSize);
  makeArena(AddrSpace::Host, 0, this->options.hostSize);
  // On-chip pools belong to a block, not to an individual core: in a MIX
  // kernel the AIC's fixpipe writes into the same UB its AIVs read.
  for (unsigned block = 0; block < this->options.blockDim; ++block) {
    makeArena(AddrSpace::UB, block, this->options.ubSize);
    makeArena(AddrSpace::L1, block, this->options.l1SizeBytes);
    makeArena(AddrSpace::L0A, block, this->options.l0aSize);
    makeArena(AddrSpace::L0B, block, this->options.l0bSize);
    makeArena(AddrSpace::L0C, block, this->options.l0cSize);
  }

  if (!this->options.traceFile.empty()) {
    std::error_code ec;
    auto stream = std::make_unique<llvm::raw_fd_ostream>(
        this->options.traceFile, ec, llvm::sys::fs::OF_Text);
    if (ec)
      llvm::errs() << "warning: cannot open trace file '"
                   << this->options.traceFile << "': " << ec.message() << '\n';
    else
      traceStream = std::move(stream);
  }
}

Interpreter::~Interpreter() = default;

llvm::raw_ostream &Interpreter::report() { return llvm::errs(); }

int Interpreter::getArenaId(AddrSpace space, const CoreId &id) {
  unsigned owner = isCoreLocal(space) ? id.blockIdx : 0;
  auto it = arenaIndex.find({static_cast<unsigned>(space), owner});
  if (it == arenaIndex.end()) {
    // Every space is created for every block up front, so this cannot happen
    // for a core the scheduler knows about. Fall back to GM rather than
    // return an id that would index out of range in a release build.
    report() << "warning: no arena for " << getAddrSpaceName(space)
             << " owned by block " << owner << "; falling back to gm\n";
    return arenaIndex[{static_cast<unsigned>(AddrSpace::GM), 0}];
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// Value environment
//===----------------------------------------------------------------------===//

RuntimeValue Interpreter::getValue(CoreState &core, Value value) {
  auto &env = core.callStack.back().env;
  auto it = env.find(value);
  if (it != env.end())
    return it->second;
  return RuntimeValue();
}

void Interpreter::setValue(CoreState &core, Value value, RuntimeValue rv) {
  core.callStack.back().env[value] = std::move(rv);
}

bool Interpreter::getMemRefOperand(CoreState &core, Value value,
                                   MemRefValue &out, Operation *op) {
  RuntimeValue rv = getValue(core, value);
  if (!rv.isMemRef()) {
    emitError(op) << "operand is not a bound memref (produced by an op the "
                     "interpreter does not model?)";
    return false;
  }
  out = rv.getMemRefValue();
  return true;
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

bool Interpreter::allocateMemRef(CoreState &core, MemRefType type,
                                 Operation *op, MemRefValue &out) {
  // Re-executing an alloc site (a loop body, a function called twice) hands
  // back the same storage, re-poisoned. Fresh bytes every time would make a
  // loop exhaust the pool; PlanMemory likewise gives each site one address.
  auto cached = allocCache.find({op, core.index});
  if (cached != allocCache.end() &&
      cached->second.sizes.size() == type.getShape().size()) {
    bool sameShape = true;
    for (auto [have, want] : llvm::zip(cached->second.sizes, type.getShape()))
      sameShape &= ShapedType::isDynamic(want) || have == want;
    if (sameShape) {
      out = cached->second;
      uint64_t bytes =
          static_cast<uint64_t>(out.getNumElements()) * out.elemBytes;
      arenas[out.arena]->poison(out.byteOffset, bytes, out.elemType);
      arenas[out.arena]->getShadow().invalidate(out.byteOffset,
                                                out.byteOffset + bytes);
      return true;
    }
  }

  AddrSpace space = getAddrSpaceOf(type);
  int arenaId = getArenaId(space, core.id);
  Arena &arena = *arenas[arenaId];

  Type elemType = type.getElementType();
  unsigned elemBytes = getStorageSize(elemType);

  SmallVector<int64_t, 4> sizes(type.getShape().begin(),
                                type.getShape().end());
  for (int64_t &s : sizes)
    if (ShapedType::isDynamic(s))
      s = 1; // A dynamic dim on an alloc without an operand: assume 1.

  int64_t numElems = 1;
  for (int64_t s : sizes)
    numElems *= s;
  uint64_t bytes = static_cast<uint64_t>(numElems) * elemBytes;

  // 256 B is the UB/L1 alignment on the supported targets; over-aligning is
  // harmless and keeps the interpreter's offsets comparable to PlanMemory's.
  uint64_t align = space == AddrSpace::GM ? 64 : 256;
  auto offset = arena.allocate(bytes, align, "", op);
  if (!offset) {
    emitError(op) << getAddrSpaceName(space) << " capacity exceeded: need "
                  << bytes << " more bytes, arena is " << arena.getCapacity()
                  << " bytes and " << arena.getHighWaterMark()
                  << " are already in use";
    return false;
  }
  arena.poison(*offset, bytes, elemType);

  SmallVector<int64_t, 4> strides;
  int64_t acc = 1;
  strides.resize(sizes.size());
  for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
    strides[d] = acc;
    acc *= sizes[d];
  }
  // A memref with an explicit strided layout keeps it.
  int64_t layoutOffset = 0;
  SmallVector<int64_t> layoutStrides;
  if (succeeded(mlir::getStridesAndOffset(type, layoutStrides, layoutOffset)) &&
      layoutStrides.size() == sizes.size()) {
    bool usable = !ShapedType::isDynamic(layoutOffset);
    for (int64_t s : layoutStrides)
      usable &= !ShapedType::isDynamic(s);
    if (usable) {
      strides.assign(layoutStrides.begin(), layoutStrides.end());
      // The declared strides may reach past the dense element count.
      int64_t span = 1;
      for (size_t d = 0; d < sizes.size(); ++d)
        span = std::max(span, (sizes[d] - 1) * strides[d] + 1);
      uint64_t needed = (static_cast<uint64_t>(layoutOffset) +
                         static_cast<uint64_t>(span)) *
                        elemBytes;
      if (needed > bytes) {
        // Grow the allocation to cover the strided span.
        auto extra = arena.allocate(needed - bytes, 1, "", op);
        if (!extra) {
          emitError(op) << getAddrSpaceName(space)
                        << " capacity exceeded while widening a strided alloc";
          return false;
        }
        arena.poison(*extra, needed - bytes, elemType);
        bytes = needed;
      }
    }
  }

  out = MemRefValue();
  out.arena = arenaId;
  out.byteOffset = *offset;
  out.baseOffset = *offset;
  out.sizes = std::move(sizes);
  out.strides = std::move(strides);
  out.elemType = elemType;
  out.elemBytes = elemBytes;
  out.space = space;
  allocCache[{op, core.index}] = out;
  return true;
}

//===----------------------------------------------------------------------===//
// Effects and flushing
//===----------------------------------------------------------------------===//

void Interpreter::collectRanges(const MemRefValue &mem,
                                SmallVectorImpl<ByteRange> &out) {
  if (!mem.isValid() || mem.getNumElements() <= 0)
    return;

  if (mem.isContiguous()) {
    auto [lo, hi] = mem.getSpannedBytes();
    out.push_back({mem.arena, lo, hi});
    return;
  }

  // Emit one range per contiguous innermost run. Cap the count so a pathological
  // stride pattern cannot blow up the shadow map; beyond the cap we widen to
  // the whole span, which is conservative (it can only over-report sharing).
  static constexpr size_t kMaxRanges = 1024;
  int64_t rank = mem.getRank();
  int64_t innerSize = rank ? mem.sizes[rank - 1] : 1;
  bool innerContiguous = rank == 0 || mem.strides[rank - 1] == 1;
  int64_t outerCount = 1;
  for (int64_t d = 0; d + 1 < rank; ++d)
    outerCount *= mem.sizes[d];

  if (!innerContiguous) {
    // A gap in the fastest dimension (a transposed or strided view). One
    // range per element is exact; falling back to the whole span would make
    // two cores working on disjoint interleaved slices look like they share
    // bytes, which is a false race rather than a missed one.
    int64_t count = mem.getNumElements();
    if (static_cast<size_t>(count) <= kMaxRanges) {
      for (int64_t n = 0; n < count; ++n) {
        uint64_t lo = mem.getByteAddrLinear(n);
        out.push_back({mem.arena, lo, lo + mem.elemBytes});
      }
      return;
    }
    auto [lo, hi] = mem.getSpannedBytes();
    out.push_back({mem.arena, lo, hi});
    return;
  }
  if (static_cast<size_t>(outerCount) > kMaxRanges) {
    auto [lo, hi] = mem.getSpannedBytes();
    out.push_back({mem.arena, lo, hi});
    return;
  }

  SmallVector<int64_t, 4> idx(rank, 0);
  for (int64_t n = 0; n < outerCount; ++n) {
    int64_t rem = n;
    for (int64_t d = rank - 2; d >= 0; --d) {
      idx[d] = rem % mem.sizes[d];
      rem /= mem.sizes[d];
    }
    idx[rank - 1] = 0;
    uint64_t lo = mem.getByteAddr(idx);
    out.push_back({mem.arena, lo, lo + static_cast<uint64_t>(innerSize) *
                                        mem.elemBytes});
  }
}

void Interpreter::prepareDirectAccess(CoreState &core, Operation *op,
                                      Pipe pipe, const ByteRange &range,
                                      bool isWrite) {
  if (options.sched != SchedMode::InOrder)
    flushPipe(core, pipe);
  issueResidentAccess(core, pipe, op, range, isWrite);
}

void Interpreter::filterAndReportRace(RaceReport &race, int arena) {
  // Two raw-pointer accesses to the same flag word are the synchronisation
  // mechanism, not a bug; reporting them buries the real findings.
  if (!options.checkRawPointerRaces && race.first.raw && race.second.raw)
    return;
  // Two hardware atomics on the same address are serialised by the hardware.
  if (race.first.atomic && race.second.atomic)
    return;
  race.space = arenas[arena]->getSpace();
  race.bufferName = arenas[arena]->describeAddress(race.lo);
  reportRace(race, lastSyncOp);
}

void Interpreter::issueEffect(CoreState &core, Pipe pipe, Operation *op,
                              ArrayRef<ByteRange> reads,
                              ArrayRef<ByteRange> writes,
                              std::function<void()> commit, bool isAtomic) {
  // The scalar unit issues every other pipe's instructions, so it can never
  // be the one waiting in a queue: a PIPE_S effect completes where it stands.
  issueEffectImpl(core, pipe, op, reads, writes, std::move(commit), isAtomic,
                  /*isRawPointer=*/false, /*completesNow=*/pipe == Pipe::S);
}

void Interpreter::issueResidentAccess(CoreState &core, Pipe pipe,
                                      Operation *op, const ByteRange &range,
                                      bool isWrite, bool isRawPointer) {
  // Keep the single range in a named local: an ArrayRef built from a brace
  // list would point at a temporary.
  ByteRange one = range;
  ArrayRef<ByteRange> touched(one);
  issueEffectImpl(core, pipe, op,
                  isWrite ? ArrayRef<ByteRange>{} : touched,
                  isWrite ? touched : ArrayRef<ByteRange>{},
                  /*commit=*/nullptr, /*isAtomic=*/false, isRawPointer,
                  /*completesNow=*/true);
}

void Interpreter::issueEffectImpl(CoreState &core, Pipe pipe, Operation *op,
                                  ArrayRef<ByteRange> reads,
                                  ArrayRef<ByteRange> writes,
                                  std::function<void()> commit, bool isAtomic,
                                  bool isRawPointer, bool completesNow) {
  // Advance program order *before* stamping the effect. Stamping first would
  // give a core's very first op an all-zero clock, which every other clock
  // dominates - and a dominated clock is never concurrent, so the first
  // access of each core would be invisible to the race detector.
  core.clock.tick(core.index);

  Effect effect;
  effect.op = op;
  effect.pipe = pipe;
  effect.reads.assign(reads.begin(), reads.end());
  effect.writes.assign(writes.begin(), writes.end());
  effect.commit = std::move(commit);
  effect.issueClock = core.clock;
  effect.isAtomic = isAtomic;
  effect.isRawPointer = isRawPointer;

  if (options.sched == SchedMode::InOrder) {
    commitEffect(core, effect);
    return;
  }

  checkPipeHazards(core, op, pipe, reads, writes, isRawPointer);

  if (!completesNow) {
    core.pipes.push(pipe, std::move(effect));
    return;
  }

  commitEffect(core, effect);
  if (!options.checkSync)
    return;

  // The data has landed, but the other pipes are not entitled to see it until
  // a set_flag on this pipe says so. What stays queued is a marker: no commit,
  // and no second shadow record when the queue eventually drains.
  effect.commit = nullptr;
  effect.isResident = true;
  if (core.pipes.coalesceResident(effect))
    return;
  if (core.pipes.getQueue(pipe).size() >= kMaxResidentMarkers) {
    if (!residentCapWarned) {
      residentCapWarned = true;
      report() << "warning: " << core.id.str() << " has more than "
               << kMaxResidentMarkers << " unretired " << getPipeName(pipe)
               << " accesses; further ones are dropped and hazards against "
                  "them will be missed\n";
    }
    return;
  }
  core.pipes.push(pipe, std::move(effect));
}

/// True when two byte ranges in the same arena overlap.
static bool overlaps(const ByteRange &a, const ByteRange &b) {
  return a.arena == b.arena && a.arena >= 0 && a.lo < b.hi && b.lo < a.hi;
}

static bool anyOverlap(ArrayRef<ByteRange> lhs, ArrayRef<ByteRange> rhs) {
  for (const ByteRange &a : lhs)
    for (const ByteRange &b : rhs)
      if (overlaps(a, b))
        return true;
  return false;
}

void Interpreter::checkPipeHazards(CoreState &core, Operation *op, Pipe pipe,
                                   ArrayRef<ByteRange> reads,
                                   ArrayRef<ByteRange> writes,
                                   bool isRawPointer) {
  if (!options.checkSync)
    return;
  // Effects on the same pipe retire in order, so only *other* pipes can hold
  // work that this op would overtake. An overlap there means the IR is
  // missing the set_flag/wait_flag pair that would order the two.
  for (unsigned p = 0; p < kNumPipes; ++p) {
    Pipe other = static_cast<Pipe>(p);
    if (other == pipe)
      continue;
    for (const Effect &pending : core.pipes.getQueue(other)) {
      if (pending.isToken)
        continue;
      // A macro op sits on two pipes at once; its halves are ordered by the
      // hardware, not by a flag the IR has to supply.
      if (pending.op == op)
        continue;
      // Two raw-pointer accesses to the same word are how the flag protocol
      // itself is written; ordering them is the protocol's job, not the IR's.
      if (!options.checkRawPointerRaces && isRawPointer && pending.isRawPointer)
        continue;
      bool war = anyOverlap(pending.reads, writes);
      bool waw = anyOverlap(pending.writes, writes);
      bool raw = anyOverlap(pending.writes, reads);
      if (!war && !waw && !raw)
        continue;
      // One report per op pair is enough to locate the missing flag, but a
      // pair we have already named must not stop the scan: a different pipe
      // may be holding a hazard nobody has seen yet.
      if (reportMissingSync(core, pending, op, pipe, /*conflictIsWrite=*/!raw))
        return;
    }
  }
}

void Interpreter::commitEffect(CoreState &core, Effect &effect) {
  // A resident marker was already committed when it was issued; draining it
  // only retires the ordering constraint it stands for.
  if (effect.isResident)
    return;
  if (options.checkRace) {
    AccessRecord record;
    record.core = core.index;
    record.clock = effect.issueClock;
    record.op = effect.op;
    record.valid = true;
    record.atomic = effect.isAtomic;
    record.raw = effect.isRawPointer;
    for (const ByteRange &r : effect.reads) {
      if (r.arena < 0)
        continue;
      auto race =
          arenas[r.arena]->getShadow().access(r.lo, r.hi, false, record);
      if (race)
        filterAndReportRace(*race, r.arena);
    }
    for (const ByteRange &r : effect.writes) {
      if (r.arena < 0)
        continue;
      auto race = arenas[r.arena]->getShadow().access(r.lo, r.hi, true, record);
      if (race)
        filterAndReportRace(*race, r.arena);
    }
  }
  if (effect.commit)
    effect.commit();
}

void Interpreter::flushPipe(CoreState &core, Pipe pipe) {
  auto &queue = core.pipes.getQueue(pipe);
  while (!queue.empty()) {
    Effect effect = std::move(queue.front());
    queue.pop_front();
    if (effect.isToken) {
      core.pipes.incFlag(effect.token, effect.issueClock);
      continue;
    }
    commitEffect(core, effect);
  }
}

void Interpreter::flushAllPipes(CoreState &core) {
  for (unsigned p = 0; p < kNumPipes; ++p)
    flushPipe(core, static_cast<Pipe>(p));
}

bool Interpreter::flushUntilToken(CoreState &core, Pipe pipe,
                                  const FlagKey &key) {
  if (!core.pipes.hasToken(pipe, key))
    return false;
  auto &queue = core.pipes.getQueue(pipe);
  while (!queue.empty()) {
    Effect effect = std::move(queue.front());
    queue.pop_front();
    if (effect.isToken) {
      core.pipes.incFlag(effect.token, effect.issueClock);
      if (effect.token == key)
        return true;
      continue;
    }
    commitEffect(core, effect);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Barrier participants
//===----------------------------------------------------------------------===//

void Interpreter::getBarrierParticipants(const CoreId &id, StringRef mode,
                                         SmallVectorImpl<unsigned> &out) {
  // Cores are stored in CoreId order, so `out` is deterministic.
  for (const CoreState &core : cores) {
    bool take = false;
    if (mode == "ALL")
      take = true;
    else if (mode == "ALL_CUBE" || mode == "BARRIER_CUBE")
      take = core.id.kind == CoreKind::AIC;
    else if (mode == "ALL_VECTOR" || mode == "BARRIER_VECTOR")
      // NOTE: the ODS description of BARRIER_VECTOR says "cube-cube
      // synchronization", duplicating BARRIER_CUBE's text. That reads as a
      // documentation typo, so we treat it as a vector-side barrier.
      take = core.id.kind == CoreKind::AIV;
    else if (mode == "ALL_SUB_VECTOR")
      // The sub-vector cores of one AIV, not every AIV in the kernel: they
      // are the group that shares a UB.
      take = core.id.kind == CoreKind::AIV && core.id.blockIdx == id.blockIdx;
    if (take)
      out.push_back(core.index);
  }
}

void Interpreter::wakeWaitersOn(StringRef crossFlagKey) {
  for (CoreState &core : cores) {
    if (core.status != CoreStatus::BlockedOnFlag)
      continue;
    if (core.blockedOn.crossFlagKey != crossFlagKey)
      continue;
    core.status = CoreStatus::Runnable;
    core.blockedOn = BlockReason();
  }
}

CoreState *Interpreter::findCore(const CoreId &id) {
  for (CoreState &core : cores)
    if (!(core.id < id) && !(id < core.id))
      return &core;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Frames
//===----------------------------------------------------------------------===//

void Interpreter::pushRegion(CoreState &core, Operation *owner, Block *block) {
  RegionFrame frame;
  frame.owner = owner;
  frame.block = block;
  frame.ip = block->begin();
  core.callStack.back().regions.push_back(std::move(frame));
}

void Interpreter::popRegion(CoreState &core) {
  if (!hasRegion(core))
    return;
  core.callStack.back().regions.pop_back();
}

LogicalResult Interpreter::pushCall(CoreState &core, func::FuncOp callee,
                                    Operation *callOp,
                                    ArrayRef<RuntimeValue> operands) {
  if (callee.isExternal()) {
    emitError(callOp ? callOp : callee.getOperation())
        << "cannot call external function '" << callee.getName() << "'";
    return failure();
  }
  if (operands.size() != callee.getNumArguments()) {
    emitError(callOp ? callOp : callee.getOperation())
        << "call has " << operands.size() << " operands but '"
        << callee.getName() << "' takes " << callee.getNumArguments();
    return failure();
  }

  CallFrame frame;
  frame.func = callee;
  frame.callOp = callOp;
  for (auto [arg, value] : llvm::zip(callee.getArguments(), operands))
    frame.env[arg] = value;
  core.callStack.push_back(std::move(frame));

  Block &entry = callee.getBody().front();
  pushRegion(core, callee.getOperation(), &entry);
  return success();
}

void Interpreter::popCall(CoreState &core, ArrayRef<RuntimeValue> results) {
  CallFrame frame = std::move(core.callStack.back());
  core.callStack.pop_back();
  if (core.callStack.empty())
    return;
  if (frame.callOp) {
    for (auto [result, value] : llvm::zip(frame.callOp->getResults(), results))
      setValue(core, result, value);
    // The caller's instruction pointer still sits on the call op.
    advanceIp(core);
  }
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

InFlightDiagnostic Interpreter::emitError(Operation *op) {
  ++errorCount;
  if (op)
    return op->emitError();
  return mlir::emitError(UnknownLoc::get(module.getContext()));
}

/// Print `AIC#0  file:line  op.name  pipe=...  vc=<...>`.
static void printAccessLine(llvm::raw_ostream &os, StringRef tag,
                            const AccessRecord &record,
                            ArrayRef<CoreState> cores) {
  os << "  " << tag << "  ";
  if (record.core < cores.size())
    os << cores[record.core].id.str();
  else
    os << "core#" << record.core;
  os << "  ";
  if (record.op)
    os << record.op->getLoc();
  else
    os << "<unknown>";
  os << "  ";
  if (record.op)
    os << record.op->getName();
  os << "  vc=";
  record.clock.print(os);
  os << '\n';
}

void Interpreter::reportRace(const RaceReport &race, Operation *nearestSync) {
  ++raceCount;
  llvm::raw_ostream &os = report();
  os << "DATA RACE on " << getAddrSpaceName(race.space) << ' '
     << race.bufferName << " [0x";
  os.write_hex(race.lo);
  os << ", 0x";
  os.write_hex(race.hi);
  os << ")\n";
  printAccessLine(os, race.firstIsWrite ? "W" : "R", race.first, cores);
  printAccessLine(os, race.secondIsWrite ? "W" : "R", race.second, cores);
  os << "  no happens-before edge between these two accesses\n";
  if (nearestSync) {
    os << "  nearest sync op: " << nearestSync->getName() << " @ "
       << nearestSync->getLoc() << '\n';
    // A plain barrier orders execution but carries no per-buffer dependency,
    // which is exactly the shape of the bug this tool exists to find.
    if (isa<hivm::SyncBlockOp>(nearestSync))
      os << "                   (barrier only - carries no data-dependency "
            "flag)\n";
  } else {
    os << "  no synchronisation op has executed yet on any core\n";
  }
  os << '\n';
}

bool Interpreter::reportMissingSync(const CoreState &core,
                                    const Effect &pending, Operation *op,
                                    Pipe issuingPipe, bool conflictIsWrite) {
  if (!reportedHazards.insert({pending.op, op}).second)
    return false;
  ++missingSyncCount;
  llvm::raw_ostream &os = report();
  os << "MISSING SYNC on " << core.id.str() << ": "
     << getPipeName(issuingPipe) << " op touches data still in flight on "
     << getPipeName(pending.pipe) << '\n';
  os << "  in flight  " << getPipeName(pending.pipe) << "  ";
  if (pending.op)
    os << pending.op->getName() << " @ " << pending.op->getLoc();
  os << '\n';
  os << "  " << (conflictIsWrite ? "overwrites" : "consumes ") << "  "
     << getPipeName(issuingPipe) << "  " << op->getName() << " @ "
     << op->getLoc() << '\n';
  os << "  the two are unordered without a set_flag[" << getPipeName(
            pending.pipe)
     << ", " << getPipeName(issuingPipe)
     << ", <id>] / wait_flag pair between them\n\n";
  return true;
}

void Interpreter::trace(CoreState &core, Operation *op) {
  if (!traceStream)
    return;
  *traceStream << core.id.str() << "  " << op->getName() << "  "
               << op->getLoc() << "  vc=";
  core.clock.print(*traceStream);
  *traceStream << '\n';
}

bool Interpreter::checkLayout(Operation *op, const MemRefValue &src,
                              const MemRefValue &dst) {
  if (src.layout == dst.layout)
    return true;
  // ND <-> fractal conversions are what nd2nz/nz2nd/fixpipe are for; a plain
  // copy between mismatched tags means somebody lost a conversion.
  emitError(op) << "layout mismatch: source is " << getLayoutTagName(src.layout)
                << " but destination is " << getLayoutTagName(dst.layout);
  return false;
}

//===----------------------------------------------------------------------===//
// Core setup
//===----------------------------------------------------------------------===//

/// Core kind declared by `hivm.func_core_type`, falling back to the name
/// suffix that SplitMixKernel produces.
static std::optional<CoreKind> getDeclaredCoreKind(func::FuncOp func) {
  if (auto attr = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
          "hivm.func_core_type")) {
    switch (attr.getFuncCoreType()) {
    case hivm::TFuncCoreType::AIC:
      return CoreKind::AIC;
    case hivm::TFuncCoreType::AIV:
      return CoreKind::AIV;
    case hivm::TFuncCoreType::MIX:
    case hivm::TFuncCoreType::AIC_OR_AIV:
      return std::nullopt;
    }
  }
  StringRef name = func.getName();
  if (name.ends_with("_mix_aic") || name.ends_with("_aic"))
    return CoreKind::AIC;
  if (name.ends_with("_mix_aiv") || name.ends_with("_aiv"))
    return CoreKind::AIV;
  return std::nullopt;
}

LogicalResult Interpreter::setupCores() {
  SmallVector<func::FuncOp> candidates;
  if (!options.entry.empty()) {
    auto func = module.lookupSymbol<func::FuncOp>(options.entry);
    if (!func) {
      report() << "error: no function named '" << options.entry << "'\n";
      return failure();
    }
    candidates.push_back(func);
  } else {
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      // `hivm.backup_function` copies are kept for later passes and are not
      // executable entry points.
      if (func->hasAttr("hivm.backup_function"))
        continue;
      if (func->hasAttr("hacc.entry"))
        candidates.push_back(func);
    }
    if (candidates.empty()) {
      // Single-function modules (hand-written lit tests) need no attribute.
      SmallVector<func::FuncOp> all;
      for (auto func : module.getOps<func::FuncOp>())
        if (!func.isExternal() && !func->hasAttr("hivm.backup_function"))
          all.push_back(func);
      if (all.size() == 1)
        candidates.push_back(all.front());
    }
  }

  if (candidates.empty()) {
    report() << "error: no entry function found; pass --entry=<name>\n";
    return failure();
  }

  for (func::FuncOp func : candidates) {
    auto kind = getDeclaredCoreKind(func);
    if (kind == CoreKind::AIC) {
      if (aicEntry && aicEntry != func) {
        report() << "error: two AIC entry functions ('" << aicEntry.getName()
                 << "' and '" << func.getName() << "'); pass --entry\n";
        return failure();
      }
      aicEntry = func;
    } else {
      if (aivEntry && aivEntry != func) {
        report() << "error: two AIV entry functions ('" << aivEntry.getName()
                 << "' and '" << func.getName() << "'); pass --entry\n";
        return failure();
      }
      aivEntry = func;
    }
  }

  // Build the core list in CoreId order: for each block, the AIC first, then
  // its sub-vector cores.
  for (unsigned block = 0; block < options.blockDim; ++block) {
    if (aicEntry) {
      CoreState core;
      core.id = {block, CoreKind::AIC, 0};
      cores.push_back(std::move(core));
    }
    if (aivEntry) {
      for (unsigned sub = 0; sub < options.subBlockNum; ++sub) {
        CoreState core;
        core.id = {block, CoreKind::AIV, sub};
        cores.push_back(std::move(core));
      }
    }
  }
  std::stable_sort(cores.begin(), cores.end(),
                   [](const CoreState &a, const CoreState &b) {
                     return a.id < b.id;
                   });
  for (unsigned i = 0; i < cores.size(); ++i) {
    cores[i].index = i;
    cores[i].clock = VectorClock(static_cast<unsigned>(cores.size()));
  }

  if (failed(materializeGlobalArgs()))
    return failure();

  for (CoreState &core : cores) {
    func::FuncOp entry = core.id.kind == CoreKind::AIC ? aicEntry : aivEntry;
    SmallVector<RuntimeValue, 4> args;
    for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
      auto it = globalArgs.find(i);
      args.push_back(it != globalArgs.end() ? it->second : RuntimeValue());
    }
    if (failed(pushCall(core, entry, /*callOp=*/nullptr, args)))
      return failure();
  }
  return success();
}

/// Parse one `--args` entry into an ArgSpec.
static ArgSpec parseArgSpec(StringRef text) {
  ArgSpec spec;
  text = text.trim();
  if (text.empty()) {
    spec.kind = ArgSpec::Kind::Zeros;
    return spec;
  }
  if (text.ends_with(".npy")) {
    spec.kind = ArgSpec::Kind::Npy;
    spec.path = text.str();
    return spec;
  }
  if (text == "zeros") {
    spec.kind = ArgSpec::Kind::Zeros;
    return spec;
  }
  if (text == "poison") {
    spec.kind = ArgSpec::Kind::Poison;
    return spec;
  }
  if (text == "arange") {
    spec.kind = ArgSpec::Kind::Arange;
    return spec;
  }
  double value = 0.0;
  if (!text.getAsDouble(value)) {
    spec.kind = ArgSpec::Kind::Scalar;
    spec.scalar = value;
    return spec;
  }
  spec.kind = ArgSpec::Kind::Zeros;
  return spec;
}

LogicalResult Interpreter::materializeGlobalArgs() {
  func::FuncOp shape = aivEntry ? aivEntry : aicEntry;
  if (aicEntry && aivEntry &&
      aicEntry.getNumArguments() != aivEntry.getNumArguments()) {
    report() << "error: AIC and AIV entries have different signatures; the "
                "interpreter cannot bind one argument list to both\n";
    return failure();
  }

  Arena &gm = *arenas[arenaIndex[{static_cast<unsigned>(AddrSpace::GM), 0}]];
  Arena &host =
      *arenas[arenaIndex[{static_cast<unsigned>(AddrSpace::Host), 0}]];

  for (unsigned i = 0, e = shape.getNumArguments(); i < e; ++i) {
    Type type = shape.getArgument(i).getType();
    ArgSpec spec;
    if (i < options.args.size())
      spec = parseArgSpec(options.args[i]);
    else
      spec.kind = ArgSpec::Kind::Zeros;

    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      // Kernel arguments live in GM; anything else (hand-written tests) gets
      // host scratch. Report the space the buffer is really in, so a later
      // diagnostic does not name a pool the bytes are not in.
      AddrSpace space =
          getAddrSpaceOf(memrefType) == AddrSpace::GM ? AddrSpace::GM
                                                      : AddrSpace::Host;
      Arena &arena = space == AddrSpace::GM ? gm : host;
      int arenaId = arenaIndex[{static_cast<unsigned>(space), 0}];
      Type elemType = memrefType.getElementType();
      unsigned elemBytes = getStorageSize(elemType);

      SmallVector<int64_t, 4> sizes(memrefType.getShape().begin(),
                                    memrefType.getShape().end());

      std::optional<NpyArray> npy;
      if (spec.kind == ArgSpec::Kind::Npy) {
        auto loaded = readNpy(spec.path);
        if (!loaded) {
          report() << "error: " << toString(loaded.takeError()) << '\n';
          return failure();
        }
        npy = std::move(*loaded);
        if (npy->elemBytes != elemBytes) {
          report() << "error: '" << spec.path << "' has " << npy->elemBytes
                   << "-byte elements but argument " << i << " needs "
                   << elemBytes << '\n';
          return failure();
        }
      }

      for (int64_t &s : sizes) {
        if (!ShapedType::isDynamic(s))
          continue;
        if (npy)
          s = npy->getNumElements();
        else if (spec.elementCount)
          s = spec.elementCount;
        else
          s = static_cast<int64_t>(options.dynGmElems);
      }

      int64_t numElems = 1;
      for (int64_t s : sizes)
        numElems *= s;
      uint64_t bytes = static_cast<uint64_t>(numElems) * elemBytes;

      std::string name = ("%arg" + Twine(i)).str();
      auto offset = arena.allocate(bytes, 64, name, shape.getOperation());
      if (!offset) {
        report() << "error: " << getAddrSpaceName(space)
                 << " exhausted while binding argument " << i << " (" << bytes
                 << " bytes); raise --gm-size\n";
        return failure();
      }

      uint8_t *dst = arena.at(*offset, bytes);
      if (npy) {
        std::memcpy(dst, npy->data.data(),
                    std::min<size_t>(npy->data.size(), bytes));
      } else if (spec.kind == ArgSpec::Kind::Poison) {
        fillPoison(dst, bytes, elemType);
      } else if (spec.kind == ArgSpec::Kind::Arange) {
        const llvm::fltSemantics *sem = getFloatSemantics(elemType);
        for (int64_t n = 0; n < numElems; ++n) {
          RuntimeValue v;
          if (sem) {
            llvm::APFloat f(*sem);
            f.convertFromAPInt(llvm::APInt(64, static_cast<uint64_t>(n)),
                               /*IsSigned=*/true,
                               llvm::APFloat::rmNearestTiesToEven);
            v = RuntimeValue::getFloat(f);
          } else {
            v = RuntimeValue::getInt(
                llvm::APInt(elemBytes * 8, static_cast<uint64_t>(n)));
          }
          storeElement(dst + n * elemBytes, elemType, v);
        }
      } else if (spec.kind == ArgSpec::Kind::Scalar) {
        // `--args=...,7,...` against a buffer means "every element is 7". The
        // same spec binds a plain scalar argument to 7, so the two agree.
        const llvm::fltSemantics *sem = getFloatSemantics(elemType);
        RuntimeValue v;
        if (sem) {
          llvm::APFloat f(spec.scalar);
          bool losesInfo = false;
          f.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
          v = RuntimeValue::getFloat(f);
        } else {
          v = RuntimeValue::getInt(
              llvm::APInt(elemBytes * 8,
                          static_cast<uint64_t>(static_cast<int64_t>(
                              spec.scalar)),
                          /*isSigned=*/true));
        }
        for (int64_t n = 0; n < numElems; ++n)
          storeElement(dst + n * elemBytes, elemType, v);
      } else {
        std::memset(dst, 0, bytes);
      }

      MemRefValue mem;
      mem.arena = arenaId;
      mem.byteOffset = *offset;
      mem.baseOffset = *offset;
      mem.sizes = sizes;
      mem.strides.resize(sizes.size());
      int64_t acc = 1;
      for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
        mem.strides[d] = acc;
        acc *= sizes[d];
      }
      mem.elemType = elemType;
      mem.elemBytes = elemBytes;
      mem.space = space;
      globalArgs[i] = RuntimeValue::getMemRef(mem);
      if (space == AddrSpace::GM)
        gmArgBuffers[i] = mem;
      continue;
    }

    // Scalars.
    if (const llvm::fltSemantics *sem = getFloatSemantics(type)) {
      llvm::APFloat value(*sem);
      if (spec.kind == ArgSpec::Kind::Scalar) {
        value = llvm::APFloat(spec.scalar);
        bool losesInfo = false;
        value.convert(*sem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      } else {
        value = llvm::APFloat::getZero(*sem);
      }
      globalArgs[i] = RuntimeValue::getFloat(value);
    } else if (auto intType = dyn_cast<IntegerType>(type)) {
      int64_t value = spec.kind == ArgSpec::Kind::Scalar
                          ? static_cast<int64_t>(spec.scalar)
                          : 0;
      globalArgs[i] = RuntimeValue::getInt(
          llvm::APInt(intType.getWidth(), static_cast<uint64_t>(value), true));
    } else if (isa<IndexType>(type)) {
      globalArgs[i] = RuntimeValue::getIndex(
          spec.kind == ArgSpec::Kind::Scalar
              ? static_cast<int64_t>(spec.scalar)
              : 0);
    } else {
      // Unmodelled argument type: leave unbound and let the first use report.
      globalArgs[i] = RuntimeValue();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

ExecResult Interpreter::step(CoreState &core) {
  CallFrame &frame = core.callStack.back();
  RegionFrame &region = frame.regions.back();

  if (region.ip == region.block->end()) {
    // A block without a terminator can only happen in malformed IR.
    emitError(region.owner) << "region block ran off the end without a "
                               "terminator";
    return ExecResult::Error;
  }

  Operation *op = &*region.ip;
  trace(core, op);
  ++core.stepCount;

  const OpHandlerFn *handler = registry.lookup(op->getName().getStringRef());
  if (!handler) {
    // Distinguish "this op is not modelled" from "this IR is not memref
    // form". The latter is a much more common mistake and a much more
    // actionable message.
    auto isTensor = [](Type type) { return isa<RankedTensorType>(type); };
    if (llvm::any_of(op->getOperandTypes(), isTensor) ||
        llvm::any_of(op->getResultTypes(), isTensor)) {
      emitError(op) << "tensor-form operand or result on '"
                    << op->getName().getStringRef()
                    << "': the interpreter only accepts fully bufferized "
                       "(memref) HIVM IR; run the remaining bufferization "
                       "passes first";
      return ExecResult::Error;
    }
    emitError(op) << "unsupported op: " << op->getName().getStringRef();
    return ExecResult::Error;
  }
  return (*handler)(*this, core, op);
}

CoreState *Interpreter::pickCore() {
  if (cores.empty())
    return nullptr;

  if (options.sched == SchedMode::Fuzz) {
    SmallVector<unsigned> runnable;
    for (const CoreState &core : cores)
      if (core.status == CoreStatus::Runnable)
        runnable.push_back(core.index);
    if (runnable.empty())
      return nullptr;
    // Indices come from an ordered scan and the engine is seeded, so the
    // choice is reproducible for a given --seed.
    unsigned pick = runnable[rng() % runnable.size()];
    return &cores[pick];
  }

  // Round-robin starting after the core that ran last, in CoreId order.
  for (unsigned n = 0; n < cores.size(); ++n) {
    unsigned idx = (currentCoreIdx + n) % cores.size();
    if (cores[idx].status == CoreStatus::Runnable)
      return &cores[idx];
  }
  return nullptr;
}

bool Interpreter::detectDeadlock() {
  bool anyBlocked = false;
  for (const CoreState &core : cores) {
    if (core.status == CoreStatus::Runnable)
      return false;
    if (core.isBlocked())
      anyBlocked = true;
  }
  if (!anyBlocked)
    return false;
  if (!options.checkDeadlock)
    return true;

  llvm::raw_ostream &os = report();

  // Distinguish the two shapes the plan calls out: a circular flag wait, and a
  // barrier that some participants never reached.
  bool barrierMismatch = false;
  for (auto &[key, state] : barriers) {
    if (state.arrived.empty() || state.arrived.size() >= state.expected)
      continue;
    barrierMismatch = true;
    os << "DEADLOCK: barrier " << key << " arrival mismatch\n";
    os << "  expected participants:";
    SmallVector<unsigned> participants;
    getBarrierParticipants(cores.front().id,
                           StringRef(key).split(':').first, participants);
    for (unsigned idx : participants)
      os << ' ' << cores[idx].id.str();
    os << " (" << state.expected << " cores)\n";
    for (size_t i = 0; i < state.arrived.size(); ++i) {
      os << "  arrived:  " << cores[state.arrived[i]].id.str();
      if (i < state.arrivedAt.size() && state.arrivedAt[i])
        os << " @ " << state.arrivedAt[i]->getLoc();
      os << '\n';
    }
    for (const CoreState &core : cores) {
      if (llvm::is_contained(state.arrived, core.index))
        continue;
      if (!llvm::is_contained(participants, core.index))
        continue;
      os << "  " << core.id.str() << " is ";
      if (core.status == CoreStatus::Done) {
        os << "Done";
        if (core.returnedAt)
          os << " (returned at " << core.returnedAt->getLoc() << ")";
        os << " and never reached " << key << '\n';
        os << "  hint: a barrier that only some cores execute is the classic "
              "symptom of a barrier cloned into a conditional region\n";
      } else {
        os << "blocked at ";
        if (core.blockedOn.op)
          os << core.blockedOn.op->getLoc() << "  " << core.blockedOn.what;
        os << '\n';
      }
    }
    os << '\n';
  }

  if (!barrierMismatch) {
    os << "DEADLOCK: circular flag wait\n";
    for (const CoreState &core : cores) {
      if (!core.isBlocked())
        continue;
      os << "  " << core.id.str() << " blocked at ";
      if (core.blockedOn.op)
        os << core.blockedOn.op->getLoc();
      os << "  " << core.blockedOn.what << '\n';
    }
    for (const CoreState &core : cores) {
      if (core.status != CoreStatus::Done)
        continue;
      os << "  " << core.id.str() << " is Done";
      if (core.returnedAt)
        os << " (returned at " << core.returnedAt->getLoc() << ')';
      os << '\n';
    }
    os << '\n';
  }
  return true;
}

void Interpreter::reportLeaks() {
  for (CoreState &core : cores) {
    size_t pending = core.pipes.pendingCount();
    if (pending)
      report() << "warning: " << core.id.str() << " finished with " << pending
               << " effect(s) still queued on its pipes\n";
    for (auto &[key, count] : core.pipes.getFlagCounts()) {
      if (count == 0)
        continue;
      report() << "warning: " << core.id.str() << " leaked flag "
               << getPipeName(key.setPipe) << "->" << getPipeName(key.waitPipe)
               << " id=" << key.eventId << " (set " << count
               << " more times than waited)\n";
    }
  }
  for (auto &[key, state] : crossFlags) {
    if (state.count != 0)
      report() << "warning: cross-core flag " << key.str() << " left at "
               << state.count << '\n';
  }
}

LogicalResult Interpreter::run() {
  if (failed(setupCores()))
    return failure();

  report() << "npuir-interp: " << cores.size() << " core(s), sched=";
  switch (options.sched) {
  case SchedMode::InOrder:
    report() << "inorder";
    break;
  case SchedMode::Lazy:
    report() << "lazy";
    break;
  case SchedMode::Fuzz:
    report() << "fuzz seed=" << options.seed;
    break;
  }
  report() << ", block-dim=" << options.blockDim
           << ", sub-block-num=" << options.subBlockNum << '\n';

  while (!aborted) {
    CoreState *core = pickCore();
    if (!core) {
      bool anyBlocked = false;
      for (const CoreState &c : cores)
        anyBlocked |= c.isBlocked();
      if (!anyBlocked)
        break; // Everyone finished.

      // Nothing is runnable but somebody is parked. Before calling that a
      // deadlock, give every blocked core one more chance whenever some core
      // has made globally visible progress since the last retry: relying
      // only on producers to wake their consumers means a single missed
      // wake-up turns a correct kernel into a reported deadlock.
      if (progressCounter != lastRetryProgress) {
        lastRetryProgress = progressCounter;
        for (CoreState &c : cores) {
          if (!c.isBlocked())
            continue;
          c.status = CoreStatus::Runnable;
          c.blockedOn = BlockReason();
        }
        continue;
      }

      // No progress since the last retry: this is genuinely stuck.
      detectDeadlock();
      return failure();
    }
    currentCoreIdx = core->index;

    // Run this core until it blocks, finishes, or (in fuzz mode) its randomly
    // chosen slice expires.
    uint64_t slice = options.sched == SchedMode::Fuzz ? (rng() % 8) + 1 : ~0ull;
    for (uint64_t n = 0; n < slice; ++n) {
      if (core->status != CoreStatus::Runnable)
        break;
      if (++totalSteps > options.maxSteps) {
        report() << "error: step limit (" << options.maxSteps
                 << ") exceeded; use --max-steps to raise it\n";
        aborted = true;
        return failure();
      }
      ExecResult result = step(*core);
      if (result == ExecResult::Error) {
        core->status = CoreStatus::Failed;
        aborted = true;
        break;
      }
      if (result == ExecResult::Block)
        break;
      if (result == ExecResult::Advance) {
        if (!core->callStack.empty())
          advanceIp(*core);
      }
    }

    if (aborted)
      break;
  }

  if (aborted)
    return failure();

  bool deadlocked = false;
  for (CoreState &core : cores)
    if (core.isBlocked())
      deadlocked = true;
  if (deadlocked)
    return failure();

  reportLeaks();

  if (missingSyncCount)
    report() << missingSyncCount
             << " missing intra-core synchronisation point(s) detected\n";
  if (raceCount)
    report() << raceCount << " data race(s) detected\n";
  if (errorCount)
    return failure();
  if (raceCount && options.checkRace)
    return failure();
  if (missingSyncCount && options.checkSync)
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

LogicalResult Interpreter::dumpOutputs() {
  if (options.outPrefix.empty())
    return success();
  for (auto &[index, mem] : gmArgBuffers) {
    Arena &arena = *arenas[mem.arena];
    uint64_t bytes = static_cast<uint64_t>(mem.getNumElements()) *
                     mem.elemBytes;
    const uint8_t *data = arena.at(mem.byteOffset, bytes);
    if (!data)
      continue;
    std::string path =
        (options.outPrefix + "arg" + Twine(index) + ".npy").str();
    auto error =
        writeNpy(path, getNpyDType(mem.elemType), mem.sizes,
                 ArrayRef<uint8_t>(data, bytes));
    if (error) {
      report() << "error: " << toString(std::move(error)) << '\n';
      return failure();
    }
    if (options.verbose)
      report() << "wrote " << path << '\n';
  }
  return success();
}

} // namespace interp
} // namespace bishengir
