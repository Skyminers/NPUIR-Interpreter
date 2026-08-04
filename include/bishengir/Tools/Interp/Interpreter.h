//===- Interpreter.h - NPUIR interpreter core -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Green-thread scheduler, per-core execution state and the op-handler
// registry. Cores are cooperative: a core runs until it blocks on a flag,
// barrier or lock, at which point the scheduler picks the next runnable core
// in CoreId order (or, in fuzz mode, a seeded random one).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_INTERPRETER_H
#define BISHENGIR_TOOLS_INTERP_INTERPRETER_H

#include "bishengir/Tools/Interp/Memory.h"
#include "bishengir/Tools/Interp/PipeEngine.h"
#include "bishengir/Tools/Interp/Value.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace bishengir {
namespace interp {

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

enum class SchedMode {
  /// Effects commit the moment they are issued. Fastest, and the reference
  /// answer for differential testing - but blind to missing synchronisation.
  InOrder,
  /// Effects commit as late as the flush rules allow. The default.
  Lazy,
  /// Like Lazy, but core interleaving and commit timing are randomised from
  /// a fixed seed.
  Fuzz
};

/// How one kernel argument is materialised.
struct ArgSpec {
  enum class Kind { Npy, Zeros, Poison, Scalar, Arange } kind = Kind::Poison;
  std::string path;         // Npy
  double scalar = 0.0;      // Scalar
  int64_t elementCount = 0; // Zeros / Poison / Arange, 0 = derive from type
};

struct InterpOptions {
  std::string entry;
  std::vector<std::string> args;
  std::string outPrefix;
  unsigned blockDim = 1;
  unsigned subBlockNum = 1;
  SchedMode sched = SchedMode::Lazy;
  uint64_t seed = 0;

  uint64_t gmSize = 64ull * 1024 * 1024;
  uint64_t ubSize = 192 * 1024;
  uint64_t l1SizeBytes = 512 * 1024;
  uint64_t l0aSize = 64 * 1024;
  uint64_t l0bSize = 64 * 1024;
  uint64_t l0cSize = 256 * 1024;
  uint64_t ssbufSize = 4 * 1024 * 1024;
  uint64_t hostSize = 16 * 1024 * 1024;
  /// Size given to a dynamically shaped (`memref<?x...>`) GM argument.
  uint64_t dynGmElems = 4096;

  bool poison = true;
  bool checkRace = true;
  bool checkDeadlock = true;
  bool checkOob = true;
  /// Report an op that touches bytes another pipe of the same core has not
  /// committed yet - i.e. a missing intra-core set_flag/wait_flag pair.
  bool checkSync = true;
  /// Also report races where both sides are raw-pointer flag traffic. Off by
  /// default: those addresses are the synchronisation medium itself.
  bool checkRawPointerRaces = false;
  bool exactLayout = false;
  bool verbose = false;
  std::string traceFile;
  uint64_t maxSteps = 200ull * 1000 * 1000;
};

//===----------------------------------------------------------------------===//
// Cores
//===----------------------------------------------------------------------===//

enum class CoreKind : uint8_t { AIC = 0, AIV = 1 };

llvm::StringRef getCoreKindName(CoreKind kind);

/// Cores are always addressed by the full triple even when only one block is
/// simulated, so that turning on multi-block later needs no data-structure
/// change.
struct CoreId {
  unsigned blockIdx = 0;
  CoreKind kind = CoreKind::AIV;
  unsigned subBlockIdx = 0;

  bool operator<(const CoreId &o) const {
    if (blockIdx != o.blockIdx)
      return blockIdx < o.blockIdx;
    if (kind != o.kind)
      return kind < o.kind;
    return subBlockIdx < o.subBlockIdx;
  }
  std::string str() const;
};

enum class CoreStatus : uint8_t {
  Runnable,
  BlockedOnFlag,
  BlockedOnBarrier,
  BlockedOnLock,
  Done,
  Failed
};

/// Why a core is parked, so the deadlock report can explain the cycle.
struct BlockReason {
  mlir::Operation *op = nullptr;
  std::string what;
  /// Cross-core flag id, when blocked on sync_block_wait.
  int64_t flagId = -1;
  /// Identity of the cross-core flag being waited on, so the producer can
  /// wake exactly the cores that care.
  std::string crossFlagKey;
  /// Barrier key, when blocked on sync_block.
  std::string barrierKey;
};

/// One region activation inside a call frame: a block plus the instruction
/// pointer into it, plus loop bookkeeping for `scf.for` / `scf.while`.
struct RegionFrame {
  mlir::Operation *owner = nullptr; // null for the function entry block
  mlir::Block *block = nullptr;
  mlir::Block::iterator ip;

  bool isForLoop = false;
  int64_t iv = 0, ub = 0, step = 1;
  llvm::SmallVector<RuntimeValue, 2> iterArgs;

  bool isWhileBefore = false;
  bool isWhileAfter = false;
};

/// One function activation. All regions of a call share one environment: MLIR
/// values are unique per op instance, and loop-carried block arguments are
/// meant to be overwritten on each iteration.
struct CallFrame {
  mlir::func::FuncOp func;
  mlir::Operation *callOp = nullptr; // null for the kernel entry
  llvm::DenseMap<mlir::Value, RuntimeValue> env;
  llvm::SmallVector<RegionFrame, 4> regions;
};

class Interpreter;

struct CoreState {
  CoreId id;
  unsigned index = 0; // dense core index, also the vector-clock slot
  llvm::SmallVector<CallFrame, 2> callStack;
  PipeEngine pipes;
  VectorClock clock;
  CoreStatus status = CoreStatus::Runnable;
  BlockReason blockedOn;
  uint64_t stepCount = 0;
  /// How many times this core has passed each barrier site. A core may leave
  /// a barrier only once the site's generation has moved past its own count,
  /// which is what stops it running ahead into the next rendezvous.
  std::map<std::string, uint64_t> barrierGeneration;
  /// Set when the core has executed its kernel entry `return`.
  mlir::Operation *returnedAt = nullptr;

  bool isBlocked() const {
    return status == CoreStatus::BlockedOnFlag ||
           status == CoreStatus::BlockedOnBarrier ||
           status == CoreStatus::BlockedOnLock;
  }
};

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

enum class ExecResult {
  /// The op completed; the scheduler advances past it.
  Advance,
  /// The handler already repositioned the instruction pointer (control flow).
  Handled,
  /// The op cannot proceed yet; retry it when the core is next scheduled.
  Block,
  /// Unrecoverable error; the run aborts.
  Error
};

using OpHandlerFn =
    std::function<ExecResult(Interpreter &, CoreState &, mlir::Operation *)>;

/// Registry keyed by the full op name. The plan proposed `ExternalModel`
/// interfaces; a name-keyed registry achieves the same zero-intrusion goal
/// (no HIVM `.td` is touched) with far less boilerplate, and keeps dispatch
/// order independent of pointer hashing.
class OpRegistry {
public:
  void add(llvm::StringRef opName, OpHandlerFn fn) {
    handlers[opName] = std::move(fn);
  }
  const OpHandlerFn *lookup(llvm::StringRef opName) const {
    auto it = handlers.find(opName);
    return it == handlers.end() ? nullptr : &it->second;
  }

  /// Replace every handler whose name starts with one of `prefixes` by
  /// `transform(handler)`. Used to lift the scalar arith/math handlers to
  /// vector operands without duplicating them. Iteration order does not
  /// matter here: each entry is rewritten independently.
  void transformMatching(llvm::ArrayRef<llvm::StringRef> prefixes,
                         llvm::function_ref<OpHandlerFn(OpHandlerFn)>
                             transform) {
    for (auto &entry : handlers)
      for (llvm::StringRef prefix : prefixes)
        if (entry.getKey().starts_with(prefix)) {
          entry.second = transform(entry.second);
          break;
        }
  }

private:
  llvm::StringMap<OpHandlerFn> handlers;
};

/// Populated by the Register*Ops functions below.
void registerCommunityOps(OpRegistry &registry);
void registerVectorOps(OpRegistry &registry);
void registerHIVMElementwiseOps(OpRegistry &registry);
void registerHIVMMemoryOps(OpRegistry &registry);
void registerHIVMSyncOps(OpRegistry &registry);
void registerHIVMMiscOps(OpRegistry &registry);
void registerHIVMShapeOps(OpRegistry &registry);
void registerHIVMIndirectOps(OpRegistry &registry);

inline void registerAllInterpOps(OpRegistry &registry) {
  registerCommunityOps(registry);
  registerVectorOps(registry);
  registerHIVMElementwiseOps(registry);
  registerHIVMMemoryOps(registry);
  registerHIVMSyncOps(registry);
  registerHIVMMiscOps(registry);
  registerHIVMShapeOps(registry);
  registerHIVMIndirectOps(registry);
}

//===----------------------------------------------------------------------===//
// Cross-core synchronisation state
//===----------------------------------------------------------------------===//

/// Key of a cross-core flag. `scope` is the block index for intra-block and
/// inter-subblock flags and -1 for inter-block ones. The ordered (tpipe, pipe)
/// pair is what distinguishes a V->C flag from the C->V flag with the same id.
struct CrossFlagKey {
  int64_t scope = 0;
  Pipe tpipe = Pipe::S;
  Pipe pipe = Pipe::S;
  int64_t flagId = 0;

  bool operator<(const CrossFlagKey &o) const {
    if (scope != o.scope)
      return scope < o.scope;
    if (tpipe != o.tpipe)
      return tpipe < o.tpipe;
    if (pipe != o.pipe)
      return pipe < o.pipe;
    return flagId < o.flagId;
  }
  std::string str() const;
};

struct CrossFlagState {
  // AIC -> AIV is a broadcast generation: every vector sub-core may consume
  // each generation once.
  int64_t aicGeneration = 0;
  VectorClock aicClock;
  mlir::Operation *aicLastSetter = nullptr;
  std::map<unsigned, int64_t> aivSeenAicGeneration;

  // AIV -> AIC is an aggregate: the cube may consume one generation only
  // after every vector sub-core in its scope has contributed a token.
  std::map<unsigned, int64_t> aivCount;
  std::map<unsigned, VectorClock> aivClock;
  std::map<unsigned, mlir::Operation *> aivLastSetter;

  // Kernels containing only one core kind keep ordinary semaphore semantics.
  std::array<int64_t, 2> sameKindCount = {0, 0};
  std::array<VectorClock, 2> sameKindClock;
};

/// Rendezvous state of one `hivm.hir.sync_block` site.
struct BarrierState {
  std::vector<unsigned> arrived;
  std::vector<mlir::Operation *> arrivedAt;
  unsigned expected = 0;
  VectorClock mergedClock;
  /// Bumped every time the barrier fires. A core passes when this has moved
  /// beyond the core's own count, which lets a barrier inside a loop rearm
  /// without letting anyone take two turns.
  uint64_t generation = 0;
};

//===----------------------------------------------------------------------===//
// Interpreter
//===----------------------------------------------------------------------===//

class Interpreter {
public:
  Interpreter(mlir::ModuleOp module, InterpOptions options);
  ~Interpreter();

  /// Set up cores and memory, then run to completion. Returns failure if the
  /// program hit an error, a deadlock or (when enabled) a data race.
  mlir::LogicalResult run();

  /// Write every GM buffer bound to a kernel argument to
  /// `<outPrefix><argIndex>.npy`.
  mlir::LogicalResult dumpOutputs();

  //=== accessors used by op handlers ====================================//

  const InterpOptions &getOptions() const { return options; }
  mlir::ModuleOp getModule() { return module; }
  llvm::raw_ostream &report();

  RuntimeValue getValue(CoreState &core, mlir::Value value);
  void setValue(CoreState &core, mlir::Value value, RuntimeValue rv);

  /// Resolve a memref-typed operand, reporting an error if it is unbound.
  bool getMemRefOperand(CoreState &core, mlir::Value value, MemRefValue &out,
                        mlir::Operation *op);

  /// Arena by id. Bounds-checked rather than asserted: release builds run
  /// with assertions off, where an out-of-range id would be silent UB.
  Arena &getArena(int id) {
    assert(id >= 0 && static_cast<size_t>(id) < arenas.size() &&
           "arena id out of range");
    if (id < 0 || static_cast<size_t>(id) >= arenas.size())
      return *arenas.front();
    return *arenas[id];
  }
  /// Arena id for `space`. Core-local spaces are keyed by the block that owns
  /// them: in a MIX kernel the AIC and its AIVs share one UB pool.
  int getArenaId(AddrSpace space, const CoreId &id);

  /// Allocate storage for `type` and return a view over it.
  bool allocateMemRef(CoreState &core, mlir::MemRefType type,
                      mlir::Operation *op, MemRefValue &out);

  //=== effects ==========================================================//

  /// Issue a pipe effect. In `inorder` mode it commits right away; otherwise
  /// it is queued on `pipe` until a flush rule drains it. PIPE_S is the
  /// exception: the scalar unit issues every other pipe's instructions and so
  /// cannot itself be waiting in a queue, and its effect completes at once
  /// (see `issueResidentAccess` for what stays behind).
  void issueEffect(CoreState &core, Pipe pipe, mlir::Operation *op,
                   llvm::ArrayRef<ByteRange> reads,
                   llvm::ArrayRef<ByteRange> writes,
                   std::function<void()> commit, bool isAtomic = false);

  /// Issue an access whose data movement the caller has already performed, or
  /// is about to perform inline: scalar `memref.load`/`memref.store` on
  /// PIPE_S, `vector.transfer_read` on PIPE_V. The shadow-memory record is
  /// taken now, but a *resident marker* stays queued on `pipe` so that a
  /// later op on another pipe touching the same bytes is still reported as
  /// missing its `set_flag[pipe, ...]` pair - which is what makes a scalar
  /// store feeding an `hivm.hir.store` without `set_flag[PIPE_S, PIPE_MTE3]`
  /// visible.
  void issueResidentAccess(CoreState &core, Pipe pipe, mlir::Operation *op,
                           const ByteRange &range, bool isWrite,
                           bool isRawPointer = false);

  /// `issueResidentAccess` preceded by a drain of `pipe` itself, for an op the
  /// pipe executes to completion before the interpreter moves on. A pipe
  /// retires in order and so does observe its own earlier work.
  void prepareDirectAccess(CoreState &core, mlir::Operation *op, Pipe pipe,
                           const ByteRange &range, bool isWrite);

  /// Byte ranges touched by a memref view, coalesced when contiguous.
  void collectRanges(const MemRefValue &mem,
                     llvm::SmallVectorImpl<ByteRange> &out);

  void flushPipe(CoreState &core, Pipe pipe);
  void flushAllPipes(CoreState &core);
  /// Drain `pipe` up to and including the first token matching `key`.
  /// Returns false when no such token is queued.
  bool flushUntilToken(CoreState &core, Pipe pipe, const FlagKey &key);

  //=== cross-core sync ==================================================//

  std::map<CrossFlagKey, CrossFlagState> &getCrossFlags() { return crossFlags; }
  std::map<std::string, BarrierState> &getBarriers() { return barriers; }

  /// Clock published by the last holder of a `sync_block_lock` variable,
  /// keyed by (arena, byte offset). Acquiring the lock joins it, so the
  /// critical sections of successive blocks are properly ordered even when a
  /// core finds the lock already free and never blocks.
  VectorClock &getLockClock(int arena, uint64_t offset) {
    return lockClocks[{arena, offset}];
  }

  /// Cores that must reach a barrier of the given mode, in CoreId order.
  void getBarrierParticipants(const CoreId &id, llvm::StringRef mode,
                              llvm::SmallVectorImpl<unsigned> &out);

  std::vector<CoreState> &getCores() { return cores; }

  /// Record that some core made globally visible progress: a cross-core flag
  /// was raised, a barrier fired, a lock changed hands. The scheduler uses
  /// this to tell a real deadlock from cores that simply have not re-tested
  /// their condition yet.
  void noteProgress() { ++progressCounter; }
  /// Wake every core parked on the given cross-core flag.
  void wakeWaitersOn(llvm::StringRef crossFlagKey);
  CoreState *findCore(const CoreId &id);
  unsigned getNumCores() const { return static_cast<unsigned>(cores.size()); }

  //=== control flow helpers used by handlers ============================//

  /// Push a region activation whose single block starts executing now.
  void pushRegion(CoreState &core, mlir::Operation *owner, mlir::Block *block);
  /// Pop the innermost region activation.
  void popRegion(CoreState &core);
  RegionFrame &currentRegion(CoreState &core) {
    return core.callStack.back().regions.back();
  }
  CallFrame &currentFrame(CoreState &core) { return core.callStack.back(); }
  /// True when the core still has a region to execute in. Handlers that pop
  /// regions check this rather than relying on assertions, which are off in
  /// release builds.
  static bool hasRegion(const CoreState &core) {
    return !core.callStack.empty() && !core.callStack.back().regions.empty();
  }

  /// Enter `callee`, binding `operands` to its arguments.
  mlir::LogicalResult pushCall(CoreState &core, mlir::func::FuncOp callee,
                               mlir::Operation *callOp,
                               llvm::ArrayRef<RuntimeValue> operands);
  /// Return from the innermost call, binding `results` in the caller.
  void popCall(CoreState &core, llvm::ArrayRef<RuntimeValue> results);

  //=== diagnostics ======================================================//

  void reportRace(const RaceReport &race, mlir::Operation *nearestSync);
  /// Report an op that reads or writes bytes still in flight on another pipe
  /// of the same core.
  /// Returns false when this op pair was already reported, so the caller can
  /// keep looking for a *different* hazard instead of stopping.
  bool reportMissingSync(const CoreState &core, const Effect &pending,
                         mlir::Operation *op, Pipe issuingPipe,
                         bool conflictIsWrite);
  mlir::InFlightDiagnostic emitError(mlir::Operation *op);

  /// The most recent synchronisation op executed anywhere, used by the race
  /// report to point at "the barrier that was not enough".
  mlir::Operation *getLastSyncOp() const { return lastSyncOp; }
  void setLastSyncOp(mlir::Operation *op) { lastSyncOp = op; }

  void trace(CoreState &core, mlir::Operation *op);

  /// Layout-tag check used by the copy family (plan §12 stage 1).
  bool checkLayout(mlir::Operation *op, const MemRefValue &src,
                   const MemRefValue &dst);

private:
  /// Discover the AIC/AIV entry points and build one CoreState per core.
  mlir::LogicalResult setupCores();
  mlir::LogicalResult bindArguments(CoreState &core, mlir::func::FuncOp func);
  /// Kernel arguments shared by every core are materialised once.
  mlir::LogicalResult materializeGlobalArgs();

  /// Execute a single op on `core`.
  ExecResult step(CoreState &core);
  /// Pick the next core to run; returns null when nothing is runnable.
  CoreState *pickCore();
  void commitEffect(CoreState &core, Effect &effect);
  bool detectDeadlock();
  void reportLeaks();

  mlir::ModuleOp module;
  InterpOptions options;
  OpRegistry registry;

  std::vector<std::unique_ptr<Arena>> arenas;
  // (space, owner) -> arena id. Ordered for deterministic reporting.
  std::map<std::pair<unsigned, unsigned>, int> arenaIndex;

  std::vector<CoreState> cores;
  std::map<CrossFlagKey, CrossFlagState> crossFlags;
  std::map<std::string, BarrierState> barriers;
  std::map<std::pair<int, uint64_t>, VectorClock> lockClocks;

  /// Kernel arguments materialised once and shared across cores, keyed by the
  /// argument index of the entry function.
  llvm::MapVector<unsigned, RuntimeValue> globalArgs;
  /// GM argument buffers, in argument order, for `dumpOutputs`.
  llvm::MapVector<unsigned, MemRefValue> gmArgBuffers;
  /// One buffer per (alloc op, core), reused when the site executes again.
  /// A `memref.alloc` inside a loop must not consume fresh on-chip storage
  /// every iteration: PlanMemory assigns one address per site, so bump
  /// allocating per execution would exhaust UB and report a capacity
  /// overflow for a kernel that fits comfortably.
  std::map<std::pair<mlir::Operation *, unsigned>, MemRefValue> allocCache;

  mlir::func::FuncOp aicEntry, aivEntry;

  /// Drop uninteresting races (raw flag-scratchpad traffic) and print the
  /// rest.
  void filterAndReportRace(RaceReport &race, int arena);

  /// Flag an op whose operands overlap effects still queued on another pipe.
  void checkPipeHazards(CoreState &core, mlir::Operation *op, Pipe pipe,
                        llvm::ArrayRef<ByteRange> reads,
                        llvm::ArrayRef<ByteRange> writes, bool isRawPointer);

  /// Shared body of `issueEffect` and `issueResidentAccess`. `completesNow`
  /// says the effect's data movement does not wait in the queue, so it is
  /// committed here and only its ranges stay behind as a resident marker.
  void issueEffectImpl(CoreState &core, Pipe pipe, mlir::Operation *op,
                       llvm::ArrayRef<ByteRange> reads,
                       llvm::ArrayRef<ByteRange> writes,
                       std::function<void()> commit, bool isAtomic,
                       bool isRawPointer, bool completesNow);

  unsigned errorCount = 0;
  unsigned raceCount = 0;
  unsigned missingSyncCount = 0;
  /// Op pairs already reported, so a hazard inside a loop is reported once.
  std::set<std::pair<mlir::Operation *, mlir::Operation *>> reportedHazards;
  /// Resident markers are dropped once a pipe holds this many, which bounds
  /// the memory a scalar loop over scattered addresses can consume. Warned
  /// about once, because past the cap the tool can miss a hazard.
  static constexpr size_t kMaxResidentMarkers = 4096;
  bool residentCapWarned = false;
  bool aborted = false;
  mlir::Operation *lastSyncOp = nullptr;
  uint64_t totalSteps = 0;
  unsigned currentCoreIdx = 0;
  uint64_t progressCounter = 0;
  uint64_t lastRetryProgress = ~0ull;

  std::mt19937_64 rng;
  std::unique_ptr<llvm::raw_ostream> traceStream;
};

//===----------------------------------------------------------------------===//
// Shared helpers for op handlers
//===----------------------------------------------------------------------===//

/// Map a `hivm::PIPE` (as an int) onto the interpreter's dense Pipe enum.
/// Returns false for PIPE_ALL / PIPE_UNASSIGNED / virtual pipes.
bool convertPipe(int32_t hivmPipe, Pipe &out);

/// Address space of a memref type, defaulting to UB when unset (HIVM vector
/// ops operate on UB and post-bufferization IR frequently omits the tag).
AddrSpace getAddrSpaceOf(mlir::MemRefType type);

/// Advance the current instruction pointer past the op just executed.
void advanceIp(CoreState &core);

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_INTERPRETER_H
