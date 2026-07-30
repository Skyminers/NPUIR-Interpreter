# `npuir-interp` — NPUIR (HIVM) CPU interpreter

[Documentation index](README.md) · [中文](Architecture_zh.md)

`npuir-interp` executes **memref-form HIVM IR** on the host. It exists for
two reasons, in priority order:

1. **Synchronisation checking.** Verify that the flags, barriers and locks the
   compiler inserted are actually sufficient. This is the part no other tool
   can do: lit tests check IR text, not runtime semantics, and a Triton-level
   interpreter cannot see HIVM synchronisation at all.
2. **Numerical reference.** Run a kernel without an Ascend device and compare
   against a numpy/torch golden.

```
npuir-interp kernel.mlir --sched=lazy --args=a.npy,b.npy,zeros --out=out.
```

---

## Why deferred commit is the whole design

On real hardware `hivm.hir.load` returns as soon as it is *issued* to MTE2.
When the data actually lands in UB is decided by the matching
`set_flag(MTE2→V)` / `wait_flag(MTE2→V)` pair.

An interpreter that executes every op synchronously in program order imposes a
much stronger ordering than the machine. Under that model **every kernel
produces the right answer whether or not the synchronisation is correct**, and
the tool is just a slow numpy.

So ops that carry a pipe do not take effect when executed. They push a
deferred `Effect` onto that pipe's queue, and the queue is drained only where
the hardware would drain it. A missing flag then shows up three ways:

- a `MISSING SYNC` diagnostic naming both ops and the pipes involved,
- poison (NaN / `0xCD`) in the result, because the producer never committed,
- a `DATA RACE` report, when the two sides are on different cores.

### Scheduling modes

| `--sched=` | Behaviour | Use |
|---|---|---|
| `inorder` | Every effect commits immediately | Fast numerical golden. Finds no synchronisation bugs *by construction*. |
| `lazy` (default) | Effects commit as late as the flush rules allow | The checking mode. |
| `fuzz --seed=N` | Lazy plus randomised core interleaving | Shaking out fragile synchronisation. |

**Differential testing is the strongest automatic signal:** run the same IR
under `inorder` and `lazy` and compare the outputs. Identical means the
synchronisation is sufficient; different means something is missing. Several
tests in `test/` do exactly this with `cmp`.

### Flush rules

| IR | Effect |
|---|---|
| op with a pipe `P` | Effect queued on `P`; same-pipe effects retire FIFO |
| op with `MacroOpPipeTrait<P1, P2>` (`mmadL1`) | Split into two effects, one per pipe — see below |
| `memref.load`/`store`, `llvm.load`/`store` | PIPE_S: the data moves at once, but a *resident marker* stays queued — see below |
| `set_flag[src, dst, id]` | Token appended to the `src` queue; the flag is raised only when the token is reached |
| `wait_flag[src, dst, id]` | Drain `src` up to that token, then join the published clock; block if there is no such token |
| `pipe_barrier[P]` | Drain `P` (`PIPE_ALL` drains everything) |
| `sync_block_set[core, tpipe, pipe] flag=N` | Drain `tpipe` **only**, bump the cross-core semaphore, publish the clock |
| `sync_block_wait[...] flag=N` | Decrement the semaphore (block if zero), join the published clock |
| `sync_block[<MODE>, id]` | Rendezvous over the mode's participants; every arriving core drains all pipes |
| `sync_block_lock` / `unlock` | Mutual exclusion on `lock_var`; a block enters when `lock_var == blockIdx` |
| kernel `return` | Drain everything — otherwise the last batch of writes disappears |

`sync_block_set` draining *only* `tpipe` is deliberate. A producer that wrote
on a different pipe stays invisible to the consumer, which is exactly the bug
worth finding.

### PIPE_S: the scalar unit is a pipe too

InjectSync emits `set_flag[PIPE_MTE2, PIPE_S]` before a `memref.load` from UB
and `set_flag[PIPE_S, PIPE_MTE3]` before the `hivm.hir.store` that drains it,
so PIPE_S has to be modelled or those pairs cannot be checked.

It cannot be modelled the same way as the others, though: the scalar unit is
the *issuing* unit, and a `memref.load` result is consumed by the very next
`arith` op. Deferring the data would be wrong. So a PIPE_S access **completes
where it stands** — data movement and shadow-memory record both — and what
stays queued is a *resident marker*: an effect with no commit and no second
shadow record, carrying only the byte ranges. Ops on other pipes scan it as
usual, and `wait_flag[S, dst]`, `pipe_barrier`, a barrier or the kernel return
retire it. All three directions are covered: MTE2→S, S→MTE3 and the anti
dependence S→MTE2.

Markers from the same op, same direction, same arena that overlap or abut are
coalesced (looking back at most eight entries and never across a token), so a
scalar loop over a buffer leaves one marker rather than one per element. What
will not coalesce is capped at 4096 per pipe with a one-time warning; widening
the range instead would invent hazards over bytes nobody touched.

### Macro ops occupy two pipes

`mmadL1` carries `MacroOpPipeTrait<PIPE_MTE1, PIPE_M>`: MTE1 stages the L1
tiles into L0A/L0B, then the cube multiplies them into L0C. Folding that onto
PIPE_M alone gets it wrong twice — a legal double-buffered kernel that releases
its A tile with `set_flag[PIPE_MTE1, PIPE_MTE2]` is reported as unsynchronised,
and a kernel that really is missing that pair is told to fix the wrong pipe.

So the staging half is queued on `getInPipe()` with the A/B reads and the
compute half on `getOutPipe()` with the C read/write. The staging commit
snapshots A and B as raw bytes; that is required, not an optimisation, because
once the MTE1 half retires the IR may refill L1 while the cube still owes a
result computed from the old tiles. Staging is idempotent, since a flush rule
may drain the cube's queue without having drained MTE1 first, and hazard
checking skips conflicts between an op and itself.

### Cross-core flags are modelled as counting semaphores

A cross-core flag is keyed by `(scope, tpipe, pipe, flag_id)`, where `scope` is
the block index — or global for `INTER_BLOCK_SYNCHRONIZATION`. The ordered
`(tpipe, pipe)` pair is what tells a V→C flag apart from the C→V flag carrying
the same id. `sync_block_set` adds one credit, `sync_block_wait` consumes one.

This matches the producer/consumer credit pattern real post-`GraphSyncSolver`
IR uses for double buffering, where the same flag id is set and waited
repeatedly around a loop. It is *not* a broadcast latch: with
`--sub-block-num=2`, two sub-vector cores both waiting on a flag that was set
once will leave the second one blocked. If your target's FFTS actually
broadcasts a collected flag back to the whole group, that model needs
revisiting; the default `--sub-block-num=1` avoids the question.

---

## Checks

All are on by default; select a subset with `--check=sync,race,deadlock,oob`.

### `sync` — missing intra-core flag

An op whose operands overlap bytes still queued on *another* pipe of the same
core. Same-pipe overlap is fine (pipes retire in order); cross-pipe overlap
without a flag is not.

```
MISSING SYNC on AIV#0.0: PIPE_V op touches data still in flight on PIPE_MTE2
  in flight  PIPE_MTE2  hivm.hir.load @ add.mlir:4:5
  consumes   PIPE_V  hivm.hir.vadd @ add.mlir:8:5
  the two are unordered without a set_flag[PIPE_MTE2, PIPE_V, <id>] / wait_flag pair between them
```

Two raw-pointer accesses are not reported against each other: that is the
compiler's own flag scratchpad, and ordering it is the flag protocol's job.
`--check-raw-pointer-races` turns the exemption off. The same rule applies to
cross-core races.

### `race` — cross-core data race

Vector clocks, not locksets. Happens-before edges come from program order,
`set_flag`/`wait_flag` pairs, `sync_block_set`/`sync_block_wait` pairs, and
barrier rendezvous. Two accesses to overlapping bytes from different cores
race when their clocks are unordered and at least one is a write.

```
DATA RACE on gm %arg1 +0 [0x40, 0x60)
  W  AIC#0    missing-c2v-flag.mlir:28:5  hivm.hir.store  vc=<4,0>
  R  AIV#0.0  missing-c2v-flag.mlir:40:5  hivm.hir.load   vc=<0,1>
  no happens-before edge between these two accesses
  nearest sync op: hivm.hir.sync_block @ kernel.mlir:180
                   (barrier only - carries no data-dependency flag)
```

UB is checked too: the sub-vector cores of one AIV share it, and in a MIX
kernel so does the AIC.

Two classes of concurrent access are deliberately *not* reported:

- **Raw flag-scratchpad traffic** — both sides being `llvm.load`/`llvm.store`
  through an `inttoptr` address. That is the compiler's own flag area, where
  concurrent access is the mechanism rather than a bug. Pass
  `--check-raw-pointer-races` to see them anyway.
- **Atomic against atomic** — `hivm.hir.store ... atomic = <add>` and friends
  are hardware read-modify-writes and serialise among themselves. An atomic
  racing a *plain* access is still reported.

### `deadlock`

Reported when no core is runnable and at least one is blocked. Two shapes are
distinguished:

- **circular flag wait** — who is blocked on which flag, and which cores
  already finished;
- **barrier arrival mismatch** — the participants a barrier expected versus
  who actually arrived. This is the runtime signature of a barrier cloned into
  a divergent conditional region, which the `AddControlFlowCondition` /
  `CloneOps` pass family can produce.

A barrier site is keyed by `(mode, flag id)` and carries a generation
counter, so one inside a loop rearms correctly: a core leaves only once the
generation has moved past its own count, which stops a fast core from taking
two turns while a slow one has taken none.

A core parked on a flag does not re-test it on its own, so producers wake
their consumers explicitly. As a safety net the scheduler also refuses to
declare a deadlock while any core has made globally visible progress since
the last retry: it wakes everyone once more first. One missed wake-up would
otherwise turn a correct kernel into a reported deadlock, which is the worst
kind of failure for a checker — it trains you to ignore it.

### `oob`

On-chip pools are fixed-capacity arenas, sized from the module's
`dlti.target_system_spec` (override with `--ub-size` and friends). An
allocation that does not fit, or an address that walks off the end of a pool,
is an error rather than silent corruption of a neighbour. Baked-in offsets
from PlanMemory are honoured as written, so a miscomputed offset hits the
arena boundary instead of being quietly "fixed" by a fresh allocation.

---

## Memory model

| Pool | Ownership |
|---|---|
| GM, SSBUF | one, shared by every core |
| UB, L1, L0A, L0B, L0C | one per **block** — in a MIX kernel the AIC's fixpipe writes into the UB its AIVs read |

Freshly allocated buffers are poison-filled (`--poison`, on by default): a
quiet NaN in the buffer's float format, or `0xCD` for integers. Combined with
lazy commit, a missing flag reads out as NaN — an order of magnitude easier to
localise than a 1e-3 discrepancy.

Cores are indexed by the full `(blockIdx, coreKind, subBlockIdx)` triple even
when only one block is simulated.

### Numerics

All arithmetic goes through `APInt` / `APFloat`, including f16 (`IEEEhalf`),
bf16 (`BFloat`) and the f8 formats. Emulating f16 with `float` gets tie
behaviour wrong and cannot express `vcast`'s round-to-odd mode at all, and a
1-ulp interpreter bug is much harder to track down than a missing op. The
transcendentals (`vexp`, `verf`, …) are the exception: they are evaluated in
double and rounded back, since the hardware's own approximations differ
anyway.

`hivm.hir.vcast` implements all six rounding modes — `rint`, `round`, `floor`,
`ceil`, `trunc`, `odd` — plus `truncwithoverflow`.

### Bit-exactness sweeps

A lit test tells you an op is wired up. Only sweeping tens of thousands of bit
patterns against an independent reference tells you the result is right down to
the last bit — and the cases that matter are the ones nobody writes by hand.
`test/Precision/` holds four sweeps (a `lit.local.cfg` keeps lit out of
that directory; they are developer tools, not regression tests):

| Script | Covers |
|---|---|
| `sweep_binary.py` | `+ - * / max min` and `abs relu sqrt rec`, f16 and f32, every edge-case pair |
| `sweep_exhaustive.py` | all 65536 f16 patterns for the unary ops; every `vcast` rounding mode |
| `sweep_integer.py` | wrap-around, `INT_MIN`, shift counts at and past the width, division and remainder by zero |
| `sweep_lowprec.py` | bf16 and both f8 formats: the whole reachable `vcast` table, the bf16 scans, `mmadL1` with bf16 operands, and the poison pattern per format |

The reference in `fp.py` computes in Python doubles and rounds once into the
target format, which is correctly rounded for `+ - * /` and `sqrt` on f16, bf16
and f32: safe double rounding needs `2p+2` bits of intermediate — 24, 18 and 50
respectively — and double carries 53. Where the reference is a *definition*
rather than an arithmetic result (round-to-odd, the integral rounding modes,
IEEE `maximum`/`minimum`), it implements the definition directly.

Three real bugs came out of this, all of which produced entirely plausible
numbers:

| Bug | Symptom | Ground truth |
|---|---|---|
| Signed zeros in `vmax`/`vmin`/`arith.maximumf`/`minimumf`/`vreduce` | `a > b ? a : b` answers wrongly for `max(+0, -0)` in both directions — neither zero compares greater, so it silently returns the second operand | `LowerToLoops` emits `arith.maximumf` for a float `vmax`, i.e. IEEE 754-2019 `maximum`, which orders `-0` below `+0` |
| `vrelu(NaN)` | Implemented as "negative ? 0 : x" it answers `+0` for a NaN whose sign bit happens to be set and NaN for one whose is not — an answer that turns on a bit carrying no meaning | `HIVMToArith` defines `vrelu` as `maximumf(0, x)` |
| Round-to-odd underflow | An input below the smallest f16 subnormal rounded to zero | Round-to-odd exists so that narrowing through it and then rounding to nearest matches one correctly rounded step; answering zero throws away exactly the information it is there to keep, so the result is the smallest subnormal |

`functional/float-edge-cases.mlir` and `functional/round-to-odd.mlir` pin those
so they stay fixed without rerunning the sweeps.
`functional/mmad-accumulator-width.mlir` pins the cube accumulator width: A
rows of `[4096, 1, 1, 1]` against an all-ones B sum to 4099 in f32, while an
f16 accumulator (ulp 4 at 4096) would answer 4096.

### Low precision

bf16 and f8 have almost no arithmetic ops of their own — the verifiers of
`vadd` and friends reject bf16, and there is nothing at all for f8 — so their
correctness lives in `vcast`. Every value of those types in a real kernel
arrives through a conversion, which makes the conversion table the subject
rather than a detail. Widening is swept exhaustively (all 65536 bf16 patterns,
all 256 of each f8) because it must be exact; narrowing is checked against the
format model in `fp.py`; and the sweep also covers the bf16 prefix scans, which
accumulate in bf16 itself, and `mmadL1` with bf16 operands into the f32 L0C.

The two f8 formats are not the same shape, and treating either as "f16 with
fewer bits" gets it wrong. f8e5m2 is IEEE-shaped (1-5-2, infinities, largest
finite 57344). f8e4m3fn has **no infinity**: its only NaN encodings are `0x7f`
and `0xff`, so the all-ones exponent is an ordinary binade except for its last
pattern, its largest finite value is 448 rather than the 240 an IEEE-shaped
reading would give, and overflow produces NaN. `functional/f8-cast.mlir` and
`functional/bf16-cast.mlir` pin both tables, including that poison is a NaN
*in the target format* — otherwise "a missing flag shows up as NaN" quietly
stops holding for these types.

The overflow convention is worth flagging as unverified: the interpreter
follows APFloat, which follows the OCP paper in producing NaN. Hardware that
saturates to the largest finite value instead would disagree, and that has not
been checked against a device.

---

## Command line

```
npuir-interp <input.mlir>
  --entry=<name>              # default: the hacc.entry function(s)
  --args=<spec>,<spec>,...    # per argument: <file>.npy | zeros | poison | arange | <number>
  --out=<prefix>              # write each GM argument to <prefix>arg<N>.npy
  --block-dim=<N>             # default 1
  --sub-block-num=<N>         # sub-vector cores per AIV, default 1
  --sched=inorder|lazy|fuzz   # default lazy
  --seed=<N>                  # fuzz
  --gm-size / --ub-size / --l1-size / --l0a-size / --l0b-size / --l0c-size
  --ssbuf-size / --host-size
  --use-target-sizes          # read pool sizes from dlti.target_system_spec (default on)
  --dyn-gm-elems=<N>          # element count assumed for memref<?x...> arguments
  --poison                    # default on
  --check=sync,race,deadlock,oob
  --check-raw-pointer-races
  --exact-layout              # not implemented yet
  --trace=<file>              # per-op execution trace
  --max-steps=<N>
```

Exit code is non-zero when a check fires, when the run deadlocks, or when an
op is unsupported.

`.npy` support covers C-order little-endian v1.0 arrays. bf16 and the f8
formats have no NumPy scalar type, so they are read and written as raw
unsigned integers of the same width (`<u2` / `|u1`); reinterpret with
`ml_dtypes` on the Python side.

---

## Coverage and limits

**Input must be memref form.** Tensor-typed HIVM operands get an explicit
error pointing at bufferization rather than a confusing "unbound operand".

Supported: `func`, `scf` (for / if / while / execute_region / index_switch),
`cf`, `arith`, `math`, `index`, `memref`, `vector` (transfer_read/write,
broadcast, splat, shape_cast, multi_reduction, extract), the `llvm` subset
that survives lowering (inttoptr / load / store / getelementptr and integer
arithmetic), `annotation`, `scope`, and from HIVM: the elementwise family
(~40 ops), `vcast` / `vcmp` / `vsel`, `vreduce`, `vbrc`, `vtranspose`, the
reordering family (`vflip`, `vconcat`, `vpad`, `vgather`, `vinterleave`,
`vdeinterleave`, `vsort`), the prefix scans (`vcumsum`, `vcumprod`, `vcummax`,
`vcummin`, forward and reversed), the copy family (`load` / `store` / `copy` /
`fixpipe` / `nd2nz` / `nz2nd` / `l12ub`), the sparse and strided family
(`indirect_load` / `indirect_store` with mask and `other`, `stride_load` /
`stride_store`), the atomics (`atomic_cas`, `atomic_xchg`), `load_scalar`,
`mmadL1` / `batchMmadL1`, the view family, the query family and the full sync
family.

An access whose addresses are data (`vgather`, the indirect family) cannot know
at issue time which bytes it will touch, so it declares the **whole** source or
destination buffer to the race detector — conservative in the direction that
over-reports sharing rather than missing it; the alternative, committing the
effect early just to read the indices, would defeat the deferred model. An
index outside the source extent is always an error rather than a fold onto some
neighbouring element: reading the wrong element quietly is exactly what this
tool exists to catch.

Deliberately not covered:

- **Performance modelling.** No cycle counts, no bank conflicts, no bandwidth.
- **Byte-exact fractal layouts.** Data is kept in logical ND order and the
  `zN` / `nZ` / `DOTx_ND` tags are tracked and checked for producer/consumer
  agreement, which catches most layout bugs at negligible cost.
  `--exact-layout` is reserved for real fractal addressing and is not
  implemented.
- **Whole-problem macro ops** (`hivm.hir.matmul`, `mix_matmul`,
  `mix_group_matmul`). These carry tiling and epilogue parameters rather than
  describing a single MAC over L1 tiles; they are left unregistered so the
  driver says so instead of computing something plausible but wrong.
- **The SIMT path.** SIMD only.

An op with no handler produces `unsupported op: <name>` and a non-zero exit.
That is intentional: a clear gap is more useful than a silent wrong answer.

### Allocation model

Each `memref.alloc` *site* owns one buffer per core, reused (and re-poisoned)
whenever the site executes again. A loop body must not consume fresh on-chip
storage every iteration: PlanMemory assigns one address per site, so bump
allocating per execution would exhaust UB and report a capacity overflow for
a kernel that fits. The consequence is that two buffers from the same alloc
op cannot be live at once — which SSA in a single core prevents anyway.

---

## Layout

```
include/bishengir/Tools/Interp/    Value.h  Memory.h  PipeEngine.h  Interpreter.h
lib/                               the implementation, one file per op family
                                   (OpsShape.cpp: reordering and scans;
                                    OpsIndirect.cpp: sparse, strided, atomic)
tools/                             driver and command line
test/functional/                   numerics; most diff inorder against lazy
test/sync/                         expect MISSING SYNC
test/race/                         expect DATA RACE
test/deadlock/                     expect DEADLOCK
test/oob/                          expect an out-of-bounds / capacity error
test/layout/                       expect a layout-tag mismatch
test/errors/                       expect a clear refusal, not a wrong answer
```

### What the functional tests cover

| Test | Exercises |
|---|---|
| `vecadd-inorder` / `vecadd-synced` | the M0 baseline, and inorder-vs-lazy agreement |
| `loop-reduce` | `scf.for` with loop-carried state, `vreduce` accumulation |
| `elementwise` | sub / mul / div / max / abs / sqrt / vcmp+vsel with an i8 mask |
| `vcast-rounding` | all five integral rounding modes, bit-exact |
| `narrow-float` | f16 arithmetic and a bf16 round trip, checked as bit patterns |
| `views` | chained `reinterpret_cast`, nested and rank-reducing subviews, strides |
| `reshape-transpose` | `expand_shape` / `collapse_shape`, `vtranspose`, `scf.if` with results |
| `vf-call` | an outlined vector function: call frames plus the `vector` dialect |
| `cube-matmul` | `nd2nz` -> `mmadL1` -> `fixpipe` across MTE2 / M / FIX |
| `mmad-macro-pipes` | the two-pipe split: L1 refilled after the MTE1 half retires, cube still sees the snapshot |
| `mmad-accumulator-width` | L0C is f32: an f16 accumulator would turn 4099 into 4096 |
| `float-edge-cases` | signed-zero ordering, NaN propagation, `relu(NaN)` |
| `round-to-odd` | round-to-odd on underflow, overflow and an inexact normal |
| `vector-reorder` | `vflip` / `vconcat` / `vpad` / `vgather` / `vinterleave` / `vdeinterleave` / `vsort` |
| `vector-scan` | the four prefix scans, forward and reversed, including rank-2 |
| `sort-nan-order` | `vsort` with NaN operands: a total order, not an undefined comparator |
| `bf16-cast` | f32 -> bf16 ties and overflow, exact widening back, bf16 -> i32 |
| `f8-cast` | both f8 formats at their boundaries, including e4m3fn's missing infinity |
| `indirect-access` | `indirect_load` / `indirect_store` with mask and `other`, `stride_load` / `stride_store` with a truncating `numel` |
| `atomic-rmw` | two blocks applying the same idempotent atomic update, without reporting each other |
| `scalar-pipe` | both PIPE_S directions flagged; markers coalesce, so no cap warning |
| `scalar-pipe-barrier` | `pipe_barrier[<PIPE_S>]` and `[<PIPE_ALL>]` retire scalar markers |
| `scalar-war-released` | `set_flag[PIPE_S, PIPE_MTE2]`: DMA may only refill after the scalar read |
| `scalar-cross-core` | `sync_block_set[<CUBE>, <PIPE_S>, ...]`: the scalar unit as cross-core producer |
| `mix-cross-core` | AIC produces, AIV consumes, ordered by a cross-core flag |
| `barrier-loop` | `sync_block` rearming over three iterations |
| `block-lock` | `sync_block_lock` serialising three blocks |
| `blockidx-tiling` | four blocks on disjoint tiles - and no false-positive race |
| `atomic-and-controlflow` | atomic `add` store from four blocks, `scf.while`, `cf.br` |

Op handlers live in a registry keyed by op name (`OpRegistry`). The HIVM
dialect is not modified — no `.td` file is touched, and nothing in
`lib/Dialect` knows the interpreter exists.

**Every checker has a negative test.** A checker with only positive tests is
indistinguishable from a checker that never runs. Each file under `sync/`,
`race/`, `deadlock/` and `oob/` is a correct kernel with one specific thing
deleted, and names what was removed.

## Testing

Beyond the lit suite there are gtest unit tests for the pieces lit cannot
reach well — `ShadowMemory`'s interval splitting, `VectorClock` ordering,
`MemRefValue` addressing, and the resident-marker coalescing rules:

```
cmake --build build --target check-npuir-interpreter-unit
```

Two sweeps are worth re-running after any change to the scheduler or the
checks:

- **Differential.** Every functional kernel under `inorder` and `lazy` must
  produce identical output files. The one deliberate exception is
  `vecadd-inorder`, which is the unsynchronised kernel.
- **Fuzz.** Every functional kernel must stay clean, and every negative test
  must still fire, across a spread of `--seed` values. A checker that only
  works on the default schedule is not a checker.

- **Precision.** After any change to an arithmetic path, run the four sweeps
  under `test/Precision/`; each must report zero mismatches.

The checked-in lit and gtest suites are the stable regression baseline. The
precision sweeps are intentionally separate because they are slower developer
checks rather than ordinary regressions.

## Determinism

Output, including diagnostics, is reproducible: `DenseMap` iteration order
never feeds a decision that affects output, cores are scheduled in `CoreId`
order, and fuzz mode draws from a `std::mt19937_64` seeded by `--seed`.
