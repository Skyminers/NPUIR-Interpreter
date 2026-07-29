#!/usr/bin/env python3
"""Low-precision sweep: bf16, f8e4m3fn and f8e5m2.

These formats have no arithmetic ops of their own in HIVM - the verifiers of
`vadd` and friends reject bf16, and there is nothing at all for f8 - so their
correctness lives almost entirely in `vcast` and in the handful of ops that do
accept them. That makes the cast table the thing to nail down: every bf16 and
f8 value in a real kernel arrives through one of these conversions.

Reachable conversions, established by probing the verifier rather than reading
the (incomplete) ODS table:

    f32  -> bf16       rint round floor ceil trunc
    bf16 -> f32        rint round
    f32  -> f8e4m3fn   rint
    f32  -> f8e5m2     rint
    f8e4m3fn -> f32    rint
    f8e5m2   -> f32    rint
    bf16 -> i32        rint round floor ceil trunc
    i8   -> bf16       rint

Widening is checked exhaustively - all 65536 bf16 patterns and all 256 of each
f8 - because it must be exact, so any mismatch is unambiguous. Narrowing is
checked against the format model in fp.py.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from fp import BF16, F16, F32, F8E4M3, F8E5M2, pack_bits, unpack_bits, \
    read_npy, write_npy

TMP = harness.scratch("lowprec")

MLIR = {'f32': 'f32', 'bf16': 'bf16', 'f16': 'f16',
        'f8e4m3fn': 'f8E4M3FN', 'f8e5m2': 'f8E5M2',
        'i32': 'i32', 'i8': 'i8'}
NPY = {'f32': '<f4', 'bf16': '<u2', 'f16': '<f2',
       'f8e4m3fn': '|u1', 'f8e5m2': '|u1', 'i32': '<i4', 'i8': '|i1'}
WIDTH = {'f32': 4, 'bf16': 2, 'f16': 2, 'f8e4m3fn': 1, 'f8e5m2': 1,
         'i32': 4, 'i8': 1}
FMT = {'f32': F32, 'bf16': BF16, 'f16': F16,
       'f8e4m3fn': F8E4M3, 'f8e5m2': F8E5M2}

CAST = """module {{
  func.func @k(%g0: memref<{n}x{s}, #hivm.address_space<gm>>,
               %g1: memref<{n}x{d}, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{
    %a = memref.alloc() : memref<{n}x{s}, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<{n}x{d}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g0 : memref<{n}x{s}, #hivm.address_space<gm>>)
                  outs(%a : memref<{n}x{s}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.vcast {{round_mode = #hivm.round_mode<{m}>}}
                   ins(%a : memref<{n}x{s}, #hivm.address_space<ub>>)
                   outs(%c : memref<{n}x{d}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%c : memref<{n}x{d}, #hivm.address_space<ub>>)
                   outs(%g1 : memref<{n}x{d}, #hivm.address_space<gm>>)
    return
  }}
}}
"""

SCAN = """module {{
  func.func @k(%g0: memref<{n}x{t}, #hivm.address_space<gm>>,
               %g1: memref<{n}x{t}, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{
    %a = memref.alloc() : memref<{n}x{t}, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<{n}x{t}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g0 : memref<{n}x{t}, #hivm.address_space<gm>>)
                  outs(%a : memref<{n}x{t}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.{op} ins(%a : memref<{n}x{t}, #hivm.address_space<ub>>)
                  outs(%c : memref<{n}x{t}, #hivm.address_space<ub>>)
                  cum_dims = [0] reverse = false
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%c : memref<{n}x{t}, #hivm.address_space<ub>>)
                   outs(%g1 : memref<{n}x{t}, #hivm.address_space<gm>>)
    return
  }}
}}
"""


def run_kernel(tag, text, inputs, out_index, out_width):
    """Write, run and read back one generated kernel."""
    src = os.path.join(TMP, tag + ".mlir")
    with open(src, "w") as f:
        f.write(text)
    out = os.path.join(TMP, tag + ".")
    rc, log = harness.run(src, inputs, out, extra=["--ub-size=8388608"])
    if rc != 0:
        return None, log
    _, payload = read_npy(out + f"arg{out_index}.npy")
    return unpack_bits(payload, out_width), None


def cast_case(src_ty, dst_ty, mode, pats):
    """Run one vcast over `pats` and diff against the format model."""
    tag = f"{src_ty}_to_{dst_ty}_{mode}"
    n = len(pats)
    a = os.path.join(TMP, tag + ".a.npy")
    write_npy(a, NPY[src_ty], [n], pack_bits(pats, WIDTH[src_ty]))
    got, err = run_kernel(tag, CAST.format(n=n, s=MLIR[src_ty], d=MLIR[dst_ty],
                                           m=mode),
                          [a, "zeros"], 1, WIDTH[dst_ty])
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]

    sfmt, dfmt = FMT[src_ty], FMT[dst_ty]
    bad = []
    for i, b in enumerate(pats):
        x = sfmt.bits_to_float(b)
        # floor/ceil/trunc/round are integral rounding modes: HIVM uses them
        # to mean "round to a whole number" (the ODS table lists them for
        # same-type and float-to-int conversions). The narrowing that follows
        # is then done in the *same* direction, so `floor` on a value above
        # the destination's range lands on its largest finite value rather
        # than overflowing to infinity. `rint` is the plain format change.
        if mode in ("floor", "ceil", "trunc", "round"):
            x = integral(x, mode)
        want = dfmt.round(x, mode)
        if dfmt.is_nan(got[i]) and dfmt.is_nan(want):
            continue
        if got[i] != want:
            bad.append(
                f"{tag} 0x{b:0{2 * WIDTH[src_ty]}x} ({x!r}) -> "
                f"got 0x{got[i]:0{2 * WIDTH[dst_ty]}x}"
                f" ({dfmt.bits_to_float(got[i])!r}) want 0x{want:0{2 * WIDTH[dst_ty]}x}"
                f" ({dfmt.bits_to_float(want)!r})")
    return bad


def integral(x, mode):
    """Round to a whole number, as a float.

    `math.floor` and friends return ints, which drops the sign of zero -
    `trunc(-0.3)` has to stay `-0.0`, and the interpreter's `roundToIntegral`
    keeps it.
    """
    import math
    if x != x or abs(x) == float("inf") or x == 0.0:
        return x
    if mode == "floor":
        r = float(math.floor(x))
    elif mode == "ceil":
        r = float(math.ceil(x))
    elif mode == "trunc":
        r = float(math.trunc(x))
    else:                                             # nearest, ties away
        r = float(math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5))
    return -0.0 if r == 0.0 and x < 0 else r


def cast_to_int(src_ty, dst_ty, mode, pats):
    tag = f"{src_ty}_to_{dst_ty}_{mode}"
    n = len(pats)
    a = os.path.join(TMP, tag + ".a.npy")
    write_npy(a, NPY[src_ty], [n], pack_bits(pats, WIDTH[src_ty]))
    got, err = run_kernel(tag, CAST.format(n=n, s=MLIR[src_ty], d=MLIR[dst_ty],
                                           m=mode),
                          [a, "zeros"], 1, WIDTH[dst_ty])
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]
    w = WIDTH[dst_ty]
    lo, hi = -(1 << (8 * w - 1)), (1 << (8 * w - 1)) - 1
    bad = []
    for i, b in enumerate(pats):
        x = FMT[src_ty].bits_to_float(b)
        if x != x or abs(x) == float("inf"):
            continue                                  # unspecified
        want = int(integral(x, "round" if mode == "round" else mode)
                   if mode != "rint" else _rint(x))
        if want < lo or want > hi:
            continue                                  # unspecified
        if got[i] != (want & ((1 << (8 * w)) - 1)):
            bad.append(f"{tag} 0x{b:04x} ({x!r}) -> got {got[i]} want {want}")
    return bad


def _rint(x):
    import math
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if int(f) % 2 == 0 else f + 1


def int_to_float(src_ty, dst_ty, mode, values):
    tag = f"{src_ty}_to_{dst_ty}_{mode}"
    n = len(values)
    w = WIDTH[src_ty]
    a = os.path.join(TMP, tag + ".a.npy")
    write_npy(a, NPY[src_ty], [n],
              pack_bits([v & ((1 << (8 * w)) - 1) for v in values], w))
    got, err = run_kernel(tag, CAST.format(n=n, s=MLIR[src_ty], d=MLIR[dst_ty],
                                           m=mode),
                          [a, "zeros"], 1, WIDTH[dst_ty])
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]
    dfmt = FMT[dst_ty]
    bad = []
    for i, v in enumerate(values):
        want = dfmt.round(float(v), "rint")
        if got[i] != want:
            bad.append(f"{tag} {v} -> got 0x{got[i]:04x} want 0x{want:04x}")
    return bad


def scan_case(op, ty, pats, combine):
    """A prefix scan whose accumulator is the element type itself."""
    tag = f"{op}_{ty}"
    n = len(pats)
    a = os.path.join(TMP, tag + ".a.npy")
    write_npy(a, NPY[ty], [n], pack_bits(pats, WIDTH[ty]))
    got, err = run_kernel(tag, SCAN.format(n=n, t=MLIR[ty], op=op),
                          [a, "zeros"], 1, WIDTH[ty])
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]
    fmt = FMT[ty]
    bad = []
    acc = None
    for i, b in enumerate(pats):
        v = fmt.bits_to_float(b)
        acc = v if i == 0 else fmt.bits_to_float(fmt.round(combine(acc, v)))
        want = fmt.round(acc)
        if fmt.is_nan(got[i]) and fmt.is_nan(want):
            acc = float("nan")
            continue
        if got[i] != want:
            bad.append(f"{tag}[{i}] -> got 0x{got[i]:04x}"
                       f" ({fmt.bits_to_float(got[i])!r})"
                       f" want 0x{want:04x} ({fmt.bits_to_float(want)!r})")
    return bad


def sample(fmt, count, seed):
    """Edge cases plus a deterministic random spread of `fmt` patterns."""
    bits = 8 * fmt.nbytes
    out = [0, fmt.sign_bit, 1, fmt.sign_bit | 1,
           fmt.max_man, 1 << fmt.mbits, fmt.max_finite_bits,
           fmt.sign_bit | fmt.max_finite_bits, fmt.nan_bits,
           fmt.bias << fmt.mbits, (fmt.bias << fmt.mbits) | 1,
           (fmt.bias - 1) << fmt.mbits, (fmt.bias + 1) << fmt.mbits]
    if fmt.inf_bits is not None:
        out += [fmt.inf_bits, fmt.sign_bit | fmt.inf_bits]
    rng = random.Random(seed)
    out += [rng.getrandbits(bits) for _ in range(count)]
    return out


MMAD = """module {{
  func.func @mm(%a: memref<{n}x{n}x{t}, #hivm.address_space<gm>>,
                %b: memref<{n}x{n}x{t}, #hivm.address_space<gm>>,
                %c: memref<{n}x{n}xf32, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>}} {{
    %true = arith.constant true
    %n = arith.constant {n} : index
    %a1 = memref.alloc() : memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>
    %b1 = memref.alloc() : memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%a : memref<{n}x{n}x{t}, #hivm.address_space<gm>>)
                   outs(%a1 : memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz ins(%b : memref<{n}x{n}x{t}, #hivm.address_space<gm>>)
                   outs(%b1 : memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %acc = memref.alloc() : memref<{n}x{n}xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a1, %b1, %true, %n, %n, %n
                        : memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>,
                          memref<{n}x{n}x{t}, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<{n}x{n}xf32, #hivm.address_space<cc>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.fixpipe {{dma_mode = #hivm.dma_mode<nz2nd>}}
        ins(%acc : memref<{n}x{n}xf32, #hivm.address_space<cc>>)
        outs(%c : memref<{n}x{n}xf32, #hivm.address_space<gm>>)
    return
  }}
}}
"""


def mmad_case(ty, n, seed):
    """A x B in `ty` accumulating into the f32 L0C.

    A product of two bf16 values needs 16 significand bits and f32 carries 24,
    so every product is exact and only the additions round - which is what
    makes an f32 oracle bit-exact rather than merely close.
    """
    fmt = FMT[ty]
    rng = random.Random(seed)
    binades = [(fmt.bias + k) << fmt.mbits for k in (-2, -1, 0, 1, 2, 4)]
    def mk():
        return [rng.choice(binades) | rng.getrandbits(fmt.mbits)
                for _ in range(n * n)]
    A, B = mk(), mk()
    tag = f"mmad_{ty}"
    a = os.path.join(TMP, tag + ".a.npy")
    b = os.path.join(TMP, tag + ".b.npy")
    write_npy(a, NPY[ty], [n, n], pack_bits(A, WIDTH[ty]))
    write_npy(b, NPY[ty], [n, n], pack_bits(B, WIDTH[ty]))
    got, err = run_kernel(tag, MMAD.format(n=n, t=MLIR[ty]),
                          [a, b, "zeros"], 2, 4)
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]

    bad = []
    for i in range(n):
        for j in range(n):
            acc = F32.round(0.0)
            for k in range(n):
                av = fmt.bits_to_float(A[i * n + k])
                bv = fmt.bits_to_float(B[k * n + j])
                prod = F32.round(av * bv)
                acc = F32.round(F32.bits_to_float(acc) +
                                F32.bits_to_float(prod))
            if got[i * n + j] != acc:
                bad.append(f"{tag}[{i}][{j}] -> got 0x{got[i * n + j]:08x}"
                           f" want 0x{acc:08x}")
    return bad


POISON = """module {{
  func.func @k(%g: memref<8x{t}, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{
    // Never written, so what lands in GM is whatever the allocator poisoned.
    %u = memref.alloc() : memref<8x{t}, #hivm.address_space<ub>>
    hivm.hir.store ins(%u : memref<8x{t}, #hivm.address_space<ub>>)
                   outs(%g : memref<8x{t}, #hivm.address_space<gm>>)
    return
  }}
}}
"""


def poison_case(ty):
    """Poison has to be a NaN *in the target format*, or the whole "a missing
    flag shows up as NaN" property quietly stops holding for that format."""
    fmt = FMT[ty]
    tag = f"poison_{ty}"
    got, err = run_kernel(tag, POISON.format(t=MLIR[ty]), ["zeros"], 0,
                          WIDTH[ty])
    if got is None:
        return [f"{tag}: {err.strip()[:300]}"]
    return [f"{tag}: lane {i} is 0x{v:x}, not a NaN"
            for i, v in enumerate(got) if not fmt.is_nan(v)]


def report(label, bad):
    print(f"{label:44s} {len(bad):5d} mismatches")
    for b in bad[:5]:
        print("    " + b)
    return len(bad)


def main():
    total = 0
    all16 = list(range(1 << 16))
    all8 = list(range(1 << 8))

    # --- widening: must be exact, so sweep it exhaustively ---------------
    total += report("bf16 -> f32          rint  (all 65536)",
                    cast_case('bf16', 'f32', 'rint', all16))
    total += report("f8e4m3fn -> f32      rint  (all 256)",
                    cast_case('f8e4m3fn', 'f32', 'rint', all8))
    total += report("f8e5m2 -> f32        rint  (all 256)",
                    cast_case('f8e5m2', 'f32', 'rint', all8))
    total += report("i8 -> bf16           rint  (all 256)",
                    int_to_float('i8', 'bf16', 'rint',
                                 [v - 128 for v in range(256)]))

    # --- narrowing -------------------------------------------------------
    f32pats = sample(F32, 60000, 20260729)
    for mode in ('rint', 'round', 'floor', 'ceil', 'trunc'):
        total += report(f"f32 -> bf16          {mode:6s}({len(f32pats)})",
                        cast_case('f32', 'bf16', mode, f32pats))
    total += report("f32 -> f8e4m3fn      rint  (%d)" % len(f32pats),
                    cast_case('f32', 'f8e4m3fn', 'rint', f32pats))
    total += report("f32 -> f8e5m2        rint  (%d)" % len(f32pats),
                    cast_case('f32', 'f8e5m2', 'rint', f32pats))

    # --- bf16 -> integer, every accepted mode ----------------------------
    for mode in ('rint', 'round', 'floor', 'ceil', 'trunc'):
        total += report(f"bf16 -> i32          {mode:6s}(all 65536)",
                        cast_to_int('bf16', 'i32', mode, all16))

    # --- bf16 arithmetic: the scans accumulate in the element type -------
    bf = [BF16.round(v) for v in
          (1.0, 0.5, 0.25, 3.0, -2.0, 1e-4, 7.5, 0.125,
           1e5, 1e-5, 2.0, -0.75, 1024.0, 1.0 / 3.0, 6.0, -1.0)]
    total += report("bf16 vcumsum         (16 values)",
                    scan_case('vcumsum', 'bf16', bf, lambda a, b: a + b))
    total += report("bf16 vcumprod        (16 values)",
                    scan_case('vcumprod', 'bf16', bf, lambda a, b: a * b))
    total += report("bf16 vcummax         (16 values)",
                    scan_case('vcummax', 'bf16', bf,
                              lambda a, b: b if a < b else a))
    total += report("bf16 vcummin         (16 values)",
                    scan_case('vcummin', 'bf16', bf,
                              lambda a, b: b if b < a else a))

    # --- bf16 through the cube, accumulating in f32 ----------------------
    total += report("bf16 mmadL1 -> f32   (16x16)", mmad_case('bf16', 16, 11))

    # --- poison must be a NaN in each format -----------------------------
    for ty in ('bf16', 'f8e4m3fn', 'f8e5m2'):
        total += report(f"{ty} poison is NaN", poison_case(ty))

    print()
    print("total mismatches:", total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
