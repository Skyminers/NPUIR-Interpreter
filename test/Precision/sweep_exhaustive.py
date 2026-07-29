#!/usr/bin/env python3
"""Deeper precision sweeps: exhaustive f16 unary, sampled f16/f32 binary,
and every vcast rounding mode."""
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from fp import F16, BF16, F32, pack_bits, unpack_bits, read_npy, write_npy

TMP = harness.scratch("exhaustive")
NPY = {'f16': '<f2', 'bf16': '<u2', 'f32': '<f4', 'i32': '<i4',
       'i16': '<i2', 'i8': '|i1'}
FMT = {'f16': F16, 'bf16': BF16, 'f32': F32}
WIDTH = {'i8': 1, 'i16': 2, 'i32': 4}


def run(src, args, outprefix):
    # A 65536-element buffer needs more UB than a real core has; the point
    # here is arithmetic, not capacity.
    return harness.run(src, args, outprefix, extra=["--ub-size=8388608"])


def emit(path, text):
    full = os.path.join(TMP, path)
    with open(full, "w") as f:
        f.write(text)
    return full


# ------------------------------------------------------- exhaustive unary

UNARY_KERNEL = """module {{
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
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%c : memref<{n}x{t}, #hivm.address_space<ub>>)
                   outs(%g1 : memref<{n}x{t}, #hivm.address_space<gm>>)
    return
  }}
}}
"""

UNARY = {
    'vabs': abs,
    'vrelu': lambda a: a if a > 0 else (0.0 if a == a else a),
    'vsqrt': lambda a: math.sqrt(a) if a >= 0 else float('nan'),
    'vrec': lambda a: (float('nan') if a != a else
                       (math.copysign(float('inf'), a) if a == 0
                        else 1.0 / a)),
}


def exhaustive_unary(op, fn, ty='f16'):
    fmt = FMT[ty]
    pats = list(range(1 << (8 * fmt.nbytes)))
    n = len(pats)
    src = emit(f"u_{op}_{ty}.mlir", UNARY_KERNEL.format(n=n, t=ty, op=op))
    a = os.path.join(TMP, f"u_{op}_{ty}.a.npy")
    write_npy(a, NPY[ty], [n], pack_bits(pats, fmt.nbytes))
    out = os.path.join(TMP, f"u_{op}_{ty}.")
    rc, log = run(src, [a, "zeros"], out)
    if rc:
        return [f"{op}/{ty}: interpreter failed\n{log}"]
    got = unpack_bits(read_npy(out + "arg1.npy")[1], fmt.nbytes)

    bad = []
    for i, b in enumerate(pats):
        av = fmt.bits_to_float(b)
        try:
            want = fmt.round(fn(av))
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        gv, wv = fmt.bits_to_float(got[i]), fmt.bits_to_float(want)
        if gv != gv and wv != wv:
            continue
        if got[i] != want:
            bad.append(f"{op}/{ty} 0x{b:04x} ({av!r}) -> got 0x{got[i]:04x}"
                       f" ({gv!r}) want 0x{want:04x} ({wv!r})")
    return bad


# ----------------------------------------------------------------- vcast

CAST_KERNEL = """module {{
  func.func @k(%g0: memref<{n}x{src}, #hivm.address_space<gm>>,
               %g1: memref<{n}x{dst}, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{
    %a = memref.alloc() : memref<{n}x{src}, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<{n}x{dst}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g0 : memref<{n}x{src}, #hivm.address_space<gm>>)
                  outs(%a : memref<{n}x{src}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.vcast {{round_mode = #hivm.round_mode<{mode}>}}
                   ins(%a : memref<{n}x{src}, #hivm.address_space<ub>>)
                   outs(%c : memref<{n}x{dst}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%c : memref<{n}x{dst}, #hivm.address_space<ub>>)
                   outs(%g1 : memref<{n}x{dst}, #hivm.address_space<gm>>)
    return
  }}
}}
"""

RMODES = ['rint', 'round', 'floor', 'ceil', 'trunc']


def round_to_int(x, mode, width):
    """Reference float->int conversion for each HIVM rounding mode."""
    if x != x:
        return None                     # NaN result is unspecified
    if x in (float('inf'), float('-inf')):
        return None
    if mode == 'rint':                  # nearest, ties to even
        f = math.floor(x)
        d = x - f
        n = f if d < 0.5 else (f + 1 if d > 0.5 else (f if int(f) % 2 == 0
                                                      else f + 1))
    elif mode == 'round':               # nearest, ties away from zero
        n = math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
    elif mode == 'floor':
        n = math.floor(x)
    elif mode == 'ceil':
        n = math.ceil(x)
    else:                               # trunc
        n = math.trunc(x)
    n = int(n)
    lo, hi = -(1 << (8 * width - 1)), (1 << (8 * width - 1)) - 1
    if n < lo or n > hi:
        return None                     # out-of-range is unspecified
    return n & ((1 << (8 * width)) - 1)


def cast_f16_to_int(mode, dst='i32'):
    fmt = F16
    pats = list(range(1 << 16))
    n = len(pats)
    src = emit(f"c_f16_{dst}_{mode}.mlir",
               CAST_KERNEL.format(n=n, src='f16', dst=dst, mode=mode))
    a = os.path.join(TMP, f"c_f16_{dst}_{mode}.a.npy")
    write_npy(a, NPY['f16'], [n], pack_bits(pats, 2))
    out = os.path.join(TMP, f"c_f16_{dst}_{mode}.")
    rc, log = run(src, [a, "zeros"], out)
    if rc:
        return [f"vcast f16->{dst} {mode}: failed\n{log}"]
    w = WIDTH[dst]
    got = unpack_bits(read_npy(out + "arg1.npy")[1], w)
    bad = []
    for i, b in enumerate(pats):
        want = round_to_int(fmt.bits_to_float(b), mode, w)
        if want is None:
            continue
        if got[i] != want:
            bad.append(f"vcast f16->{dst} {mode} 0x{b:04x} "
                       f"({fmt.bits_to_float(b)!r}) -> got {got[i]} want {want}")
    return bad


def cast_f32_to_f16(mode):
    """f32 -> f16 narrowing. Only `rint` is round-to-nearest-even; `odd` is
    the round-to-odd mode used for two-step narrowing."""
    random.seed(20260729)
    pats = [random.getrandbits(32) for _ in range(60000)]
    pats += [0, 0x80000000, 1, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000,
             0x3f800000, 0x33000000, 0x33800000, 0x34000000, 0x38800000,
             0x387fe000, 0x387ff000, 0x33000001]
    n = len(pats)
    src = emit(f"n_f32_f16_{mode}.mlir",
               CAST_KERNEL.format(n=n, src='f32', dst='f16', mode=mode))
    a = os.path.join(TMP, f"n_f32_f16_{mode}.a.npy")
    write_npy(a, NPY['f32'], [n], pack_bits(pats, 4))
    out = os.path.join(TMP, f"n_f32_f16_{mode}.")
    rc, log = run(src, [a, "zeros"], out)
    if rc:
        return [f"vcast f32->f16 {mode}: failed\n{log}"]
    got = unpack_bits(read_npy(out + "arg1.npy")[1], 2)
    bad = []
    for i, b in enumerate(pats):
        x = F32.bits_to_float(b)
        want = F16.round(x)
        gv, wv = F16.bits_to_float(got[i]), F16.bits_to_float(want)
        if gv != gv and wv != wv:
            continue
        if got[i] != want:
            bad.append(f"vcast f32->f16 {mode} 0x{b:08x} ({x!r}) -> "
                       f"got 0x{got[i]:04x} ({gv!r}) want 0x{want:04x} ({wv!r})")
    return bad


def cast_f32_to_f16_odd():
    random.seed(20260729)
    pats = [random.getrandbits(32) for _ in range(60000)]
    pats += [0, 0x80000000, 1, 0x80000001, 0x2EDBE6FF, 0xAEDBE6FF,
             0x33000000, 0x33800001, 0x3F800001, 0x7F7FFFFF, 0xFF7FFFFF,
             0x7F800000, 0xFF800000, 0x7FC00000]
    n = len(pats)
    src = emit("o_f32_f16.mlir",
               CAST_KERNEL.format(n=n, src='f32', dst='f16', mode='odd'))
    a = os.path.join(TMP, "o_f32_f16.a.npy")
    write_npy(a, NPY['f32'], [n], pack_bits(pats, 4))
    out = os.path.join(TMP, "o_f32_f16.")
    rc, log = run(src, [a, "zeros"], out)
    if rc:
        return [f"vcast f32->f16 odd: failed\n{log}"]
    got = unpack_bits(read_npy(out + "arg1.npy")[1], 2)
    bad = []
    for i, b in enumerate(pats):
        x = F32.bits_to_float(b)
        want = F16.round(x, 'odd')
        gv, wv = F16.bits_to_float(got[i]), F16.bits_to_float(want)
        if gv != gv and wv != wv:
            continue
        if got[i] != want:
            bad.append(f"vcast f32->f16 odd 0x{b:08x} ({x!r}) -> "
                       f"got 0x{got[i]:04x} ({gv!r}) want 0x{want:04x} ({wv!r})")
    return bad


def main():
    total = 0
    for op, fn in sorted(UNARY.items()):
        bad = exhaustive_unary(op, fn)
        total += len(bad)
        print(f"exhaustive f16 {op:8s} {len(bad):6d} mismatches "
              f"(65536 patterns)")
        for b in bad[:6]:
            print("    " + b)

    for mode in RMODES:
        bad = cast_f16_to_int(mode)
        total += len(bad)
        print(f"vcast f16->i32 {mode:6s} {len(bad):6d} mismatches "
              f"(65536 patterns)")
        for b in bad[:6]:
            print("    " + b)

    bad = cast_f32_to_f16('rint')
    total += len(bad)
    print(f"vcast f32->f16 rint   {len(bad):6d} mismatches (60015 patterns)")
    for b in bad[:6]:
        print("    " + b)

    bad = cast_f32_to_f16_odd()
    total += len(bad)
    print(f"vcast f32->f16 odd    {len(bad):6d} mismatches (60014 patterns)")
    for b in bad[:6]:
        print("    " + b)

    print()
    print("total mismatches:", total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
