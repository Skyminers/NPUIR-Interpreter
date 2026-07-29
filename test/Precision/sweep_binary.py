#!/usr/bin/env python3
"""Bit-exactness sweep of the interpreter's float arithmetic.

Builds a one-op kernel per (op, format), feeds it a table of edge-case bit
patterns, and compares every output bit against the Python oracle in fp.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from fp import F16, BF16, F32, pack_bits, unpack_bits, read_npy, write_npy

TMP = harness.scratch("binary")

MLIR_TYPE = {'f16': 'f16', 'bf16': 'bf16', 'f32': 'f32'}
NPY_DTYPE = {'f16': '<f2', 'bf16': '<u2', 'f32': '<f4'}
FMT = {'f16': F16, 'bf16': BF16, 'f32': F32}

# Most HIVM vector ops reject bf16 in their verifier: bf16 goes through vcast
# first. Only list a type here where the op actually accepts it.
TYPES_FOR = {}


def interesting(fmt):
    """Edge-case bit patterns for `fmt`, plus a spread of ordinary values."""
    mb, eb = fmt.mbits, fmt.ebits
    top = (1 << eb) - 1
    out = [
        0,                                      # +0
        1 << (eb + mb),                         # -0
        1,                                      # smallest subnormal
        (1 << mb) - 1,                          # largest subnormal
        1 << mb,                                # smallest normal
        (top - 1) << mb | ((1 << mb) - 1),      # largest finite
        top << mb,                              # +inf
        (1 << (eb + mb)) | top << mb,           # -inf
        top << mb | (1 << (mb - 1)),            # quiet NaN
        fmt.bias << mb,                         # 1.0
        (fmt.bias << mb) | 1,                   # 1 + 1ulp
        (fmt.bias - 1) << mb,                   # 0.5
        (fmt.bias + 1) << mb,                   # 2.0
        (1 << (eb + mb)) | (fmt.bias << mb),    # -1.0
        (fmt.bias << mb) | ((1 << mb) - 1),     # just under 2
        ((fmt.bias - mb - 1) << mb),            # 2^-(mb+1): forces a tie
    ]
    # A deterministic spread over the whole range.
    step = max(1, (1 << (eb + mb + 1)) // 97)
    out += [(b % (1 << (eb + mb + 1))) for b in range(0, 1 << (eb + mb + 1), step)][:96]
    return out


BINARY = {
    'vadd': lambda a, b: a + b,
    'vsub': lambda a, b: a - b,
    'vmul': lambda a, b: a * b,
    'vdiv': lambda a, b: _div(a, b),
    # LowerToLoops turns a float vmax/vmin into arith.maximumf/minimumf, i.e.
    # IEEE 754-2019 maximum/minimum: NaN propagates and -0 sorts below +0.
    'vmax': lambda a, b: _maximum(a, b),
    'vmin': lambda a, b: _minimum(a, b),
}

UNARY = {
    'vabs': abs,
    'vrelu': lambda a: a if a > 0 else (0.0 if a == a else a),
    'vsqrt': lambda a: math.sqrt(a) if a >= 0 else float('nan'),
    'vrec': lambda a: (float('nan') if a != a else
                       (math.copysign(float('inf'), a) if a == 0 else 1.0 / a)),
}


def _signbit(x):
    import struct
    return struct.pack('<d', x)[7] & 0x80 != 0


def _div(a, b):
    if a != a or b != b:
        return float('nan')
    if b == 0:
        if a == 0 or a == float('inf') or a == float('-inf'):
            return float('nan') if a == 0 else math.copysign(
                float('inf'), a) * (-1.0 if _signbit(b) else 1.0)
        return math.copysign(float('inf'), a) * (-1.0 if _signbit(b) else 1.0)
    return a / b


def _maximum(a, b):
    if a != a:
        return a
    if b != b:
        return b
    if a == 0 and b == 0 and _signbit(a) != _signbit(b):
        return b if _signbit(a) else a
    return b if a < b else a


def _minimum(a, b):
    if a != a:
        return a
    if b != b:
        return b
    if a == 0 and b == 0 and _signbit(a) != _signbit(b):
        return a if _signbit(a) else b
    return b if b < a else a


def kernel(op, ty, n, binary):
    mt = MLIR_TYPE[ty]
    ins = ("%a, %b" if binary else "%a")
    types = ((f"memref<{n}x{mt}, #hivm.address_space<ub>>, "
              f"memref<{n}x{mt}, #hivm.address_space<ub>>")
             if binary else f"memref<{n}x{mt}, #hivm.address_space<ub>>")
    args = (f"%g0: memref<{n}x{mt}, #hivm.address_space<gm>>, "
            f"%g1: memref<{n}x{mt}, #hivm.address_space<gm>>, "
            f"%g2: memref<{n}x{mt}, #hivm.address_space<gm>>")
    loads = f"""
    %a = memref.alloc() : memref<{n}x{mt}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g0 : memref<{n}x{mt}, #hivm.address_space<gm>>)
                  outs(%a : memref<{n}x{mt}, #hivm.address_space<ub>>)
    %b = memref.alloc() : memref<{n}x{mt}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g1 : memref<{n}x{mt}, #hivm.address_space<gm>>)
                  outs(%b : memref<{n}x{mt}, #hivm.address_space<ub>>)
    %c = memref.alloc() : memref<{n}x{mt}, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]"""
    return f"""module {{
  func.func @k({args})
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{{loads}
    hivm.hir.{op} ins({ins} : {types})
                  outs(%c : memref<{n}x{mt}, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%c : memref<{n}x{mt}, #hivm.address_space<ub>>)
                   outs(%g2 : memref<{n}x{mt}, #hivm.address_space<gm>>)
    return
  }}
}}
"""


def run_case(op, ty, fn, binary):
    fmt = FMT[ty]
    pats = interesting(fmt)
    if binary:
        lhs, rhs = [], []
        for a in pats:
            for b in pats:
                lhs.append(a)
                rhs.append(b)
    else:
        lhs = pats
        rhs = pats
    n = len(lhs)

    tag = f"{op}_{ty}"
    src = os.path.join(TMP, tag + ".mlir")
    with open(src, "w") as f:
        f.write(kernel(op, ty, n, binary))
    a_npy = os.path.join(TMP, tag + ".a.npy")
    b_npy = os.path.join(TMP, tag + ".b.npy")
    write_npy(a_npy, NPY_DTYPE[ty], [n], pack_bits(lhs, fmt.nbytes))
    write_npy(b_npy, NPY_DTYPE[ty], [n], pack_bits(rhs, fmt.nbytes))

    out = os.path.join(TMP, tag + ".out.")
    rc, log = harness.run(src, [a_npy, b_npy, "zeros"], out)
    if rc != 0:
        return [f"{tag}: interpreter failed\n{log}"]

    _, payload = read_npy(out + "arg2.npy")
    got = unpack_bits(payload, fmt.nbytes)

    problems = []
    for i in range(n):
        av = fmt.bits_to_float(lhs[i])
        bv = fmt.bits_to_float(rhs[i])
        try:
            want = fmt.round(fn(av, bv) if binary else fn(av))
        except (ValueError, OverflowError, ZeroDivisionError):
            continue                                   # oracle cannot say
        g = got[i]
        # Any NaN encoding is acceptable; only NaN-ness is architectural.
        gv = fmt.bits_to_float(g)
        wv = fmt.bits_to_float(want)
        if gv != gv and wv != wv:
            continue
        if g != want:
            problems.append(
                f"{tag}[{i}]: {av!r} op {bv!r} -> got 0x{g:0{2*fmt.nbytes}x}"
                f" ({gv!r}) want 0x{want:0{2*fmt.nbytes}x} ({wv!r})")
    return problems


def main():
    only = sys.argv[1:] or None
    total = 0
    for op, fn in sorted(BINARY.items()):
        for ty in TYPES_FOR.get(op, ('f16', 'f32')):
            if only and op not in only:
                continue
            problems = run_case(op, ty, fn, True)
            total += len(problems)
            print(f"{op:8s} {ty:5s}  {len(problems):5d} mismatches")
            for p in problems[:6]:
                print("    " + p)
    for op, fn in sorted(UNARY.items()):
        for ty in TYPES_FOR.get(op, ('f16', 'f32')):
            if only and op not in only:
                continue
            problems = run_case(op, ty, fn, False)
            total += len(problems)
            print(f"{op:8s} {ty:5s}  {len(problems):5d} mismatches")
            for p in problems[:6]:
                print("    " + p)
    print()
    print("total mismatches:", total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
