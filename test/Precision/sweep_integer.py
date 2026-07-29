#!/usr/bin/env python3
"""Integer edge-case sweep: wrap-around, INT_MIN, shift counts at and past the
width, and division/remainder by zero."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from fp import pack_bits, unpack_bits, read_npy, write_npy

TMP = harness.scratch("integer")
NPY = {'i8': '|i1', 'i16': '<i2', 'i32': '<i4'}
BYTES = {'i8': 1, 'i16': 2, 'i32': 4}

KERNEL = """module {{
  func.func @k(%g0: memref<{n}x{t}, #hivm.address_space<gm>>,
               %g1: memref<{n}x{t}, #hivm.address_space<gm>>,
               %g2: memref<{n}x{t}, #hivm.address_space<gm>>)
      attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{
    %a = memref.alloc() : memref<{n}x{t}, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<{n}x{t}, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<{n}x{t}, #hivm.address_space<ub>>
    hivm.hir.load ins(%g0 : memref<{n}x{t}, #hivm.address_space<gm>>)
                  outs(%a : memref<{n}x{t}, #hivm.address_space<ub>>)
    hivm.hir.load ins(%g1 : memref<{n}x{t}, #hivm.address_space<gm>>)
                  outs(%b : memref<{n}x{t}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.{op} ins(%a, %b : memref<{n}x{t}, #hivm.address_space<ub>>,
                               memref<{n}x{t}, #hivm.address_space<ub>>)
                  outs(%c : memref<{n}x{t}, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store ins(%c : memref<{n}x{t}, #hivm.address_space<ub>>)
                   outs(%g2 : memref<{n}x{t}, #hivm.address_space<gm>>)
    return
  }}
}}
"""


def sgn(v, w):
    """Interpret a `w`-byte two's-complement pattern as a signed integer."""
    bits = 8 * w
    return v - (1 << bits) if v >> (bits - 1) else v


def wrap(v, w):
    return v & ((1 << (8 * w)) - 1)


def patterns(w):
    bits = 8 * w
    hi = (1 << (bits - 1)) - 1
    out = [0, 1, 2, 3, wrap(-1, w), wrap(-2, w), hi, wrap(-hi - 1, w),
           wrap(hi // 2, w), wrap(-(hi // 2), w),
           bits - 1, bits, bits + 1, 2 * bits, wrap(-bits, w), 7, 8, 16, 31,
           32, 63, 64]
    return sorted(set(wrap(v, w) for v in out))


# Reference semantics. Division and remainder by zero answer 0, matching what
# the interpreter documents; the hardware has no defined behaviour there and
# trapping would make an otherwise runnable kernel unusable.
def ref(op, a, b, w):
    bits = 8 * w
    if op == 'vadd':
        return wrap(a + b, w)
    if op == 'vsub':
        return wrap(a - b, w)
    if op == 'vmul':
        return wrap(a * b, w)
    if op == 'vmax':
        return wrap(max(a, b), w)
    if op == 'vmin':
        return wrap(min(a, b), w)
    if op == 'vand':
        return wrap(a & b, w)
    if op == 'vor':
        return wrap(a | b, w)
    if op == 'vxor':
        return wrap(a ^ b, w)
    if op == 'vshl':
        # APInt::shl saturates to zero once the count reaches the width.
        return 0 if b < 0 or b >= bits else wrap(a << b, w)
    if op == 'vshr':
        # Arithmetic shift right; past the width every bit is the sign bit.
        if b < 0 or b >= bits:
            return wrap(-1 if a < 0 else 0, w)
        return wrap(a >> b, w)
    if op == 'vmod':
        if b == 0:
            return 0
        # C-style remainder: sign follows the dividend, matching APInt::srem.
        r = abs(a) % abs(b)
        return wrap(-r if a < 0 else r, w)
    raise KeyError(op)


OPS = ['vadd', 'vsub', 'vmul', 'vmax', 'vmin', 'vand', 'vor', 'vxor',
       'vshl', 'vshr', 'vmod']


def run_case(op, ty):
    w = BYTES[ty]
    pats = patterns(w)
    lhs, rhs = [], []
    for a in pats:
        for b in pats:
            lhs.append(a)
            rhs.append(b)
    n = len(lhs)

    tag = f"{op}_{ty}"
    src = os.path.join(TMP, tag + ".mlir")
    with open(src, "w") as f:
        f.write(KERNEL.format(n=n, t=ty, op=op))
    a_npy = os.path.join(TMP, tag + ".a.npy")
    b_npy = os.path.join(TMP, tag + ".b.npy")
    write_npy(a_npy, NPY[ty], [n], pack_bits(lhs, w))
    write_npy(b_npy, NPY[ty], [n], pack_bits(rhs, w))
    out = os.path.join(TMP, tag + ".out.")
    rc, log = harness.run(src, [a_npy, b_npy, "zeros"], out)
    if rc != 0:
        return [f"{tag}: rejected\n{log.strip()[:400]}"]

    got = unpack_bits(read_npy(out + "arg2.npy")[1], w)
    bad = []
    for i in range(n):
        want = ref(op, sgn(lhs[i], w), sgn(rhs[i], w), w)
        if got[i] != want:
            bad.append(f"{tag}[{i}]: {sgn(lhs[i], w)} op {sgn(rhs[i], w)}"
                       f" -> got {sgn(got[i], w)} want {sgn(want, w)}")
    return bad


def main():
    total = 0
    for op in OPS:
        for ty in ('i8', 'i16', 'i32'):
            bad = run_case(op, ty)
            if bad and bad[0].endswith(("rejected", "")) and "rejected" in bad[0]:
                print(f"{op:6s} {ty:4s}  (not accepted for this type)")
                continue
            total += len(bad)
            print(f"{op:6s} {ty:4s}  {len(bad):5d} mismatches")
            for b in bad[:5]:
                print("    " + b)
    print()
    print("total mismatches:", total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
