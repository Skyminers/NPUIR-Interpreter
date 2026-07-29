// Cases where the obvious implementation is subtly wrong. Each of the three
// blocks below pins a bug that a bit-exactness sweep against an IEEE oracle
// turned up; all three produce plausible-looking numbers when wrong.
//
// The inputs are hand-built f16 bit patterns so the test needs no .npy files:
//   0x0000 +0    0x8000 -0    0x7e00 qNaN   0x3c00 1.0    0xbc00 -1.0
//
// 1. maximum(+0, -0) is +0 and minimum(+0, -0) is -0. `a > b ? a : b` gets
//    both wrong, because neither zero compares greater than the other, so it
//    silently answers with the *second* operand either way. HIVM's float
//    vmax/vmin are IEEE 754-2019 maximum/minimum (LowerToLoops emits
//    arith.maximumf), which orders the zeros.
// 2. relu(NaN) is NaN. Implementing relu as "negative ? 0 : x" answers +0 for
//    a NaN whose sign bit happens to be set and NaN for one whose is not -
//    an answer that depends on a bit that carries no meaning. HIVMToArith
//    defines vrelu as maximumf(0, x), which propagates.
// 3. Both min/max propagate NaN rather than returning the other operand.

// RUN: npuir-interp %s --sched=lazy --args=zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -tx2 -j128 -N24 %t.arg0.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=zeros --out=%t.inorder.
// RUN: cmp %t.arg0.npy %t.inorder.arg0.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// out[0] = max(+0, -0)  = +0
// out[1] = max(-0, +0)  = +0
// out[2] = min(+0, -0)  = -0
// out[3] = min(-0, +0)  = -0
// out[4] = max(1.0, NaN) = NaN
// out[5] = min(NaN, 1.0) = NaN
// out[6] = relu(-0)      = +0
// out[7] = relu(NaN)     = NaN   (sign bit set: 0xfe00 in, NaN out)
// out[8] = relu(-1.0)    = +0
// out[9] = relu(1.0)     = 1.0
// out[10] = max(-1.0, -0) = -0
// out[11] = min(-1.0, -0) = -1.0
// DATA:      0000 0000 8000 8000 7e00 7e00 0000 fe00
// DATA-NEXT: 0000 3c00 8000 bc00

module {
  func.func @edges(%out: memref<12xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %pos0 = arith.constant 0.0 : f16
    %neg0 = arith.constant -0.0 : f16
    %one = arith.constant 1.0 : f16
    %negone = arith.constant -1.0 : f16
    // 0x7e00 is the f16 quiet NaN; there is no literal for it.
    %nanbits = arith.constant 32256 : i16
    %nan = arith.bitcast %nanbits : i16 to f16
    // 0xfe00: the same NaN with its sign bit set.
    %nnanbits = arith.constant -512 : i16
    %nnan = arith.bitcast %nnanbits : i16 to f16

    %a = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    %r = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index

    // %a and %b hold the operand pairs; %c holds relu's input.
    memref.store %pos0,   %a[%c0]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %b[%c0]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %a[%c1]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %pos0,   %b[%c1]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %pos0,   %a[%c2]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %b[%c2]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %a[%c3]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %pos0,   %b[%c3]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %one,    %a[%c4]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %nan,    %b[%c4]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %nan,    %a[%c5]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %one,    %b[%c5]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %negone, %a[%c10] : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %b[%c10] : memref<12xf16, #hivm.address_space<ub>>
    memref.store %negone, %a[%c11] : memref<12xf16, #hivm.address_space<ub>>
    memref.store %neg0,   %b[%c11] : memref<12xf16, #hivm.address_space<ub>>

    memref.store %neg0,   %c[%c6]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %nnan,   %c[%c7]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %negone, %c[%c8]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %one,    %c[%c9]  : memref<12xf16, #hivm.address_space<ub>>

    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // Lanes 0..5 and 10..11 come from vmax/vmin, lanes 6..9 from vrelu; each
    // op writes the whole buffer and the unused lanes are overwritten below.
    %vmax = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    %vmin = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    %vrelu = memref.alloc() : memref<12xf16, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%a, %b : memref<12xf16, #hivm.address_space<ub>>,
                               memref<12xf16, #hivm.address_space<ub>>)
                  outs(%vmax : memref<12xf16, #hivm.address_space<ub>>)
    hivm.hir.vmin ins(%a, %b : memref<12xf16, #hivm.address_space<ub>>,
                               memref<12xf16, #hivm.address_space<ub>>)
                  outs(%vmin : memref<12xf16, #hivm.address_space<ub>>)
    hivm.hir.vrelu ins(%c : memref<12xf16, #hivm.address_space<ub>>)
                   outs(%vrelu : memref<12xf16, #hivm.address_space<ub>>)

    hivm.hir.pipe_barrier[<PIPE_ALL>]

    %m0 = memref.load %vmax[%c0]  : memref<12xf16, #hivm.address_space<ub>>
    %m1 = memref.load %vmax[%c1]  : memref<12xf16, #hivm.address_space<ub>>
    %m2 = memref.load %vmin[%c2]  : memref<12xf16, #hivm.address_space<ub>>
    %m3 = memref.load %vmin[%c3]  : memref<12xf16, #hivm.address_space<ub>>
    %m4 = memref.load %vmax[%c4]  : memref<12xf16, #hivm.address_space<ub>>
    %m5 = memref.load %vmin[%c5]  : memref<12xf16, #hivm.address_space<ub>>
    %m6 = memref.load %vrelu[%c6] : memref<12xf16, #hivm.address_space<ub>>
    %m7 = memref.load %vrelu[%c7] : memref<12xf16, #hivm.address_space<ub>>
    %m8 = memref.load %vrelu[%c8] : memref<12xf16, #hivm.address_space<ub>>
    %m9 = memref.load %vrelu[%c9] : memref<12xf16, #hivm.address_space<ub>>
    %m10 = memref.load %vmax[%c10] : memref<12xf16, #hivm.address_space<ub>>
    %m11 = memref.load %vmin[%c11] : memref<12xf16, #hivm.address_space<ub>>

    memref.store %m0,  %r[%c0]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m1,  %r[%c1]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m2,  %r[%c2]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m3,  %r[%c3]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m4,  %r[%c4]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m5,  %r[%c5]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m6,  %r[%c6]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m7,  %r[%c7]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m8,  %r[%c8]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m9,  %r[%c9]  : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m10, %r[%c10] : memref<12xf16, #hivm.address_space<ub>>
    memref.store %m11, %r[%c11] : memref<12xf16, #hivm.address_space<ub>>

    hivm.hir.pipe_barrier[<PIPE_S>]
    hivm.hir.store ins(%r : memref<12xf16, #hivm.address_space<ub>>)
                   outs(%out : memref<12xf16, #hivm.address_space<gm>>)
    return
  }
}
