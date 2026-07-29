// The two f8 formats are not the same shape, and treating them as "f16 with
// fewer bits" gets both of them wrong.
//
// f8e5m2 is IEEE-shaped: 1-5-2, infinities, max finite 57344.
// f8e4m3fn is not: 1-4-3 with **no infinity**. Its only NaN encodings are
// 0x7f and 0xff, so the all-ones exponent is an ordinary binade except for its
// last pattern - which puts its largest finite value at 448, not the 240 an
// IEEE-shaped reading of the exponent field would give. Overflow therefore
// produces NaN, because there is no infinity to reach for.
//
//   in            f8e4m3fn        f8e5m2
//   1.0           38  1.0         3c  1.0
//   448.0         7e  448.0       5f  448.0    e4m3's largest finite
//   464.0         7e  448.0       5f  448.0    tie at the top -> even
//   480.0         7f  NaN         60  512.0    e4m3 overflows to NaN
//   57344.0       7f  NaN         7b  57344.0  e5m2's largest finite
//   61440.0       7f  NaN         7c  +inf     e5m2 overflows to infinity
//   -480.0        ff  NaN         e0  -512.0   overflow keeps the sign
//   2^-9          01  2^-9        18  2^-9     e4m3's smallest subnormal
//   2^-10         00  0.0         14  2^-10    exactly half of it -> even
//   -0.0          80  -0.0        80  -0.0
//   +inf          7f  NaN         7c  +inf     e4m3 has no infinity
//   NaN           7f  NaN         7e  NaN

// RUN: npuir-interp %s --sched=lazy --args=%S/Inputs/f8_cast_f32.npy,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -tx1 -j128 %t.arg1.npy | FileCheck %s --check-prefix=E4M3
// RUN: od -v -An -tx1 -j128 %t.arg2.npy | FileCheck %s --check-prefix=E5M2
// RUN: od -v -An -tx4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=WIDEN
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/f8_cast_f32.npy,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// E4M3: 38 7e 7e 7f 7f 7f ff 01 00 80 7f 7f
// E5M2: 3c 5f 5f 60 7b 7c e0 18 14 80 7c 7e

// Widening f8e5m2 back to f32 is exact. Lane 5 is the infinity e5m2 produced
// for 61440, lane 11 the NaN; both widen unchanged.
// WIDEN:      3f800000 43e00000 43e00000 44000000
// WIDEN-NEXT: 47600000 7f800000 c4000000 3b000000
// WIDEN-NEXT: 3a800000 80000000 7f800000 7fc00000

module {
  func.func @f8cast(%in: memref<12xf32, #hivm.address_space<gm>>,
                    %e4m3: memref<12xf8E4M3FN, #hivm.address_space<gm>>,
                    %e5m2: memref<12xf8E5M2, #hivm.address_space<gm>>,
                    %widen: memref<12xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %a = memref.alloc() : memref<12xf32, #hivm.address_space<ub>>
    %b4 = memref.alloc() : memref<12xf8E4M3FN, #hivm.address_space<ub>>
    %b5 = memref.alloc() : memref<12xf8E5M2, #hivm.address_space<ub>>
    %w = memref.alloc() : memref<12xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<12xf32, #hivm.address_space<gm>>)
                  outs(%a : memref<12xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    hivm.hir.vcast {round_mode = #hivm.round_mode<rint>}
                   ins(%a : memref<12xf32, #hivm.address_space<ub>>)
                   outs(%b4 : memref<12xf8E4M3FN, #hivm.address_space<ub>>)
    hivm.hir.vcast {round_mode = #hivm.round_mode<rint>}
                   ins(%a : memref<12xf32, #hivm.address_space<ub>>)
                   outs(%b5 : memref<12xf8E5M2, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast {round_mode = #hivm.round_mode<rint>}
                   ins(%b5 : memref<12xf8E5M2, #hivm.address_space<ub>>)
                   outs(%w : memref<12xf32, #hivm.address_space<ub>>)

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%b4 : memref<12xf8E4M3FN, #hivm.address_space<ub>>)
                   outs(%e4m3 : memref<12xf8E4M3FN, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b5 : memref<12xf8E5M2, #hivm.address_space<ub>>)
                   outs(%e5m2 : memref<12xf8E5M2, #hivm.address_space<gm>>)
    hivm.hir.store ins(%w : memref<12xf32, #hivm.address_space<ub>>)
                   outs(%widen : memref<12xf32, #hivm.address_space<gm>>)
    return
  }
}
