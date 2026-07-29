// `round_mode = <odd>` is round-to-odd (Von Neumann rounding): round toward
// zero, then force the significand's least significant bit to 1 whenever the
// conversion lost anything. Its whole reason to exist is that narrowing
// through it and then rounding to nearest gives the same answer as narrowing
// in one correctly-rounded step, so no intermediate may collapse a nonzero
// value to zero - a later round-to-nearest could no longer tell the two
// apart.
//
// The interesting case is therefore underflow: an f32 smaller than the
// smallest f16 subnormal must come out as that subnormal (0x0001), not as
// zero. Rounding toward zero and stopping there gets every other case right,
// which is why this one survives casual testing.
//
//   in                       expected f16
//   0x2edbe6ff  1e-10        0x0001  underflow: smallest subnormal, not 0
//   0xaedbe6ff -1e-10        0x8001  ... with the sign kept
//   0x33000000  2^-25        0x0001  exactly half the smallest subnormal
//   0x33800000  2^-24        0x0001  exact: already the smallest subnormal
//   0x3f800001  1+1ulp(f32)  0x3c01  inexact normal: LSB forced to 1
//   0x3f800000  1.0          0x3c00  exact: left alone, LSB stays 0
//   0x7f7fffff  f32 max      0x7bff  overflow saturates to max finite, odd
//   0x7f800000  +inf         0x7c00  infinity is passed through

// RUN: npuir-interp %s --sched=lazy --args=%S/Inputs/odd_f32.npy,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -tx2 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/odd_f32.npy,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy

// CHECK-NOT: MISSING SYNC
// DATA: 0001 8001 0001 0001 3c01 3c00 7bff 7c00

module {
  func.func @narrow(%in: memref<8xf32, #hivm.address_space<gm>>,
                    %out: memref<8xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %a = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%a : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast {round_mode = #hivm.round_mode<odd>}
                   ins(%a : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%c : memref<8xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%c : memref<8xf16, #hivm.address_space<ub>>)
                   outs(%out : memref<8xf16, #hivm.address_space<gm>>)
    return
  }
}
