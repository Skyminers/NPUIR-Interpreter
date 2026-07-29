// The Cube path end to end: nd2nz brings A and B from GM into L1, mmadL1
// accumulates into the L0C f32 accumulator, and fixpipe writes the result out
// with an nz2nd layout change. Every step is on a different pipe
// (MTE2 / M / FIX), so the flags between them are load-bearing.
//
// A = arange(16) as 4x4 f16, B = all ones, so C[i][j] = row_sum(A, i):
//   row 0: 0+1+2+3 = 6, row 1: 22, row 2: 38, row 3: 54.
// Accumulation happens in f32 even though the inputs are f16, matching the
// L0C accumulator.

// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy    --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg2.npy %t.inorder.arg2.npy
// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. && \
// RUN:   od -An -tf4 -j128 -N64 %t.arg2.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA:      6.000000e+00 6.000000e+00 6.000000e+00 6.000000e+00
// DATA-NEXT: 2.200000e+01 2.200000e+01 2.200000e+01 2.200000e+01
// DATA-NEXT: 3.800000e+01 3.800000e+01 3.800000e+01 3.800000e+01
// DATA-NEXT: 5.400000e+01 5.400000e+01 5.400000e+01 5.400000e+01

module {
  func.func @matmul(%a: memref<4x4xf16, #hivm.address_space<gm>>,
                    %b: memref<4x4xf16, #hivm.address_space<gm>>,
                    %c: memref<4x4xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %true = arith.constant true
    %c4 = arith.constant 4 : index

    %a_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%a : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz ins(%b : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%b_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)

    // MTE2 -> M: L1 must be filled before the cube unit reads it.
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_M>, <EVENT_ID0>]

    %acc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    // M -> FIX: the accumulator must be complete before fixpipe drains it.
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]

    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
        outs(%c : memref<4x4xf32, #hivm.address_space<gm>>)
    return
  }
}
