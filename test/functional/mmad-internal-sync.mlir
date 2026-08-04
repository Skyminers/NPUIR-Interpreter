// GraphSyncSolver lowers dependencies on the virtual L1A/L1B pipes into
// `mmadL1`'s seven sync_related_args.  They are real wait/set operations
// executed inside mma_tile, not metadata that the interpreter may ignore.
// This test exercises both the L1 producer/release pair and the two hoisted
// M->MTE1 ping-pong events.

// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.lazy. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg2.npy %t.inorder.arg2.npy
// RUN: od -An -tf4 -j128 -N64 %t.lazy.arg2.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// CHECK-NOT: DEADLOCK
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
    %c-1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64

    %a_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%a : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz ins(%b : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%b_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    // These initial credits are normally hoisted around a group of mmads.
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID3>]

    %acc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
                    sync_related_args(%c0_i64, %c-1_i64,
                                      %c1_i64, %c-1_i64, %c-1_i64,
                                      %c2_i64, %c3_i64
                                      : i64, i64, i64, i64, i64, i64, i64)

    // The macro publishes all three of these events internally.
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID3>]

    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
        outs(%c : memref<4x4xf32, #hivm.address_space<gm>>)
    return
  }
}
