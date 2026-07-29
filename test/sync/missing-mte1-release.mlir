// Negative counterpart of functional/mmad-macro-pipes.mlir. The second nd2nz
// refills the A tile in L1 while the mmad's MTE1 half has not been released,
// so the values the cube ends up staging are whichever of the two arrive
// first. The missing pair is set_flag[PIPE_MTE1, PIPE_MTE2].
//
// This is only reachable because a macro op occupies two pipes: fold mmadL1
// onto PIPE_M alone and the conflict is reported against the wrong pipe, so
// the fix the report suggests would not be the fix the hardware needs.

// RUN: not npuir-interp %s --sched=lazy --args=arange,%S/../functional/Inputs/ones4x4_f16.npy,%S/../functional/Inputs/ones4x4_f16.npy,zeros 2>&1 | FileCheck %s

// CHECK: MISSING SYNC on AIC#0: PIPE_MTE2 op touches data still in flight on PIPE_MTE1
// CHECK:   in flight  PIPE_MTE1  hivm.hir.mmadL1
// CHECK:   overwrites  PIPE_MTE2  hivm.hir.nd2nz
// CHECK: set_flag[PIPE_MTE1, PIPE_MTE2, <id>]
// CHECK: missing intra-core synchronisation point(s) detected

module {
  func.func @matmul(%a0: memref<4x4xf16, #hivm.address_space<gm>>,
                    %a1: memref<4x4xf16, #hivm.address_space<gm>>,
                    %b: memref<4x4xf16, #hivm.address_space<gm>>,
                    %c: memref<4x4xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %true = arith.constant true
    %false = arith.constant false
    %c4 = arith.constant 4 : index

    %a_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%a0 : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz ins(%b : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%b_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %acc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    // Missing set_flag[<PIPE_MTE1>, <PIPE_MTE2>] here.
    hivm.hir.nd2nz ins(%a1 : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]

    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %false, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
        outs(%c : memref<4x4xf32, #hivm.address_space<gm>>)
    return
  }
}
