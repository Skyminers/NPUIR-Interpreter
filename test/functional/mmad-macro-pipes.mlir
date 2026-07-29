// `mmadL1` carries MacroOpPipeTrait<MTE1, M>: it is two hardware instructions,
// not one. MTE1 stages the L1 tiles into L0A/L0B, then the cube multiplies
// them into L0C. Modelling that split is what makes this kernel legal:
// `set_flag[PIPE_MTE1, PIPE_MTE2]` releases the A tile as soon as it has been
// staged, so MTE2 may refill L1 for the next step while the cube is still
// working on the values it captured.
//
// A0 = arange(16) as 4x4 f16 -> row sums  6, 22, 38, 54
// A1 = ones(4x4)             -> row sums  4,  4,  4,  4
// B  = ones(4x4), so C = row_sum(A0) + row_sum(A1) = 10, 26, 42, 58.
//
// Fold the two halves into one effect and the first mmad reads A *after* the
// refill, which would give 8 8 8 8 - and the release flag would be reported as
// not matching anything in flight.

// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -tf4 -j128 -N64 %t.arg3.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=arange,%S/Inputs/ones4x4_f16.npy,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.inorder.
// RUN: cmp %t.arg3.npy %t.inorder.arg3.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA:      1.000000e+01 1.000000e+01 1.000000e+01 1.000000e+01
// DATA-NEXT: 2.600000e+01 2.600000e+01 2.600000e+01 2.600000e+01
// DATA-NEXT: 4.200000e+01 4.200000e+01 4.200000e+01 4.200000e+01
// DATA-NEXT: 5.800000e+01 5.800000e+01 5.800000e+01 5.800000e+01

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

    // MTE2 -> MTE1: L1 must be filled before the tiles are staged into L0A/L0B.
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %acc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    // MTE1 -> MTE2: staging is done, the A tile in L1 may be reused. The cube
    // half of the mmad above is still outstanding at this point.
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]

    hivm.hir.nd2nz ins(%a1 : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]

    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %false, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    // M -> FIX: the accumulator must be complete before fixpipe drains it.
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID3>]

    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
        outs(%c : memref<4x4xf32, #hivm.address_space<gm>>)
    return
  }
}
