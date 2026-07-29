// The L0C accumulator is f32 even when A and B are f16, and the products are
// exact - f16 has an 11-bit significand, so a product needs 22 bits and f32
// carries 24. Getting this wrong is invisible on well-conditioned data, so the
// input is chosen to make it loud.
//
// A rows are [4096, 1, 1, 1] and B is all ones, so each output is
// 4096 + 1 + 1 + 1. In f32 that is 4099. In f16 the ulp at 4096 is 4, so each
// +1 rounds straight back to 4096 and an f16 accumulator would answer 4096 -
// off by three units in a value that looks entirely plausible.

// RUN: npuir-interp %s --sched=lazy --args=%S/Inputs/accum4x4_f16.npy,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -tf4 -j128 -N64 %t.arg2.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/accum4x4_f16.npy,%S/Inputs/ones4x4_f16.npy,zeros --out=%t.inorder.
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA:      4.099000e+03 4.099000e+03 4.099000e+03 4.099000e+03
// DATA-NEXT: 4.099000e+03 4.099000e+03 4.099000e+03 4.099000e+03
// DATA-NEXT: 4.099000e+03 4.099000e+03 4.099000e+03 4.099000e+03
// DATA-NEXT: 4.099000e+03 4.099000e+03 4.099000e+03 4.099000e+03

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
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    %acc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c4, %c4, %c4
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<4x4xf32, #hivm.address_space<cc>>)

    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<4x4xf32, #hivm.address_space<cc>>)
        outs(%c : memref<4x4xf32, #hivm.address_space<gm>>)
    return
  }
}
