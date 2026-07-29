// mmadL1 carries real_m / real_k / real_n operands because L1 tiles are
// padded up to whole fractal blocks. Reading the loop bounds off the buffer
// extents instead folds the padding into the accumulator - and the padding
// holds whatever was there before, so the answer is quietly wrong rather than
// obviously broken.
//
// A and B are 4x4 tiles; the real problem is K = 2. With A = arange(16) and
// B = ones, C[i][j] = A[i][0] + A[i][1]:
//   C[0][*] = 0 + 1 = 1,  C[1][*] = 4 + 5 = 9.
// Using the padded K = 4 would give 6 and 22 instead.

// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --args=arange,%S/Inputs/ones4x4_f16.npy,zeros --out=%t. && \
// RUN:   od -An -tf4 -j128 -N16 %t.arg2.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: DATA RACE
// DATA: 1.000000e+00 1.000000e+00 9.000000e+00 9.000000e+00

module {
  func.func @padded(%a: memref<4x4xf16, #hivm.address_space<gm>>,
                    %b: memref<4x4xf16, #hivm.address_space<gm>>,
                    %c: memref<2x2xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %true = arith.constant true
    %c2 = arith.constant 2 : index

    %a_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    %b_l1 = memref.alloc() : memref<4x4xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%a : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%a_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz ins(%b : memref<4x4xf16, #hivm.address_space<gm>>)
                   outs(%b_l1 : memref<4x4xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_M>, <EVENT_ID0>]

    %acc = memref.alloc() : memref<2x2xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c2, %c2, %c2
                        : memref<4x4xf16, #hivm.address_space<cbuf>>,
                          memref<4x4xf16, #hivm.address_space<cbuf>>,
                          i1, index, index, index)
                    outs(%acc : memref<2x2xf32, #hivm.address_space<cc>>)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%acc : memref<2x2xf32, #hivm.address_space<cc>>)
        outs(%c : memref<2x2xf32, #hivm.address_space<gm>>)
    return
  }
}
