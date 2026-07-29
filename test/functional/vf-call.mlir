// An outlined vectorized function called with UB memref arguments, modelled
// on the `_swa_fwd_kernel_outlined_*_vf_*` bodies real SIMD compilation
// produces: scf.for over tiles, memref.subview, vector.transfer_read /
// transfer_write, and arith on vector values.
//
// Exercises the call frame (a nested func activation with its own
// environment) together with the vector lifting of the scalar arith handlers.
//
// out = (in * 2) + 1, elementwise over 32 f32 values.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy    --args=arange,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg1.npy %t.inorder.arg1.npy
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. && \
// RUN:   od -An -tf4 -j128 -N32 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA:      1.000000e+00 3.000000e+00 5.000000e+00 7.000000e+00
// DATA-NEXT: 9.000000e+00 1.100000e+01 1.300000e+01 1.500000e+01

module {
  // The vectorized body, as SplitMixKernel / OutlineScope would leave it.
  func.func @kernel_outlined_vf_0(%src: memref<32xf32, #hivm.address_space<ub>>,
                                  %dst: memref<32xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %pad = arith.constant 0.000000e+00 : f32
    %two = arith.constant dense<2.000000e+00> : vector<8xf32>
    %one = arith.constant dense<1.000000e+00> : vector<8xf32>

    scf.for %i = %c0 to %c32 step %c8 {
      %in = memref.subview %src[%i] [8] [1]
          : memref<32xf32, #hivm.address_space<ub>>
         to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %out = memref.subview %dst[%i] [8] [1]
          : memref<32xf32, #hivm.address_space<ub>>
         to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %v = vector.transfer_read %in[%c0], %pad {in_bounds = [true]}
          : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>,
            vector<8xf32>
      %scaled = arith.mulf %v, %two : vector<8xf32>
      %biased = arith.addf %scaled, %one : vector<8xf32>
      vector.transfer_write %biased, %out[%c0] {in_bounds = [true]}
          : vector<8xf32>,
            memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }

  func.func @kernel(%in: memref<32xf32, #hivm.address_space<gm>>,
                    %out: memref<32xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %src = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<32xf32, #hivm.address_space<gm>>)
                  outs(%src : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    func.call @kernel_outlined_vf_0(%src, %dst)
        {hivm.vector_function, no_inline}
        : (memref<32xf32, #hivm.address_space<ub>>,
           memref<32xf32, #hivm.address_space<ub>>) -> ()

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%dst : memref<32xf32, #hivm.address_space<ub>>)
                   outs(%out : memref<32xf32, #hivm.address_space<gm>>)
    return
  }
}
