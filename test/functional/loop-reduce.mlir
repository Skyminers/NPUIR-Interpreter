// A tiled kernel with an scf.for loop and a hivm.hir.vreduce, exercising the
// frame-based executor (loop-carried state, region entry/exit) and the f32
// reduction path. Sums arange(64) tile by tile: 0+1+...+63 == 2016.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy    --args=arange,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg1.npy %t.inorder.arg1.npy
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. && \
// RUN:   od -An -tf4 -j128 -N4 %t.arg1.npy | FileCheck %s --check-prefix=SUM

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// SUM: 2.016000e+03

module {
  func.func @tiled_sum(%in: memref<64xf32, #hivm.address_space<gm>>,
                       %out: memref<1xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    %acc = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    %zero = arith.constant 0.000000e+00 : f32
    hivm.hir.vbrc ins(%zero : f32) outs(%acc : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]

    %tile = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c64 step %c16 {
      %slice = memref.subview %in[%i] [16] [1]
          : memref<64xf32, #hivm.address_space<gm>>
         to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%slice : memref<16xf32, strided<[1], offset: ?>,
                                        #hivm.address_space<gm>>)
                    outs(%tile : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

      // already_initialize_init makes vreduce accumulate into the destination,
      // so this folds the whole tile into the running total.
      hivm.hir.vreduce {already_initialize_init} <sum>
          ins(%tile : memref<16xf32, #hivm.address_space<ub>>)
          outs(%acc : memref<1xf32, #hivm.address_space<ub>>)
          unsigned_src = false reduce_dims = [0]
      // The next iteration's load targets the same tile buffer.
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    }

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
    hivm.hir.store ins(%acc : memref<1xf32, #hivm.address_space<ub>>)
                   outs(%out : memref<1xf32, #hivm.address_space<gm>>)
    return
  }
}
