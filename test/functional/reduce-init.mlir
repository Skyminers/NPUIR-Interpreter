// VReduce starts from the operation's identity value when its destination was
// not explicitly initialized. The destination arena is poison-filled, so this
// test catches implementations that accidentally use its old contents.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg1.npy %t.inorder.arg1.npy
// RUN: od -An -tf4 -j128 -N4 %t.lazy.arg1.npy | FileCheck %s

// CHECK: 2.016000e+03

module {
  func.func @reduce_default_init(
      %in: memref<64xf32, #hivm.address_space<gm>>,
      %out: memref<1xf32, #hivm.address_space<gm>>) attributes {hacc.entry} {
    %tile = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.load
        ins(%in : memref<64xf32, #hivm.address_space<gm>>)
        outs(%tile : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    %sum = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum>
        ins(%tile : memref<64xf32, #hivm.address_space<ub>>)
        outs(%sum : memref<1xf32, #hivm.address_space<ub>>)
        unsigned_src = false reduce_dims = [0]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.store
        ins(%sum : memref<1xf32, #hivm.address_space<ub>>)
        outs(%out : memref<1xf32, #hivm.address_space<gm>>)
    return
  }
}
