// VArange applies one stride per destination dimension. For offset 1,
// strides [1, 2], and shape 2x4, the rows are [1,3,5,7] and [2,4,6,8].

// RUN: npuir-interp %s --sched=lazy --args=zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg0.npy %t.inorder.arg0.npy
// RUN: od -An -td4 -j128 -N32 %t.lazy.arg0.npy | FileCheck %s

// CHECK: 1 3 5 7
// CHECK-NEXT: 2 4 6 8

module {
  func.func @varange(%out: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %tile = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    hivm.hir.varange offset[%c1] strides[%c1, %c2]
        outs(%tile : memref<2x4xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %flat = memref.collapse_shape %tile [[0, 1]]
        : memref<2x4xi32, #hivm.address_space<ub>>
          into memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.store
        ins(%flat : memref<8xi32, #hivm.address_space<ub>>)
        outs(%out : memref<8xi32, #hivm.address_space<gm>>)
    return
  }
}
