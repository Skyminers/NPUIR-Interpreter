// Exercise dimension and symbol operands together with the non-trivial
// floordiv/mod forms emitted by real tiled kernels.

// RUN: npuir-interp %s --sched=inorder --args=zeros --out=%t.
// RUN: od -An -td8 -j128 -N32 %t.arg0.npy | FileCheck %s --check-prefix=DATA

// DATA: 23 1
// DATA-NEXT: 8 0

#linear = affine_map<(d0)[s0] -> (d0 * 3 + s0)>
#ceildiv = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#mod = affine_map<()[s0] -> (s0 mod 16)>
#negative = affine_map<()[s0] -> ((-s0 + 15) floordiv 16)>

module {
  func.func @affine_apply(%out: memref<4xindex, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %c5 = arith.constant 5 : index
    %c8 = arith.constant 8 : index
    %0 = affine.apply #linear(%c5)[%c8]
    %1 = affine.apply #ceildiv()[%c8]
    %2 = affine.apply #mod()[%c8]
    %3 = affine.apply #negative()[%c8]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    memref.store %0, %out[%c0] : memref<4xindex, #hivm.address_space<gm>>
    memref.store %1, %out[%c1] : memref<4xindex, #hivm.address_space<gm>>
    memref.store %2, %out[%c2] : memref<4xindex, #hivm.address_space<gm>>
    memref.store %3, %out[%c3] : memref<4xindex, #hivm.address_space<gm>>
    return
  }
}
