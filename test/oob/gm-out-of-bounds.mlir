// Negative test: a reinterpret_cast whose offset walks past the end of the
// GM pool. A miscomputed base address shows up as an out-of-bounds access
// rather than as silent corruption of a neighbouring buffer.

// RUN: not npuir-interp %s --sched=inorder --gm-size=65536 2>&1 | FileCheck %s

// CHECK: error: out of bounds access to gm at byte
// CHECK-SAME: arena capacity 65536

module {
  func.func @oob(%arg0: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %far = memref.reinterpret_cast %arg0 to offset: [1048576], sizes: [16],
                                   strides: [1]
        : memref<16xi16, #hivm.address_space<gm>>
       to memref<16xi16, strided<[1], offset: 1048576>,
                 #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%far : memref<16xi16, strided<[1], offset: 1048576>,
                                     #hivm.address_space<gm>>)
                  outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    return
  }
}
