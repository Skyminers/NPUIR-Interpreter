// Negative test: the UB working set does not fit the device's UB. The
// interpreter allocates from a fixed-capacity arena rather than from the
// host heap precisely so that a PlanMemory result that overflows UB fails
// here instead of quietly working.

// RUN: not npuir-interp %s --sched=inorder --ub-size=1024 2>&1 | FileCheck %s

// CHECK: error: ub capacity exceeded
// CHECK-SAME: arena is 1024 bytes

module {
  func.func @big(%arg0: memref<4096xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %alloc = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<4096xf32, #hivm.address_space<gm>>)
                  outs(%alloc : memref<4096xf32, #hivm.address_space<ub>>)
    return
  }
}
