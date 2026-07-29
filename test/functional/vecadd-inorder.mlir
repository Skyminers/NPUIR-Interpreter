// The plan's M0 acceptance case: run the VecAdd integration kernel on the
// host and check the numerics. `inorder` commits every effect immediately, so
// this exercises the value model rather than the synchronisation model.

// RUN: npuir-interp %s --sched=inorder --args=arange,arange --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,arange --out=%t. && \
// RUN:   od -An -td2 -j128 %t.arg2.npy | FileCheck %s --check-prefix=DATA

// CHECK: npuir-interp: 1 core(s), sched=inorder
// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// arange + arange == 0 2 4 ... 30 (od wraps after eight i16 values)
// DATA:      0 2 4 6 8 10 12 14
// DATA-NEXT: 16 18 20 22 24 26 28 30

module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>,
                 %arg1: memref<16xi16, #hivm.address_space<gm>>,
                 %arg2: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>,
                                          memref<16xi16, #hivm.address_space<ub>>)
                  outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
