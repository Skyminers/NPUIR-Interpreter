// Negative test: intra-core pipes with no flag between them. The vadd on
// PIPE_V reads UB bytes the PIPE_MTE2 load has not delivered yet, and the
// store on PIPE_MTE3 reads a sum PIPE_V has not produced yet.
//
// This is the raw form of bishengir/test/Integration/HIVM/VecAdd/add.mlir,
// before auto-sync runs. Under --sched=inorder it "works"; the whole point of
// the deferred-commit model is that it does not work here.

// RUN: not npuir-interp %s --sched=lazy --args=arange,arange 2>&1 | FileCheck %s
// The same IR is numerically fine once every effect is committed eagerly.
// RUN: npuir-interp %s --sched=inorder --args=arange,arange

// CHECK: MISSING SYNC on AIV#0.0: PIPE_V op touches data still in flight on PIPE_MTE2
// CHECK:   in flight  PIPE_MTE2  hivm.hir.load
// CHECK:   consumes   PIPE_V  hivm.hir.vadd
// CHECK: set_flag[PIPE_MTE2, PIPE_V, <id>]
// CHECK: MISSING SYNC on AIV#0.0: PIPE_MTE3 op touches data still in flight on PIPE_V
// CHECK: 2 missing intra-core synchronisation point(s) detected

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
