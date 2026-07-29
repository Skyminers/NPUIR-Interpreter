// Negative counterpart of functional/scalar-pipe.mlir: the same kernel with
// every PIPE_S flag removed. Each of the three directions InjectSync would
// have covered is reported separately.
//
//   MTE2 -> S    the scalar load reads UB the DMA has not delivered
//   MTE2 -> S    the scalar store overwrites the same undelivered bytes
//   S -> MTE3    the store-out drains UB the scalar unit has not published
//
// Under --sched=inorder every effect commits where it stands, so the same IR
// produces the right numbers and no report - which is exactly why the deferred
// model is the one that checks anything.

// RUN: not npuir-interp %s --sched=lazy --args=arange,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros

// CHECK: MISSING SYNC on AIV#0.0: PIPE_S op touches data still in flight on PIPE_MTE2
// CHECK:   in flight  PIPE_MTE2  hivm.hir.load
// CHECK:   consumes   PIPE_S  memref.load
// CHECK: set_flag[PIPE_MTE2, PIPE_S, <id>]
// CHECK: MISSING SYNC on AIV#0.0: PIPE_S op touches data still in flight on PIPE_MTE2
// CHECK:   overwrites  PIPE_S  memref.store
// CHECK: MISSING SYNC on AIV#0.0: PIPE_MTE3 op touches data still in flight on PIPE_S
// CHECK:   in flight  PIPE_S  memref.store
// CHECK:   consumes   PIPE_MTE3  hivm.hir.store
// CHECK: set_flag[PIPE_S, PIPE_MTE3, <id>]
// CHECK: 3 missing intra-core synchronisation point(s) detected

module {
  func.func @scalar_increment(%in: memref<16xi32, #hivm.address_space<gm>>,
                              %out: memref<16xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %one = arith.constant 1 : i32

    %buf = memref.alloc() : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<16xi32, #hivm.address_space<gm>>)
                  outs(%buf : memref<16xi32, #hivm.address_space<ub>>)
    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %buf[%i] : memref<16xi32, #hivm.address_space<ub>>
      %w = arith.addi %v, %one : i32
      memref.store %w, %buf[%i] : memref<16xi32, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%buf : memref<16xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<16xi32, #hivm.address_space<gm>>)
    return
  }
}
