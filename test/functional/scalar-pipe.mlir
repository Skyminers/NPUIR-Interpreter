// The scalar unit is PIPE_S, and it is a pipe like any other: a value MTE2
// has not delivered yet is not readable by `memref.load`, and a value the
// scalar unit has written is not visible to MTE3 until a flag says so. This
// is the shape InjectSync produces for @test_mem_memref_load_store in
// test/Dialect/HIVM/inject-sync.mlir.
//
// out = in + 1, elementwise, done by the scalar unit in UB.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -td4 -j128 -N64 %t.arg1.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// A scalar loop must not queue one unretired marker per iteration; the markers
// coalesce, so this run stays well under the cap and prints no warning.
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros 2>&1 | FileCheck %s --check-prefix=NOWARN

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// NOWARN-NOT: warning
// DATA:      1 2 3 4
// DATA-NEXT: 5 6 7 8
// DATA-NEXT: 9 10 11 12
// DATA-NEXT: 13 14 15 16

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

    // MTE2 -> S: the scalar unit may not read UB before the load lands.
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]

    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %buf[%i] : memref<16xi32, #hivm.address_space<ub>>
      %w = arith.addi %v, %one : i32
      memref.store %w, %buf[%i] : memref<16xi32, #hivm.address_space<ub>>
    }

    // S -> MTE3: the scalar stores must be visible before MTE3 drains UB.
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]

    hivm.hir.store ins(%buf : memref<16xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<16xi32, #hivm.address_space<gm>>)
    return
  }
}
