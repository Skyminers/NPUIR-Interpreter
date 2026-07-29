// A `pipe_barrier` retires the scalar unit's outstanding accesses just like a
// flag pair does, in both directions and for both the targeted and the
// PIPE_ALL form. Without that, every kernel that orders its scalar traffic
// with barriers instead of flags would be reported as unsynchronised.
//
// out = in + 1 again, but MTE2 -> S is ordered by pipe_barrier[<PIPE_ALL>] and
// S -> MTE3 by pipe_barrier[<PIPE_S>].

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -td4 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// CHECK-NOT: warning
// DATA: 1 2 3 4

module {
  func.func @barrier_ordered(%in: memref<4xi32, #hivm.address_space<gm>>,
                             %out: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %one = arith.constant 1 : i32

    %buf = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<4xi32, #hivm.address_space<gm>>)
                  outs(%buf : memref<4xi32, #hivm.address_space<ub>>)

    hivm.hir.pipe_barrier[<PIPE_ALL>]

    scf.for %i = %c0 to %c4 step %c1 {
      %v = memref.load %buf[%i] : memref<4xi32, #hivm.address_space<ub>>
      %w = arith.addi %v, %one : i32
      memref.store %w, %buf[%i] : memref<4xi32, #hivm.address_space<ub>>
    }

    hivm.hir.pipe_barrier[<PIPE_S>]

    hivm.hir.store ins(%buf : memref<4xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<4xi32, #hivm.address_space<gm>>)
    return
  }
}
