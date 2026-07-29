// The same kernel with the intra-core flags auto-sync would insert. Under
// `lazy` this must produce the same numbers as `inorder` and report nothing.
// That agreement between the two schedules is the strongest automatic signal
// that the synchronisation is sufficient (plan section 15.3).

// RUN: npuir-interp %s --sched=lazy --args=arange,arange --out=%t.lazy. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy    --args=arange,arange --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,arange --out=%t.inorder.
// RUN: cmp %t.lazy.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// CHECK-NOT: DEADLOCK

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

    // MTE2 -> V: the loads must land before the vector unit reads UB.
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>,
                                          memref<16xi16, #hivm.address_space<ub>>)
                  outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)

    // V -> MTE3: the sum must be in UB before the store reads it.
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]

    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
