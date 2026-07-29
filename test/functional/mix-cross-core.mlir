// A MIX kernel where the AIC produces a GM buffer and the AIV consumes it,
// ordered by a cross-core flag pair. The positive counterpart of
// race/missing-c2v-flag.mlir: nothing should be reported here.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros 2>&1 | FileCheck %s
// The consumer may be scheduled before the producer. A cross-core flag that
// is raised while its waiter is already parked has to wake that waiter, or a
// correct kernel gets reported as a deadlock on most seeds.
// RUN: npuir-interp %s --sched=fuzz --seed=0 --args=arange,zeros,zeros 2>&1 | FileCheck %s --check-prefix=FUZZ
// RUN: npuir-interp %s --sched=fuzz --seed=3 --args=arange,zeros,zeros 2>&1 | FileCheck %s --check-prefix=FUZZ
// RUN: npuir-interp %s --sched=fuzz --seed=6 --args=arange,zeros,zeros 2>&1 | FileCheck %s --check-prefix=FUZZ
// RUN: npuir-interp %s --sched=lazy    --args=arange,zeros,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg2.npy %t.inorder.arg2.npy

// CHECK: 2 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DEADLOCK

// FUZZ-NOT: DEADLOCK
// FUZZ-NOT: DATA RACE

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // AIC: %in -> %mid
  func.func @k_mix_aic(%in: memref<16xi16, #hivm.address_space<gm>>,
                       %mid: memref<16xi16, #hivm.address_space<gm>>,
                       %out: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %buf = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%buf : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%buf : memref<16xi16, #hivm.address_space<ub>>)
                   outs(%mid : memref<16xi16, #hivm.address_space<gm>>)
    // Publish: everything MTE3 has written is now visible to the vector side.
    hivm.hir.sync_block_set [<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
    return
  }

  // AIV: %mid -> %out, gated on the AIC's flag.
  func.func @k_mix_aiv(%in: memref<16xi16, #hivm.address_space<gm>>,
                       %mid: memref<16xi16, #hivm.address_space<gm>>,
                       %out: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
    %buf = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%mid : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%buf : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%buf : memref<16xi16, #hivm.address_space<ub>>)
                   outs(%out : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
