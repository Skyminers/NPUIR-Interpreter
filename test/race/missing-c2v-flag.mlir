// Negative test: functional/mix-cross-core.mlir with the cross-core
// sync_block_set/wait pair deleted, which is what a pass that drops a real
// cross-core memory dependency produces. The AIV's load of %mid is then
// unordered with the AIC's store to it.
//
// A checker with only positive tests is indistinguishable from a checker that
// never runs, so every check in this tool has a case like this one.

// RUN: not npuir-interp %s --sched=lazy --args=arange,zeros,zeros 2>&1 | FileCheck %s

// CHECK: DATA RACE on gm
// CHECK:   W  AIC#0
// CHECK-SAME: hivm.hir.store
// CHECK:   R  AIV#0
// CHECK-SAME: hivm.hir.load
// CHECK: no happens-before edge between these two accesses

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
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
    // MISSING: hivm.hir.sync_block_set [<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
    return
  }

  func.func @k_mix_aiv(%in: memref<16xi16, #hivm.address_space<gm>>,
                       %mid: memref<16xi16, #hivm.address_space<gm>>,
                       %out: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // MISSING: hivm.hir.sync_block_wait [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
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
