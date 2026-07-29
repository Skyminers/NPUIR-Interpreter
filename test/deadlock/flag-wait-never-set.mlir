// Negative test: the AIV waits on a cross-core flag the AIC never sets.

// RUN: not npuir-interp %s --sched=lazy 2>&1 | FileCheck %s

// CHECK: DEADLOCK: circular flag wait
// CHECK: AIV#0.0 blocked at
// CHECK-SAME: sync_block_wait
// CHECK-SAME: flag=7
// CHECK: AIC#0 is Done

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @k_mix_aic(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // The matching sync_block_set [<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 7
    // is missing.
    return
  }

  func.func @k_mix_aiv(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 7
    return
  }
}
