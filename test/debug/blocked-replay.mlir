// A failed session is still replayable and carries the exact synchronization
// object on which each core is parked.

// RUN: not npuir-interp-debug %s --debug-output=%t.jsonl
// RUN: %python %S/check_debug_session.py %t.jsonl --mode=blocked

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @k_mix_aic(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    return
  }

  func.func @k_mix_aiv(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 7
    return
  }
}
