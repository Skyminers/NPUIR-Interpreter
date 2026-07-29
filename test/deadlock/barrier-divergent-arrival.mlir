// Negative test: a `sync_block[<ALL>]` that only the AIC reaches, because the
// AIV's copy sits inside a conditional that is false. This is the runtime
// shape of the AddControlFlowCondition/CloneOps bug class - a barrier cloned
// into a divergent region - and it is invisible to IR-text lit checks.

// RUN: not npuir-interp %s --sched=lazy 2>&1 | FileCheck %s

// CHECK: DEADLOCK: barrier ALL:1 arrival mismatch
// CHECK: expected participants: AIC#0 AIV#0.0
// CHECK: arrived:  AIC#0
// CHECK: AIV#0.0 is Done
// CHECK-SAME: never reached ALL:1
// CHECK: barrier that only some cores execute

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @k_mix_aic(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %always = arith.constant true
    scf.if %always {
      hivm.hir.sync_block [<ALL>, 1] tcube_pipe = <PIPE_M> tvector_pipe = <PIPE_MTE3>
    }
    return
  }

  func.func @k_mix_aiv(%gm: memref<16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // The condition the cloning pass failed to keep in sync with the AIC's.
    %never = arith.constant false
    scf.if %never {
      hivm.hir.sync_block [<ALL>, 1] tcube_pipe = <PIPE_M> tvector_pipe = <PIPE_MTE3>
    }
    return
  }
}
