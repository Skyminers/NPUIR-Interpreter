// Deterministic schedules model concurrent core progress by executing one IR
// operation per runnable core before rotating in CoreId order.

// RUN: npuir-interp %s --sched=lazy --sub-block-num=2 --trace=%t.lazy.trace
// RUN: FileCheck %s --check-prefix=TRACE --input-file=%t.lazy.trace
// RUN: npuir-interp %s --sched=inorder --sub-block-num=2 --trace=%t.inorder.trace
// RUN: FileCheck %s --check-prefix=TRACE --input-file=%t.inorder.trace

// TRACE:      AIC#0  arith.constant
// TRACE-NEXT: AIV#0.0  arith.constant
// TRACE-NEXT: AIV#0.1  arith.constant
// TRACE-NEXT: AIC#0  arith.constant
// TRACE-NEXT: AIV#0.0  arith.constant
// TRACE-NEXT: AIV#0.1  arith.constant
// TRACE-NEXT: AIC#0  func.return
// TRACE-NEXT: AIV#0.0  func.return
// TRACE-NEXT: AIV#0.1  func.return

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @round_robin_aic()
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    return
  }

  func.func @round_robin_aiv()
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    return
  }
}
