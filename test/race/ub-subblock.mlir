// The two split AIV lanes have distinct UB address spaces even though each
// uses the same local address. Their UB writes therefore do not race. Both
// lanes deliberately store to the same GM output, which remains a real race.
//
// This guards the distinction between lane-private UB and launch-global GM.

// RUN: not npuir-interp %s --sched=lazy --sub-block-num=2 --args=zeros 2>&1 | FileCheck %s

// CHECK-NOT: DATA RACE on ub
// CHECK: DATA RACE on gm
// CHECK: AIV#0.0
// CHECK: AIV#0.1
// CHECK: no happens-before edge between these two accesses

module {
  func.func @shared_ub(%out: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %addr = arith.constant 4096 : i64
    %shared = hivm.hir.pointer_cast(%addr)
        : memref<8xi32, #hivm.address_space<ub>>

    // Every sub-block writes its own index to its private local UB.
    %sub64 = hivm.hir.get_sub_block_idx -> i64
    %sub = arith.trunci %sub64 : i64 to i32
    hivm.hir.vbrc ins(%sub : i32)
                  outs(%shared : memref<8xi32, #hivm.address_space<ub>>)

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%shared : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<8xi32, #hivm.address_space<gm>>)
    return
  }
}
