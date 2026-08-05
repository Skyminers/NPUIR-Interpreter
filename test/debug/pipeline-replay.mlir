// Debug replay must expose an issued MTE2 task, the memory commit caused by
// wait_flag, and the final completed core state.

// RUN: npuir-interp-debug %s --args=arange,zeros --debug-output=%t.jsonl
// RUN: %python %S/check_debug_session.py %t.jsonl --mode=pipeline
// RUN: npuir-interp-debug %s --args=arange,zeros --block-dim=2 \
// RUN:   --check=none --debug-core=AIV#0.0 --debug-output=%t.filtered.jsonl
// RUN: %python %S/check_debug_session.py %t.filtered.jsonl --mode=pipeline \
// RUN:   --core-filter=AIV#0.0

module {
  func.func @copy(%src: memref<8xi16, #hivm.address_space<gm>>,
                  %dst: memref<8xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %six = arith.muli %two, %three : i32
    %tmp = memref.alloc() : memref<8xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<8xi16, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xi16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%tmp : memref<8xi16, #hivm.address_space<ub>>)
                   outs(%dst : memref<8xi16, #hivm.address_space<gm>>)
    return
  }
}
