// Minimal debugger walkthrough: AIC publishes 42, then AIV consumes it and
// writes 43. The deliberately short kernel makes every UI transition easy to
// compare with the IR and mathematical result.

// RUN: npuir-interp-debug %s --sched=lazy --args=zeros,zeros \
// RUN:   --debug-output=%t.jsonl --out=%t.
// RUN: %python %S/check_debug_walkthrough.py %t.jsonl
// RUN: od -An -td4 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// DATA: 43 0 0 0

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @walkthrough_aic(
      %mailbox: memref<4xi32, #hivm.address_space<gm>>,
      %output: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    memref.store %c42, %mailbox[%c0]
        : memref<4xi32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set [<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = 1
    return
  }

  func.func @walkthrough_aiv(
      %mailbox: memref<4xi32, #hivm.address_space<gm>>,
      %output: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = 1
    %c0 = arith.constant 0 : index
    %value = memref.load %mailbox[%c0]
        : memref<4xi32, #hivm.address_space<gm>>
    %c1 = arith.constant 1 : i32
    %result = arith.addi %value, %c1 : i32
    memref.store %result, %output[%c0]
        : memref<4xi32, #hivm.address_space<gm>>
    return
  }
}
