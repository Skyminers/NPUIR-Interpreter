// An index vector is data, so nothing about the IR constrains it. A gather
// whose index escapes the source extent must be reported, not folded to some
// neighbouring element - reading the wrong element quietly is exactly the
// class of bug an interpreter is supposed to catch for you.
//
// The indices here are arange(8) scaled by 2, so lanes 4..7 ask for elements
// 8, 10, 12 and 14 of an 8-element source.

// RUN: not npuir-interp %s --sched=lazy --args=arange,zeros 2>&1 | FileCheck %s

// CHECK: error: vgather index 8 is outside the source extent 8 on axis 0

module {
  func.func @oob(%in: memref<8xi32, #hivm.address_space<gm>>,
                 %out: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %two = arith.constant 2 : i32
    %a = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %idx = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %d = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xi32, #hivm.address_space<gm>>)
                  outs(%a : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%a, %two : memref<8xi32, #hivm.address_space<ub>>, i32)
                  outs(%idx : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vgather ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                     indices(%idx : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%d : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.store ins(%d : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<8xi32, #hivm.address_space<gm>>)
    return
  }
}
