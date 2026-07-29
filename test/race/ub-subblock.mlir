// Negative test: the two sub-vector cores of one AIV share a UB, so an
// unsynchronised write there is a real race. It is tempting to skip UB in a
// race detector on the grounds that it is "core-private"; it is not, and this
// is the case that proves the checker does not skip it.
//
// Both sub-blocks address the same UB buffer through hivm.hir.pointer_cast at
// a baked absolute address - the shape PlanMemory leaves once on-chip offsets
// are fixed, and the only way two cores can name the same UB bytes.

// RUN: not npuir-interp %s --sched=lazy --sub-block-num=2 --args=zeros 2>&1 | FileCheck %s

// CHECK: DATA RACE on ub
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

    // Every sub-block writes its own index over the whole buffer. Nothing
    // orders the two, and they share the pool.
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
