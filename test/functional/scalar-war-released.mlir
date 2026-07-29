// Positive counterpart of sync/missing-scalar-war.mlir: the same two-tile
// kernel with the S -> MTE2 release flag in place. Both tiles are read, so the
// result is in0[0] + in1[0] rather than 2 * in0[0].
//
// in0 = arange(16), so in0[0] = 0; in1 is filled with 7. out[0] must be 7,
// and out[1] must still be 0 - nothing writes it.

// RUN: npuir-interp %s --sched=lazy --args=arange,7,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -td4 -j128 -N8 %t.arg2.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=inorder --args=arange,7,zeros --out=%t.inorder.
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA: 7 0

module {
  func.func @two_tiles(%in0: memref<16xi32, #hivm.address_space<gm>>,
                       %in1: memref<16xi32, #hivm.address_space<gm>>,
                       %out: memref<16xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index

    %buf = memref.alloc() : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in0 : memref<16xi32, #hivm.address_space<gm>>)
                  outs(%buf : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %v0 = memref.load %buf[%c0] : memref<16xi32, #hivm.address_space<ub>>

    // S -> MTE2: the scalar read is done, MTE2 may reuse the buffer.
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]

    hivm.hir.load ins(%in1 : memref<16xi32, #hivm.address_space<gm>>)
                  outs(%buf : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID2>]
    %v1 = memref.load %buf[%c0] : memref<16xi32, #hivm.address_space<ub>>

    %sum = arith.addi %v0, %v1 : i32
    memref.store %sum, %out[%c0] : memref<16xi32, #hivm.address_space<gm>>
    return
  }
}
