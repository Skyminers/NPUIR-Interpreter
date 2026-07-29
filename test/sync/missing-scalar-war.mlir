// The other direction of the PIPE_S model: a scalar *read* also has to be
// released before a DMA is allowed to refill the bytes it took. Anti
// dependences are the half of auto-sync that is easiest to drop, because the
// program still looks like it only ever reads.
//
// Without set_flag[PIPE_S, PIPE_MTE2] the second load may land before the
// first read has happened, and the kernel adds the same tile twice. The
// positive counterpart is functional/scalar-war-released.mlir.

// RUN: not npuir-interp %s --sched=lazy --args=arange,arange,zeros 2>&1 | FileCheck %s

// CHECK: MISSING SYNC on AIV#0.0: PIPE_MTE2 op touches data still in flight on PIPE_S
// CHECK:   in flight  PIPE_S  memref.load
// CHECK:   overwrites  PIPE_MTE2  hivm.hir.load
// CHECK: set_flag[PIPE_S, PIPE_MTE2, <id>]

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

    // Missing set_flag[<PIPE_S>, <PIPE_MTE2>] / wait_flag pair here.
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
