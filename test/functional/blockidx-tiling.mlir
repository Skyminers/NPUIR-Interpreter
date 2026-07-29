// Multi-block execution: each block computes its own tile, chosen by
// hivm.hir.get_block_idx. Because the tiles are disjoint there is no race,
// which is the point - the race detector must stay quiet on correctly tiled
// code, or it is useless on real kernels.
//
// 4 blocks x 8 elements: out[i] = in[i] + 100.

// RUN: npuir-interp %s --sched=lazy --block-dim=4 --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --block-dim=4 --args=arange,zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N128 %t.arg1.npy | FileCheck %s --check-prefix=DATA
// Two blocks that both write tile 0 would be a race; four disjoint ones are not.
// RUN: npuir-interp %s --sched=fuzz --seed=3 --block-dim=4 --args=arange,zeros

// CHECK: 4 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC
// DATA:      100 101 102 103
// DATA-NEXT: 104 105 106 107
// DATA-NEXT: 108 109 110 111
// DATA-NEXT: 112 113 114 115
// DATA-NEXT: 116 117 118 119
// DATA-NEXT: 120 121 122 123
// DATA-NEXT: 124 125 126 127
// DATA-NEXT: 128 129 130 131

module {
  func.func @tiled(%in: memref<32xi32, #hivm.address_space<gm>>,
                   %out: memref<32xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %block64 = hivm.hir.get_block_idx -> i64
    %block = arith.index_cast %block64 : i64 to index
    %c8 = arith.constant 8 : index
    %offset = arith.muli %block, %c8 : index

    %inTile = memref.subview %in[%offset] [8] [1]
        : memref<32xi32, #hivm.address_space<gm>>
       to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %outTile = memref.subview %out[%offset] [8] [1]
        : memref<32xi32, #hivm.address_space<gm>>
       to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>

    %buf = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%inTile : memref<8xi32, strided<[1], offset: ?>,
                                       #hivm.address_space<gm>>)
                  outs(%buf : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    %bias = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %c100 = arith.constant 100 : i32
    hivm.hir.vbrc ins(%c100 : i32)
                  outs(%bias : memref<8xi32, #hivm.address_space<ub>>)
    %sum = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%buf, %bias : memref<8xi32, #hivm.address_space<ub>>,
                                     memref<8xi32, #hivm.address_space<ub>>)
                  outs(%sum : memref<8xi32, #hivm.address_space<ub>>)

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%sum : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%outTile : memref<8xi32, strided<[1], offset: ?>,
                                           #hivm.address_space<gm>>)
    return
  }
}
