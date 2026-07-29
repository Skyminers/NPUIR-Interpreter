// The ops that move elements around rather than compute on them. Each one is
// fed arange(8) (or its reverse) so the expected output can be read off by
// eye, and every result goes out to its own GM buffer.
//
//   arg1  vconcat dim 0 of [0..7] with its reverse
//   arg2  vpad low 1 high 3 with 99
//   arg3  vgather of [0..7] at the reversed indices
//   arg4  vinterleave of [0..7] and its reverse
//   arg5  vdeinterleave channel 1 of that interleaving, i.e. the reverse back
//   arg6  vsort ascending of the reverse
//   arg7  the indices vsort reports for that
//   arg8  vflip of a 2x4, flipped along dim 1

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg4.npy %t.inorder.arg4.npy
// RUN: cmp %t.arg8.npy %t.inorder.arg8.npy
// RUN: od -An -td4 -j128 %t.arg1.npy | FileCheck %s --check-prefix=CONCAT
// RUN: od -An -td4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=PAD
// RUN: od -An -td4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=GATHER
// RUN: od -An -td4 -j128 %t.arg4.npy | FileCheck %s --check-prefix=ILV
// RUN: od -An -td4 -j128 %t.arg5.npy | FileCheck %s --check-prefix=DEILV
// RUN: od -An -td4 -j128 %t.arg6.npy | FileCheck %s --check-prefix=SORTV
// RUN: od -An -td4 -j128 %t.arg7.npy | FileCheck %s --check-prefix=SORTI
// RUN: od -An -td4 -j128 %t.arg8.npy | FileCheck %s --check-prefix=FLIP2D

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// CONCAT:      0 1 2 3
// CONCAT-NEXT: 4 5 6 7
// CONCAT-NEXT: 7 6 5 4
// CONCAT-NEXT: 3 2 1 0
// PAD:         99 0 1 2
// PAD-NEXT:    3 4 5 6
// PAD-NEXT:    7 99 99 99
// GATHER:      7 6 5 4
// GATHER-NEXT: 3 2 1 0
// ILV:         0 7 1 6
// ILV-NEXT:    2 5 3 4
// ILV-NEXT:    4 3 5 2
// ILV-NEXT:    6 1 7 0
// DEILV:       7 6 5 4
// DEILV-NEXT:  3 2 1 0
// SORTV:       0 1 2 3
// SORTV-NEXT:  4 5 6 7
// SORTI:       7 6 5 4
// SORTI-NEXT:  3 2 1 0
// FLIP2D:      3 2 1 0
// FLIP2D-NEXT: 7 6 5 4

module {
  func.func @reorder(%in: memref<8xi32, #hivm.address_space<gm>>,
                     %cat: memref<16xi32, #hivm.address_space<gm>>,
                     %pad: memref<12xi32, #hivm.address_space<gm>>,
                     %gat: memref<8xi32, #hivm.address_space<gm>>,
                     %ilv: memref<16xi32, #hivm.address_space<gm>>,
                     %deilv: memref<8xi32, #hivm.address_space<gm>>,
                     %sortv: memref<8xi32, #hivm.address_space<gm>>,
                     %sorti: memref<8xi32, #hivm.address_space<gm>>,
                     %flip2d: memref<2x4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %pv = arith.constant 99 : i32

    %a = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %rev = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xi32, #hivm.address_space<gm>>)
                  outs(%a : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    hivm.hir.vflip ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%rev : memref<8xi32, #hivm.address_space<ub>>)
                   flip_axis = 0

    // The reversed buffer feeds the ops below, so PIPE_V has to retire it
    // first even though everything here is on the same pipe - vgather reads
    // it as an index vector, and the interpreter models that read exactly.
    hivm.hir.pipe_barrier[<PIPE_V>]

    %c16 = memref.alloc() : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.vconcat dim(0)
        ins(%a, %rev : memref<8xi32, #hivm.address_space<ub>>,
                       memref<8xi32, #hivm.address_space<ub>>)
        outs(%c16 : memref<16xi32, #hivm.address_space<ub>>)

    %p12 = memref.alloc() : memref<12xi32, #hivm.address_space<ub>>
    hivm.hir.vpad ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                  outs(%p12 : memref<12xi32, #hivm.address_space<ub>>)
                  low[1] high[3] pad_value %pv : i32

    %g8 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vgather ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                     indices(%rev : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%g8 : memref<8xi32, #hivm.address_space<ub>>)

    %i16 = memref.alloc() : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.vinterleave
        ins(%a, %rev : memref<8xi32, #hivm.address_space<ub>>,
                       memref<8xi32, #hivm.address_space<ub>>)
        outs(%i16 : memref<16xi32, #hivm.address_space<ub>>)
        interleave_channel_nums = 2

    %sv = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %si = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vsort ins(%rev : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%sv, %si : memref<8xi32, #hivm.address_space<ub>>,
                                   memref<8xi32, #hivm.address_space<ub>>)
                   descending = false sort_axis = 0

    hivm.hir.pipe_barrier[<PIPE_V>]

    %d0 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %d1 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vdeinterleave ins(%i16 : memref<16xi32, #hivm.address_space<ub>>)
        outs(%d0, %d1 : memref<8xi32, #hivm.address_space<ub>>,
                        memref<8xi32, #hivm.address_space<ub>>)
        channel_num = 2

    // A rank-2 flip, to show `flip_axis` is not hardcoded to the last dim of
    // a 1-D buffer.
    %m = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    %f2 = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    %a2 = memref.reinterpret_cast %a to offset: [0], sizes: [2, 4],
              strides: [4, 1]
        : memref<8xi32, #hivm.address_space<ub>>
       to memref<2x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a2 : memref<2x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>)
                  outs(%m : memref<2x4xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.vflip ins(%m : memref<2x4xi32, #hivm.address_space<ub>>)
                   outs(%f2 : memref<2x4xi32, #hivm.address_space<ub>>)
                   flip_axis = 1

    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.store ins(%c16 : memref<16xi32, #hivm.address_space<ub>>)
                   outs(%cat : memref<16xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%p12 : memref<12xi32, #hivm.address_space<ub>>)
                   outs(%pad : memref<12xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%g8 : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%gat : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%i16 : memref<16xi32, #hivm.address_space<ub>>)
                   outs(%ilv : memref<16xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d1 : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%deilv : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%sv : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%sortv : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%si : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%sorti : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%f2 : memref<2x4xi32, #hivm.address_space<ub>>)
                   outs(%flip2d : memref<2x4xi32, #hivm.address_space<gm>>)
    return
  }
}
