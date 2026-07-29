// vector.transfer_read / transfer_write with a non-identity permutation_map
// and with out-of-bounds lanes.
//
// Real vectorized bodies read a scalar row and splat it across a vector with
// `permutation_map = affine_map<(d0) -> (d0, 0)>`: the second vector
// dimension is a broadcast that does not move in memory. A minor-identity
// map (reading a 1-D vector out of a 2-D memref) is the other common shape.
//
// in = arange(8) f32, viewed as 2x4:
//     0 1 2 3
//     4 5 6 7
//   out0: row 1 read with a minor-identity map -> 4 5 6 7
//   out1: element [1,0] broadcast across 4 lanes -> 4 4 4 4
//   out2: an in_bounds=false read of 6 lanes starting at column 2 of row 0,
//         padded with -1 -> 2 3 -1 -1 -1 -1

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros --out=%t.
// RUN: od -An -tf4 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=ROW
// RUN: od -An -tf4 -j128 -N16 %t.arg2.npy | FileCheck %s --check-prefix=BCAST
// RUN: od -An -tf4 -j128 -N24 %t.arg3.npy | FileCheck %s --check-prefix=PAD

// CHECK-NOT: MISSING SYNC
// ROW:   4.000000e+00 5.000000e+00 6.000000e+00 7.000000e+00
// BCAST: 4.000000e+00 4.000000e+00 4.000000e+00 4.000000e+00
// PAD:   2.000000e+00 3.000000e+00 -1.000000e+00 -1.000000e+00
// PAD-NEXT: -1.000000e+00 -1.000000e+00

#minor = affine_map<(d0, d1) -> (d1)>
#bcast = affine_map<(d0, d1) -> (0)>

module {
  func.func @perm(%in: memref<8xf32, #hivm.address_space<gm>>,
                  %row: memref<4xf32, #hivm.address_space<gm>>,
                  %bcast: memref<4xf32, #hivm.address_space<gm>>,
                  %padded: memref<6xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.000000e+00 : f32
    %minusone = arith.constant -1.000000e+00 : f32

    %flat = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%flat : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %m = memref.expand_shape %flat [[0, 1]] output_shape [2, 4]
        : memref<8xf32, #hivm.address_space<ub>>
       into memref<2x4xf32, #hivm.address_space<ub>>

    %d0 = memref.alloc() : memref<4xf32, #hivm.address_space<ub>>
    %d1 = memref.alloc() : memref<4xf32, #hivm.address_space<ub>>
    %d2 = memref.alloc() : memref<6xf32, #hivm.address_space<ub>>

    // Minor identity: the vector follows the second memref dimension.
    %r = vector.transfer_read %m[%c1, %c0], %zero
        {in_bounds = [true], permutation_map = #minor}
        : memref<2x4xf32, #hivm.address_space<ub>>, vector<4xf32>
    vector.transfer_write %r, %d0[%c0] {in_bounds = [true]}
        : vector<4xf32>, memref<4xf32, #hivm.address_space<ub>>

    // Broadcast: no memref dimension varies, so every lane is m[1, 0].
    %b = vector.transfer_read %m[%c1, %c0], %zero
        {in_bounds = [true], permutation_map = #bcast}
        : memref<2x4xf32, #hivm.address_space<ub>>, vector<4xf32>
    vector.transfer_write %b, %d1[%c0] {in_bounds = [true]}
        : vector<4xf32>, memref<4xf32, #hivm.address_space<ub>>

    // Past the end of the row: the masked-off lanes take the padding value.
    %p = vector.transfer_read %m[%c0, %c2], %minusone
        {in_bounds = [false], permutation_map = #minor}
        : memref<2x4xf32, #hivm.address_space<ub>>, vector<6xf32>
    vector.transfer_write %p, %d2[%c0] {in_bounds = [true]}
        : vector<6xf32>, memref<6xf32, #hivm.address_space<ub>>

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%d0 : memref<4xf32, #hivm.address_space<ub>>)
                   outs(%row : memref<4xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d1 : memref<4xf32, #hivm.address_space<ub>>)
                   outs(%bcast : memref<4xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d2 : memref<6xf32, #hivm.address_space<ub>>)
                   outs(%padded : memref<6xf32, #hivm.address_space<gm>>)
    return
  }
}
