// The prefix-scan family. `cum_dims` names the axis to scan and `reverse`
// runs the scan from the far end, so element i accumulates everything at or
// after it instead of at or before it.
//
//   arg1  cumsum  of [0..7]              prefix sums
//   arg2  cumsum  of [0..7], reversed    suffix sums
//   arg3  cumprod of [1..8]              factorials, kept small enough for i32
//   arg4  cummax  of the reverse         a running maximum that never moves
//   arg5  cummin  of [0..7]              a running minimum that never moves
//   arg6  cumsum  of a 2x4 along dim 1: prefix sums within each row

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros,zeros,zeros,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg6.npy %t.inorder.arg6.npy
// RUN: od -An -td4 -j128 %t.arg1.npy | FileCheck %s --check-prefix=SUM
// RUN: od -An -td4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=RSUM
// RUN: od -An -td4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=PROD
// RUN: od -v -An -td4 -j128 %t.arg4.npy | FileCheck %s --check-prefix=MAX
// RUN: od -v -An -td4 -j128 %t.arg5.npy | FileCheck %s --check-prefix=MIN
// RUN: od -An -td4 -j128 %t.arg6.npy | FileCheck %s --check-prefix=SUM2D

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// SUM:        0 1 3 6
// SUM-NEXT:   10 15 21 28
// RSUM:       28 28 27 25
// RSUM-NEXT:  22 18 13 7
// PROD:       1 2 6 24
// PROD-NEXT:  120 720 5040 40320
// MAX:        7 7 7 7
// MAX-NEXT:   7 7 7 7
// MIN:        0 0 0 0
// MIN-NEXT:   0 0 0 0
// Row-wise scan of [[0,1,2,3],[4,5,6,7]]: each row scanned independently.
// SUM2D:      0 1 3 6
// SUM2D-NEXT: 4 9 15 22

module {
  func.func @scans(%in: memref<8xi32, #hivm.address_space<gm>>,
                   %osum: memref<8xi32, #hivm.address_space<gm>>,
                   %orsum: memref<8xi32, #hivm.address_space<gm>>,
                   %oprod: memref<8xi32, #hivm.address_space<gm>>,
                   %omax: memref<8xi32, #hivm.address_space<gm>>,
                   %omin: memref<8xi32, #hivm.address_space<gm>>,
                   %osum2d: memref<2x4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %one = arith.constant 1 : i32

    %a = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xi32, #hivm.address_space<gm>>)
                  outs(%a : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    // cumprod over 0..7 would be all zeros; shift to 1..8.
    %plus1 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%a, %one : memref<8xi32, #hivm.address_space<ub>>, i32)
                  outs(%plus1 : memref<8xi32, #hivm.address_space<ub>>)
    %rev = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vflip ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%rev : memref<8xi32, #hivm.address_space<ub>>)
                   flip_axis = 0
    hivm.hir.pipe_barrier[<PIPE_V>]

    %sum = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vcumsum ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%sum : memref<8xi32, #hivm.address_space<ub>>)
                     cum_dims = [0] reverse = false
    %rsum = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vcumsum ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%rsum : memref<8xi32, #hivm.address_space<ub>>)
                     cum_dims = [0] reverse = true
    %prod = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vcumprod ins(%plus1 : memref<8xi32, #hivm.address_space<ub>>)
                      outs(%prod : memref<8xi32, #hivm.address_space<ub>>)
                      cum_dims = [0] reverse = false
    %max = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vcummax ins(%rev : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%max : memref<8xi32, #hivm.address_space<ub>>)
                     cum_dims = [0] reverse = false
    %min = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vcummin ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%min : memref<8xi32, #hivm.address_space<ub>>)
                     cum_dims = [0] reverse = false

    %m = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    %s2 = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    %a2 = memref.reinterpret_cast %a to offset: [0], sizes: [2, 4],
              strides: [4, 1]
        : memref<8xi32, #hivm.address_space<ub>>
       to memref<2x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a2 : memref<2x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>)
                  outs(%m : memref<2x4xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.vcumsum ins(%m : memref<2x4xi32, #hivm.address_space<ub>>)
                     outs(%s2 : memref<2x4xi32, #hivm.address_space<ub>>)
                     cum_dims = [1] reverse = false

    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.store ins(%sum : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%osum : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%rsum : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%orsum : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%prod : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%oprod : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%max : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%omax : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%min : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%omin : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%s2 : memref<2x4xi32, #hivm.address_space<ub>>)
                   outs(%osum2d : memref<2x4xi32, #hivm.address_space<gm>>)
    return
  }
}
