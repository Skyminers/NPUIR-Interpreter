// `vsort` has to hand `std::stable_sort` a strict weak ordering, and `<` on
// floats is not one: every comparison against a NaN is false, so with a NaN in
// the input the comparator becomes inconsistent and the sort is undefined -
// in practice a scrambled result or a walk off the end of the range. Ordering
// by the IEEE total order instead puts NaNs at one end and leaves everything
// else where `<` would.
//
// in  = [3, NaN, 1, -inf, NaN, 0, inf, -2]
// out = [-inf, -2, 0, 1, 3, inf, NaN, NaN] and the indices that produced it.
// The two NaNs keep their input order because the sort is stable, so the index
// output is reproducible rather than "some permutation".

// RUN: npuir-interp %s --sched=lazy --args=%S/Inputs/sort_nan_f32.npy,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -tf4 -j128 %t.arg1.npy | FileCheck %s --check-prefix=VALUES
// RUN: od -v -An -td4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=INDICES
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/sort_nan_f32.npy,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// VALUES:       -inf -2.000000e+00 0.000000e+00 1.000000e+00
// VALUES-NEXT:  3.000000e+00 inf nan nan
// INDICES:      3 7 5 2
// INDICES-NEXT: 0 6 1 4

module {
  func.func @sortnan(%in: memref<8xf32, #hivm.address_space<gm>>,
                     %values: memref<8xf32, #hivm.address_space<gm>>,
                     %indices: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %a = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %v = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %i = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%a : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsort ins(%a : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%v, %i : memref<8xf32, #hivm.address_space<ub>>,
                                 memref<8xi32, #hivm.address_space<ub>>)
                   descending = false sort_axis = 0
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%v : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%values : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%i : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%indices : memref<8xi32, #hivm.address_space<gm>>)
    return
  }
}
