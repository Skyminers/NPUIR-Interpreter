// expand_shape/collapse_shape preserve a padded outer stride when the result
// types spell that layout out explicitly.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t.
// RUN: od -An -td4 -j128 -N32 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// DATA: 0 1 2 3
// DATA-NEXT: 5 6 7 8

module {
  func.func @strided_reshape(
      %in: memref<10xi32, #hivm.address_space<gm>>,
      %out: memref<2x4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %view = memref.reinterpret_cast %in to offset: [0], sizes: [2, 4],
        strides: [5, 1] : memref<10xi32, #hivm.address_space<gm>> to
        memref<2x4xi32, strided<[5, 1]>, #hivm.address_space<gm>>
    %expanded = memref.expand_shape %view [[0], [1, 2]] output_shape [2, 2, 2] :
        memref<2x4xi32, strided<[5, 1]>, #hivm.address_space<gm>> into
        memref<2x2x2xi32, strided<[5, 2, 1]>, #hivm.address_space<gm>>
    %collapsed = memref.collapse_shape %expanded [[0], [1, 2]] :
        memref<2x2x2xi32, strided<[5, 2, 1]>, #hivm.address_space<gm>> into
        memref<2x4xi32, strided<[5, 1]>, #hivm.address_space<gm>>
    %buf = memref.alloc() : memref<2x4xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapsed : memref<2x4xi32, strided<[5, 1]>,
                                          #hivm.address_space<gm>>)
                  outs(%buf : memref<2x4xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%buf : memref<2x4xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<2x4xi32, #hivm.address_space<gm>>)
    return
  }
}
