// Negative test for stage-1 layout checking. nd2nz leaves its L1 destination
// in fractal nZ order; the convert_layout that reads it back claims the
// source is still ND, which is the shape of a lost layout conversion.
//
// Data stays in logical ND order inside the interpreter, so the *values*
// would look perfectly fine here - only the tag catches this. That is the
// cheap half of the plan's two-stage layout strategy: no fractal addressing,
// but a producer/consumer disagreement is caught for free.

// RUN: not npuir-interp %s --sched=lazy --args=arange,zeros 2>&1 | FileCheck %s

// CHECK: error: layout mismatch: convert_layout declares its source is ND
// CHECK-SAME: but the producer left it as nZ

module {
  func.func @layout(%in: memref<128x128xf16, #hivm.address_space<gm>>,
                    %out: memref<128x128xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %l1 = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%in : memref<128x128xf16, #hivm.address_space<gm>>)
                   outs(%l1 : memref<128x128xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]

    // Claims ND, but nd2nz produced nZ.
    %a = hivm.hir.convert_layout %l1 output_shape [128, 128]
        {srcLayout = #hivm.data_layout<ND>,
         dstLayout = #hivm.data_layout<dotA_ND, transpose = false>}
        : (memref<128x128xf16, #hivm.address_space<cbuf>>)
       -> memref<128x128xf16, #hivm.address_space<cbuf>>
    return
  }
}
