// A missing op handler must say so and stop. Reporting the gap plainly beats
// guessing: a wrong numerical answer from a silently skipped op is far more
// expensive to chase than a one-line "not supported".

// RUN: not npuir-interp %s --sched=inorder --args=zeros 2>&1 | FileCheck %s

// CHECK: error: unsupported op: hivm.hir.mix_matmul

module {
  func.func @macro(%gm: memref<16x16xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c16 = arith.constant 16 : index
    hivm.hir.mix_matmul ins(%gm, %gm : memref<16x16xf16, #hivm.address_space<gm>>,
                                        memref<16x16xf16, #hivm.address_space<gm>>)
                        outs(%gm : memref<16x16xf16, #hivm.address_space<gm>>)
    return
  }
}
