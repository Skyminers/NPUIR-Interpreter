// A short ND tail is padded to a complete NZ tile. Converting only the first
// two logical rows back to ND must preserve the valid row and expose a zero
// row, not poison left in the padded part of L1.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -td2 -j128 -N64 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA: 0 1 2 3 4 5 6 7
// DATA-NEXT: 8 9 10 11 12 13 14 15
// DATA-NEXT: 0 0 0 0 0 0 0 0
// DATA-NEXT: 0 0 0 0 0 0 0 0

module {
  func.func @tail(%in: memref<1x16xi16, #hivm.address_space<gm>>,
                  %out: memref<2x16xi16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %tile = memref.alloc() : memref<1x2x16x16xi16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz ins(%in : memref<1x16xi16, #hivm.address_space<gm>>)
                   outs(%tile : memref<1x2x16x16xi16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.nz2nd ins(%tile : memref<1x2x16x16xi16, #hivm.address_space<cbuf>>)
                   outs(%out : memref<2x16xi16, #hivm.address_space<gm>>)
    return
  }
}
