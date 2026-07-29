// Narrow float formats. Routing arithmetic through APFloat rather than
// `float` is what keeps f16 rounding and ties correct, and what lets bf16
// exist at all - it has an 8-bit mantissa that `float` silently widens.
//
// f16 leg:  out0 = in + in over arange(8), checked as raw bit patterns
//           0 -> 0x0000, 2 -> 0x4000, 4 -> 0x4400, 6 -> 0x4600,
//           8 -> 0x4800, 10 -> 0x4900, 12 -> 0x4a00, 14 -> 0x4b00
// bf16 leg: a load / copy / store round trip must preserve the bit pattern
//           of arange(8) exactly: 0, 1, 2, 3, 4, 5, 6, 7 as bf16.

// RUN: npuir-interp %s --sched=lazy --args=arange,arange,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --args=arange,arange,zeros,zeros --out=%t. && \
// RUN:   od -An -tx2 -j128 -N16 %t.arg2.npy | FileCheck %s --check-prefix=F16
// RUN: npuir-interp %s --sched=lazy --args=arange,arange,zeros,zeros --out=%t. && \
// RUN:   od -An -tx2 -j128 -N16 %t.arg3.npy | FileCheck %s --check-prefix=BF16

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// F16:  0000 4000 4400 4600 4800 4900 4a00 4b00
// BF16: 0000 3f80 4000 4040 4080 40a0 40c0 40e0

module {
  func.func @narrow(%f16in: memref<8xf16, #hivm.address_space<gm>>,
                    %bf16in: memref<8xbf16, #hivm.address_space<gm>>,
                    %f16out: memref<8xf16, #hivm.address_space<gm>>,
                    %bf16out: memref<8xbf16, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %a = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<8xbf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%f16in : memref<8xf16, #hivm.address_space<gm>>)
                  outs(%a : memref<8xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%bf16in : memref<8xbf16, #hivm.address_space<gm>>)
                  outs(%b : memref<8xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    %sum = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%a, %a : memref<8xf16, #hivm.address_space<ub>>,
                                memref<8xf16, #hivm.address_space<ub>>)
                  outs(%sum : memref<8xf16, #hivm.address_space<ub>>)
    // vadd does not take bf16; a UB-to-UB copy still exercises the bf16
    // load/store path end to end.
    %bcopy = memref.alloc() : memref<8xbf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%b : memref<8xbf16, #hivm.address_space<ub>>)
                  outs(%bcopy : memref<8xbf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.store ins(%sum : memref<8xf16, #hivm.address_space<ub>>)
                   outs(%f16out : memref<8xf16, #hivm.address_space<gm>>)
    hivm.hir.store ins(%bcopy : memref<8xbf16, #hivm.address_space<ub>>)
                   outs(%bf16out : memref<8xbf16, #hivm.address_space<gm>>)
    return
  }
}
