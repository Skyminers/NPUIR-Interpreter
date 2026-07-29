// Casts applied to vector values, which is how real vectorized bodies spell
// them: `arith.truncf ... : vector<1x64xf32> to vector<1x64xbf16>` appears
// verbatim in post-pass IR.
//
// Vector lifting feeds the scalar handlers one lane at a time, but the
// operand and result *types* stay `vector<...>`. A handler that asks whether
// its result type is a float gets "no" for `vector<8xf16>` and silently falls
// through to the integer path, so these must be checked as bit patterns
// rather than trusted to look plausible.
//
// in = arange(8) f32.
//   out0 = truncf to f16   : 0 1 2 3 4 5 6 7  -> 0000 3c00 4000 4200 4400 4500 4600 4700
//   out1 = truncf to bf16  :                  -> 0000 3f80 4000 4040 4080 40a0 40c0 40e0
//   out2 = fptosi to i32   : 0 1 2 3 4 5 6 7
//   out3 = bitcast to i32  : the raw f32 bit patterns of 0..7

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros,zeros --out=%t.
// RUN: od -An -tx2 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=F16
// RUN: od -An -tx2 -j128 -N16 %t.arg2.npy | FileCheck %s --check-prefix=BF16
// RUN: od -An -td4 -j128 -N32 %t.arg3.npy | FileCheck %s --check-prefix=SI
// RUN: od -An -tx4 -j128 -N32 %t.arg4.npy | FileCheck %s --check-prefix=BITS

// CHECK-NOT: MISSING SYNC
// F16:  0000 3c00 4000 4200 4400 4500 4600 4700
// BF16: 0000 3f80 4000 4040 4080 40a0 40c0 40e0
// SI:        0 1 2 3
// SI-NEXT:   4 5 6 7
// BITS:      00000000 3f800000 40000000 40400000
// BITS-NEXT: 40800000 40a00000 40c00000 40e00000

module {
  func.func @casts(%in: memref<8xf32, #hivm.address_space<gm>>,
                   %f16out: memref<8xf16, #hivm.address_space<gm>>,
                   %bf16out: memref<8xbf16, #hivm.address_space<gm>>,
                   %siout: memref<8xi32, #hivm.address_space<gm>>,
                   %bitsout: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.000000e+00 : f32

    %src = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %d0 = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
    %d1 = memref.alloc() : memref<8xbf16, #hivm.address_space<ub>>
    %d2 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %d3 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%src : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    %v = vector.transfer_read %src[%c0], %pad {in_bounds = [true]}
        : memref<8xf32, #hivm.address_space<ub>>, vector<8xf32>
    %half = arith.truncf %v : vector<8xf32> to vector<8xf16>
    %bhalf = arith.truncf %v : vector<8xf32> to vector<8xbf16>
    %ints = arith.fptosi %v : vector<8xf32> to vector<8xi32>
    %bits = arith.bitcast %v : vector<8xf32> to vector<8xi32>
    vector.transfer_write %half, %d0[%c0] {in_bounds = [true]}
        : vector<8xf16>, memref<8xf16, #hivm.address_space<ub>>
    vector.transfer_write %bhalf, %d1[%c0] {in_bounds = [true]}
        : vector<8xbf16>, memref<8xbf16, #hivm.address_space<ub>>
    vector.transfer_write %ints, %d2[%c0] {in_bounds = [true]}
        : vector<8xi32>, memref<8xi32, #hivm.address_space<ub>>
    vector.transfer_write %bits, %d3[%c0] {in_bounds = [true]}
        : vector<8xi32>, memref<8xi32, #hivm.address_space<ub>>

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%d0 : memref<8xf16, #hivm.address_space<ub>>)
                   outs(%f16out : memref<8xf16, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d1 : memref<8xbf16, #hivm.address_space<ub>>)
                   outs(%bf16out : memref<8xbf16, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d2 : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%siout : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%d3 : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%bitsout : memref<8xi32, #hivm.address_space<gm>>)
    return
  }
}
