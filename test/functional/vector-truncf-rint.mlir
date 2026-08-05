// RUN: npuir-interp %s --sched=inorder --args=0.391605616,zeros --out=%t.
// RUN: od -An -tx2 -j128 -N2 %t.arg1.npy | FileCheck %s

// `rint` controls the format conversion rounding.  It must not be confused
// with the `round` mode merely because the attribute type is named
// `round_mode`; 0.391605616 narrows to the f16 bit pattern 0x3644, not zero.
// CHECK: 3644

module {
  func.func @truncf_rint(
      %in: memref<1xf32, #hivm.address_space<gm>>,
      %out: memref<1xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.000000e+00 : f32
    %value = vector.transfer_read %in[%c0], %pad {in_bounds = [true]}
        : memref<1xf32, #hivm.address_space<gm>>, vector<1xf32>
    %half = arith.truncf %value {
      enable_saturate = false,
      round_mode = #hfusion.round_mode<rint>,
      unsigned_mode = #hfusion.unsigned_mode<si2si>
    } : vector<1xf32> to vector<1xf16>
    vector.transfer_write %half, %out[%c0] {in_bounds = [true]}
        : vector<1xf16>, memref<1xf16, #hivm.address_space<gm>>
    return
  }
}
