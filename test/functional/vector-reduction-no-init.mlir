// RUN: npuir-interp %s --sched=inorder --args=zeros --out=%t.
// RUN: od -An -tfF -j128 -N4 %t.arg0.npy | FileCheck %s

// CHECK: 1.000000e+01

module {
  func.func @reduction_no_init(
      %out: memref<1xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %source = arith.constant dense<[[1.0, 2.0, 3.0, 4.0]]>
        : vector<1x4xf32>
    // This value must not participate in a reduction marked no-init.
    %poison = arith.constant dense<-1000.0> : vector<1xf32>
    %sum = vector.multi_reduction <add>, %source, %poison
        {withoutInitMergeOp} [1] : vector<1x4xf32> to vector<1xf32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]}
        : vector<1xf32>, memref<1xf32, #hivm.address_space<gm>>
    return
  }
}
