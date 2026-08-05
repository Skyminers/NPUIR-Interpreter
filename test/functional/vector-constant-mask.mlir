// RUN: npuir-interp %s --args=zeros --out=%t.
// RUN: od -An -tu1 -j128 -N4 %t.arg0.npy | FileCheck %s

// CHECK: 1 1 0 0

module {
  func.func @constant_mask(%out: memref<4xi8, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %mask = vector.constant_mask [2] : vector<4xi1>
    %m0 = vector.extract %mask[0] : i1 from vector<4xi1>
    %m1 = vector.extract %mask[1] : i1 from vector<4xi1>
    %m2 = vector.extract %mask[2] : i1 from vector<4xi1>
    %m3 = vector.extract %mask[3] : i1 from vector<4xi1>
    %v0 = arith.extui %m0 : i1 to i8
    %v1 = arith.extui %m1 : i1 to i8
    %v2 = arith.extui %m2 : i1 to i8
    %v3 = arith.extui %m3 : i1 to i8
    memref.store %v0, %out[%c0] : memref<4xi8, #hivm.address_space<gm>>
    memref.store %v1, %out[%c1] : memref<4xi8, #hivm.address_space<gm>>
    memref.store %v2, %out[%c2] : memref<4xi8, #hivm.address_space<gm>>
    memref.store %v3, %out[%c3] : memref<4xi8, #hivm.address_space<gm>>
    return
  }
}
