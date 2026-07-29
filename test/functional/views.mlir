// View arithmetic, modelled on the addressing real post-pass kernels use:
// reinterpret_cast on a dynamically shaped GM argument, subview of a subview,
// and a rank-reducing subview.
//
// The reinterpret_cast cases matter: MLIR keeps the source's base pointer and
// *sets* the result's offset, so chaining two casts must not accumulate the
// offsets. Getting that wrong reads the right values from the wrong place and
// is invisible in any test that only ever casts an argument once.

// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.
// RUN: od -An -td4 -j128 -N32 %t.arg1.npy | FileCheck %s --check-prefix=DATA

//   out[0] = in[3]                        reinterpret_cast offset 3
//   out[1] = in[5]                        cast offset 3, then cast offset 5:
//                                         absolute, so 5 and not 8
//   out[2] = in[2+4]                      subview[2] of subview[4]
//   out[3] = in[9]                        rank-reducing subview of a 2-D view
//   out[4] = in[1]                        cast off a subview rebases to the
//                                         allocation, so not in[4+1]
//   out[5] = in[7]                        memref.load through a strided view
//   out[6] = in[12]
//   out[7] = in[15]
// DATA:      3 5 6 9
// DATA-NEXT: 1 7 12 15

module {
  func.func @views(%in: memref<16xi32, #hivm.address_space<gm>>,
                   %out: memref<8xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %c0 = arith.constant 0 : index

    // --- reinterpret_cast, then a second one: offsets must not accumulate ---
    %r3 = memref.reinterpret_cast %in to offset: [3], sizes: [1], strides: [1]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<1xi32, strided<[1], offset: 3>, #hivm.address_space<gm>>
    %r5 = memref.reinterpret_cast %r3 to offset: [5], sizes: [1], strides: [1]
        : memref<1xi32, strided<[1], offset: 3>, #hivm.address_space<gm>>
       to memref<1xi32, strided<[1], offset: 5>, #hivm.address_space<gm>>
    %v0 = memref.load %r3[%c0] : memref<1xi32, strided<[1], offset: 3>,
                                        #hivm.address_space<gm>>
    %v1 = memref.load %r5[%c0] : memref<1xi32, strided<[1], offset: 5>,
                                        #hivm.address_space<gm>>

    // --- subview of a subview: these *do* compose ---
    %s4 = memref.subview %in[4] [8] [1]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<8xi32, strided<[1], offset: 4>, #hivm.address_space<gm>>
    %s2 = memref.subview %s4[2] [4] [1]
        : memref<8xi32, strided<[1], offset: 4>, #hivm.address_space<gm>>
       to memref<4xi32, strided<[1], offset: 6>, #hivm.address_space<gm>>
    %v2 = memref.load %s2[%c0] : memref<4xi32, strided<[1], offset: 6>,
                                        #hivm.address_space<gm>>

    // --- rank-reducing subview of a 2-D reinterpretation: row 2, col 1 ---
    %m = memref.reinterpret_cast %in to offset: [0], sizes: [4, 4],
                                 strides: [4, 1]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<4x4xi32, strided<[4, 1]>, #hivm.address_space<gm>>
    %row = memref.subview %m[2, 1] [1, 1] [1, 1]
        : memref<4x4xi32, strided<[4, 1]>, #hivm.address_space<gm>>
       to memref<1xi32, strided<[1], offset: 9>, #hivm.address_space<gm>>
    %v3 = memref.load %row[%c0] : memref<1xi32, strided<[1], offset: 9>,
                                         #hivm.address_space<gm>>

    // --- cast off a subview: the cast rebases to the *allocation*, so
    //     offset 1 lands on in[1] and not on in[4+1] ---
    %sub4 = memref.subview %in[4] [8] [1]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<8xi32, strided<[1], offset: 4>, #hivm.address_space<gm>>
    %fromsub = memref.reinterpret_cast %sub4 to offset: [1], sizes: [1],
                                       strides: [1]
        : memref<8xi32, strided<[1], offset: 4>, #hivm.address_space<gm>>
       to memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
    %v4 = memref.load %fromsub[%c0] : memref<1xi32, strided<[1], offset: 1>,
                                             #hivm.address_space<gm>>

    // --- strided load: stride 5 from offset 7 gives in[7], in[12] ---
    %strided = memref.reinterpret_cast %in to offset: [7], sizes: [2],
                                       strides: [5]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<2xi32, strided<[5], offset: 7>, #hivm.address_space<gm>>
    %c1 = arith.constant 1 : index
    %v5 = memref.load %strided[%c0] : memref<2xi32, strided<[5], offset: 7>,
                                             #hivm.address_space<gm>>
    %v6 = memref.load %strided[%c1] : memref<2xi32, strided<[5], offset: 7>,
                                             #hivm.address_space<gm>>
    %c15 = arith.constant 15 : index
    %v7 = memref.load %in[%c15] : memref<16xi32, #hivm.address_space<gm>>

    memref.store %v0, %out[%c0] : memref<8xi32, #hivm.address_space<gm>>
    memref.store %v1, %out[%c1] : memref<8xi32, #hivm.address_space<gm>>
    %c2 = arith.constant 2 : index
    memref.store %v2, %out[%c2] : memref<8xi32, #hivm.address_space<gm>>
    %c3 = arith.constant 3 : index
    memref.store %v3, %out[%c3] : memref<8xi32, #hivm.address_space<gm>>
    %c4 = arith.constant 4 : index
    memref.store %v4, %out[%c4] : memref<8xi32, #hivm.address_space<gm>>
    %c5 = arith.constant 5 : index
    memref.store %v5, %out[%c5] : memref<8xi32, #hivm.address_space<gm>>
    %c6 = arith.constant 6 : index
    memref.store %v6, %out[%c6] : memref<8xi32, #hivm.address_space<gm>>
    %c7 = arith.constant 7 : index
    memref.store %v7, %out[%c7] : memref<8xi32, #hivm.address_space<gm>>
    return
  }
}
