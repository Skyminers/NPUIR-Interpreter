// Accesses whose addresses are data rather than shape. The interpreter cannot
// know at issue time which bytes these reach, so each declares the whole
// source (or destination) buffer to the race detector - conservative in the
// direction that over-reports sharing rather than missing it.
//
//   arg1  indirect_load of [0..7] through reversed offsets, mask all-true
//   arg2  indirect_store of that back through the same offsets: the identity
//   arg3  indirect_load with a mask that is false on the odd lanes, so those
//         come from `other` (-1) instead
//   arg4  stride_load  dst[i][j] = in[1 + 4i + 2j]
//   arg5  stride_store out[4i + j] = dst[i][j], the rest untouched
//   arg6  stride_load where numels is smaller than the destination, so the
//         tail is filled with `other` rather than read out of bounds

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros,zeros,zeros,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros,zeros,zeros,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg4.npy %t.inorder.arg4.npy
// RUN: cmp %t.arg5.npy %t.inorder.arg5.npy
// RUN: od -v -An -td4 -j128 %t.arg1.npy | FileCheck %s --check-prefix=ILOAD
// RUN: od -v -An -td4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=ISTORE
// RUN: od -v -An -td4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=MASKED
// RUN: od -v -An -td4 -j128 %t.arg4.npy | FileCheck %s --check-prefix=SLOAD
// RUN: od -v -An -td4 -j128 %t.arg5.npy | FileCheck %s --check-prefix=SSTORE
// RUN: od -v -An -td4 -j128 %t.arg6.npy | FileCheck %s --check-prefix=SPAD

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// ILOAD:        7 6 5 4
// ILOAD-NEXT:   3 2 1 0
// ISTORE:       0 1 2 3
// ISTORE-NEXT:  4 5 6 7
// MASKED:       7 -1 5 -1
// MASKED-NEXT:  3 -1 1 -1
// SLOAD:        1 3 5 7
// SLOAD-NEXT:   9 11
// SSTORE:       1 3 0 0
// SSTORE-NEXT:  5 7 0 0
// SSTORE-NEXT:  9 11 0 0
// SSTORE-NEXT:  0 0 0 0
// numels [2, 1] over a 3x2 destination: only dst[0][0] and dst[1][0] are read.
// SPAD:         1 -1 5 -1
// SPAD-NEXT:    -1 -1

module {
  func.func @indirect(%in: memref<16xi32, #hivm.address_space<gm>>,
                      %iload: memref<8xi32, #hivm.address_space<gm>>,
                      %istore: memref<8xi32, #hivm.address_space<gm>>,
                      %masked: memref<8xi32, #hivm.address_space<gm>>,
                      %sload: memref<3x2xi32, #hivm.address_space<gm>>,
                      %sstore: memref<16xi32, #hivm.address_space<gm>>,
                      %spad: memref<3x2xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %i2 = arith.constant 2 : i32
    %i3 = arith.constant 3 : i32
    %i4 = arith.constant 4 : i32
    %neg1 = arith.constant -1 : i32
    %true8 = arith.constant 1 : i8
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c7 = arith.constant 7 : index
    %false8 = arith.constant 0 : i8

    %a = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %in8 = memref.reinterpret_cast %in to offset: [0], sizes: [8], strides: [1]
        : memref<16xi32, #hivm.address_space<gm>>
       to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%in8 : memref<8xi32, strided<[1]>, #hivm.address_space<gm>>)
                  outs(%a : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    %off = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %mask = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %other = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vflip ins(%a : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%off : memref<8xi32, #hivm.address_space<ub>>)
                   flip_axis = 0
    hivm.hir.vbrc ins(%true8 : i8)
                  outs(%mask : memref<8xi8, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%neg1 : i32)
                  outs(%other : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]

    %d = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.indirect_load
        ins(%a : memref<8xi32, #hivm.address_space<ub>>,
            %off : memref<8xi32, #hivm.address_space<ub>>,
            %mask : memref<8xi8, #hivm.address_space<ub>>,
            %other : memref<8xi32, #hivm.address_space<ub>>)
        outs(%d : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]

    %sc = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.indirect_store
        ins(%d : memref<8xi32, #hivm.address_space<ub>>,
            %off : memref<8xi32, #hivm.address_space<ub>>)
        outs(%sc : memref<8xi32, #hivm.address_space<ub>>)

    // Turn the odd lanes of the mask off, so those come from `other`.
    memref.store %false8, %mask[%c1] : memref<8xi8, #hivm.address_space<ub>>
    memref.store %false8, %mask[%c3] : memref<8xi8, #hivm.address_space<ub>>
    memref.store %false8, %mask[%c5] : memref<8xi8, #hivm.address_space<ub>>
    memref.store %false8, %mask[%c7] : memref<8xi8, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    %dm = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.indirect_load
        ins(%a : memref<8xi32, #hivm.address_space<ub>>,
            %off : memref<8xi32, #hivm.address_space<ub>>,
            %mask : memref<8xi8, #hivm.address_space<ub>>,
            %other : memref<8xi32, #hivm.address_space<ub>>)
        outs(%dm : memref<8xi32, #hivm.address_space<ub>>)

    %sl = memref.alloc() : memref<3x2xi32, #hivm.address_space<ub>>
    hivm.hir.stride_load ins(%in : memref<16xi32, #hivm.address_space<gm>>)
                         outs(%sl : memref<3x2xi32, #hivm.address_space<ub>>)
                         offset(%i1 : i32) other(%neg1 : i32)
                         strides([%i4, %i2 : i32, i32])
                         numels([%i3, %i2 : i32, i32])
    %sp = memref.alloc() : memref<3x2xi32, #hivm.address_space<ub>>
    hivm.hir.stride_load ins(%in : memref<16xi32, #hivm.address_space<gm>>)
                         outs(%sp : memref<3x2xi32, #hivm.address_space<ub>>)
                         offset(%i1 : i32) other(%neg1 : i32)
                         strides([%i4, %i2 : i32, i32])
                         numels([%i2, %i1 : i32, i32])
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.stride_store ins(%sl : memref<3x2xi32, #hivm.address_space<ub>>)
                          outs(%sstore : memref<16xi32, #hivm.address_space<gm>>)
                          offset(%i0 : i32)
                          strides([%i4, %i1 : i32, i32])
                          numels([%i3, %i2 : i32, i32])

    hivm.hir.store ins(%d : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%iload : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%sc : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%istore : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%dm : memref<8xi32, #hivm.address_space<ub>>)
                   outs(%masked : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%sl : memref<3x2xi32, #hivm.address_space<ub>>)
                   outs(%sload : memref<3x2xi32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%sp : memref<3x2xi32, #hivm.address_space<ub>>)
                   outs(%spad : memref<3x2xi32, #hivm.address_space<gm>>)
    return
  }
}
