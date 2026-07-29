// The remaining shape-manipulation ops: memref.collapse_shape and
// expand_shape, hivm.hir.vtranspose, and an scf.if that yields results (the
// deadlock tests only cover the result-less form).
//
// in = arange(12) viewed as 3x4:
//     0  1  2  3
//     4  5  6  7
//     8  9 10 11
// transpose -> 4x3, then collapse back to 12 elements in row-major order:
//     0 4 8 1 5 9 2 6 10 3 7 11
// The scf.if picks the transposed result over a straight copy.

// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy    --args=arange,zeros --out=%t.lazy.
// RUN: npuir-interp %s --sched=inorder --args=arange,zeros --out=%t.inorder.
// RUN: cmp %t.lazy.arg1.npy %t.inorder.arg1.npy
// RUN: npuir-interp %s --sched=lazy --args=arange,zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N48 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE
// DATA:      0 4 8 1
// DATA-NEXT: 5 9 2 6
// DATA-NEXT: 10 3 7 11

module {
  func.func @reshape(%in: memref<12xi32, #hivm.address_space<gm>>,
                     %out: memref<12xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %flat = memref.alloc() : memref<12xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<12xi32, #hivm.address_space<gm>>)
                  outs(%flat : memref<12xi32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // 12 -> 3x4
    %expanded = memref.expand_shape %flat [[0, 1]] output_shape [3, 4]
        : memref<12xi32, #hivm.address_space<ub>>
       into memref<3x4xi32, #hivm.address_space<ub>>

    %t = memref.alloc() : memref<4x3xi32, #hivm.address_space<ub>>
    %copy = memref.alloc() : memref<4x3xi32, #hivm.address_space<ub>>

    // An scf.if that yields a memref: pick the transposed buffer.
    %true = arith.constant true
    %picked = scf.if %true -> (memref<4x3xi32, #hivm.address_space<ub>>) {
      hivm.hir.vtranspose ins(%expanded : memref<3x4xi32,
                                                 #hivm.address_space<ub>>)
                          outs(%t : memref<4x3xi32,
                                           #hivm.address_space<ub>>)
                          permutation = [1, 0]
      scf.yield %t : memref<4x3xi32, #hivm.address_space<ub>>
    } else {
      scf.yield %copy : memref<4x3xi32, #hivm.address_space<ub>>
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // 4x3 -> 12
    %collapsed = memref.collapse_shape %picked [[0, 1]]
        : memref<4x3xi32, #hivm.address_space<ub>>
       into memref<12xi32, #hivm.address_space<ub>>
    hivm.hir.store ins(%collapsed : memref<12xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<12xi32, #hivm.address_space<gm>>)
    return
  }
}
