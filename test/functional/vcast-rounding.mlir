// The plan's M1 acceptance criterion: hivm.hir.vcast's rounding modes must be
// bit-exact. Emulating f16/bf16 with `float` gets ties wrong and cannot
// express round-to-odd at all, so everything goes through APFloat.
//
// Input (arg0, f32): -2.5  -1.5  -0.5  0.5  1.5  2.5  3.5  -3.5
//
//   rint  (nearest, ties to even) : -2 -2 -0  0  2  2  4 -4
//   round (nearest, ties away)    : -3 -2 -1  1  2  3  4 -4
//   floor                         : -3 -2 -1  0  1  2  3 -4
//   ceil                          : -2 -1 -0  1  2  3  4 -3
//   trunc (toward zero)           : -2 -1 -0  0  1  2  3 -3

// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/halves.npy,zeros,zeros,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/halves.npy,zeros,zeros,zeros,zeros,zeros --out=%t.
// RUN: od -An -td1 -j128 -N8 %t.arg1.npy | FileCheck %s --check-prefix=RINT
// RUN: od -An -td1 -j128 -N8 %t.arg2.npy | FileCheck %s --check-prefix=ROUND
// RUN: od -An -td1 -j128 -N8 %t.arg3.npy | FileCheck %s --check-prefix=FLOOR
// RUN: od -An -td1 -j128 -N8 %t.arg4.npy | FileCheck %s --check-prefix=CEIL
// RUN: od -An -td1 -j128 -N8 %t.arg5.npy | FileCheck %s --check-prefix=TRUNC

// CHECK-NOT: error
// RINT:  -2 -2 0 0 2 2 4 -4
// ROUND: -3 -2 -1 1 2 3 4 -4
// FLOOR: -3 -2 -1 0 1 2 3 -4
// CEIL:  -2 -1 0 1 2 3 4 -3
// TRUNC: -2 -1 0 0 1 2 3 -3

module {
  func.func @casts(%in: memref<8xf32, #hivm.address_space<gm>>,
                   %rint: memref<8xi8, #hivm.address_space<gm>>,
                   %round: memref<8xi8, #hivm.address_space<gm>>,
                   %floor: memref<8xi8, #hivm.address_space<gm>>,
                   %ceil: memref<8xi8, #hivm.address_space<gm>>,
                   %trunc: memref<8xi8, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %src = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%src : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    %r0 = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %r1 = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %r2 = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %r3 = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %r4 = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>

    hivm.hir.vcast ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r0 : memref<8xi8, #hivm.address_space<ub>>)
                   round_mode = <rint>
    hivm.hir.vcast ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r1 : memref<8xi8, #hivm.address_space<ub>>)
                   round_mode = <round>
    hivm.hir.vcast ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r2 : memref<8xi8, #hivm.address_space<ub>>)
                   round_mode = <floor>
    hivm.hir.vcast ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r3 : memref<8xi8, #hivm.address_space<ub>>)
                   round_mode = <ceil>
    hivm.hir.vcast ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r4 : memref<8xi8, #hivm.address_space<ub>>)
                   round_mode = <trunc>
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.store ins(%r0 : memref<8xi8, #hivm.address_space<ub>>)
                   outs(%rint : memref<8xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r1 : memref<8xi8, #hivm.address_space<ub>>)
                   outs(%round : memref<8xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r2 : memref<8xi8, #hivm.address_space<ub>>)
                   outs(%floor : memref<8xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r3 : memref<8xi8, #hivm.address_space<ub>>)
                   outs(%ceil : memref<8xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r4 : memref<8xi8, #hivm.address_space<ub>>)
                   outs(%trunc : memref<8xi8, #hivm.address_space<gm>>)
    return
  }
}
