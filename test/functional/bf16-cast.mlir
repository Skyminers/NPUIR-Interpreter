// bf16 has no arithmetic ops of its own in HIVM - the verifiers of `vadd` and
// friends reject it - so every bf16 value in a real kernel arrives through
// `vcast`. That makes the conversion the whole of bf16 correctness, and it is
// the one place where an 8-bit significand behaves unlike f16's 11-bit one:
// bf16 keeps f32's exponent range but throws away 16 mantissa bits, so ties
// are common and underflow is not.
//
// Narrowing (arg1), widening straight back (arg2), and bf16 -> i32 (arg3).
// Widening must be exact, so arg2 is the top 16 bits of each input with the
// low 16 zeroed - which is only true because narrowing rounded correctly.
//
//   in           value                    bf16    why
//   0x3fc00000   1.5                      3fc0    exactly representable
//   0x3fc08000   1.50390625               3fc0    tie -> even (down)
//   0x3fc18000   1.51171875               3fc2    tie -> even (up)
//   0x3fc0ffff   1.5078123807907104       3fc1    just under the tie
//   0x3fc10001   1.5078126192092896       3fc1    just over the tie
//   0x7f7fffff   f32 max                  7f80    above bf16 max -> +inf
//   0xff7fffff   -f32 max                 ff80    -> -inf
//   0x00000001   smallest f32 subnormal   0000    underflows bf16
//   0x80000000   -0                       8000    the sign survives
//   0x7fc00000   NaN                      7fc0    stays NaN
//   0x7f800000   +inf                     7f80    passed through
//   0x00800000   smallest f32 normal      0080    exact: bf16 keeps the range

// RUN: npuir-interp %s --sched=lazy --args=%S/Inputs/bf16_cast_f32.npy,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -tx2 -j128 %t.arg1.npy | FileCheck %s --check-prefix=NARROW
// RUN: od -v -An -tx4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=WIDEN
// RUN: od -v -An -td4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=TOINT
// RUN: npuir-interp %s --sched=inorder --args=%S/Inputs/bf16_cast_f32.npy,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DATA RACE

// NARROW:      3fc0 3fc0 3fc2 3fc1 3fc1 7f80 ff80 0000
// NARROW-NEXT: 8000 7fc0 7f80 0080

// Widening is exact, so each result is its bf16 pattern in the high half.
// WIDEN:      3fc00000 3fc00000 3fc20000 3fc10000
// WIDEN-NEXT: 3fc10000 7f800000 ff800000 00000000
// WIDEN-NEXT: 80000000 7fc00000 7f800000 00800000

// bf16 -> i32 with trunc. The infinities and NaN are out of range, so their
// results are unspecified and not checked; the first five all truncate to 1.
// TOINT: 1 1 1 1

module {
  func.func @bf16cast(%in: memref<12xf32, #hivm.address_space<gm>>,
                      %narrow: memref<12xbf16, #hivm.address_space<gm>>,
                      %widen: memref<12xf32, #hivm.address_space<gm>>,
                      %toint: memref<12xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %a = memref.alloc() : memref<12xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<12xbf16, #hivm.address_space<ub>>
    %w = memref.alloc() : memref<12xf32, #hivm.address_space<ub>>
    %i = memref.alloc() : memref<12xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<12xf32, #hivm.address_space<gm>>)
                  outs(%a : memref<12xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

    hivm.hir.vcast {round_mode = #hivm.round_mode<rint>}
                   ins(%a : memref<12xf32, #hivm.address_space<ub>>)
                   outs(%b : memref<12xbf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast {round_mode = #hivm.round_mode<rint>}
                   ins(%b : memref<12xbf16, #hivm.address_space<ub>>)
                   outs(%w : memref<12xf32, #hivm.address_space<ub>>)
    hivm.hir.vcast {round_mode = #hivm.round_mode<trunc>}
                   ins(%b : memref<12xbf16, #hivm.address_space<ub>>)
                   outs(%i : memref<12xi32, #hivm.address_space<ub>>)

    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%b : memref<12xbf16, #hivm.address_space<ub>>)
                   outs(%narrow : memref<12xbf16, #hivm.address_space<gm>>)
    hivm.hir.store ins(%w : memref<12xf32, #hivm.address_space<ub>>)
                   outs(%widen : memref<12xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%i : memref<12xi32, #hivm.address_space<ub>>)
                   outs(%toint : memref<12xi32, #hivm.address_space<gm>>)
    return
  }
}
