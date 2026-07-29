// A sweep over the elementwise family so the shared driver (broadcasting,
// destination-shape iteration, integer/float dispatch) is exercised on more
// than vadd.
//
// in = arange(8) as f32: 0 1 2 3 4 5 6 7, and a splat of 2.0.
//   out0 = in - 2         : -2 -1 0 1 2 3 4 5
//   out1 = in * 2         :  0  2 4 6 8 10 12 14
//   out2 = in / 2         :  0 .5 1 1.5 2 2.5 3 3.5
//   out3 = max(in, 2)     :  2  2 2 3 4 5 6 7
//   out4 = |in - 4|       :  4  3 2 1 0 1 2 3
//   out5 = sqrt(in)                          (checked at 0, 1, 4)
//   out6 = (in > 2) ? in : 2  via vcmp + vsel

// RUN: npuir-interp %s --sched=inorder --args=arange,zeros,zeros,zeros,zeros,zeros,zeros,zeros --out=%t.
// RUN: od -An -tf4 -j128 -N32 %t.arg1.npy | FileCheck %s --check-prefix=SUB
// RUN: od -An -tf4 -j128 -N32 %t.arg2.npy | FileCheck %s --check-prefix=MUL
// RUN: od -An -tf4 -j128 -N32 %t.arg3.npy | FileCheck %s --check-prefix=DIV
// RUN: od -An -tf4 -j128 -N32 %t.arg4.npy | FileCheck %s --check-prefix=MAX
// RUN: od -An -tf4 -j128 -N32 %t.arg5.npy | FileCheck %s --check-prefix=ABS
// RUN: od -An -tf4 -j128 -N32 %t.arg6.npy | FileCheck %s --check-prefix=SQRT
// RUN: od -An -tf4 -j128 -N32 %t.arg7.npy | FileCheck %s --check-prefix=SEL

// SUB:       -2.000000e+00 -1.000000e+00 0.000000e+00 1.000000e+00
// SUB-NEXT:   2.000000e+00 3.000000e+00 4.000000e+00 5.000000e+00
// MUL:        0.000000e+00 2.000000e+00 4.000000e+00 6.000000e+00
// MUL-NEXT:   8.000000e+00 1.000000e+01 1.200000e+01 1.400000e+01
// DIV:        0.000000e+00 5.000000e-01 1.000000e+00 1.500000e+00
// DIV-NEXT:   2.000000e+00 2.500000e+00 3.000000e+00 3.500000e+00
// MAX:        2.000000e+00 2.000000e+00 2.000000e+00 3.000000e+00
// MAX-NEXT:   4.000000e+00 5.000000e+00 6.000000e+00 7.000000e+00
// ABS:        4.000000e+00 3.000000e+00 2.000000e+00 1.000000e+00
// ABS-NEXT:   0.000000e+00 1.000000e+00 2.000000e+00 3.000000e+00
// SQRT:       0.000000e+00 1.000000e+00 1.414214e+00 1.732051e+00
// SQRT-NEXT:  2.000000e+00
// SEL:        2.000000e+00 2.000000e+00 2.000000e+00 3.000000e+00
// SEL-NEXT:   4.000000e+00 5.000000e+00 6.000000e+00 7.000000e+00

module {
  func.func @sweep(%in: memref<8xf32, #hivm.address_space<gm>>,
                   %o0: memref<8xf32, #hivm.address_space<gm>>,
                   %o1: memref<8xf32, #hivm.address_space<gm>>,
                   %o2: memref<8xf32, #hivm.address_space<gm>>,
                   %o3: memref<8xf32, #hivm.address_space<gm>>,
                   %o4: memref<8xf32, #hivm.address_space<gm>>,
                   %o5: memref<8xf32, #hivm.address_space<gm>>,
                   %o6: memref<8xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %two = arith.constant 2.000000e+00 : f32
    %four = arith.constant 4.000000e+00 : f32

    %src = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%src : memref<8xf32, #hivm.address_space<ub>>)
    %k2 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%two : f32)
                  outs(%k2 : memref<8xf32, #hivm.address_space<ub>>)
    %k4 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%four : f32)
                  outs(%k4 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    %r0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r2 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r3 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r4 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r5 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %r6 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %mask = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %diff = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>

    hivm.hir.vsub ins(%src, %k2 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r0 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vmul ins(%src, %k2 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r1 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vdiv ins(%src, %k2 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r2 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vmax ins(%src, %k2 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r3 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vsub ins(%src, %k4 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%diff : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vabs ins(%diff : memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r4 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vsqrt ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%r5 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vcmp ins(%src, %k2 : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%mask : memref<8xi8, #hivm.address_space<ub>>)
                  compare_mode = <gt>
    hivm.hir.vsel ins(%mask, %src, %k2 : memref<8xi8, #hivm.address_space<ub>>,
                                          memref<8xf32, #hivm.address_space<ub>>,
                                          memref<8xf32, #hivm.address_space<ub>>)
                  outs(%r6 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    hivm.hir.store ins(%r0 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o0 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r1 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o1 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r2 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o2 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r3 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o3 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r4 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o4 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r5 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o5 : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%r6 : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%o6 : memref<8xf32, #hivm.address_space<gm>>)
    return
  }
}
