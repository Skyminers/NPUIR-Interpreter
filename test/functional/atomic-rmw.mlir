// `atomic_cas` and `atomic_xchg` are read-modify-writes on GM. Two blocks
// apply the same idempotent update to the same buffer, which is the pattern
// they exist for - and the race detector must not report the two atomics
// against each other, because the hardware serialises them. Everything that
// is *not* atomic here (the seeding, the scalar readback) is confined to
// block 0 and separated by a barrier, so any report is a real finding.
//
//   arg1  cas expecting 4 everywhere: only lane 4 of arange holds 4, so only
//         that lane takes 99
//   arg2  xchg with 99 under a mask that is off on lane 2, which keeps its
//         original value
//   arg3  what load_scalar reads from GM offsets 0 and 4, i.e. %arg0[0..1]

// RUN: npuir-interp %s --sched=lazy --block-dim=2 --args=arange,zeros,zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -v -An -td4 -j128 %t.arg1.npy | FileCheck %s --check-prefix=CAS
// RUN: od -v -An -td4 -j128 %t.arg2.npy | FileCheck %s --check-prefix=XCHG
// RUN: od -v -An -td4 -j128 %t.arg3.npy | FileCheck %s --check-prefix=SCALAR
// RUN: npuir-interp %s --sched=fuzz --seed=2 --block-dim=2 --args=arange,zeros,zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=fuzz --seed=9 --block-dim=2 --args=arange,zeros,zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --block-dim=2 --args=arange,zeros,zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy
// RUN: cmp %t.arg2.npy %t.inorder.arg2.npy

// CHECK: 2 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DEADLOCK
// CAS:        0 1 2 3
// CAS-NEXT:   99 5 6 7
// XCHG:       99 99 2 99
// SCALAR:     0 1

module {
  func.func @rmw(%in: memref<8xi32, #hivm.address_space<gm>>,
                 %cas: memref<8xi32, #hivm.address_space<gm>>,
                 %xchg: memref<4xi32, #hivm.address_space<gm>>,
                 %scalar: memref<2xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c4_i64 = arith.constant 4 : i64
    %four = arith.constant 4 : i32
    %v99 = arith.constant 99 : i32
    %on = arith.constant 1 : i8
    %off = arith.constant 0 : i8
    %zero64 = arith.constant 0 : i64

    %block = hivm.hir.get_block_idx -> i64
    %isFirst = arith.cmpi eq, %block, %zero64 : i64

    // Only block 0 seeds, so the seeding writes are unshared. A gm-to-gm copy
    // is not something the DMA can do, hence the trip through UB.
    scf.if %isFirst {
      %stage8 = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.load ins(%in : memref<8xi32, #hivm.address_space<gm>>)
                    outs(%stage8 : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.store ins(%stage8 : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%cas : memref<8xi32, #hivm.address_space<gm>>)
      %stage4 = memref.reinterpret_cast %stage8 to offset: [0], sizes: [4],
                    strides: [1]
          : memref<8xi32, #hivm.address_space<ub>>
         to memref<4xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.store ins(%stage4 : memref<4xi32, strided<[1]>, #hivm.address_space<ub>>)
                     outs(%xchg : memref<4xi32, #hivm.address_space<gm>>)
    }
    hivm.hir.sync_block [<ALL_VECTOR>, 0] tvector_pipe = <PIPE_MTE3>

    %expect = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %desire = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%four : i32)
                  outs(%expect : memref<8xi32, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%v99 : i32)
                  outs(%desire : memref<8xi32, #hivm.address_space<ub>>)

    %newval = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    %mask = memref.alloc() : memref<4xi8, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%v99 : i32)
                  outs(%newval : memref<4xi32, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%on : i8)
                  outs(%mask : memref<4xi8, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    memref.store %off, %mask[%c2] : memref<4xi8, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_S>]

    hivm.hir.atomic_cas
        ins(%expect, %desire : memref<8xi32, #hivm.address_space<ub>>,
                               memref<8xi32, #hivm.address_space<ub>>)
        outs(%cas : memref<8xi32, #hivm.address_space<gm>>)
    hivm.hir.atomic_xchg ins(%newval : memref<4xi32, #hivm.address_space<ub>>)
                         outs(%xchg : memref<4xi32, #hivm.address_space<gm>>)
                         mask(%mask : memref<4xi8, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block [<ALL_VECTOR>, 1] tvector_pipe = <PIPE_MTE3>

    // One block reads back through the scalar unit. The address is absolute
    // within the GM arena, where %arg0 sits at offset 0.
    scf.if %isFirst {
      %p0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<1>
      %p1 = llvm.inttoptr %c4_i64 : i64 to !llvm.ptr<1>
      %r0 = hivm.hir.load_scalar %p0 : !llvm.ptr<1> -> i32
      %r1 = hivm.hir.load_scalar %p1 : !llvm.ptr<1> -> i32
      memref.store %r0, %scalar[%c0] : memref<2xi32, #hivm.address_space<gm>>
      memref.store %r1, %scalar[%c1] : memref<2xi32, #hivm.address_space<gm>>
    }
    return
  }
}
