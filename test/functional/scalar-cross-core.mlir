// PIPE_S is also a legal `tpipe` for a cross-core flag, and InjectSync does
// emit `sync_block_set[<CUBE>, <PIPE_S>, ...]` when the producer of the value
// is the scalar unit. Raising the flag has to drain the scalar unit's own
// outstanding accesses first, or the AIV's consumption of %mid is reported as
// racing with the AIC's scalar stores.

// RUN: npuir-interp %s --sched=lazy --args=zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: od -An -td4 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=DATA
// RUN: npuir-interp %s --sched=fuzz --seed=1 --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=fuzz --seed=5 --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=inorder --args=zeros,zeros --out=%t.inorder.
// RUN: cmp %t.arg1.npy %t.inorder.arg1.npy

// CHECK: 2 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC
// CHECK-NOT: DEADLOCK
// DATA: 100 101 102 103

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // AIC: fills %mid with 100, 101, 102, 103 using scalar stores only.
  func.func @k_mix_aic(%mid: memref<4xi32, #hivm.address_space<gm>>,
                       %out: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : i32
    scf.for %i = %c0 to %c4 step %c1 {
      %i32 = arith.index_cast %i : index to i32
      %v = arith.addi %c100, %i32 : i32
      memref.store %v, %mid[%i] : memref<4xi32, #hivm.address_space<gm>>
    }
    hivm.hir.sync_block_set [<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = 1
    return
  }

  // AIV: copies %mid to %out through UB, gated on the AIC's flag.
  func.func @k_mix_aiv(%mid: memref<4xi32, #hivm.address_space<gm>>,
                       %out: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = 1
    %buf = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%mid : memref<4xi32, #hivm.address_space<gm>>)
                  outs(%buf : memref<4xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%buf : memref<4xi32, #hivm.address_space<ub>>)
                   outs(%out : memref<4xi32, #hivm.address_space<gm>>)
    return
  }
}
