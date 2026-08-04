// MIX cross-core flags have group semantics when one cube is paired with
// multiple vector sub-cores: AIV -> AIC collects one contribution from every
// sub-core, while AIC -> AIV broadcasts the released generation to all of
// them.  This is the handshake used by lowered SIMD MIX kernels.

// RUN: npuir-interp %s --sched=lazy --sub-block-num=2 --args=zeros,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=fuzz --seed=0 --sub-block-num=2 --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=fuzz --seed=5 --sub-block-num=2 --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: od -An -td4 -j128 -N12 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK: 3 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: DEADLOCK
// CHECK-NOT: cross-core flag
// DATA: 21 21 21

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @subblocks_aic(%ready: memref<2xi32, #hivm.address_space<gm>>,
                           %out: memref<3xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIC>,
                  hivm.part_of_mix,
                  parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // The wait must collect both AIV publications before either load is
    // allowed to become visible to this core.
    hivm.hir.sync_block_wait [<CUBE>, <PIPE_S>, <PIPE_S>] flag = 1
    %lhs = memref.load %ready[%c0] : memref<2xi32, #hivm.address_space<gm>>
    %rhs = memref.load %ready[%c1] : memref<2xi32, #hivm.address_space<gm>>
    %sum = arith.addi %lhs, %rhs : i32
    memref.store %sum, %out[%c0] : memref<3xi32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set [<CUBE>, <PIPE_S>, <PIPE_S>] flag = 1
    return
  }

  func.func @subblocks_aiv(%ready: memref<2xi32, #hivm.address_space<gm>>,
                           %out: memref<3xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.part_of_mix,
                  parallel_mode = "simd"} {
    %c1_i64 = arith.constant 1 : i64
    %c10_i32 = arith.constant 10 : i32
    %c0 = arith.constant 0 : index
    %sub64 = hivm.hir.get_sub_block_idx -> i64
    %sub = arith.index_cast %sub64 : i64 to index
    %value = arith.index_cast %sub : index to i32
    %value10 = arith.addi %value, %c10_i32 : i32
    memref.store %value10, %ready[%sub]
        : memref<2xi32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set [<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait [<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1
    %sum = memref.load %out[%c0] : memref<3xi32, #hivm.address_space<gm>>
    %outIdx64 = arith.addi %sub64, %c1_i64 : i64
    %outIdx = arith.index_cast %outIdx64 : i64 to index
    memref.store %sum, %out[%outIdx] : memref<3xi32, #hivm.address_space<gm>>
    return
  }
}
