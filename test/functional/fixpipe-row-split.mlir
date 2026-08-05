// A dual-destination fixpipe sends consecutive row halves to the same local
// UB address in the two split AIV lanes.  The lanes must not alias one shared
// UB pool: lane 0 receives 10,11 while lane 1 receives 20,21.

// RUN: npuir-interp %s --sched=lazy --sub-block-num=2 --args=zeros --out=%t.
// RUN: od -An -tfF -j128 -N16 %t.arg0.npy | FileCheck %s

// CHECK: 1.000000e+01 1.100000e+01 2.000000e+01 2.100000e+01

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<MIX>
} {
  func.func @row_split_aic(
      %out: memref<4xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>,
                  hivm.part_of_mix, parallel_mode = "simd"} {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %v10 = arith.constant 10.0 : f32
    %v11 = arith.constant 11.0 : f32
    %v20 = arith.constant 20.0 : f32
    %v21 = arith.constant 21.0 : f32
    %src = hivm.hir.pointer_cast(%c0_i64)
        : memref<1x1x4x1xf32, #hivm.address_space<cc>>
    %src1d = memref.collapse_shape %src [[0, 1, 2, 3]]
        : memref<1x1x4x1xf32, #hivm.address_space<cc>>
          into memref<4xf32, #hivm.address_space<cc>>
    memref.store %v10, %src1d[%c0]
        : memref<4xf32, #hivm.address_space<cc>>
    memref.store %v11, %src1d[%c1]
        : memref<4xf32, #hivm.address_space<cc>>
    memref.store %v20, %src1d[%c2]
        : memref<4xf32, #hivm.address_space<cc>>
    memref.store %v21, %src1d[%c3]
        : memref<4xf32, #hivm.address_space<cc>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_FIX>, <EVENT_ID0>]
    %dst = hivm.hir.pointer_cast(%c0_i64)
        : memref<2x1xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%src : memref<1x1x4x1xf32, #hivm.address_space<cc>>)
        outs(%dst : memref<2x1xf32, #hivm.address_space<ub>>)
        dual_dst_mode = <ROW_SPLIT>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    return
  }

  func.func @row_split_aiv(
      %out: memref<4xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.part_of_mix, parallel_mode = "simd"} {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i64 = arith.constant 2 : i64
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %src = hivm.hir.pointer_cast(%c0_i64)
        : memref<2x1xf32, #hivm.address_space<ub>>
    %src1d = memref.collapse_shape %src [[0, 1]]
        : memref<2x1xf32, #hivm.address_space<ub>>
          into memref<2xf32, #hivm.address_space<ub>>
    %lhs = memref.load %src1d[%c0]
        : memref<2xf32, #hivm.address_space<ub>>
    %rhs = memref.load %src1d[%c1]
        : memref<2xf32, #hivm.address_space<ub>>
    %sub = hivm.hir.get_sub_block_idx -> i64
    %base64 = arith.muli %sub, %c2_i64 : i64
    %base = arith.index_cast %base64 : i64 to index
    %next = arith.addi %base, %c1 : index
    memref.store %lhs, %out[%base]
        : memref<4xf32, #hivm.address_space<gm>>
    memref.store %rhs, %out[%next]
        : memref<4xf32, #hivm.address_space<gm>>
    return
  }
}
