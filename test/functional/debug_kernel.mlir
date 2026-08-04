// End-to-end TTAdapter validation: lower through the repository's production
// HFusion/HIVM pipeline, stop after the synchronized memref HIVM stage, then
// compare the interpreter's ordered and deferred schedules.  N=1500 spans two
// 1024-element blocks, so the second block also exercises tail padding/masking.

// RUN: %python %S/../lower_ttadapter_for_interp.py \
// RUN:   bishengir-compile %s %t.hivm.mlir -- --target=Ascend910B4 \
// RUN:   --enable-triton-kernel-compile=true --enable-hfusion-compile=true \
// RUN:   --enable-hivm-compile=true --enable-lir-compile=false \
// RUN:   --enable-hivm-graph-sync-solver=true
// RUN: FileCheck %s --check-prefix=LOWERED \
// RUN:   --implicit-check-not='tensor<' \
// RUN:   --implicit-check-not='bufferization.' \
// RUN:   --input-file=%t.hivm.mlir
// RUN: npuir-interp %t.hivm.mlir --sched=lazy --block-dim=2 \
// RUN:   --dyn-gm-elems=2048 \
// RUN:   --args=0,zeros,zeros,arange,1,zeros,1500,2,1,1 \
// RUN:   --out=%t.lazy. 2>&1 | FileCheck %s --check-prefix=LAZY
// RUN: npuir-interp %t.hivm.mlir --sched=inorder --block-dim=2 \
// RUN:   --dyn-gm-elems=2048 \
// RUN:   --args=0,zeros,zeros,arange,1,zeros,1500,2,1,1 \
// RUN:   --out=%t.inorder.
// RUN: cmp %t.lazy.arg5.npy %t.inorder.arg5.npy
// RUN: npuir-interp %t.hivm.mlir --sched=fuzz --seed=17 --block-dim=2 \
// RUN:   --dyn-gm-elems=2048 \
// RUN:   --args=0,zeros,zeros,arange,1,zeros,1500,2,1,1 \
// RUN:   2>&1 | FileCheck %s --check-prefix=FUZZ
// RUN: od -An -tf4 -j6112 -N64 %t.lazy.arg5.npy | \
// RUN:   FileCheck %s --check-prefix=TAIL

// LOWERED-LABEL: func.func @add_kernel
// LOWERED: hivm.hir.load
// LOWERED: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
// LOWERED: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
// LOWERED: hivm.hir.vadd
// LOWERED: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// LOWERED: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// LOWERED: hivm.hir.store

// LAZY: 2 core(s), sched=lazy
// LAZY-NOT: MISSING SYNC
// LAZY-NOT: DATA RACE
// LAZY-NOT: DEADLOCK

// FUZZ: 2 core(s), sched=fuzz seed=17
// FUZZ-NOT: MISSING SYNC
// FUZZ-NOT: DATA RACE
// FUZZ-NOT: DEADLOCK

// TAIL:      1.497000e+03 1.498000e+03 1.499000e+03 1.500000e+03
// TAIL-NEXT: 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00

#loc = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0)
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @add_kernel(%arg0: memref<?xi8> loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg1: memref<?xi8> loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32} loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32} loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32} loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg5: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg6: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg7: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg8: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg9: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg10: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0), %arg11: i32 loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":20:0)) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c1024 = arith.constant 1024 : index loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc3)
    %0 = arith.muli %arg9, %c1024_i32 : i32 loc(#loc3)
    %1 = arith.index_cast %0 : i32 to index loc(#loc3)
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc4)
    %alloc = memref.alloc() : memref<1024xf32> loc(#loc2)
    %2 = arith.addi %1, %c1024 : index loc(#loc2)
    %3 = arith.index_cast %arg5 : i32 to index loc(#loc)
    %4 = arith.maxsi %1, %3 : index loc(#loc2)
    %5 = arith.minsi %2, %4 : index loc(#loc2)
    %6 = arith.subi %5, %1 : index loc(#loc2)
    %7 = arith.cmpi slt, %6, %c1024 : index loc(#loc2)
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>) loc(#loc2)
    } {hivm.unlikely_condition} loc(#loc2)
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc2)
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>> loc(#loc2)
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>> loc(#loc2)
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> loc(#loc2)
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc5)
    %alloc_2 = memref.alloc() : memref<1024xf32> loc(#loc1)
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1024xf32>) loc(#loc1)
    } {hivm.unlikely_condition} loc(#loc1)
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc1)
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>> loc(#loc1)
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>> loc(#loc1)
    %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32> loc(#loc1)
    gpu.barrier loc(#loc6)
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>> loc(#loc7)
    %10 = arith.addf %8, %9 : tensor<1024xf32> loc(#loc8)
    %extracted_slice = tensor.extract_slice %10[0] [%6] [1] : tensor<1024xf32> to tensor<?xf32> loc(#loc9)
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> loc(#loc9)
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> () loc(#loc9)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":25:16)
#loc2 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":24:16)
#loc3 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":22:17)
#loc4 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":24:24)
#loc5 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":25:24)
#loc6 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":30:4)
#loc7 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":31:23)
#loc8 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":31:33)
#loc9 = loc("/Users/sky_miner/Documents/Project/huawei/github-triton-ascend/dump_ttadapter.py":31:29)
