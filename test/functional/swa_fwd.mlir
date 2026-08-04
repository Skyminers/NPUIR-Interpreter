// This is a real lowered SIMD MIX kernel rather than a hand-written unit
// kernel.  The dump is taken after PlanMemory and cross-core synchronisation,
// but before the HIVM intra-core GraphSyncSolver.  Inorder therefore validates
// execution coverage at this stage; lazy sync checking belongs on the result
// after GraphSyncSolver has materialised set_flag/wait_flag operations.

// RUN: npuir-interp %s --sched=inorder --block-dim=1 --sub-block-num=2 --dyn-gm-elems=200000 \
// RUN:   --args=zeros,zeros,zeros,zeros,zeros,1,1,1,arange,arange,1,1536,128,1536,128,1,1536,128,512,128,512,128,1,12,1,1 2>&1 | FileCheck %s --check-prefix=BASELINE

// BASELINE: 3 core(s), sched=inorder
// BASELINE-NOT: unsupported op
// BASELINE-NOT: DEADLOCK

#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 + s3)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map4 = affine_map<()[s0] -> (s0 floordiv 16)>
#map5 = affine_map<()[s0] -> (s0 mod 16)>
#map6 = affine_map<(d0) -> (d0 * 16)>
#map7 = affine_map<()[s0] -> (s0 * 64)>
#map8 = affine_map<()[s0] -> (s0 * 4)>
#map9 = affine_map<()[s0] -> (s0 * 64 + 64)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map11 = affine_map<()[s0] -> (s0 + 128)>
#map12 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1 * 2816)>
#map14 = affine_map<()[s0, s1] -> (s0 - s1 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_swa_fwd_kernel_mix_aiv_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x128xf32, #hivm.address_space<ub>>, %arg5: memref<64x128xbf16, #hivm.address_space<ub>>, %arg6: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    scf.for %arg7 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg7] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg8 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg4[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg6[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %4 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %6 = arith.divf %4, %5 : vector<1x64xf32>
        vector.transfer_write %6, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg5[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %7 = arith.truncf %6 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : vector<1x64xf32> to vector<1x64xbf16>
        vector.transfer_write %7, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xbf16>, memref<1x64xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_swa_fwd_kernel_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: f32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg23: i32, %arg24: i32, %arg25: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false]> : vector<26xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c131072_i64 = arith.constant 131072 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c173056_i64 = arith.constant 173056 : i64
    %c140288_i64 = arith.constant 140288 : i64
    %c107520_i64 = arith.constant 107520 : i64
    %c74752_i64 = arith.constant 74752 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.mlir.constant(1024 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c12_i32 = arith.constant 12 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1023_i32 = arith.constant 1023 : i32
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c3_i32 = arith.constant 3 : i32
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(1028 : i64) : i64
    %5 = llvm.mlir.constant(8 : i64) : i64
    %6 = llvm.mlir.constant(1032 : i64) : i64
    %7 = llvm.mlir.constant(12 : i64) : i64
    %8 = llvm.mlir.constant(1036 : i64) : i64
    %9 = llvm.mlir.constant(16 : i64) : i64
    %10 = llvm.mlir.constant(1040 : i64) : i64
    %11 = llvm.mlir.constant(20 : i64) : i64
    %12 = llvm.mlir.constant(1044 : i64) : i64
    %13 = llvm.mlir.constant(24 : i64) : i64
    %14 = llvm.mlir.constant(1048 : i64) : i64
    %15 = llvm.mlir.constant(28 : i64) : i64
    %16 = llvm.mlir.constant(1052 : i64) : i64
    %17 = llvm.mlir.constant(32 : i64) : i64
    %18 = llvm.mlir.constant(1056 : i64) : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c28_i32 = arith.constant 28 : i32
    %19 = arith.muli %arg23, %arg24 : i32
    %20 = arith.muli %19, %arg25 : i32
    annotation.mark %20 {logical_block_num} : i32
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.trunci %21 : i64 to i32
    %23 = llvm.inttoptr %1 : i64 to !llvm.ptr<11>
    %24 = llvm.inttoptr %0 : i64 to !llvm.ptr<11>
    %25 = llvm.inttoptr %3 : i64 to !llvm.ptr<11>
    %26 = llvm.inttoptr %4 : i64 to !llvm.ptr<11>
    %27 = llvm.inttoptr %5 : i64 to !llvm.ptr<11>
    %28 = llvm.inttoptr %6 : i64 to !llvm.ptr<11>
    %29 = llvm.inttoptr %7 : i64 to !llvm.ptr<11>
    %30 = llvm.inttoptr %8 : i64 to !llvm.ptr<11>
    %31 = llvm.inttoptr %9 : i64 to !llvm.ptr<11>
    %32 = llvm.inttoptr %10 : i64 to !llvm.ptr<11>
    %33 = llvm.inttoptr %11 : i64 to !llvm.ptr<11>
    %34 = llvm.inttoptr %12 : i64 to !llvm.ptr<11>
    %35 = llvm.inttoptr %13 : i64 to !llvm.ptr<11>
    %36 = llvm.inttoptr %14 : i64 to !llvm.ptr<11>
    %37 = llvm.inttoptr %15 : i64 to !llvm.ptr<11>
    %38 = llvm.inttoptr %16 : i64 to !llvm.ptr<11>
    %39 = llvm.inttoptr %17 : i64 to !llvm.ptr<11>
    %40 = llvm.inttoptr %18 : i64 to !llvm.ptr<11>
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg8 to offset: [1], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg9 to offset: [1], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
    %41 = arith.index_cast %arg16 : i32 to index
    scf.for %arg26 = %22 to %20 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %42 = arith.remsi %arg26, %arg23 : i32
      llvm.store volatile %2, %23 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %24 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %25 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %26 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %27 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %28 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %29 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %30 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %31 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %32 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %33 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %34 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %35 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %36 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %37 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %38 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %39 : i32, !llvm.ptr<11>
      llvm.store volatile %2, %40 : i32, !llvm.ptr<11>
      %43 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
      %44 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
      %45 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
      %46 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
      %47 = arith.subi %44, %43 : i32
      %48 = arith.subi %46, %45 : i32
      %49 = arith.subi %48, %47 : i32
      %50 = arith.addi %47, %c127_i32 : i32
      %51 = arith.divsi %50, %c128_i32 : i32
      %52 = arith.muli %51, %c12_i32 : i32
      %53 = arith.remsi %42, %arg23 : i32
      %54 = arith.muli %43, %arg16 : i32
      %55 = arith.muli %45, %arg18 : i32
      %56 = arith.muli %45, %arg20 : i32
      %57 = arith.index_cast %54 : i32 to index
      %58 = arith.index_cast %47 : i32 to index
      %59 = arith.index_cast %55 : i32 to index
      %60 = arith.index_cast %56 : i32 to index
      scf.for %arg27 = %53 to %52 step %arg23  : i32 {
        %61 = arith.divsi %arg27, %c12_i32 : i32
        %62 = arith.muli %61, %c128_i32 : i32
        %63 = arith.addi %62, %c128_i32 : i32
        %64 = arith.minsi %63, %47 : i32
        %65 = arith.subi %64, %62 : i32
        %66 = arith.addi %62, %49 : i32
        %67 = arith.addi %66, %65 : i32
        %68 = arith.addi %67, %c127_i32 : i32
        %69 = arith.divsi %68, %c128_i32 : i32
        %70 = arith.minsi %69, %c1_i32 : i32
        %71 = arith.subi %66, %c1023_i32 : i32
        %72 = arith.maxsi %71, %c0_i32 : i32
        %73 = arith.divsi %72, %c128_i32 : i32
        %74 = arith.maxsi %70, %73 : i32
        %75 = arith.subi %69, %74 : i32
        %76 = arith.addi %70, %75 : i32
        %77 = arith.remsi %arg27, %c12_i32 : i32
        %78 = arith.divsi %77, %c3_i32 : i32
        %79 = arith.muli %77, %arg17 : i32
        %80 = arith.index_cast %79 : i32 to index
        %81 = arith.maxsi %62, %c0_i32 : i32
        %82 = arith.index_cast %81 : i32 to index
        %83 = affine.apply #map1()[%82, %41, %57, %80]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%83], sizes: [128, 128], strides: [%41, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %84 = affine.apply #map2()[%58, %82]
        %85 = arith.maxsi %84, %c0 : index
        %86 = arith.minsi %85, %c128 : index
        %87 = arith.subi %c0_i32, %62 : i32
        %88 = arith.maxsi %87, %c0_i32 : i32
        %89 = arith.index_cast %88 : i32 to index
        %90 = arith.minsi %89, %86 : index
        %91 = affine.apply #map2()[%86, %90]
        %92 = arith.cmpi slt, %91, %c128 : index
        %subview = memref.subview %reinterpret_cast_3[0, 0] [%91, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %93 = arith.subi %74, %70 : i32
        %94 = arith.muli %78, %arg19 : i32
        %95 = arith.index_cast %94 : i32 to index
        %96 = arith.muli %78, %arg21 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %cast = memref.cast %98 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        %99 = affine.apply #map3()[%86, %90]
        %100 = affine.apply #map4()[%90]
        %101 = affine.apply #map5()[%90]
        %subview_4 = memref.subview %98[0, %100, %101, 0] [8, %99, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        %cast_5 = memref.cast %subview_4 : memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %92 {
          %collapse_shape = memref.collapse_shape %98 [[0, 1, 2, 3]] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> into memref<16384xbf16, #hivm.address_space<cbuf>>
          hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<16384xbf16, #hivm.address_space<cbuf>>)
        } {hivm.unlikely_condition}
        hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
        %102 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %102 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = -1 : index} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %103 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %103 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = -1 : index} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        %104 = hivm.hir.pointer_cast(%c74752_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %104 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %105 = hivm.hir.pointer_cast(%c107520_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %105 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c140288_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %106 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %107 = hivm.hir.pointer_cast(%c173056_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %107 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %108 = arith.addi %76, %c3_i32 : i32
        %109:3 = scf.for %arg28 = %c0_i32 to %108 step %c1_i32 iter_args(%arg29 = %c0_i32, %arg30 = %c0_i32, %arg31 = %c0_i32) -> (i32, i32, i32)  : i32 {
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          %110 = arith.cmpi slt, %arg29, %76 : i32
          %111 = scf.if %110 -> (i32) {
            %174 = arith.addi %arg29, %c1_i32 : i32
            scf.yield %174 : i32
          } else {
            scf.yield %arg29 : i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 8 : i32}
          %112 = llvm.load volatile %23 : !llvm.ptr<11> -> i32
          %113 = llvm.load volatile %24 : !llvm.ptr<11> -> i32
          %114 = arith.cmpi slt, %112, %c2_i32 : i32
          %115 = arith.cmpi slt, %113, %c2_i32 : i32
          %116 = arith.andi %114, %115 : i1
          %117 = llvm.load volatile %27 : !llvm.ptr<11> -> i32
          %118 = llvm.load volatile %28 : !llvm.ptr<11> -> i32
          %119 = arith.cmpi slt, %117, %c2_i32 : i32
          %120 = arith.cmpi slt, %118, %c2_i32 : i32
          %121 = arith.andi %119, %120 : i1
          %122 = arith.andi %116, %121 : i1
          %123 = llvm.load volatile %25 : !llvm.ptr<11> -> i32
          %124 = llvm.load volatile %26 : !llvm.ptr<11> -> i32
          %125 = arith.cmpi slt, %123, %c2_i32 : i32
          %126 = arith.cmpi slt, %124, %c2_i32 : i32
          %127 = arith.andi %125, %126 : i1
          %128 = arith.andi %122, %127 : i1
          %129 = arith.cmpi slt, %arg30, %76 : i32
          %130 = arith.andi %128, %129 : i1
          %131 = scf.if %130 -> (i32) {
            %174 = arith.cmpi sge, %arg30, %70 : i32
            %175 = arith.extui %174 : i1 to i32
            %176 = arith.muli %175, %93 : i32
            %177 = arith.addi %arg30, %176 : i32
            %178 = arith.muli %177, %c128_i32 : i32
            %179 = arith.maxsi %178, %c0_i32 : i32
            %180 = arith.index_cast %179 : i32 to index
            %181 = arith.index_cast %arg18 : i32 to index
            %182 = arith.index_cast %48 : i32 to index
            %183 = affine.apply #map2()[%182, %180]
            %184 = arith.maxsi %183, %c0 : index
            %185 = arith.minsi %184, %c128 : index
            %186 = arith.subi %c0_i32, %178 : i32
            %187 = arith.maxsi %186, %c0_i32 : i32
            %188 = arith.index_cast %187 : i32 to index
            %189 = arith.minsi %188, %185 : index
            %190 = affine.apply #map2()[%185, %189]
            %191 = arith.cmpi slt, %190, %c128 : index
            %192 = hivm.hir.pointer_cast(%c98304_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
            %cast_6 = memref.cast %192 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
            %193 = affine.apply #map1()[%180, %181, %59, %95]
            %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%193], sizes: [128, 128], strides: [%181, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%190, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %194 = affine.apply #map3()[%185, %189]
            %195 = affine.apply #map4()[%189]
            %196 = affine.apply #map5()[%189]
            %subview_9 = memref.subview %192[0, %195, %196, 0] [8, %194, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            %cast_10 = memref.cast %subview_9 : memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
            scf.if %191 {
              %collapse_shape = memref.collapse_shape %192 [[0, 1, 2, 3]] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> into memref<16384xbf16, #hivm.address_space<cbuf>>
              hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<16384xbf16, #hivm.address_space<cbuf>>)
            } {hivm.unlikely_condition}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_10 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %197 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
            %cast_11 = memref.cast %197 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, normalized_in_L0C} ins(%cast, %cast_6, %true, %c128, %c128, %c128 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_11 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
            %198 = arith.remsi %arg30, %c2_i32 : i32
            %199 = arith.cmpi eq, %198, %c0_i32 : i32
            scf.if %199 {
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
              hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%197 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%104 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
              hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
            } else {
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
              hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%197 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%105 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
              hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
            }
            %200 = llvm.load volatile %23 : !llvm.ptr<11> -> i32
            %201 = llvm.load volatile %24 : !llvm.ptr<11> -> i32
            %202 = arith.addi %200, %c1_i32 : i32
            %203 = arith.addi %201, %c1_i32 : i32
            llvm.store volatile %202, %23 : i32, !llvm.ptr<11>
            llvm.store volatile %203, %24 : i32, !llvm.ptr<11>
            %204 = llvm.load volatile %27 : !llvm.ptr<11> -> i32
            %205 = llvm.load volatile %28 : !llvm.ptr<11> -> i32
            %206 = arith.addi %204, %c1_i32 : i32
            %207 = arith.addi %205, %c1_i32 : i32
            llvm.store volatile %206, %27 : i32, !llvm.ptr<11>
            llvm.store volatile %207, %28 : i32, !llvm.ptr<11>
            %208 = llvm.load volatile %25 : !llvm.ptr<11> -> i32
            %209 = llvm.load volatile %26 : !llvm.ptr<11> -> i32
            %210 = arith.addi %208, %c1_i32 : i32
            %211 = arith.addi %209, %c1_i32 : i32
            llvm.store volatile %210, %25 : i32, !llvm.ptr<11>
            llvm.store volatile %211, %26 : i32, !llvm.ptr<11>
            %212 = arith.addi %arg30, %c1_i32 : i32
            scf.yield %212 : i32
          } else {
            scf.yield %arg30 : i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 5 : i32}
          %132 = llvm.load volatile %37 : !llvm.ptr<11> -> i32
          %133 = llvm.load volatile %38 : !llvm.ptr<11> -> i32
          %134 = arith.cmpi sgt, %132, %c0_i32 : i32
          %135 = arith.cmpi sgt, %133, %c0_i32 : i32
          %136 = arith.andi %134, %135 : i1
          %137 = llvm.load volatile %35 : !llvm.ptr<11> -> i32
          %138 = llvm.load volatile %36 : !llvm.ptr<11> -> i32
          %139 = arith.cmpi sgt, %137, %c0_i32 : i32
          %140 = arith.cmpi sgt, %138, %c0_i32 : i32
          %141 = arith.andi %139, %140 : i1
          %142 = arith.andi %136, %141 : i1
          %143 = llvm.load volatile %39 : !llvm.ptr<11> -> i32
          %144 = llvm.load volatile %40 : !llvm.ptr<11> -> i32
          %145 = arith.cmpi sgt, %143, %c0_i32 : i32
          %146 = arith.cmpi sgt, %144, %c0_i32 : i32
          %147 = arith.andi %145, %146 : i1
          %148 = arith.andi %142, %147 : i1
          %149 = llvm.load volatile %31 : !llvm.ptr<11> -> i32
          %150 = llvm.load volatile %32 : !llvm.ptr<11> -> i32
          %151 = arith.cmpi slt, %149, %c2_i32 : i32
          %152 = arith.cmpi slt, %150, %c2_i32 : i32
          %153 = arith.andi %151, %152 : i1
          %154 = arith.andi %148, %153 : i1
          %155 = llvm.load volatile %29 : !llvm.ptr<11> -> i32
          %156 = llvm.load volatile %30 : !llvm.ptr<11> -> i32
          %157 = arith.cmpi slt, %155, %c2_i32 : i32
          %158 = arith.cmpi slt, %156, %c2_i32 : i32
          %159 = arith.andi %157, %158 : i1
          %160 = arith.andi %154, %159 : i1
          %161 = llvm.load volatile %33 : !llvm.ptr<11> -> i32
          %162 = llvm.load volatile %34 : !llvm.ptr<11> -> i32
          %163 = arith.cmpi slt, %161, %c2_i32 : i32
          %164 = arith.cmpi slt, %162, %c2_i32 : i32
          %165 = arith.andi %163, %164 : i1
          %166 = arith.andi %160, %165 : i1
          %167 = arith.cmpi sge, %arg30, %76 : i32
          %168 = arith.cmpi sge, %arg30, %c2_i32 : i32
          %169 = arith.ori %167, %168 : i1
          %170 = arith.cmpi slt, %arg31, %76 : i32
          %171 = arith.andi %166, %169 : i1
          %172 = arith.andi %171, %170 : i1
          %173 = scf.if %172 -> (i32) {
            %174 = arith.cmpi sge, %arg31, %70 : i32
            %175 = arith.extui %174 : i1 to i32
            %176 = arith.muli %175, %93 : i32
            %177 = arith.addi %arg31, %176 : i32
            %178 = arith.muli %177, %c128_i32 : i32
            %179 = arith.maxsi %178, %c0_i32 : i32
            %180 = arith.index_cast %179 : i32 to index
            %181 = arith.index_cast %48 : i32 to index
            %182 = arith.index_cast %arg20 : i32 to index
            %183 = arith.subi %c0_i32, %178 : i32
            %184 = arith.maxsi %183, %c0_i32 : i32
            %185 = arith.index_cast %184 : i32 to index
            %186 = affine.apply #map2()[%181, %180]
            %187 = arith.maxsi %186, %c0 : index
            %188 = arith.minsi %187, %c128 : index
            %189 = arith.minsi %185, %188 : index
            %190 = affine.apply #map2()[%188, %189]
            %191 = arith.cmpi slt, %190, %c128 : index
            %192 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
            %cast_6 = memref.cast %192 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
            %193 = affine.apply #map1()[%180, %182, %60, %97]
            %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%193], sizes: [128, 128], strides: [%182, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%190, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %194 = affine.apply #map3()[%188, %189]
            %195 = affine.apply #map4()[%189]
            %196 = affine.apply #map5()[%189]
            %subview_9 = memref.subview %192[0, %195, %196, 0] [8, %194, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            %cast_10 = memref.cast %subview_9 : memref<8x?x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
            scf.if %191 {
              %collapse_shape = memref.collapse_shape %192 [[0, 1, 2, 3]] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> into memref<16384xbf16, #hivm.address_space<cbuf>>
              hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<16384xbf16, #hivm.address_space<cbuf>>)
            } {hivm.unlikely_condition}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_10 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %197 = arith.remsi %arg31, %c2_i32 : i32
            %198 = arith.cmpi eq, %197, %c0_i32 : i32
            %199 = arith.select %198, %102, %103 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
            scf.if %198 {
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            } else {
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
            }
            %200 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
            %cast_11 = memref.cast %200 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%199, %cast_6, %true, %c128, %c128, %c128 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_11 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
            scf.if %198 {
              hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
              hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%200 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%106 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
              hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
            } else {
              hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
              hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%200 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%107 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
              hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
            }
            %201 = llvm.load volatile %37 : !llvm.ptr<11> -> i32
            %202 = llvm.load volatile %38 : !llvm.ptr<11> -> i32
            %203 = arith.subi %201, %c1_i32 : i32
            %204 = arith.subi %202, %c1_i32 : i32
            llvm.store volatile %203, %37 : i32, !llvm.ptr<11>
            llvm.store volatile %204, %38 : i32, !llvm.ptr<11>
            %205 = llvm.load volatile %35 : !llvm.ptr<11> -> i32
            %206 = llvm.load volatile %36 : !llvm.ptr<11> -> i32
            %207 = arith.subi %205, %c1_i32 : i32
            %208 = arith.subi %206, %c1_i32 : i32
            llvm.store volatile %207, %35 : i32, !llvm.ptr<11>
            llvm.store volatile %208, %36 : i32, !llvm.ptr<11>
            %209 = llvm.load volatile %39 : !llvm.ptr<11> -> i32
            %210 = llvm.load volatile %40 : !llvm.ptr<11> -> i32
            %211 = arith.subi %209, %c1_i32 : i32
            %212 = arith.subi %210, %c1_i32 : i32
            llvm.store volatile %211, %39 : i32, !llvm.ptr<11>
            llvm.store volatile %212, %40 : i32, !llvm.ptr<11>
            %213 = llvm.load volatile %31 : !llvm.ptr<11> -> i32
            %214 = llvm.load volatile %32 : !llvm.ptr<11> -> i32
            %215 = arith.addi %213, %c1_i32 : i32
            %216 = arith.addi %214, %c1_i32 : i32
            llvm.store volatile %215, %31 : i32, !llvm.ptr<11>
            llvm.store volatile %216, %32 : i32, !llvm.ptr<11>
            %217 = llvm.load volatile %29 : !llvm.ptr<11> -> i32
            %218 = llvm.load volatile %30 : !llvm.ptr<11> -> i32
            %219 = arith.addi %217, %c1_i32 : i32
            %220 = arith.addi %218, %c1_i32 : i32
            llvm.store volatile %219, %29 : i32, !llvm.ptr<11>
            llvm.store volatile %220, %30 : i32, !llvm.ptr<11>
            %221 = llvm.load volatile %33 : !llvm.ptr<11> -> i32
            %222 = llvm.load volatile %34 : !llvm.ptr<11> -> i32
            %223 = arith.addi %221, %c1_i32 : i32
            %224 = arith.addi %222, %c1_i32 : i32
            llvm.store volatile %223, %33 : i32, !llvm.ptr<11>
            llvm.store volatile %224, %34 : i32, !llvm.ptr<11>
            %225 = arith.addi %arg31, %c1_i32 : i32
            scf.yield %225 : i32
          } else {
            scf.yield %arg31 : i32
          } {hivm.matmul_limited_in_cube, ssbuffer.if = 7 : i32}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          scf.yield %111, %131, %173 : i32, i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv_outlined_vf_0(%arg0: memref<64x128xi8, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: f32, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x128xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<-1.000000e+06> : vector<1x64xf32>
    %cst_0 = arith.constant dense<0> : vector<1x64xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg7 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg7] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %3 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %cst_1) -> (vector<1x64xf32>) {
        %subview_3 = memref.subview %arg0[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_4 = memref.subview %arg1[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_5 = memref.subview %arg4[%arg7, %arg8] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %6 = vector.transfer_read %subview_3[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi8>
        %7 = vector.transfer_read %subview_4[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %8 = vector.broadcast %arg2 : f32 to vector<1x64xf32>
        %9 = arith.mulf %7, %8 : vector<1x64xf32>
        %10 = arith.cmpi ne, %6, %cst_0 : vector<1x64xi8>
        %11 = arith.select %10, %9, %cst : vector<1x64xi1>, vector<1x64xf32>
        vector.transfer_write %11, %subview_5[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %12 = arith.maximumf %11, %arg9 {reductionOp} : vector<1x64xf32>
        scf.yield %12 : vector<1x64xf32>
      } {reductionLoop}
      %4 = vector.transfer_read %subview[%c0], %cst_2 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %5 = vector.multi_reduction <maximumf>, %3, %4 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %5, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    %0 = vector.transfer_read %arg5[%c0], %cst_2 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg3[%c0], %cst_2 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = arith.maximumf %0, %1 : vector<64xf32>
    vector.transfer_write %2, %arg6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv_outlined_vf_1(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>, %arg7: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    scf.for %arg8 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg0[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %8 = scf.for %arg9 = %c0 to %c8 step %c4 iter_args(%arg10 = %cst) -> (vector<1x64xf32>) {
        %11 = affine.apply #map6(%arg9)
        %subview_2 = memref.subview %arg1[%arg8, %11] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %12 = vector.transfer_read %subview_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %13 = vector.transfer_read %subview_1[%c0], %cst_0 {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %14 = arith.subf %12, %13 : vector<1x64xf32>
        %15 = math.exp %14 : vector<1x64xf32>
        %16 = arith.truncf %15 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : vector<1x64xf32> to vector<1x64xbf16>
        %subview_3 = memref.subview %arg2[%arg9, %arg8, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %17 = vector.shape_cast %16 : vector<1x64xbf16> to vector<4x1x16xbf16>
        vector.transfer_write %17, %subview_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xbf16>, memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %18 = arith.addf %15, %arg10 {reductionOp} : vector<1x64xf32>
        scf.yield %18 : vector<1x64xf32>
      } {reductionLoop, unroll_for_vsstb}
      %9 = vector.transfer_read %subview[%c0], %cst_0 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %10 = vector.multi_reduction <add>, %8, %9 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %10, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    %0 = vector.transfer_read %arg5[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg0[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = arith.subf %0, %1 : vector<64xf32>
    %3 = math.exp %2 : vector<64xf32>
    vector.transfer_write %3, %arg7[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %4 = vector.transfer_read %arg4[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %5 = vector.transfer_read %arg3[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %6 = arith.mulf %4, %3 : vector<64xf32>
    %7 = arith.addf %6, %5 : vector<64xf32>
    vector.transfer_write %7, %arg6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv_outlined_vf_2(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>, %arg3: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg5 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg1[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %1 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %2 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %3 = arith.mulf %1, %2 : vector<1x64xf32>
        %4 = arith.addf %0, %3 : vector<1x64xf32>
        vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv_outlined_vf_3(%arg0: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c64 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        vector.transfer_write %cst, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv_outlined_vf_6(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst_0, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    vector.transfer_write %cst, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_swa_fwd_kernel_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: f32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg23: i32, %arg24: i32, %arg25: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false]> : vector<26xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c41472_i64 = arith.constant 41472 : i64
    %c25088_i64 = arith.constant 25088 : i64
    %c57856_i64 = arith.constant 57856 : i64
    %c57600_i64 = arith.constant 57600 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c58112_i64 = arith.constant 58112 : i64
    %c24832_i64 = arith.constant 24832 : i64
    %c206592_i64 = arith.constant 206592 : i64
    %c0_i64 = arith.constant 0 : i64
    %c239872_i64 = arith.constant 239872 : i64
    %c206848_i64 = arith.constant 206848 : i64
    %c207104_i64 = arith.constant 207104 : i64
    %c206336_i64 = arith.constant 206336 : i64
    %c206080_i64 = arith.constant 206080 : i64
    %c205824_i64 = arith.constant 205824 : i64
    %c173056_i64 = arith.constant 173056 : i64
    %c140288_i64 = arith.constant 140288 : i64
    %c107520_i64 = arith.constant 107520 : i64
    %c74752_i64 = arith.constant 74752 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c74496_i64 = arith.constant 74496 : i64
    %c74240_i64 = arith.constant 74240 : i64
    %c64 = arith.constant 64 : index
    %c28_i32 = arith.constant 28 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i64 = arith.constant 32 : i64
    %c24_i64 = arith.constant 24 : i64
    %c28_i64 = arith.constant 28 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c20_i64 = arith.constant 20 : i64
    %c12_i64 = arith.constant 12 : i64
    %c16_i64 = arith.constant 16 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %0 = llvm.mlir.constant(1056 : i64) : i64
    %1 = llvm.mlir.constant(32 : i64) : i64
    %2 = llvm.mlir.constant(1052 : i64) : i64
    %3 = llvm.mlir.constant(28 : i64) : i64
    %4 = llvm.mlir.constant(1048 : i64) : i64
    %5 = llvm.mlir.constant(24 : i64) : i64
    %6 = llvm.mlir.constant(1044 : i64) : i64
    %7 = llvm.mlir.constant(20 : i64) : i64
    %8 = llvm.mlir.constant(1040 : i64) : i64
    %9 = llvm.mlir.constant(16 : i64) : i64
    %10 = llvm.mlir.constant(1036 : i64) : i64
    %11 = llvm.mlir.constant(12 : i64) : i64
    %12 = llvm.mlir.constant(1032 : i64) : i64
    %13 = llvm.mlir.constant(8 : i64) : i64
    %14 = llvm.mlir.constant(1028 : i64) : i64
    %15 = llvm.mlir.constant(4 : i64) : i64
    %c3_i32 = arith.constant 3 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128 = arith.constant 128 : index
    %c1023_i32 = arith.constant 1023 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c12_i32 = arith.constant 12 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2176_i32 = arith.constant 2176 : i32
    %c2304_i32 = arith.constant 2304 : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.constant(1024 : i64) : i64
    %c0 = arith.constant 0 : index
    %19 = arith.muli %arg23, %arg24 : i32
    %20 = arith.muli %19, %arg25 : i32
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.trunci %21 : i64 to i32
    %23 = llvm.inttoptr %17 : i64 to !llvm.ptr<11>
    %24 = llvm.inttoptr %18 : i64 to !llvm.ptr<11>
    %25 = llvm.inttoptr %15 : i64 to !llvm.ptr<11>
    %26 = llvm.inttoptr %14 : i64 to !llvm.ptr<11>
    %27 = llvm.inttoptr %13 : i64 to !llvm.ptr<11>
    %28 = llvm.inttoptr %12 : i64 to !llvm.ptr<11>
    %29 = llvm.inttoptr %11 : i64 to !llvm.ptr<11>
    %30 = llvm.inttoptr %10 : i64 to !llvm.ptr<11>
    %31 = llvm.inttoptr %9 : i64 to !llvm.ptr<11>
    %32 = llvm.inttoptr %8 : i64 to !llvm.ptr<11>
    %33 = llvm.inttoptr %7 : i64 to !llvm.ptr<11>
    %34 = llvm.inttoptr %6 : i64 to !llvm.ptr<11>
    %35 = llvm.inttoptr %5 : i64 to !llvm.ptr<11>
    %36 = llvm.inttoptr %4 : i64 to !llvm.ptr<11>
    %37 = llvm.inttoptr %3 : i64 to !llvm.ptr<11>
    %38 = llvm.inttoptr %2 : i64 to !llvm.ptr<11>
    %39 = llvm.inttoptr %1 : i64 to !llvm.ptr<11>
    %40 = llvm.inttoptr %0 : i64 to !llvm.ptr<11>
    %41 = hivm.hir.get_sub_block_idx -> i64
    %42 = arith.muli %41, %c1024_i64 : i64
    %43 = arith.addi %42, %c16_i64 : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr<11>
    %45 = arith.addi %42, %c12_i64 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr<11>
    %47 = arith.addi %42, %c20_i64 : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr<11>
    %49 = llvm.inttoptr %42 : i64 to !llvm.ptr<11>
    %50 = arith.addi %42, %c8_i64 : i64
    %51 = llvm.inttoptr %50 : i64 to !llvm.ptr<11>
    %52 = arith.addi %42, %c4_i64 : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr<11>
    %54 = arith.addi %42, %c28_i64 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr<11>
    %56 = arith.addi %42, %c24_i64 : i64
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr<11>
    %58 = arith.addi %42, %c32_i64 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr<11>
    %60 = hivm.hir.pointer_cast(%c74240_i64) : memref<64xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c74496_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @_swa_fwd_kernel_mix_aiv_outlined_vf_6(%60, %61) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg8 to offset: [1], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg9 to offset: [1], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
    %62 = arith.index_cast %arg11 : i32 to index
    %63 = arith.index_cast %arg13 : i32 to index
    %64 = arith.index_cast %41 : i64 to index
    %65 = affine.apply #map7()[%64]
    %66 = affine.apply #map8()[%64]
    annotation.mark %20 {logical_block_num} : i32
    %67 = affine.apply #map9()[%64]
    scf.for %arg26 = %22 to %20 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %68 = arith.remsi %arg26, %arg23 : i32
      llvm.store volatile %16, %23 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %24 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %25 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %26 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %27 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %28 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %29 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %30 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %31 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %32 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %33 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %34 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %35 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %36 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %37 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %38 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %39 : i32, !llvm.ptr<11>
      llvm.store volatile %16, %40 : i32, !llvm.ptr<11>
      %69 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
      %70 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
      %71 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
      %72 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
      %73 = arith.subi %70, %69 : i32
      %74 = arith.subi %72, %71 : i32
      %75 = arith.subi %74, %73 : i32
      %76 = arith.addi %73, %c127_i32 : i32
      %77 = arith.divsi %76, %c128_i32 : i32
      %78 = arith.muli %77, %c12_i32 : i32
      %79 = arith.remsi %68, %arg23 : i32
      %80 = arith.muli %69, %arg11 : i32
      %81 = arith.muli %69, %arg13 : i32
      %82 = arith.index_cast %69 : i32 to index
      %83 = arith.index_cast %73 : i32 to index
      %84 = arith.index_cast %80 : i32 to index
      %85 = arith.index_cast %81 : i32 to index
      scf.for %arg27 = %79 to %78 step %arg23  : i32 {
        %86 = arith.divsi %arg27, %c12_i32 : i32
        %87 = arith.muli %86, %c128_i32 : i32
        %88 = arith.addi %87, %c128_i32 : i32
        %89 = arith.minsi %88, %73 : i32
        %90 = arith.subi %89, %87 : i32
        %91 = arith.addi %87, %75 : i32
        %92 = arith.addi %91, %90 : i32
        %93 = arith.addi %92, %c127_i32 : i32
        %94 = arith.divsi %93, %c128_i32 : i32
        %95 = arith.minsi %94, %c1_i32 : i32
        %96 = arith.subi %91, %c1023_i32 : i32
        %97 = arith.maxsi %96, %c0_i32 : i32
        %98 = arith.divsi %97, %c128_i32 : i32
        %99 = arith.maxsi %95, %98 : i32
        %100 = arith.subi %94, %99 : i32
        %101 = arith.addi %95, %100 : i32
        %102 = arith.subi %99, %95 : i32
        %103 = arith.cmpi sge, %91, %74 : i32
        %104 = arith.extui %103 : i1 to i32
        %105 = arith.minsi %91, %c2176_i32 : i32
        %106 = arith.subi %c1_i32, %104 : i32
        %107 = arith.muli %106, %105 : i32
        %108 = arith.muli %104, %c2304_i32 : i32
        %109 = arith.addi %107, %108 : i32
        %110 = arith.subi %91, %109 : i32
        %111 = arith.cmpi ne, %110, %c0_i32 : i32
        %112 = arith.extui %111 : i1 to i32
        %113 = arith.index_cast %109 : i32 to index
        %114 = arith.remsi %arg27, %c12_i32 : i32
        %115 = arith.muli %114, %arg15 : i32
        %116 = arith.index_cast %115 : i32 to index
        %117 = arith.maxsi %87, %c0_i32 : i32
        %118 = arith.index_cast %117 : i32 to index
        %119 = affine.apply #map2()[%83, %118]
        %120 = arith.subi %c0_i32, %87 : i32
        %121 = arith.maxsi %120, %c0_i32 : i32
        %122 = arith.index_cast %121 : i32 to index
        %123 = arith.index_cast %87 : i32 to index
        %124 = affine.apply #map10()[%123, %116, %82]
        %125 = affine.apply #map11()[%123]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%124], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %126 = arith.maxsi %123, %83 : index
        %127 = arith.minsi %125, %126 : index
        %128 = affine.apply #map2()[%127, %123]
        %129 = arith.minsi %65, %128 : index
        %130 = affine.apply #map12()[%129, %127, %123]
        %131 = arith.minsi %130, %c64 : index
        %subview = memref.subview %reinterpret_cast_3[%65] [%131] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %132 = arith.muli %114, %arg12 : i32
        %133 = arith.index_cast %132 : i32 to index
        %134 = affine.apply #map1()[%118, %62, %84, %133]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%134], sizes: [128, 128], strides: [%62, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %135 = arith.maxsi %119, %c0 : index
        %136 = arith.minsi %135, %c128 : index
        %137 = arith.minsi %122, %136 : index
        %138 = affine.apply #map2()[%136, %137]
        %139 = arith.minsi %65, %138 : index
        %140 = affine.apply #map12()[%139, %136, %137]
        %141 = arith.minsi %140, %c64 : index
        %subview_5 = memref.subview %reinterpret_cast_4[%65, 0] [%141, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %142 = arith.muli %114, %arg14 : i32
        %143 = arith.index_cast %142 : i32 to index
        %144 = affine.apply #map1()[%118, %63, %85, %143]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%144], sizes: [128, 128], strides: [%63, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[%65, 0] [%141, 128] [1, 1] : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %145 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %subview_8 = memref.subview %145[0, %66, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x4x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        annotation.mark %145 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %146 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %subview_9 = memref.subview %146[0, %66, 0, 0] [8, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<8x4x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        annotation.mark %146 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 1 : index} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %147 = hivm.hir.pointer_cast(%c74752_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %147 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %148 = hivm.hir.pointer_cast(%c107520_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %148 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        %149 = hivm.hir.pointer_cast(%c140288_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %149 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        %150 = hivm.hir.pointer_cast(%c173056_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        annotation.mark %150 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        %151 = hivm.hir.pointer_cast(%c205824_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %151 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64xf32, #hivm.address_space<ub>>
        %152 = hivm.hir.pointer_cast(%c206080_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %152 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64xf32, #hivm.address_space<ub>>
        %153 = hivm.hir.pointer_cast(%c206336_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %153 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        %154 = arith.addi %101, %c3_i32 : i32
        %155 = hivm.hir.pointer_cast(%c207104_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        func.call @_swa_fwd_kernel_mix_aiv_outlined_vf_3(%155) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>) -> ()
        %156 = hivm.hir.pointer_cast(%c206848_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%60 : memref<64xf32, #hivm.address_space<ub>>) outs(%156 : memref<64xf32, #hivm.address_space<ub>>)
        %157 = hivm.hir.pointer_cast(%c239872_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%61 : memref<64xf32, #hivm.address_space<ub>>) outs(%157 : memref<64xf32, #hivm.address_space<ub>>)
        %158:6 = scf.for %arg28 = %c0_i32 to %154 step %c1_i32 iter_args(%arg29 = %155, %arg30 = %156, %arg31 = %157, %arg32 = %c0_i32, %arg33 = %c0_i32, %arg34 = %c0_i32) -> (memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, i32, i32, i32)  : i32 {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %167 = llvm.load volatile %49 : !llvm.ptr<11> -> i32
          %168 = arith.cmpi sgt, %167, %c0_i32 : i32
          %169 = llvm.load volatile %51 : !llvm.ptr<11> -> i32
          %170 = arith.cmpi sgt, %169, %c0_i32 : i32
          %171 = arith.andi %168, %170 : i1
          %172 = llvm.load volatile %53 : !llvm.ptr<11> -> i32
          %173 = arith.cmpi sgt, %172, %c0_i32 : i32
          %174 = arith.andi %171, %173 : i1
          %175 = llvm.load volatile %55 : !llvm.ptr<11> -> i32
          %176 = arith.cmpi slt, %175, %c2_i32 : i32
          %177 = arith.andi %174, %176 : i1
          %178 = llvm.load volatile %57 : !llvm.ptr<11> -> i32
          %179 = arith.cmpi slt, %178, %c2_i32 : i32
          %180 = arith.andi %177, %179 : i1
          %181 = llvm.load volatile %59 : !llvm.ptr<11> -> i32
          %182 = arith.cmpi slt, %181, %c2_i32 : i32
          %183 = arith.andi %180, %182 : i1
          %184 = arith.cmpi slt, %arg34, %c3_i32 : i32
          %185 = arith.cmpi slt, %arg32, %101 : i32
          %186 = arith.andi %183, %184 : i1
          %187 = arith.andi %186, %185 : i1
          %188:4 = scf.if %187 -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, i32, i32) {
            %203 = arith.remsi %arg32, %c2_i32 : i32
            %204 = arith.cmpi eq, %203, %c0_i32 : i32
            %205 = arith.select %204, %147, %148 : memref<64x128xf32, #hivm.address_space<ub>>
            scf.if %204 {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
            } else {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
            }
            %206 = arith.cmpi sge, %arg32, %95 : i32
            %207 = arith.extui %206 : i1 to i32
            %208 = arith.muli %207, %102 : i32
            %209 = arith.addi %arg32, %208 : i32
            %210 = arith.muli %209, %c128_i32 : i32
            %211 = arith.cmpi slt, %210, %c4_i32 : i32
            %212 = arith.extui %211 : i1 to i32
            %213 = arith.subi %91, %210 : i32
            %214 = arith.cmpi sle, %213, %c1023_i32 : i32
            %215 = arith.extui %214 : i1 to i32
            %216 = arith.subi %c1_i32, %212 : i32
            %217 = arith.muli %212, %215 : i32
            %218 = arith.addi %216, %217 : i32
            %219 = arith.muli %112, %218 : i32
            %220 = arith.subi %c1_i32, %219 : i32
            %221 = arith.muli %220, %210 : i32
            %222 = arith.subi %210, %110 : i32
            %223 = arith.maxsi %222, %c5_i32 : i32
            %224 = arith.muli %219, %223 : i32
            %225 = arith.addi %221, %224 : i32
            %226 = arith.muli %225, %106 : i32
            %227 = arith.index_cast %226 : i32 to index
            %228 = affine.apply #map13()[%227, %113]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg22 to offset: [%228], sizes: [128, 128], strides: [2816, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<128x128xi8, strided<[2816, 1], offset: ?>, #hivm.address_space<gm>>
            %subview_14 = memref.subview %reinterpret_cast_13[%65, 0] [64, 128] [1, 1] : memref<128x128xi8, strided<[2816, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xi8, strided<[2816, 1], offset: ?>, #hivm.address_space<gm>>
            %229 = hivm.hir.pointer_cast(%c0_i64) : memref<64x128xi8, #hivm.address_space<ub>>
            annotation.mark %229 {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload} : memref<64x128xi8, #hivm.address_space<ub>>
            hivm.hir.load ins(%subview_14 : memref<64x128xi8, strided<[2816, 1], offset: ?>, #hivm.address_space<gm>>) outs(%229 : memref<64x128xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
            %230 = hivm.hir.pointer_cast(%c206592_i64) : memref<64xf32, #hivm.address_space<ub>>
            %231 = hivm.hir.pointer_cast(%c24832_i64) : memref<64x128xf32, #hivm.address_space<ub>>
            %232 = hivm.hir.pointer_cast(%c58112_i64) : memref<64xf32, #hivm.address_space<ub>>
            func.call @_swa_fwd_kernel_mix_aiv_outlined_vf_0(%229, %205, %arg10, %230, %231, %arg31, %232) {hivm.vector_function, no_inline} : (memref<64x128xi8, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, f32, memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
            %233 = hivm.hir.pointer_cast(%c8192_i64) : memref<8x65x16x1xbf16, #hivm.address_space<ub>>
            %subview_15 = memref.subview %233[0, 0, 0, 0] [8, 64, 16, 1] [1, 1, 1, 1] : memref<8x65x16x1xbf16, #hivm.address_space<ub>> to memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>
            %234 = hivm.hir.pointer_cast(%c57600_i64) : memref<64xf32, #hivm.address_space<ub>>
            %235 = hivm.hir.pointer_cast(%c206848_i64) : memref<64xf32, #hivm.address_space<ub>>
            %236 = hivm.hir.pointer_cast(%c57856_i64) : memref<64xf32, #hivm.address_space<ub>>
            func.call @_swa_fwd_kernel_mix_aiv_outlined_vf_1(%232, %231, %subview_15, %234, %arg30, %arg31, %235, %236) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
            %expand_shape = memref.expand_shape %subview_15 [[0], [1, 2], [3]] output_shape [8, 4, 16, 16] : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> into memref<8x4x16x16xbf16, strided<[1040, 256, 16, 1]>, #hivm.address_space<ub>>
            %237 = arith.remsi %arg32, %c3_i32 : i32
            %238 = arith.cmpi eq, %237, %c0_i32 : i32
            scf.if %238 {
              hivm.hir.copy ins(%236 : memref<64xf32, #hivm.address_space<ub>>) outs(%151 : memref<64xf32, #hivm.address_space<ub>>) {tiled_op}
            } else {
              %253 = arith.cmpi eq, %237, %c1_i32 : i32
              scf.if %253 {
                hivm.hir.copy ins(%236 : memref<64xf32, #hivm.address_space<ub>>) outs(%152 : memref<64xf32, #hivm.address_space<ub>>) {tiled_op}
              } else {
                hivm.hir.copy ins(%236 : memref<64xf32, #hivm.address_space<ub>>) outs(%153 : memref<64xf32, #hivm.address_space<ub>>) {tiled_op}
              }
            }
            scf.if %204 {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
              %collapse_shape = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<8x4x16x16xbf16, strided<[1040, 256, 16, 1]>, #hivm.address_space<ub>> into memref<8x1024xbf16, strided<[1040, 1]>, #hivm.address_space<ub>>
              %collapse_shape_16 = memref.collapse_shape %subview_8 [[0], [1, 2, 3]] : memref<8x4x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<8x1024xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
              hivm.hir.copy ins(%collapse_shape : memref<8x1024xbf16, strided<[1040, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8x1024xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>) {tiled_op}
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
            } else {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
              %collapse_shape = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<8x4x16x16xbf16, strided<[1040, 256, 16, 1]>, #hivm.address_space<ub>> into memref<8x1024xbf16, strided<[1040, 1]>, #hivm.address_space<ub>>
              %collapse_shape_16 = memref.collapse_shape %subview_9 [[0], [1, 2, 3]] : memref<8x4x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<8x1024xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
              hivm.hir.copy ins(%collapse_shape : memref<8x1024xbf16, strided<[1040, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8x1024xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>) {tiled_op}
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
            }
            %239 = llvm.load volatile %49 : !llvm.ptr<11> -> i32
            %240 = arith.subi %239, %c1_i32 : i32
            llvm.store volatile %240, %49 : i32, !llvm.ptr<11>
            %241 = llvm.load volatile %51 : !llvm.ptr<11> -> i32
            %242 = arith.subi %241, %c1_i32 : i32
            llvm.store volatile %242, %51 : i32, !llvm.ptr<11>
            %243 = llvm.load volatile %53 : !llvm.ptr<11> -> i32
            %244 = arith.subi %243, %c1_i32 : i32
            llvm.store volatile %244, %53 : i32, !llvm.ptr<11>
            %245 = llvm.load volatile %55 : !llvm.ptr<11> -> i32
            %246 = arith.addi %245, %c1_i32 : i32
            llvm.store volatile %246, %55 : i32, !llvm.ptr<11>
            %247 = llvm.load volatile %57 : !llvm.ptr<11> -> i32
            %248 = arith.addi %247, %c1_i32 : i32
            llvm.store volatile %248, %57 : i32, !llvm.ptr<11>
            %249 = llvm.load volatile %59 : !llvm.ptr<11> -> i32
            %250 = arith.addi %249, %c1_i32 : i32
            llvm.store volatile %250, %59 : i32, !llvm.ptr<11>
            %251 = arith.addi %arg34, %c1_i32 : i32
            %252 = arith.addi %arg32, %c1_i32 : i32
            scf.yield %232, %235, %251, %252 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, i32, i32
          } else {
            %203 = hivm.hir.pointer_cast(%c58112_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%arg31 : memref<64xf32, #hivm.address_space<ub>>) outs(%203 : memref<64xf32, #hivm.address_space<ub>>)
            scf.yield %203, %arg30, %arg34, %arg32 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, i32, i32
          }
          %189 = llvm.load volatile %44 : !llvm.ptr<11> -> i32
          %190 = arith.cmpi sgt, %189, %c0_i32 : i32
          %191 = llvm.load volatile %46 : !llvm.ptr<11> -> i32
          %192 = arith.cmpi sgt, %191, %c0_i32 : i32
          %193 = arith.andi %190, %192 : i1
          %194 = llvm.load volatile %48 : !llvm.ptr<11> -> i32
          %195 = arith.cmpi sgt, %194, %c0_i32 : i32
          %196 = arith.andi %193, %195 : i1
          %197 = arith.cmpi sgt, %188#2, %c0_i32 : i32
          %198 = arith.cmpi slt, %arg33, %101 : i32
          %199 = arith.andi %196, %197 : i1
          %200 = arith.andi %199, %198 : i1
          %201:3 = scf.if %200 -> (memref<64x128xf32, #hivm.address_space<ub>>, i32, i32) {
            %203 = arith.remsi %arg33, %c2_i32 : i32
            %204 = arith.cmpi eq, %203, %c0_i32 : i32
            %205 = arith.select %204, %149, %150 : memref<64x128xf32, #hivm.address_space<ub>>
            scf.if %204 {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            } else {
              hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
            }
            %206 = arith.remsi %arg33, %c3_i32 : i32
            %207 = arith.cmpi eq, %206, %c0_i32 : i32
            %208 = scf.if %207 -> (memref<64xf32, #hivm.address_space<ub>>) {
              scf.yield %151 : memref<64xf32, #hivm.address_space<ub>>
            } else {
              %218 = arith.cmpi eq, %206, %c1_i32 : i32
              %219 = arith.select %218, %152, %153 : memref<64xf32, #hivm.address_space<ub>>
              scf.yield %219 : memref<64xf32, #hivm.address_space<ub>>
            }
            %209 = hivm.hir.pointer_cast(%c207104_i64) : memref<64x128xf32, #hivm.address_space<ub>>
            func.call @_swa_fwd_kernel_mix_aiv_outlined_vf_2(%208, %205, %arg29, %209) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
            scf.if %204 {
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
            } else {
              hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
            } {ssbuffer.cross_buffer = 1 : i32}
            %210 = llvm.load volatile %44 : !llvm.ptr<11> -> i32
            %211 = arith.subi %210, %c1_i32 : i32
            llvm.store volatile %211, %44 : i32, !llvm.ptr<11>
            %212 = llvm.load volatile %46 : !llvm.ptr<11> -> i32
            %213 = arith.subi %212, %c1_i32 : i32
            llvm.store volatile %213, %46 : i32, !llvm.ptr<11>
            %214 = llvm.load volatile %48 : !llvm.ptr<11> -> i32
            %215 = arith.subi %214, %c1_i32 : i32
            llvm.store volatile %215, %48 : i32, !llvm.ptr<11>
            %216 = arith.subi %188#2, %c1_i32 : i32
            %217 = arith.addi %arg33, %c1_i32 : i32
            scf.yield %209, %216, %217 : memref<64x128xf32, #hivm.address_space<ub>>, i32, i32
          } else {
            scf.yield %arg29, %188#2, %arg33 : memref<64x128xf32, #hivm.address_space<ub>>, i32, i32
          }
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %202 = hivm.hir.pointer_cast(%c239872_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%188#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%202 : memref<64xf32, #hivm.address_space<ub>>)
          scf.yield %201#0, %188#1, %202, %188#3, %201#2, %201#1 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, i32, i32, i32
        }
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        %159 = hivm.hir.pointer_cast(%c24832_i64) : memref<64xf32, #hivm.address_space<ub>>
        %160 = hivm.hir.pointer_cast(%c25088_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
        %161 = hivm.hir.pointer_cast(%c41472_i64) : memref<64x128xf32, #hivm.address_space<ub>>
        func.call @_swa_fwd_kernel_mix_aiv_outlined_merged_vf_0(%158#2, %158#1, %159, %158#1, %158#0, %160, %161) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xbf16, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
        %subview_10 = memref.subview %159[0] [%131] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.store ins(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        %162 = arith.maxsi %137, %65 : index
        %163 = arith.minsi %136, %67 : index
        %164 = arith.maxsi %162, %163 : index
        %165 = affine.apply #map2()[%164, %162]
        %166 = affine.apply #map14()[%162, %64]
        %subview_11 = memref.subview %160[%166, 0] [%165, 128] [1, 1] : memref<64x128xbf16, #hivm.address_space<ub>> to memref<?x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.store ins(%subview_11 : memref<?x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?x128xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        %subview_12 = memref.subview %161[%166, 0] [%165, 128] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.store ins(%subview_12 : memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    return
  }
}
