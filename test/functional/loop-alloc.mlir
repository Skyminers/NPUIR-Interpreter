// A memref.alloc inside a loop must not consume fresh on-chip storage on
// every iteration. PlanMemory assigns one address per alloc site, so bump
// allocating per execution would exhaust UB and report a capacity overflow
// for a kernel that fits comfortably - a false alarm that would make the
// tool unusable on any tiled kernel.
//
// 32 iterations, 1 KiB per iteration, in a 8 KiB UB. Per-site reuse fits;
// per-execution allocation would need 32 KiB.
//
// Each iteration loads a tile, adds 1, and stores it back: out = in + 1.

// RUN: npuir-interp %s --sched=lazy --ub-size=8192 --args=arange,zeros --out=%t. 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --ub-size=8192 --args=arange,zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N32 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK-NOT: capacity exceeded
// CHECK-NOT: MISSING SYNC
// DATA:      1 2 3 4
// DATA-NEXT: 5 6 7 8

module {
  func.func @tiled(%in: memref<256xi32, #hivm.address_space<gm>>,
                   %out: memref<256xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %one = arith.constant 1 : i32

    scf.for %i = %c0 to %c256 step %c8 {
      %inTile = memref.subview %in[%i] [8] [1]
          : memref<256xi32, #hivm.address_space<gm>>
         to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %outTile = memref.subview %out[%i] [8] [1]
          : memref<256xi32, #hivm.address_space<gm>>
         to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>

      // 256 i32 = 1 KiB, allocated fresh on every one of the 32 iterations.
      %buf = memref.alloc() : memref<256xi32, #hivm.address_space<ub>>
      %ones = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
      %sum = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
      %tile = memref.subview %buf[0] [8] [1]
          : memref<256xi32, #hivm.address_space<ub>>
         to memref<8xi32, strided<[1]>, #hivm.address_space<ub>>

      hivm.hir.load ins(%inTile : memref<8xi32, strided<[1], offset: ?>,
                                         #hivm.address_space<gm>>)
                    outs(%tile : memref<8xi32, strided<[1]>,
                                        #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%one : i32)
                    outs(%ones : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vadd ins(%tile, %ones : memref<8xi32, strided<[1]>,
                                              #hivm.address_space<ub>>,
                                        memref<8xi32, #hivm.address_space<ub>>)
                    outs(%sum : memref<8xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.store ins(%sum : memref<8xi32, #hivm.address_space<ub>>)
                     outs(%outTile : memref<8xi32, strided<[1], offset: ?>,
                                             #hivm.address_space<gm>>)
      // The next iteration reuses these buffers.
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    }
    return
  }
}
