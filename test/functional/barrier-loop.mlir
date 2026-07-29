// A sync_block[<ALL>] barrier reached repeatedly from inside a loop. Both
// cores must pass it three times: the barrier has to rearm once the last
// participant is through, or the second iteration deadlocks.
//
// Also checks that the barrier's clock merge is a real happens-before edge:
// the AIC writes the buffer the AIV reads with nothing but barriers between
// them, so a missing edge would show up as a spurious DATA RACE.
//
// Two barriers per iteration, not one. With a single barrier the AIC's write
// for iteration i+1 sits in the same barrier interval as the AIV's read for
// iteration i, and those really are concurrent - the second barrier is what
// keeps the producer from running ahead into the consumer's window.

// RUN: npuir-interp %s --sched=lazy --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --args=zeros,zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N16 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK: 2 core(s)
// CHECK-NOT: DEADLOCK
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC

// The AIC bumps the counter once per iteration; the AIV copies it out after
// the barrier, so the last value it sees is 3.
// DATA: 3 3 3 3

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @k_mix_aic(%scratch: memref<4xi32, #hivm.address_space<gm>>,
                       %out: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %one = arith.constant 1 : i32
    scf.for %i = %c0 to %c3 step %c1 {
      %cur = memref.load %scratch[%c0] : memref<4xi32, #hivm.address_space<gm>>
      %next = arith.addi %cur, %one : i32
      memref.store %next, %scratch[%c0]
          : memref<4xi32, #hivm.address_space<gm>>
      memref.store %next, %scratch[%c1]
          : memref<4xi32, #hivm.address_space<gm>>
      // Data ready.
      hivm.hir.sync_block [<ALL>, 1] tcube_pipe = <PIPE_M>
                                     tvector_pipe = <PIPE_MTE3>
      // Consumer done; safe to overwrite on the next iteration.
      hivm.hir.sync_block [<ALL>, 2] tcube_pipe = <PIPE_M>
                                     tvector_pipe = <PIPE_MTE3>
    }
    return
  }

  func.func @k_mix_aiv(%scratch: memref<4xi32, #hivm.address_space<gm>>,
                       %out: memref<4xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      hivm.hir.sync_block [<ALL>, 1] tcube_pipe = <PIPE_M>
                                     tvector_pipe = <PIPE_MTE3>
      // Ordered after the AIC's store purely by the barrier.
      %v = memref.load %scratch[%c0] : memref<4xi32, #hivm.address_space<gm>>
      memref.store %v, %out[%c0] : memref<4xi32, #hivm.address_space<gm>>
      memref.store %v, %out[%c1] : memref<4xi32, #hivm.address_space<gm>>
      memref.store %v, %out[%c2] : memref<4xi32, #hivm.address_space<gm>>
      memref.store %v, %out[%c3] : memref<4xi32, #hivm.address_space<gm>>
      hivm.hir.sync_block [<ALL>, 2] tcube_pipe = <PIPE_M>
                                     tvector_pipe = <PIPE_MTE3>
    }
    return
  }
}
