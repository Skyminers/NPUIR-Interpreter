// hivm.hir.sync_block_lock / unlock serialise a critical section across
// blocks: a block enters only when lock_var equals its own index, and unlock
// hands the lock to the next one.
//
// Three blocks each read-modify-write the same GM counter. Serialisation is
// the only thing making that safe, so this doubles as a check that acquiring
// the lock carries a happens-before edge from the previous holder - without
// one the accumulation would be reported as a race.

// RUN: npuir-interp %s --sched=lazy --block-dim=3 --args=zeros,zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --block-dim=3 --args=zeros,zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N4 %t.arg1.npy | FileCheck %s --check-prefix=DATA

// CHECK: 3 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: DEADLOCK
// Three blocks, each adding one.
// DATA: 3

module {
  func.func @serialised(%lockbuf: memref<?xi8, #hivm.address_space<gm>>
                            {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
                        %counter: memref<1xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %lock = hivm.hir.create_sync_block_lock from %lockbuf
        : from memref<?xi8, #hivm.address_space<gm>> to memref<1xi64>

    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    %cur = memref.load %counter[%c0] : memref<1xi32, #hivm.address_space<gm>>
    %next = arith.addi %cur, %one : i32
    memref.store %next, %counter[%c0]
        : memref<1xi32, #hivm.address_space<gm>>
    hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    return
  }
}
