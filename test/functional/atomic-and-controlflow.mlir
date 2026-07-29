// Two things at once, both otherwise untested:
//
//  * `hivm.hir.store ... atomic = <add>` from several blocks accumulating
//    into one GM cell. Atomic stores are read-modify-write, so the four
//    blocks must sum rather than overwrite - and because the hardware does
//    the combine, this is *not* a race.
//  * scf.while and cf.br, the control-flow shapes the frame-based executor
//    handles besides scf.for / scf.if.
//
// Each block contributes (blockIdx + 1), so the total is 1+2+3+4 = 10. The
// while loop computes that contribution by counting up, and the cf.br chain
// selects the buffer to store from.

// RUN: npuir-interp %s --sched=lazy --block-dim=4 --args=zeros 2>&1 | FileCheck %s
// RUN: npuir-interp %s --sched=lazy --block-dim=4 --args=zeros --out=%t. && \
// RUN:   od -An -td4 -j128 -N4 %t.arg0.npy | FileCheck %s --check-prefix=DATA

// CHECK: 4 core(s)
// CHECK-NOT: DATA RACE
// CHECK-NOT: MISSING SYNC
// DATA: 10

module {
  func.func @atomic_sum(%acc: memref<1xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    %block64 = hivm.hir.get_block_idx -> i64
    %block = arith.trunci %block64 : i64 to i32
    %limit = arith.addi %block, %one : i32

    // scf.while: count 0, 1, ... until we reach blockIdx + 1.
    %contribution = scf.while (%i = %zero) : (i32) -> i32 {
      %continue = arith.cmpi slt, %i, %limit : i32
      scf.condition(%continue) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %one : i32
      scf.yield %next : i32
    }

    %buf = memref.alloc() : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%contribution : i32)
                  outs(%buf : memref<1xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]

    // cf.br into a shared tail block, carrying the buffer as an argument.
    cf.br ^store(%buf : memref<1xi32, #hivm.address_space<ub>>)
  ^store(%src: memref<1xi32, #hivm.address_space<ub>>):
    hivm.hir.store ins(%src : memref<1xi32, #hivm.address_space<ub>>)
                   outs(%acc : memref<1xi32, #hivm.address_space<gm>>)
                   atomic = <add>
    return
  }
}
