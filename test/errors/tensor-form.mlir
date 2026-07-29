// The interpreter only runs memref-form HIVM. Tensor operands must produce a
// message that names the actual problem, not a downstream "unbound operand"
// from whichever handler happened to look at the value first.

// RUN: not npuir-interp %s --sched=inorder --args=zeros,zeros 2>&1 | FileCheck %s

// CHECK: error: tensor-form operand or result on 'bufferization.to_tensor'
// CHECK-SAME: only accepts fully bufferized (memref) HIVM IR

module {
  func.func @half_bufferized(%in: memref<8xf32, #hivm.address_space<gm>>,
                             %out: memref<8xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry} {
    %t = bufferization.to_tensor %in restrict
        : memref<8xf32, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%t : tensor<8xf32>)
                  outs(%dst : memref<8xf32, #hivm.address_space<ub>>)
    return
  }
}
