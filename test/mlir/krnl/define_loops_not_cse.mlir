// RUN: onnx-mlir-opt --cse %s -split-input-file | FileCheck %s

// krnl.define_loops hands out fresh loop references, whose only distinguishing
// property is their identity. Two such ops are structurally identical -- no
// operands, same result types -- so a purity trait on this op is an invitation
// for CSE to merge them and alias two unrelated loop nests.
//
// This is a wrong-answer miscompile rather than an error. In the nest below the
// merge makes both krnl.iterate ops drive the same two loop references, and the
// inner krnl.get_induction_var_value -- identical to the outer one once its
// operands alias -- is merged into it as well, leaving
// krnl.load %arg0[%a, %b, %a, %b]. Dimension 0 then has extent 4 but is indexed
// by a loop running to 6, so the lowered code reads and writes past the end of
// the buffer. Hence KrnlDefineLoopsOp carries no NoMemoryEffect: it is not
// redundant with anything, ever.
//
// Nothing is lost by that. The op is erased by ConvertKrnlToAffine itself rather
// than by dead-code elimination, hoisting a loop reference out of a loop buys
// nothing since the op does not survive that pass, and krnl.block, krnl.collapse,
// krnl.permute and krnl.unroll have always been opaque in exactly the same way.
func.func @two_define_loops_are_not_redundant(%arg0: memref<4x5x6x7xf32>) -> memref<4x5x6x7xf32> {
  %alloc = memref.alloc() : memref<4x5x6x7xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %kk, %ll = krnl.define_loops 2
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      krnl.store %v, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>

  // Both define_loops survive, and the inner nest keeps its own two references
  // and its own induction variable query.
  // CHECK-LABEL: two_define_loops_are_not_redundant
  // CHECK:         [[OUTER:%.+]]:2 = krnl.define_loops 2
  // CHECK:         krnl.iterate([[OUTER]]#0, [[OUTER]]#1) with ([[OUTER]]#0 -> %{{.+}} = 0 to 4, [[OUTER]]#1 -> %{{.+}} = 0 to 5)
  // CHECK:           [[OUTER_IV:%.+]]:2 = krnl.get_induction_var_value([[OUTER]]#0, [[OUTER]]#1)
  // CHECK:           [[INNER:%.+]]:2 = krnl.define_loops 2
  // CHECK:           krnl.iterate([[INNER]]#0, [[INNER]]#1) with ([[INNER]]#0 -> %{{.+}} = 0 to 6, [[INNER]]#1 -> %{{.+}} = 0 to 7)
  // CHECK:             [[INNER_IV:%.+]]:2 = krnl.get_induction_var_value([[INNER]]#0, [[INNER]]#1)
  // CHECK:             krnl.load %arg0{{.}}[[OUTER_IV]]#0, [[OUTER_IV]]#1, [[INNER_IV]]#0, [[INNER_IV]]#1] : memref<4x5x6x7xf32>
}
