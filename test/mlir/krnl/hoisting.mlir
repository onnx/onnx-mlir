// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

// Hoist invariant instructions outside of the loop.
func.func @simple_block(%arg0 : memref<?xf32>) {
  // CHECK-LABEL: simple_block
  // CHECK-NEXT:  arith.constant
  // CHECK-NEXT:  memref.dim 
  // CHECK-NEXT:  affine.for
  // CHECK-NEXT:    affine.for
  // CHECK-NEXT:      affine.for
  // CHECK-NEXT:        arith.addi
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }

  %c0 = arith.constant 0 : index
  %ii = krnl.define_loops 1
  krnl.iterate(%ii) with (%ii -> %i = 0 to 1) {
    %0 = memref.dim %arg0, %c0 : memref<?xf32>
    %jj = krnl.define_loops 1
    %jb, %jl = krnl.block %jj 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    krnl.iterate(%jb, %jl) with (%jj -> %j = 0 to %0) {
      %foo = arith.addi %j, %j : index
    }
  }
  return
}

