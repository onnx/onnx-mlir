// RUN: onnx-mlir-opt --lower-krnl %s -split-input-file | FileCheck %s

func @simple_permute() {
  %ii, %jj = krnl.define_loops 2
  krnl.permute(%ii, %jj) [1, 0] : !krnl.loop, !krnl.loop
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %foo = addi %i, %i : index
  }

  // CHECK-LABEL: simple_permute
  // CHECK-NEXT: affine.for [[OUTER_LOOP_IV:%.+]] = 0 to 20 {
  // CHECK-NEXT:   affine.for [[INNER_LOOP_IV:%.+]] = 0 to 10 {
  // CHECK-NEXT:     [[ADD:%.+]] = addi [[INNER_LOOP_IV]], [[INNER_LOOP_IV]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}

// -----

func @tiling() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb, %il, %jl) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %foo = addi %i, %i : index
  }

  // CHECK-LABEL: tiling
  // CHECK-NEXT: affine.for [[I_BLOCK_IV:%.+]] = 0 to 10 step 5 {
  // CHECK-NEXT:   affine.for [[J_BLOCK_IV:%.+]] = 0 to 20 step 4 {
  // CHECK-NEXT:     affine.for [[I_LOCAL_IV:%.+]] = #map{{.*}}([[I_BLOCK_IV]]) to #map{{.*}}([[I_BLOCK_IV]]) {
  // CHECK-NEXT:       affine.for [[J_LOCAL_IV:%.+]] = #map{{.*}}([[J_BLOCK_IV]]) to #map{{.*}}([[J_BLOCK_IV]]) {
  // CHECK-NEXT:         %0 = addi %arg2, %arg2 : index
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}