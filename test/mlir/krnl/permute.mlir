// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @simple_permute() {
  %ii, %jj = krnl.define_loops 2
  krnl.permute(%ii, %jj) [1, 0] : !krnl.loop, !krnl.loop
  krnl.iterate(%jj, %ii) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %foo = arith.addi %i, %i : index
  }

  // CHECK-LABEL: simple_permute
  // CHECK-NEXT: affine.for [[OUTER_LOOP_IV:%.+]] = 0 to 20 {
  // CHECK-NEXT:   affine.for [[INNER_LOOP_IV:%.+]] = 0 to 10 {
  // CHECK-NEXT:     [[ADD:%.+]] = arith.addi [[INNER_LOOP_IV]], [[INNER_LOOP_IV]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}

// -----

func.func @tiling() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb, %il, %jl) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %foo = arith.addi %i, %i : index
  }

  // CHECK-LABEL: tiling
  // CHECK-NEXT: affine.for [[I_BLOCK_IV:%.+]] = 0 to 10 step 5 {
  // CHECK-NEXT:   affine.for [[J_BLOCK_IV:%.+]] = 0 to 20 step 4 {
  // CHECK-NEXT:     affine.for [[I_LOCAL_IV:%.+]] = #map{{.*}}([[I_BLOCK_IV]]) to #map{{.*}}([[I_BLOCK_IV]]) {
  // CHECK-NEXT:       affine.for [[J_LOCAL_IV:%.+]] = #map{{.*}}([[J_BLOCK_IV]]) to #map{{.*}}([[J_BLOCK_IV]]) {
  // CHECK-NEXT:         %0 = arith.addi %arg2, %arg2 : index
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}

// -----

func.func @tiling3d() {
  %ii, %jj, %kk = krnl.define_loops 3
  // Blocking each loop by a factor of 4.
  %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %jj 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %kb, %kl = krnl.block %kk 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // Move iteration over tile coordinates to be the outer loops and iterateion over
  // the inter-tile elements to be the inner loops.
  krnl.permute(%ib, %il, %jb, %jl, %kb, %kl) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %il, %jb, %jl, %kb, %kl) with (%ii -> %i = 0 to 1024, %jj -> %j = 0 to 2048, %kk -> %k = 0 to 4096)  {
  }

  // CHECK-LABEL: tiling3d
  // CHECK-NEXT:  affine.for [[I_BLOCK_IV:%.+]] = 0 to 1024 step 4 {
  // CHECK-NEXT:    affine.for [[J_BLOCK_IV:%.+]] = 0 to 2048 step 4 {
  // CHECK-NEXT:      affine.for [[K_BLOCK_IV:%.+]] = 0 to 4096 step 4 {
  // CHECK-NEXT:        affine.for [[I_INNER_IV:%.+]] = #map([[I_BLOCK_IV]]) to #map{{.*}}([[I_BLOCK_IV]]) {
  // CHECK-NEXT:          affine.for [[J_INNER_IV:%.+]] = #map([[J_BLOCK_IV]]) to #map{{.*}}([[J_BLOCK_IV]]) {
  // CHECK-NEXT:            affine.for [[K_INNER_IV:%.+]] = #map([[K_BLOCK_IV]]) to #map{{.*}}([[K_BLOCK_IV]]) {
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  return
}
