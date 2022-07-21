// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 4, 10)>
// CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (d1 + 2, d0 + 4, 10)>

func.func @simple_block() {
  // CHECK-LABEL: simple_block
  // CHECK-NEXT: affine.for [[OUTER_LOOP:%.+]] = 0 to 10 step 2 {
  // CHECK-NEXT:   affine.for [[INNER_LOOP:%.+]] = #map{{.*}}([[OUTER_LOOP]]) to #map{{.*}}([[OUTER_LOOP]]) {
  // CHECK-NEXT:     %0 = arith.addi [[INNER_LOOP]], [[INNER_LOOP]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib, %il) with (%ii -> %i = 0 to 10) {
    %foo = arith.addi %i, %i : index
  }

  return
}

func.func @block_nested() {
  // CHECK-LABEL: block_nested
  // CHECK-NEXT: affine.for [[OUTER_LOOP:%.+]] = 0 to 10 step 4 {
  // CHECK-NEXT:   affine.for [[MIDDLE_LOOP:%.+]] = #map{{.*}}([[OUTER_LOOP]]) to min #map{{.*}}([[OUTER_LOOP]]) step 2 {
  // CHECK-NEXT:     affine.for [[INNER_LOOP:%.+]] = #map{{.*}}([[MIDDLE_LOOP]]) to min #map{{.*}}([[OUTER_LOOP]], [[MIDDLE_LOOP]]) {
  // CHECK-NEXT:       %0 = arith.addi [[INNER_LOOP]], [[INNER_LOOP]] : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %ilb, %ill = krnl.block %il 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib, %ilb, %ill) with (%ii -> %i = 0 to 10) {
    %foo = arith.addi %i, %i : index
  }

  return
}