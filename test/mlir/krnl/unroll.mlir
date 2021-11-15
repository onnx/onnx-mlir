// RUN: onnx-mlir-opt --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func @simple_unroll() {
  %ii = krnl.define_loops 1
  krnl.unroll %ii : !krnl.loop
  krnl.iterate(%ii) with (%ii -> %i = 0 to 4) {
    %c1 = arith.constant 1 : index
    %foo = arith.addi %i, %c1 : index
  }

  // CHECK-LABEL: simple_unroll
  // CHECK-NEXT: [[CONST_IV_INIT:%.+]] = arith.constant 0 : index
  // CHECK-NEXT: [[CONST_ONE_0:%.+]] = arith.constant 1 : index
  // CHECK-NEXT: [[FIRST_RES:%.+]] = arith.addi [[CONST_IV_INIT]], [[CONST_ONE_0]] : index
  //CHECK-NEST:  [[IV_TWO:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_1:%.+]] = arith.constant 1 : index
  //CHECK-NEST:  %2 = arith.addi %1, [[CONST_ONE_1]] : index
  //CHECK-NEST:  [[IV_THREE:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_2:%.+]] = arith.constant 1 : index
  //CHECK-NEST:  %4 = arith.addi %3, [[CONST_ONE_2]] : index
  //CHECK-NEST:  [[IV_FOUR:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_3:%.+]] = arith.constant 1 : index
  //CHECK-NEST:  %6 = arith.addi %5, [[CONST_ONE_3]] : index
  return
}

// -----

func @unroll_with_block() {
  %ii = krnl.define_loops 1
  %ii1, %ii2 = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.unroll %ii2 : !krnl.loop
  krnl.iterate(%ii1) with (%ii -> %i = 0 to 8) {
    krnl.iterate(%ii2) with () {
      %i2 = krnl.get_induction_var_value(%ii2) : (!krnl.loop) -> index
      %foo = arith.addi %i2, %i2 : index
    }
  }
  return

  // CHECK: #map = affine_map<(d0) -> (d0 + 1)>
  // CHECK-LABEL: unroll_with_block
  // CHECK:       affine.for [[IV:%.+]] = 0 to 8 step 2 {
  // CHECK-NEXT:    [[FOO_UNROLL_0:%.+]] = arith.addi [[IV]], [[IV]] : index
  // CHECK-NEXT:    [[IV_PLUS_1:%.+]] = affine.apply #map([[IV]])
  // CHECK-NEXT:    [[FOO_UNROLL_1:%.+]] = arith.addi [[IV_PLUS_1]], [[IV_PLUS_1]]  : index
  // CHECK-NEXT:  }
}

// -----

func @unroll_with_block_get_iv(%arg0 : memref<8xf32>) {
  %ii = krnl.define_loops 1
  %ii1, %ii2 = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.unroll %ii2 : !krnl.loop
  krnl.iterate(%ii1) with (%ii -> %i = 0 to 8) {
    %i1 = krnl.get_induction_var_value(%ii1) : (!krnl.loop) -> index
    krnl.iterate(%ii2) with () {
      %i2 = krnl.get_induction_var_value(%ii2) : (!krnl.loop) -> index
      %foo = arith.addi %i1, %i2 : index
    }
  }
  return

  // CHECK: #map = affine_map<(d0) -> (d0 + 1)>
  // CHECK-LABEL: unroll_with_block_get_iv
  // CHECK:       affine.for [[IV:%.+]] = 0 to 8 step 2 {
  // CHECK-NEXT:    [[FOO_UNROLL_0:%.+]] = arith.addi [[IV]], [[IV]] : index
  // CHECK-NEXT:    [[IV_PLUS_1:%.+]] = affine.apply #map([[IV]])
  // CHECK-NEXT:    [[FOO_UNROLL_1:%.+]] = arith.addi [[IV]], [[IV_PLUS_1]] : index
  // CHECK-NEXT:  }
}
