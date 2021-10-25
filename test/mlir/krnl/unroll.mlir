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