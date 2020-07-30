// RUN: onnx-mlir-opt --lower-krnl %s -split-input-file | FileCheck %s

func @simple_unroll() {
  %ii = krnl.define_loops 1
  krnl.unroll %ii : !krnl.loop
  krnl.iterate(%ii) with (%ii -> %i = 0 to 4) {
    %c1 = constant 1 : index
    %foo = addi %i, %c1 : index
  }

  // CHECK-LABEL: simple_unroll
  // CHECK-NEXT: [[CONST_IV_INIT:%.+]] = constant 0 : index
  // CHECK-NEXT: [[CONST_ONE_0:%.+]] = constant 1 : index
  // CHECK-NEXT: [[FIRST_RES:%.+]] = addi [[CONST_IV_INIT]], [[CONST_ONE_0]] : index
  //CHECK-NEST:  [[IV_TWO:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_1:%.+]] = constant 1 : index
  //CHECK-NEST:  %2 = addi %1, [[CONST_ONE_1]] : index
  //CHECK-NEST:  [[IV_THREE:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_2:%.+]] = constant 1 : index
  //CHECK-NEST:  %4 = addi %3, [[CONST_ONE_2]] : index
  //CHECK-NEST:  [[IV_FOUR:%.+]] = affine.apply #map{{.+}}([[CONST_IV_INIT]])
  //CHECK-NEST:  [[CONST_ONE_3:%.+]] = constant 1 : index
  //CHECK-NEST:  %6 = addi %5, [[CONST_ONE_3]] : index
  return
}