// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @simple_unroll() {
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

func.func @unroll_with_block() {
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

  // CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
  // CHECK-LABEL: unroll_with_block
  // CHECK:       affine.for [[IV:%.+]] = 0 to 8 step 2 {
  // CHECK-NEXT:    [[FOO_UNROLL_0:%.+]] = arith.addi [[IV]], [[IV]] : index
  // CHECK-NEXT:    [[IV_PLUS_1:%.+]] = affine.apply #map([[IV]])
  // CHECK-NEXT:    [[FOO_UNROLL_1:%.+]] = arith.addi [[IV_PLUS_1]], [[IV_PLUS_1]]  : index
  // CHECK-NEXT:  }
}

// -----

func.func @unroll_with_block_get_iv(%arg0 : memref<8xf32>) {
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

  // CHECK-DAG: [[MAP:#.+]] = affine_map<(d0) -> (d0 + 1)>
  // CHECK-LABEL: unroll_with_block_get_iv
  // CHECK:       affine.for [[IV:%.+]] = 0 to 8 step 2 {
  // CHECK-NEXT:    [[FOO_UNROLL_0:%.+]] = arith.addi [[IV]], [[IV]] : index
  // CHECK-NEXT:    [[IV_PLUS_1:%.+]] = affine.apply [[MAP]]([[IV]])
  // CHECK-NEXT:    [[FOO_UNROLL_1:%.+]] = arith.addi [[IV]], [[IV_PLUS_1]] : index
  // CHECK-NEXT:  }
}

// -----

func.func @unroll_with_block_and_permute() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.unroll %jl : !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %b1, %b2 = krnl.get_induction_var_value(%ib, %jb) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%il, %jl) with () {
      %l1, %l2 = krnl.get_induction_var_value(%il, %jl) : (!krnl.loop, !krnl.loop) -> (index, index)
      %foo = arith.addi %l1, %l2 : index
      %bar = arith.addi %b1, %l2 : index
    }
  }
  return

  // CHECK-DAG: #map = affine_map<(d0) -> (d0)>
  // CHECK-DAG: #map1 = affine_map<(d0) -> (d0 + 5)>
  // CHECK-DAG: #map2 = affine_map<(d0) -> (d0 + 1)>
  // CHECK-DAG: #map3 = affine_map<(d0) -> (d0 + 2)>
  // CHECK-DAG: #map4 = affine_map<(d0) -> (d0 + 3)>
  // CHECK-LABEL:  unroll_with_block_and_permute
  // CHECK:        affine.for [[I_0_:%.+]] = 0 to 10 step 5 {
  // CHECK:          affine.for [[I_1_:%.+]] = 0 to 20 step 4 {
  // CHECK:            affine.for [[I_2_:%.+]] = #map([[I_0_]]) to #map1([[I_0_]]) {
  // CHECK-NEXT:         [[VAR_0_:%.+]] = arith.addi [[I_2_]], [[I_1_]] : index
  // CHECK-NEXT:         [[VAR_1_:%.+]] = arith.addi [[I_0_]], [[I_1_]] : index
  // CHECK-NEXT:         [[VAR_2_:%.+]] = affine.apply #map2([[I_1_]])
  // CHECK-NEXT:         [[VAR_3_:%.+]] = arith.addi [[I_2_]], [[VAR_2_]] : index
  // CHECK-NEXT:         [[VAR_4_:%.+]] = arith.addi [[I_0_]], [[VAR_2_]] : index
  // CHECK-NEXT:         [[VAR_5_:%.+]] = affine.apply #map3([[I_1_]])
  // CHECK-NEXT:         [[VAR_6_:%.+]] = arith.addi [[I_2_]], [[VAR_5_]] : index
  // CHECK-NEXT:         [[VAR_7_:%.+]] = arith.addi [[I_0_]], [[VAR_5_]] : index
  // CHECK-NEXT:         [[VAR_8_:%.+]] = affine.apply #map4([[I_1_]])
  // CHECK-NEXT:         [[VAR_9_:%.+]] = arith.addi [[I_2_]], [[VAR_8_]] : index
  // CHECK-NEXT:         [[VAR_10_:%.+]] = arith.addi [[I_0_]], [[VAR_8_]] : index
  // CHECK:            }
  // CHECK:          }
  // CHECK:        }
}
