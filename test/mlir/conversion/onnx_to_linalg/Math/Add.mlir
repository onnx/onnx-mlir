// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Add' %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=DISABLED --implicit-check-not=linalg.generic

// -----

func.func @test_add_same_shape(%arg0: tensor<2x3xf32> {onnx.name = "A"},
    %arg1: tensor<2x3xf32> {onnx.name = "B"})
    -> (tensor<2x3xf32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: test_add_same_shape
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[A:[^:]+]]: f32, %[[B:[^:]+]]: f32,
  // CHECK: %[[SUM:[^ ]+]] = arith.addf %[[A]], %[[B]] : f32
  // CHECK-NEXT: linalg.yield %[[SUM]] : f32

  // DISABLED-LABEL: func.func @test_add_same_shape
  // DISABLED: "onnx.Add"
  // DISABLED: return
}

// -----

func.func @test_add_reject_broadcast(%arg0: tensor<2x3xf32> {onnx.name = "A"},
    %arg1: tensor<1x3xf32> {onnx.name = "B"})
    -> (tensor<2x3xf32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // CHECK-LABEL: test_add_reject_broadcast
  // CHECK-NOT: linalg.generic
  // CHECK: "onnx.Add"
  // CHECK-NOT: linalg.generic
  // CHECK: return
}

// -----

func.func @test_add_integer(%arg0: tensor<2x3xi32> {onnx.name = "A"},
    %arg1: tensor<2x3xi32> {onnx.name = "B"})
    -> (tensor<2x3xi32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>

  // CHECK-LABEL: func.func @test_add_integer
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32,
  // CHECK: %[[SUM:[^ ]+]] = arith.addi %[[A]], %[[B]] : i32
  // CHECK-NEXT: linalg.yield %[[SUM]] : i32
}

// -----

func.func @test_add_unsigned(%arg0: tensor<2x3xui32> {onnx.name = "A"},
    %arg1: tensor<2x3xui32> {onnx.name = "B"})
    -> (tensor<2x3xui32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3xui32>, tensor<2x3xui32>) -> tensor<2x3xui32>
  return %0 : tensor<2x3xui32>

  // CHECK-LABEL: func.func @test_add_unsigned
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[A:[^:]+]]: ui32, %[[B:[^:]+]]: ui32,
  // CHECK: %[[AI:[^ ]+]] = builtin.unrealized_conversion_cast %[[A]] : ui32 to i32
  // CHECK: %[[BI:[^ ]+]] = builtin.unrealized_conversion_cast %[[B]] : ui32 to i32
  // CHECK: %[[SUM:[^ ,]+]], {{%[^ ]+}} = arith.addui_extended %[[AI]], %[[BI]] : i32, i1
  // CHECK: %[[RESULT:[^ ]+]] = builtin.unrealized_conversion_cast %[[SUM]] : i32 to ui32
  // CHECK-NEXT: linalg.yield %[[RESULT]] : ui32
}

// -----

func.func @test_add_reject_dynamic(%arg0: tensor<?x3xf32> {onnx.name = "A"},
    %arg1: tensor<?x3xf32> {onnx.name = "B"})
    -> (tensor<?x3xf32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>

  // CHECK-LABEL: func.func @test_add_reject_dynamic
  // CHECK-NOT: linalg.generic
  // CHECK: "onnx.Add"
  // CHECK-NOT: linalg.generic
  // CHECK: return
}

// -----

func.func @test_add_reject_all_dynamic(%arg0: tensor<?x?xf32> {onnx.name = "A"},
    %arg1: tensor<?x?xf32> {onnx.name = "B"})
    -> (tensor<?x?xf32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>

  // CHECK-LABEL: func.func @test_add_reject_all_dynamic
  // CHECK-NOT: linalg.generic
  // CHECK: "onnx.Add"
  // CHECK-NOT: linalg.generic
  // CHECK: return
}

// -----

func.func @test_add_reject_rank_expansion(%arg0: tensor<1x3xf32> {onnx.name = "A"},
    %arg1: tensor<3xf32> {onnx.name = "B"})
    -> (tensor<1x3xf32> {onnx.name = "Y"}) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>

  // CHECK-LABEL: func.func @test_add_reject_rank_expansion
  // CHECK-NOT: linalg.generic
  // CHECK: "onnx.Add"
  // CHECK-NOT: linalg.generic
  // CHECK: return
}
