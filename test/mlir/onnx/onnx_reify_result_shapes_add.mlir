// RUN: onnx-mlir-opt --shape-inference --test-onnx-reify-result-shapes %s -split-input-file | FileCheck %s

// -----

func.func @test_add_reify_dynamic_batch(%arg0: tensor<?x2xf32>, %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: func.func @test_add_reify_dynamic_batch
// CHECK-DAG: [[SHAPE:%.+]] = shape.shape_of %arg{{.*}} : tensor<?x2xf32> -> tensor<2xindex>
// CHECK-DAG: shape.get_extent [[SHAPE]]
// CHECK-DAG: shape.get_extent [[SHAPE]]
// CHECK: "onnx.Add"(%arg0, %arg1)

// -----

func.func @test_add_reify_static(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @test_add_reify_static
// CHECK-DAG: arith.constant 3 : index
// CHECK-DAG: arith.constant 4 : index
// CHECK: "onnx.Add"(%arg0, %arg1)
