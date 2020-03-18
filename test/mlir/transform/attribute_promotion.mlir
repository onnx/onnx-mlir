// RUN: onnx-mlir-opt --attribute-promotion %s -split-input-file | FileCheck %s

func @test_should_promote_to_attribute(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %shape = constant dense<[6, 7, 42]> : tensor<3xi32>
  %0 = "onnx.Reshape"(%arg0, %shape) : (tensor<?x10xf32>, tensor<3xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: test_should_promote_to_attribute
// CHECK: [[NONE:%.+]] = constant unit
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"(%{{.*}}, [[NONE]]) {shape = dense<[6, 7, 42]> : tensor<3xi32>} : (tensor<?x10xf32>, none) -> tensor<*xf32>
// CHECK: return [[RESHAPE]] : tensor<*xf32>

func @test_should_not_promote_to_attribute(%arg0 : tensor<?x10xf32>, %arg1 : tensor<*xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<*xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: test_should_not_promote_to_attribute
// CHECK: [[RESHAPE:%.+]] = "onnx.Reshape"(%{{.*}}, %{{.*}}) : (tensor<?x10xf32>, tensor<*xi64>) -> tensor<*xf32>
// CHECK: return [[RESHAPE]] : tensor<*xf32>