// RUN: onnx-mlir-opt --attribute-promotion %s -split-input-file | FileCheck %s

func @test_should_promote_to_attribute(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %shape = constant dense<[6, 7, 42]> : tensor<3xi32>
  %0 = "onnx.Reshape"(%arg0, %shape) : (tensor<?x10xf32>, tensor<3xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_should_promote_to_attribute
  // CHECK-NEXT: [[NONE:%.+]] = constant unit
  // CHECK-NEXT: [[RESHAPE:%.+]] = "onnx.Reshape"(%{{.*}}, [[NONE]]) {shape = dense<[6, 7, 42]> : tensor<3xi32>} : (tensor<?x10xf32>, none) -> tensor<*xf32>
  // CHECK-NEXT: return [[RESHAPE]] : tensor<*xf32>
}

func @test_should_not_promote_to_attribute(%arg0 : tensor<?x10xf32>, %arg1 : tensor<*xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<*xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_should_not_promote_to_attribute
  // CHECK-NEXT: [[RESHAPE:%.+]] = "onnx.Reshape"(%{{.*}}, %{{.*}}) : (tensor<?x10xf32>, tensor<*xi64>) -> tensor<*xf32>
  // CHECK-NEXT: return [[RESHAPE]] : tensor<*xf32>
}

func @test_promote_to_attribute_without_removing_const_op(%arg0 : tensor<?x10xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %shape = constant dense<[6, 7, 42]> : tensor<3xi32>
  %0 = "onnx.Reshape"(%arg0, %shape) : (tensor<?x10xf32>, tensor<3xi32>) -> tensor<*xf32>
  %1 = "onnx.Identity"(%shape) : (tensor<3xi32>) -> tensor<*xf32>
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()
  // CHECK-LABEL: test_promote_to_attribute_without_removing_const_op
  // CHECK-NEXT: [[NONE:%.+]] = constant unit
  // CHECK-NEXT: [[SHAPE:%.+]] = constant dense<[6, 7, 42]> : tensor<3xi32>
  // CHECK-NEXT: [[RESHAPE:%.+]] = "onnx.Reshape"(%{{.*}}, [[NONE]]) {shape = dense<[6, 7, 42]> : tensor<3xi32>} : (tensor<?x10xf32>, none) -> tensor<*xf32>
  // CHECK-NEXT: [[IDENTITY:%.+]] = "onnx.Identity"([[SHAPE]]) : (tensor<3xi32>) -> tensor<*xf32>
  // CHECK-NEXT: return [[RESHAPE]], [[IDENTITY]] : tensor<*xf32>, tensor<*xf32>
}