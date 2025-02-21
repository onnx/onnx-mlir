// RUN: onnx-mlir-opt --device-placement=use-faster=true --march=z16 --maccel=NNPA --split-input-file %s | FileCheck %s
// -----

// Shape is such that this op is nearly guaranteed to be faster on CPU.
func.func @add_cpu(%arg0: tensor<1024x32x1xf32>) -> tensor<1024x32x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<1024x32x1xf32>, tensor<1024x32x1xf32>) -> tensor<1024x32x1xf32>
  return %0 : tensor<1024x32x1xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_cpu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1024x32x1xf32>) -> tensor<1024x32x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) {device = "cpu"} : (tensor<1024x32x1xf32>, tensor<1024x32x1xf32>) -> tensor<1024x32x1xf32>
// CHECK:           return [[VAR_0_]] : tensor<1024x32x1xf32>
// CHECK:         }
}

// -----

// Shape is such that this op is nearly guaranteed to be faster on NNPA; so no device="cpu" here.

func.func @matmul_nnpa(%arg0: tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg0) : (tensor<1024x1024x1024xf32>, tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32>
  return %0 : tensor<1024x1024x1024xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @matmul_nnpa
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_0_]]) {device = "nnpa"} : (tensor<1024x1024x1024xf32>, tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32>
// CHECK:           return [[VAR_0_]] : tensor<1024x1024x1024xf32>
// CHECK:         }
}

