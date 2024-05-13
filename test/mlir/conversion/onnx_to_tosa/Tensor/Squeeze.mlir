// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa --canonicalize %s -split-input-file | FileCheck %s

func.func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<16x32x64xf32>)
  "func.return"(%1) : (tensor<16x32x64xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 16, 32, 64>} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<16x32x64xf32>
// CHECK:         }
}

func.func @test_squeeze_unknown_dimensions(%arg0 : tensor<1x1x32x1x64xf32>) -> tensor<32x64xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x1x32x1x64xf32>, none) -> (tensor<32x64xf32>)
  "func.return"(%1) : (tensor<32x64xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeeze_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x32x1x64xf32>) -> tensor<32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 32, 64>} : (tensor<1x1x32x1x64xf32>) -> tensor<32x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<32x64xf32>
// CHECK:         }
}

// -----

func.func @squeeze_runtime(%arg0: tensor<1x3x4x5xf32> , %arg1: tensor<1xi64> ) -> tensor<3x4x5xf32> {
  %0 = "onnx.Squeeze"(%arg0, %arg1) : (tensor<1x3x4x5xf32>, tensor<1xi64>) -> tensor<3x4x5xf32>
  return %0 : tensor<3x4x5xf32>
// CHECK-LABEL: squeeze_runtime
// CHECK: tosa.reshape {{.*}} {new_shape = array<i64: 3, 4, 5>} : (tensor<1x3x4x5xf32>) -> tensor<3x4x5xf32>
}

// -----

func.func @squeeze_dynamic(%arg0: tensor<1x3x4x5xf32> , %arg1: tensor<1xi64> ) -> tensor<?x?x?xf32> {
  %0 = "onnx.Squeeze"(%arg0, %arg1) : (tensor<1x3x4x5xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
// CHECK-LABEL: squeeze_dynamic
// CHECK: onnx.Squeeze
}
