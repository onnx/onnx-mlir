// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_default_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [3, 2, 1, 0] : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
// CHECK:           return [[VAR_0_]] : tensor<32x1x5x5xf32>
// CHECK:         }

// -----

func.func @test_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<5x1x32x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [0, 2, 3, 1] : (tensor<5x5x1x32xf32>) -> tensor<5x1x32x5xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x1x32x5xf32>
// CHECK:         }

// -----

func.func @test_transpose_dyn(%arg0 : tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<?x?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_transpose_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [0, 2, 3, 1] : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?x?xf32>
// CHECK:         }
