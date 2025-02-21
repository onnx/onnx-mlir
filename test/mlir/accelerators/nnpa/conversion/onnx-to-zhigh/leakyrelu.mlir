// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_leakyrelu(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) { alpha = 0.02:f32 } : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_leakyrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.LeakyRelu"([[VAR_0_]]) {alpha = 2.000000e-02 : f32} : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func.func @test_leakyrelu2(%arg0 : tensor<2x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) { alpha = 0.01:f32 } : (tensor<2x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_leakyrelu2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x10xf32>) -> tensor<2x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<2x10xf32>) -> tensor<2x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.LeakyRelu"([[VAR_0_]]) {alpha = 0.00999999977 : f32} : (tensor<2x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<2x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<2x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<2x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<2x10xf32>
// CHECK:         }
}

// -----

func.func @test_leakyrelu_default(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_leakyrelu_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.LeakyRelu"([[VAR_0_]]) {alpha = 0.00999999977 : f32} : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10xf32>
// CHECK:         }
}

