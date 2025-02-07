// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func.func @test_reduce_max_axes_defined_noop_0(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
   %cst = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64> } : () -> tensor<1xi64>
   %0 ="onnx.ReduceMax"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_reduce_max_axes_defined_noop_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2x1xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<3x2x2xf32>) -> tensor<3x2x2xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.ReduceMax"([[VAR_1_]]) : (tensor<3x2x2xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<3x2x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<3x2x1xf32>
// CHECK:         }
}

// -----

func.func @test_reduce_max_axes_minus_one(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
   %cst = "onnx.Constant"() {value = dense<-1> : tensor<1xi64> } : () -> tensor<1xi64>
   %0 ="onnx.ReduceMax"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reduce_max_axes_minus_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<3x2x1xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<3x2x2xf32>) -> tensor<3x2x2xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.ReduceMax"([[VAR_1_]]) : (tensor<3x2x2xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<3x2x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<3x2x1xf32>
// CHECK:         }
}

// -----

func.func @test_reduce_max_not_lowered_unknown_axis(%arg0 : tensor<3x2x2xf32>, %arg1: tensor<1xi64>) -> tensor<*xf32> {
   %0 ="onnx.ReduceMax"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reduce_max_not_lowered_unknown_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_reduce_max_axes_not_lowered_not_innermost_axis(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
   %cst = "onnx.Constant"() {value = dense<0> : tensor<1xi64> } : () -> tensor<1xi64>
   %0 ="onnx.ReduceMax"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reduce_max_axes_not_lowered_not_innermost_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<1x2x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>) -> tensor<1x2x2xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x2x2xf32>
// CHECK:         }
}

// -----

func.func @test_reduce_max_axes_not_lowered_not_multiple_axes(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
   %cst = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
   %0 ="onnx.ReduceMax"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<2xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reduce_max_axes_not_lowered_not_multiple_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2x2xf32>) -> tensor<1x2x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceMax"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<2xi64>) -> tensor<1x2x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x2x1xf32>
// CHECK:         }
}
