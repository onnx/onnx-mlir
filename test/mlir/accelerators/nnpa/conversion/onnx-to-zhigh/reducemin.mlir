// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

 func.func @test_reduce_min_axes_defined_noop_0(%arg0: tensor<1x2x4xf32>) -> tensor<*xf32> {
   %0 = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64> } : () -> tensor<1xi64>
   %1 ="onnx.ReduceMin"(%arg0, %0) {keepdims = 1: si64, noop_with_empty_axes = 0: si64} : (tensor<1x2x4xf32>, tensor<1xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_reduce_min_axes_defined_noop_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x4xf32>) -> tensor<1x2x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<1x2x4xf32>) -> tensor<1x2x4xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.ReduceMin"([[VAR_1_]]) {op_type = "REDUCE_OP_MINIMUM"} : (tensor<1x2x4xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x2x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x2x1xf32>
// CHECK:         }
}