// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_softplus(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) {axis = 1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_softplus
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Exp"([[VAR_0_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<10x10xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Add"([[VAR_1_]], [[VAR_3_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Log"([[VAR_4_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_6_]] : tensor<10x10xf32>
// CHECK:         }
}

