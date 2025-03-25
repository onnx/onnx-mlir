// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --nnpa-enable-scalar-bcast-binary %s -split-input-file | FileCheck %s

// COM: Division by a scalar in case of dynamic dimensions.
func.func @test_div_unknown_scalar1(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<8.000000e+00> : tensor<f32>
  %1 = "onnx.Div"(%arg0, %0) : (tensor<?x10xf32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_div_unknown_scalar1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<8.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<?x10xf32>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x10xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x10xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.StickifiedConstantOfShape"([[VAR_4_]]) {layout = "2D", value = 8.000000e+00 : f32} : (tensor<2xi64>) -> tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Div"([[VAR_1_]], [[VAR_5_]]) : (tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<?x10xf32>
// CHECK:           return [[VAR_7_]] : tensor<?x10xf32>
// CHECK:         }
}

// -----

// COM: Division by a scalar in case of dynamic dimensions.
func.func @test_div_unknown_scalar2(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<8.000000e+00> : tensor<f32>
  %1 = "onnx.Div"(%0, %arg0) : (tensor<f32>, tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_div_unknown_scalar2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<8.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 0 : si64} : (tensor<?x10xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Dim"([[PARAM_0_]]) {axis = 1 : si64} : (tensor<?x10xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_2_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.StickifiedConstantOfShape"([[VAR_3_]]) {layout = "2D", value = 8.000000e+00 : f32} : (tensor<2xi64>) -> tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<?x10xf32>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Div"([[VAR_4_]], [[VAR_5_]]) : (tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_6_]]) : (tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<?x10xf32>
// CHECK:           return [[VAR_7_]] : tensor<?x10xf32>
// CHECK:         }
}

