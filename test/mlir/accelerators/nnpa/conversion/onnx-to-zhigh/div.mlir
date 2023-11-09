// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Div"([[VAR_0_]], [[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

// COM: Binary ops use 3DS by default for rank 3.
func.func @test_div_3ds(%arg0 : tensor<10x10x10xf32>, %arg1 : tensor<10x10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_div_3ds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<10x10x10xf32>) -> tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<10x10x10xf32>) -> tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Div"([[VAR_0_]], [[VAR_1_]]) : (tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<10x10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10x10xf32>
// CHECK:         }
}

// -----

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

// -----

// COM:  Do not lower broadcasting onnx.Div to zHigh.
func.func @test_div_not_lowered_diff_shape(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div_not_lowered_diff_shape
}

// -----

/// Do not lower onnx.Div to zHigh if inputs have unknown dimensions
/// because we cannot statically check whether it is really broadcasting or not.
func.func @test_div_lowered_unknown_dims(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div_lowered_unknown_dims
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_div(%arg0 : tensor<32769x10xf32>, %arg1 : tensor<32769x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<32769x10xf32>, tensor<32769x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_div
// CHECK:        "onnx.Div"
}
