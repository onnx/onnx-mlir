// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --onnx-dim-analysis %s -split-input-file | FileCheck %s

// COM: test zdnn unary operations. Use Relu as a sample.
func.func @test_stick_unary_unstick(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  %1 = "zhigh.Stick"(%0) {layout = "3D"} : (tensor<?x3x?xf32>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
  %2 = "zhigh.Relu"(%1) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf32>
  "onnx.Return"(%3) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_stick_unary_unstick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]]  : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3D"} : (tensor<?x3x?xf32>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Relu"([[VAR_1_]]) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           "onnx.DimGroup"([[VAR_2_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

// COM: test zdnn binary operations. Use Add as a sample.
func.func @test_stick_binary_unstick(%arg0 : tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  %1 = "zhigh.Stick"(%0) {layout = "3D"} : (tensor<?x3x?xf32>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>

  %2 = "zhigh.Add"(%1, %1) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>

  %3 = "zhigh.Unstick"(%2) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf32>
  "onnx.Return"(%3) : (tensor<?x3x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_stick_binary_unstick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
// CHECK:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3D"} : (tensor<?x3x?xf32>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[VAR_1_]], [[VAR_1_]]) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>
// CHECK:           "onnx.DimGroup"([[VAR_2_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> ()
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<?x3x?xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<?x3x?xf32>
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x3x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x3x?xf32>
// CHECK:         }
}

// -----

// COM: Test NHWC layout, dimensions must be correctly transposed from NCHW to NHWC.
func.func @test_nhwc_layout(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = "zhigh.Stick"(%0) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x?x?x?xf32>
  "onnx.Return"(%2) : (tensor<?x?x?x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_nhwc_layout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 3 : si64, group_id = [[GROUP_3_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 3 : si64, group_id = [[GROUP_3_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = [[GROUP_3_]] : si64} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 1 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 3 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<?x?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 3 : si64, group_id = [[GROUP_3_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_2_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_stick_matmul_unstick(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "zhigh.Stick"(%0) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  %2 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}: (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  %none = "onnx.NoValue"() {value} : () -> none
  %4 = "zhigh.MatMul"(%1, %3, %none) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  %5 = "zhigh.Unstick"(%4) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf32>
  "onnx.Return"(%5) : (tensor<?x?x?xf32>) -> ()

// CHECK-LABEL:  func.func @test_stick_matmul_unstick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[PARAM_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_:.*]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sigmoid"([[PARAM_0_]]) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_0_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 2 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_1_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK:           [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1]} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 2 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 1 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_2_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_3_]]) {axis = 2 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_3_]]) {axis = 1 : si64, group_id = [[GROUP_2_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_3_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK:           [[VAR_4_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_5_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 2 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_5_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> ()
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf32>
// CHECK-DAG:       "onnx.DimGroup"([[VAR_6_]]) {axis = 1 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_6_]]) {axis = 2 : si64, group_id = [[GROUP_1_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK-DAG:       "onnx.DimGroup"([[VAR_6_]]) {axis = 0 : si64, group_id = [[GROUP_0_]] : si64} : (tensor<?x?x?xf32>) -> ()
// CHECK:           onnx.Return [[VAR_6_]] : tensor<?x?x?xf32>
// CHECK:         }
}
