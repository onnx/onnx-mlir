// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func.func @test_onnx_to_matmul_unstacked(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul_unstacked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<4x16xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_matmul_stacked(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul_stacked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<100x8x16xf32>) -> tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

// not nnpa functionality for z16

func.func @test_onnx_to_matmul_bcast1(%arg0 : tensor<4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_bcast1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_0_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_matmul_bcast23(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_bcast23
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf16>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

// Fused add

func.func @test_onnx_to_matmul_add_unstacked(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_unstacked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<16xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "1D"} : (tensor<16xf32>) -> tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_5_]] : tensor<4x16xf32>
// CHECK:         }
}

// -----

// Fused add disabled because expected 1D but got 2D

func.func @test_onnx_to_matmul_add_unstacked_fail(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<4x16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<4x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_unstacked_fail
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_2_]], [[VAR_3_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<*xf16>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "2D"} : (tensor<4x16xf32>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "2D"} : (tensor<4x16xf32>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Add"([[VAR_6_]], [[VAR_7_]]) : (tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Unstick"([[VAR_8_]]) : (tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_9_]] : tensor<4x16xf32>
// CHECK:         }
}

// -----

// stacked adds a [s x p] constant, but to satisfy shape inference, it has to be [s x 1 x p].

func.func @test_onnx_to_matmul_add_stacked(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<100x1x16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<100x1x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_stacked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<100x16xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<100x8x16xf32>) -> tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "2DS"} : (tensor<100x16xf32>) -> tensor<100x16xf16, #zhigh.layout<{dataLayout = "2DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x16xf16, #zhigh.layout<{dataLayout = "2DS"}>>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_5_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

// stacked adds a [s x p] constant, but to satisfy shape inference, it has to be [s x 1 x p].
// But here fails to merge multiply and add as we don't get the "1".  Both sent separately to NNPA

func.func @test_onnx_to_matmul_add_stacked_failed(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<100x4x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_stacked_failed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<100x8x16xf32>) -> tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_2_]], [[VAR_3_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x8x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<*xf16>) -> tensor<100x4x16xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Add"([[VAR_6_]], [[VAR_7_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Unstick"([[VAR_8_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_9_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

// not nnpa functionality for z16

func.func @test_onnx_to_matmul_add_bcast1(%arg0 : tensor<4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<100x4x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_bcast1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_3_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_5_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----
// const has wrong shape for bcast23
func.func @test_onnx_to_matmul_add_bcast23(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<*xf32>, tensor<100x4x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_add_bcast23
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<100x4x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_2_]], [[VAR_3_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<*xf16>) -> tensor<100x4x16xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Add"([[VAR_6_]], [[VAR_7_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Unstick"([[VAR_8_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_9_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

/// Do not lower onnx.MatMul to zHigh if inputs have inadequate static shapes
/// for matrix multiply because zDNN does not support broadcasting.
func.func @test_matmul_not_lowered_inadequate_shape(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
  "func.return"(%0) : (tensor<4xf32>) -> ()

  // CHECK-LABEL: test_matmul_not_lowered_inadequate_shape
  // CHECK: {{.*}} = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
}

// -----

// COM: In this test, matmul and add will be combined together to be lowered to
// COM: zhigh.MatMul.

func.func @test_onnx_matmul_add_to_zhigh_1D_bias(
    %arg0 : tensor<4x8xf32>,
    %arg1 : tensor<8x16xf32>,
    %arg2 : tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg2, %0) : (tensor<16xf32>,tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_matmul_add_to_zhigh_1D_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<16xf32>) -> tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x16xf32>
// CHECK:         }
// CHECK-NOT: "onnx.Add"
}

// -----

// COM: In this test, matmul and add will be combined together to be lowered to
// COM: zhigh.MatMul.
// COM: add(bias, matmul(x,y)) will be normalized to add(matmul(x,y), bias)
// COM: before the lowering.

func.func @test_onnx_matmul_add_to_zhigh_1D_bias_normalized(
    %arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>,
    %arg2 : tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_matmul_add_to_zhigh_1D_bias_normalized
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<16xf32>) -> tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x16xf32>
// CHECK:         }
// CHECK-NOT: "onnx.Add"
}

// -----

// COM: In this test, add is not combined with matmul to be lowered together.
// COM: It's because zhigh.Matmul expects bias to have the shape of <10x16> instead of <4x16>.

func.func @test_onnx_matmul_add_to_zhigh_not_lower_add_since_bias_dims(
    %arg0 : tensor<10x4x8xf32>,
    %arg1 : tensor<10x8x16xf32>,
    %arg2 : tensor<4x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x4x8xf32>, tensor<10x8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg2, %0) : (tensor<4x16xf32>,tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK: func @test_onnx_matmul_add_to_zhigh_not_lower_add_since_bias_dims
  // CHECK: "zhigh.MatMul"
  // CHECK: "onnx.Add"
}

// -----

// COM: In this test, add is not combined with matmul to be lowered together.
// COM: It's because zhigh.Matmul does not support broadcasting for bias.
func.func @test_onnx_matmul_add_to_zhigh_1D_bias_not_lower_add_since_broadcasting_bias(
    %arg0 : tensor<4x8xf32>,
    %arg1 : tensor<8x16xf32>,
    %arg2 : tensor<1xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg2, %0) : (tensor<1xf32>,tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK: func @test_onnx_matmul_add_to_zhigh_1D_bias_not_lower_add_since_broadcasting_bias
  // CHECK: "zhigh.MatMul"
  // CHECK: "onnx.Add"
}

// -----

// COM: Lower onnx.MatMul to zHigh if input matrices are 2D x 2D and have unknown
// COM: dimensions, assuming they meet requirement in matrix shape.
// COM: If they don't meet the requirement, get runtime error.
func.func @test_onnx_to_matmul_unstacked_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul_unstacked_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_matmul_stacked_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul_stacked_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_matmul_bcast23_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_bcast23_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf16>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

// Not supported on z16

func.func @test_onnx_to_matmul_bcast1_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_matmul_bcast1_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

/// Do not lower onnx.MatMul to zHigh if inputs have inadequate shapes (2D x 1D)
/// for matrix multiply because broadcasting is not supported
func.func @test_matmul_not_lowered_inadequate_shape_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()

  // CHECK-LABEL: func @test_matmul_not_lowered_inadequate_shape_dyn
  // CHECK: onnx.MatMul
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_matmul2d(%arg0 : tensor<4x32769xf32>, %arg1 : tensor<32769x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x32769xf32>, tensor<32769x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_matmul2d
// CHECK:        "onnx.MatMul"
}
