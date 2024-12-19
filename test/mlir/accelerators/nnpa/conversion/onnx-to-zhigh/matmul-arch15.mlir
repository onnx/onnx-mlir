// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

// COM: In these tests, matmul and transpose will be combined together to be lowered to
// COM: zhigh.MatMul.

func.func @test_onnx_transposea_matmul_to_zhigh(%arg0 : tensor<8x4xf32>, %arg1 : tensor<8x4xf32>) -> tensor<*xf32> {
   %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]}: (tensor<8x4xf32>) -> tensor<4x8xf32>
   %1 = "onnx.MatMul"(%0, %arg1) : (tensor<4x8xf32>,tensor<8x4xf32>) -> tensor<*xf32>
   "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_transposea_matmul_to_zhigh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x4xf32>, [[PARAM_1_:%.+]]: tensor<8x4xf32>) -> tensor<4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<8x4xf32>) -> tensor<8x4xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x4xf32>) -> tensor<8x4xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) {transposeA = 1 : si64, transposeB = 0 : si64} : (tensor<8x4xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x4xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<4x4xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<4x4xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x4xf32>
// CHECK:           return [[VAR_3_]] : tensor<4x4xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_transposeb_matmul_to_zhigh(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg1) {perm = [1, 0]}: (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<4x8xf32>,tensor<8x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_transposeb_matmul_to_zhigh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<4x8xf32>) -> tensor<4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 1 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<4x4xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<4x4xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<4x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x4xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_transposeab_matmul_to_zhigh(%arg0 : tensor<4x8xf32>, %arg1 : tensor<16x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {permA = [1, 0]}: (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "onnx.Transpose"(%arg1) {permB = [1, 0]}: (tensor<16x4xf32>) -> tensor<4x16xf32>
  %2 = "onnx.MatMul"(%0, %1) : (tensor<8x4xf32>,tensor<4x16xf32>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_transposeab_matmul_to_zhigh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<16x4xf32>) -> tensor<8x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<16x4xf32>) -> tensor<16x4xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 1 : si64, transposeB = 1 : si64} : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16x4xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<8x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<8x16xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_transposea_matmul_to_zhigh_3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x16x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}: (tensor<100x4x8xf32>) -> tensor<100x8x4xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<100x8x4xf32>, tensor<100x16x8xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_transposea_matmul_to_zhigh_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x16x8xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1]} : (tensor<100x4x8xf32>) -> tensor<100x8x4xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[PARAM_1_]]) : (tensor<100x8x4xf32>, tensor<100x16x8xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_transposeb_matmul_to_zhigh_3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x16x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1]}: (tensor<100x16x8xf32>) -> tensor<100x8x16xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_transposeb_matmul_to_zhigh_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x16x8xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<100x16x8xf32>) -> tensor<100x16x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {transposeA = 0 : si64, transposeB = 1 : si64} : (tensor<100x4x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<100x16x8xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none) -> tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<100x4x16xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_to_transposeab_matmul_to_zhigh_3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {permA = [0, 2, 1]}: (tensor<100x4x8xf32>) -> tensor<100x8x4xf32>
  %1 = "onnx.Transpose"(%arg1) {permB = [0, 2, 1]}: (tensor<100x8x16xf32>) -> tensor<100x16x8xf32>
  %2 = "onnx.MatMul"(%0, %1) : (tensor<100x8x4xf32>,tensor<100x16x8xf32>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_to_transposeab_matmul_to_zhigh_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<*xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [2, 1, 0], permA = [0, 2, 1]} : (tensor<100x4x8xf32>) -> tensor<100x8x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 1, 0], permB = [0, 2, 1]} : (tensor<100x8x16xf32>) -> tensor<100x16x8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_1_]]) : (tensor<100x8x4xf32>, tensor<100x16x8xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_2_]] : tensor<*xf32>
// CHECK:         }
}

// -----

