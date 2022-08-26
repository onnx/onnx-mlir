// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @test_matmul_2d(%arg0 : tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg1 : tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_matmul_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, [[PARAM_1_:%.+]]: tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<4x16xf32, #zhigh.encoding<{dataLayout = "2D"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<4x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:           return [[VAR_0_]] : tensor<4x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:         }
}

// -----

func.func @test_matmul_3d_broadcast(%arg0 : tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1 : tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_matmul_3d_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @test_matmul_3d_stack(%arg0 : tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1 : tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_matmul_3d_stack
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x4x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @test_matmul_2d_unknown_dims(%arg0 : tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg1 : tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_matmul_2d_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, [[PARAM_1_:%.+]]: tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:         }
}

// -----

func.func @test_matmul_3d_broadcast_unknown_dims(%arg0 : tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1 : tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>

  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_matmul_3d_broadcast_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<?x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @test_matmul_3d_stack_unknown_dims(%arg0 : tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1 : tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MatMul"(%arg0, %arg1, %cst) : (tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_matmul_3d_stack_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MatMul"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) : (tensor<2x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x?x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}
