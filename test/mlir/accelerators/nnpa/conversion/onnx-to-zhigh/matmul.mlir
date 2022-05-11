// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<4x16xf32>
// CHECK:         }
}

// -----

func @test_onnx_to_matmul3d(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<100x8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<100x8x16xf32>) -> tensor<100x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<100x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<100x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

func @test_onnx_to_matmul3dbcast(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul3dbcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<100x4x8xf32>) -> tensor<100x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<100x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<100x4x16xf32>
// CHECK:         }
}

// -----

/// Do not lower onnx.MatMul to zHigh if inputs have inadequate static shapes
/// for matrix multiply because zDNN does not support broadcasting.
func @test_matmul_not_lowered_inadequate_shape(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
  "func.return"(%0) : (tensor<4xf32>) -> ()

  // CHECK-LABEL: test_matmul_not_lowered_inadequate_shape
  // CHECK: {{.*}} = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
}

// -----

// COM: In this test, matmul and add will be combined together to be lowered to
// COM: zhigh.MatMul.

func @test_onnx_matmul_add_to_zhigh_1D_bias(
    %arg0 : tensor<4x8xf32>,
    %arg1 : tensor<8x16xf32>,
    %arg2 : tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg2, %0) : (tensor<16xf32>,tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_matmul_add_to_zhigh_1D_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<16xf32>) -> tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf32>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x16xf32>
// CHECK:         }
// CHECK-NOT: "onnx.Add"
}

// -----

// COM: In this test, matmul and add will be combined together to be lowered to
// COM: zhigh.MatMul.
// COM: add(bias, matmul(x,y)) will be normalized to add(matmul(x,y), bias)
// COM: before the lowering. 

func @test_onnx_matmul_add_to_zhigh_1D_bias_normalized(
    %arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>,
    %arg2 : tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_matmul_add_to_zhigh_1D_bias_normalized
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<4x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<4x8xf32>) -> tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<8x16xf32>) -> tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<16xf32>) -> tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf32>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<4x16xf32>
// CHECK:         }
// CHECK-NOT: "onnx.Add"
}

// -----

// COM: In this test, add is not combined with matmul to be lowered together.
// COM: It's because zhigh.Matmul expects bias to have the shape of <10x16> instead of <4x16>.

func @test_onnx_matmul_add_to_zhigh_not_lower_add_since_bias_dims(
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
func @test_onnx_matmul_add_to_zhigh_1D_bias_not_lower_add_since_broadcasting_bias(
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
func @test_onnx_to_matmul2d_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul2d_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?xf32>
// CHECK:         }
}

// -----

func @test_onnx_to_matmul3d_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul3d_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?xf32>
// CHECK:         }
}

// -----

func @test_onnx_to_matmul3dbcast_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_to_matmul3dbcast_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<?x?xf32>) -> tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_cst_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<?x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, none) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf32>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?xf32>
// CHECK:         }
}


// -----

/// Do not lower onnx.MatMul to zHigh if inputs have inadequate shapes (2D x 1D)
/// for matrix multiply because broadcasting is not supported
func @test_matmul_not_lowered_inadequate_shape_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()

  // CHECK-LABEL: func @test_matmul_not_lowered_inadequate_shape_dyn
  // CHECK: onnx.MatMul
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func @test_exceed_limit_matmul2d(%arg0 : tensor<4x32769xf32>, %arg1 : tensor<32769x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x32769xf32>, tensor<32769x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_matmul2d
// CHECK:        "onnx.MatMul"
}
