// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func @test_gemm(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x5xf32>) -> tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_4_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func @test_gemm_transA(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_transA
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x10xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0]} : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "2D"} : (tensor<10x5xf32>) -> tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) : (tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<*xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func @test_gemm_transB(%arg0 : tensor<10x5xf32>, %arg1 : tensor<10x5xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_transB
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<10x5xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x5xf32>) -> tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [1, 0]} : (tensor<10x5xf32>) -> tensor<5x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "2D"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_2_]], [[VAR_3_]]) : (tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<*xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func @test_gemm_transAB(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<5xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_transAB
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<5xf32>) -> tensor<5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0]} : (tensor<10x5xf32>) -> tensor<5x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "2D"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [1, 0]} : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "2D"} : (tensor<10x5xf32>) -> tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<5xf32>) -> tensor<5xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.MatMul"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) : (tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<5xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<*xf32>) -> tensor<5x5xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x5xf32>
// CHECK:         }
}

// -----

// COM: Lower ONNXGemm to ZHigh without verifying the actual size of an unknown
// COM: dimension. Accelerator will emit an error if the dimension size is
// COM: beyond the supported value.
func @test_gemm_unknown_dims(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_gemm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<?x5xf32>) -> tensor<?x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<5x10xf32>) -> tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<?x5xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<5x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<*xf32>) -> tensor<?x10xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x10xf32>
// CHECK:         }
}

// -----

// COM: Not support because we cannot check bias broadcasting at compile time.

func @test_gemm_unknown_dims_not_lowered(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

 // CHECK-LABEL: test_gemm_unknown_dims_not_lowered
 // CHECK: onnx.Gemm
}

// -----

// COM: Not support since alpha != 1.0
func @test_gemm_not_lowered(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

 // CHECK-LABEL: test_gemm_not_lowered
 // CHECK: onnx.Gemm
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func @test_exceed_limit_gemm(%arg0 : tensor<32769x5xf32>, %arg1 : tensor<5x32769xf32>, %arg2: tensor<32769xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<32769x5xf32>, tensor<5x32769xf32>, tensor<32769xf32>) -> tensor<*xf32>
 "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_gemm
// CHECK:        "onnx.Gemm"

}
