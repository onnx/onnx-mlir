// RUN: onnx-mlir  --useOnnxModelTypes=false --fuse-parallel-onnx-gemm --EmitONNXIR --printIR %s | FileCheck %s

func.func @test_gemm_concat_simple(%arg0: tensor<1x4xf32>) -> tensor<1x6xf32> {
  %0 = onnx.Constant dense<5.5>: tensor<4x3xf32>
  %1 = onnx.Constant dense<0.2> : tensor<3xf32>
  %2 = onnx.Constant dense<4.5>: tensor<4x3xf32>
  %3 = onnx.Constant dense<0.5> : tensor<3xf32>
  %4 = "onnx.Gemm"(%arg0, %0, %1) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_1", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %5 = "onnx.Gemm"(%arg0, %2, %3) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_2", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %6 = "onnx.Concat"(%4, %5) {axis = 1 : si64, onnx_node_name = "Concat"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x6xf32>
  return %6 : tensor<1x6xf32>

  // CHECK-LABEL: func @test_gemm_concat_simple
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xf32>) -> tensor<1x6xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<4x6xf32>
  
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<6xf32>

  // CHECK:     [[VAR_2_:%.+]] = "onnx.Gemm"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     : (tensor<1x4xf32>, tensor<4x6xf32>, tensor<6xf32>) -> tensor<1x6xf32>
  // CHECK-NEXT:     return [[VAR_2_]] : tensor<1x6xf32>

}

func.func @test_combine_gemm_split(%arg0: tensor<1x4xf32>) -> tensor<1x12xf32> {
  %0 = onnx.Constant dense<1.6> : tensor<4x3xf32>
  %1 = onnx.Constant dense<2.7> : tensor<4x3xf32>
  %2 = onnx.Constant dense<3.7> : tensor<4x3xf32>
  %3 = onnx.Constant dense<4.6> : tensor<4x3xf32>
  %4 = onnx.Constant dense<0.1> : tensor<3xf32>
  %5 = onnx.Constant dense<0.9> : tensor<3xf32>
  %6 = onnx.Constant dense<0.2> : tensor<3xf32>
  %7 = onnx.Constant dense<0.8> : tensor<3xf32>
  %8 = "onnx.Gemm"(%arg0, %0, %4) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_1", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %9 = "onnx.Gemm"(%arg0, %1, %5) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_2", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %10 = "onnx.Gemm"(%arg0, %2, %6) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_3", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %11 = "onnx.Gemm"(%arg0, %3, %7) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_4", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %12 = "onnx.Relu"(%8) {onnx_node_name = "ReLU_1"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %13 = "onnx.Sigmoid"(%9) {onnx_node_name = "Sigmoid_2"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %14 = "onnx.Tanh"(%10) {onnx_node_name = "Tanh_3"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %15 = "onnx.LeakyRelu"(%11) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyReLU_4"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %16 = "onnx.Concat"(%12, %13, %14, %15) {axis = 1 : si64, onnx_node_name = "Concat"} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x12xf32>
  return %16 : tensor<1x12xf32>

// CHECK-LABEL: func @test_combine_gemm_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xf32>) -> tensor<1x12xf32> {
// CHECK:      [[CONST_SPLIT_:%.+]] = onnx.Constant dense<3> : tensor<4xi64>
// CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<4x12xf32>
// CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<12xf32>
// CHECK:      [[GEMM_OUT_:%.+]] = "onnx.Gemm"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME:     : (tensor<1x4xf32>, tensor<4x12xf32>, tensor<12xf32>) -> tensor<1x12xf32>
// CHECK: [[VAR_2_:[^ ]+]]:4 = "onnx.Split"([[GEMM_OUT_]], [[CONST_SPLIT_]]) {axis = 1 : si64, onnx_node_name = "onnx.Split_2"} : (tensor<1x12xf32>, tensor<4xi64>) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>)
// CHECK: [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]#0) {onnx_node_name = "ReLU_1"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: [[VAR_4_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]#3) {onnx_node_name = "Sigmoid_2"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: [[VAR_5_:%.+]] = "onnx.Tanh"([[VAR_2_]]#2) {onnx_node_name = "Tanh_3"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: [[VAR_6_:%.+]] = "onnx.LeakyRelu"([[VAR_2_]]#1) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyReLU_4"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: [[FINAL_OUT:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]]) {axis = 1 : si64, onnx_node_name = "Concat"} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x12xf32>
// CHECK: return [[FINAL_OUT]] : tensor<1x12xf32>


}
