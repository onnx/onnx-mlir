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

func.func @test_gemm_concat_complex(%arg0: tensor<1x4xf32>) -> tensor<1x18xf32> {
  %0 = onnx.Constant dense<[[0.204779208, 0.695178091, 0.239361823], [0.996994256, 0.601588786, 0.190346241], [0.842002928, 0.739568233, 0.994108259], [0.905652821, 0.834119677, 0.303750187]]> : tensor<4x3xf32>
  %1 = onnx.Constant dense<[[0.793336808, 0.967174768, 0.98079878], [0.761894762, 0.102106638, 0.039635919], [0.00603901641, 0.923491775, 0.357523948], [0.696550369, 0.308858335, 0.0805873647]]> : tensor<4x3xf32>
  %2 = onnx.Constant dense<[[0.544325054, 0.151464358, 0.934764087], [0.478074521, 0.161221609, 0.71641761], [0.50913018, 0.756769299, 0.904207945], [0.0835523381, 0.918578445, 0.835795641]]> : tensor<4x3xf32>
  %3 = onnx.Constant dense<[[0.41472131, 0.492292702, 0.088731639], [0.903954088, 0.128603399, 0.769681036], [0.953823149, 0.836306929, 0.9627828], [0.800210654, 0.308792889, 0.314317614]]> : tensor<4x3xf32>
  %4 = onnx.Constant dense<[[0.12443202, 0.226671219, 0.148676723], [0.616570889, 0.962450921, 0.134999171], [0.184063375, 0.764316678, 0.414653629], [0.0643175319, 0.148418352, 0.596157073]]> : tensor<4x3xf32>
  %5 = onnx.Constant dense<[[0.391361624, 0.664259791, 0.618797242], [0.672276973, 0.0329957306, 0.00447194278], [0.732442378, 0.597825587, 0.0171195511], [0.568968296, 0.778787076, 0.921517431]]> : tensor<4x3xf32>
  %6 = onnx.Constant dense<[0.276767612, 0.952775657, 0.301255673]> : tensor<3xf32>
  %7 = onnx.Constant dense<[0.889294981, 0.491430521, 0.142108783]> : tensor<3xf32>
  %8 = onnx.Constant dense<[0.790298938, 0.401669294, 0.446535289]> : tensor<3xf32>
  %9 = onnx.Constant dense<[0.3797189, 0.496988833, 0.511586726]> : tensor<3xf32>
  %10 = onnx.Constant dense<[0.721806407, 0.0192602724, 0.322999328]> : tensor<3xf32>
  %11 = onnx.Constant dense<[0.969116449, 4.448790e-01, 0.668284774]> : tensor<3xf32>
  %12 = "onnx.Gemm"(%arg0, %0, %6) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_1", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %13 = "onnx.Gemm"(%arg0, %1, %7) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_2", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %14 = "onnx.Gemm"(%arg0, %2, %8) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_3", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %15 = "onnx.Gemm"(%arg0, %3, %9) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_4", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %16 = "onnx.Gemm"(%arg0, %4, %10) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_5", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %17 = "onnx.Gemm"(%arg0, %5, %11) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_6", transA = 0 : si64, transB = 0 : si64} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %18 = "onnx.Concat"(%12, %13, %14, %15, %16, %17) {axis = 1 : si64, onnx_node_name = "Concat"} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x18xf32>
  return %18 : tensor<1x18xf32>

  // CHECK-LABEL: func @test_gemm_concat_complex
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xf32>) -> tensor<1x18xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<4x18xf32>
  
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<18xf32>

  // CHECK:     [[VAR_2_:%.+]] = "onnx.Gemm"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     : (tensor<1x4xf32>, tensor<4x18xf32>, tensor<18xf32>) -> tensor<1x18xf32>
  // CHECK-NEXT:     return [[VAR_2_]] : tensor<1x18xf32>

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
