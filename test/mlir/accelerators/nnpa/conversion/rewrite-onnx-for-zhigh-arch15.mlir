// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --rewrite-onnx-for-zhigh --canonicalize %s -split-input-file | FileCheck %s

// -----

// Do not Split MatMul because a dimension does not exceeds NNPAGetMaxForDim for e2 of 1048576.

func.func @test_matmul_no_splitting_arch15_A(%arg0: tensor<?x1048576x768xf32>, %arg1: tensor<768x1024xf32>) -> (tensor<?x1048576x1024xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x1048576x768xf32>, tensor<768x1024xf32>) -> tensor<?x1048576x1024xf32>
  return %0 : tensor<?x1048576x1024xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_no_splitting_arch15_A
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x1048576x768xf32>, [[B_:%.+]]: tensor<768x1024xf32>) -> tensor<?x1048576x1024xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[A_]], [[B_]]) : (tensor<?x1048576x768xf32>, tensor<768x1024xf32>) -> tensor<?x1048576x1024xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x1048576x1024xf32>
// CHECK:         }
}

// -----

// Split MatMul because a dimension exceeds NNPAGetMaxForDim for e2 on arch15 of 1048576: use 2097152

func.func @test_matmul_splitting_arch15_A(%arg0: tensor<?x2097152x768xf32>, %arg1: tensor<768x1024xf32>) -> (tensor<?x2097152x1024xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x2097152x768xf32>, tensor<768x1024xf32>) -> tensor<?x2097152x1024xf32>
  return %0 : tensor<?x2097152x1024xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_arch15_A
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x2097152x768xf32>, [[B_:%.+]]: tensor<768x1024xf32>) -> tensor<?x2097152x1024xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1048576> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]]:2 = "onnx.Split"([[A_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<?x2097152x768xf32>, tensor<2xi64>) -> (tensor<?x1048576x768xf32>, tensor<?x1048576x768xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[B_]]) : (tensor<?x1048576x768xf32>, tensor<768x1024xf32>) -> tensor<?x1048576x1024xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[B_]]) : (tensor<?x1048576x768xf32>, tensor<768x1024xf32>) -> tensor<?x1048576x1024xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) {axis = 1 : si64} : (tensor<?x1048576x1024xf32>, tensor<?x1048576x1024xf32>) -> tensor<?x2097152x1024xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x2097152x1024xf32>
// CHECK:         }
}

// -----

// Do not split MatMul because a dimension does not exceeds NNPAGetMaxForDim e1 on arch15 of 2097152.

func.func @test_matmul_no_splitting_arch15_B(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768x2097152xf32>) -> (tensor<?x?x2097152xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768x2097152xf32>) -> tensor<?x?x2097152xf32>
  return %0 : tensor<?x?x2097152xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_no_splitting_arch15_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x?x768xf32>, [[B_:%.+]]: tensor<768x2097152xf32>) -> tensor<?x?x2097152xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[A_]], [[B_]]) : (tensor<?x?x768xf32>, tensor<768x2097152xf32>) -> tensor<?x?x2097152xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x2097152xf32>
// CHECK:         }
}

// -----

// Split MatMul because a dimension exceeds NNPAGetMaxForDim e1 on arch15 of 2097152: use 4194304.

func.func @test_matmul_splitting_arch15_B(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768x4194304xf32>) -> (tensor<?x?x4194304xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768x4194304xf32>) -> tensor<?x?x4194304xf32>
  return %0 : tensor<?x?x4194304xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_arch15_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x?x768xf32>, [[B_:%.+]]: tensor<768x4194304xf32>) -> tensor<?x?x4194304xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<2097152> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]]:2 = "onnx.Split"([[B_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<768x4194304xf32>, tensor<2xi64>) -> (tensor<768x2097152xf32>, tensor<768x2097152xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[A_]], [[VAR_1_]]#0) : (tensor<?x?x768xf32>, tensor<768x2097152xf32>) -> tensor<?x?x2097152xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.MatMul"([[A_]], [[VAR_1_]]#1) : (tensor<?x?x768xf32>, tensor<768x2097152xf32>) -> tensor<?x?x2097152xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) {axis = 2 : si64} : (tensor<?x?x2097152xf32>, tensor<?x?x2097152xf32>) -> tensor<?x?x4194304xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x?x4194304xf32>
// CHECK:         }
}

// -----

// No split MatMul because a dimension does not exceeds NNPAGetMaxForDim for e2/e1 on arch15 of 1048576 / 2097152

func.func @test_matmul_no_splitting_arch15_A_B(%arg0: tensor<?x1048576x768xf32>, %arg1: tensor<768x2097152xf32>) -> (tensor<?x1048576x2097152xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
  return %0 : tensor<?x1048576x2097152xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_no_splitting_arch15_A_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x1048576x768xf32>, [[B_:%.+]]: tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[A_]], [[B_]]) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x1048576x2097152xf32>
// CHECK:         }
}

// -----

// Split MatMul because a dimension exceeds NNPAGetMaxForDim for e2/e1 on arch15 of 1048576 / 2097152: use 2097152 and 4194304

func.func @test_matmul_splitting_arch15_A_B(%arg0: tensor<?x2097152x768xf32>, %arg1: tensor<768x4194304xf32>) -> (tensor<?x2097152x4194304xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x2097152x768xf32>, tensor<768x4194304xf32>) -> tensor<?x2097152x4194304xf32>
  return %0 : tensor<?x2097152x4194304xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_arch15_A_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x2097152x768xf32>, [[B_:%.+]]: tensor<768x4194304xf32>) -> tensor<?x2097152x4194304xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1048576> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]]:2 = "onnx.Split"([[A_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<?x2097152x768xf32>, tensor<2xi64>) -> (tensor<?x1048576x768xf32>, tensor<?x1048576x768xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2097152> : tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]]:2 = "onnx.Split"([[B_]], [[VAR_2_]]) {axis = 1 : si64} : (tensor<768x4194304xf32>, tensor<2xi64>) -> (tensor<768x2097152xf32>, tensor<768x2097152xf32>)
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[VAR_3_]]#0) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[VAR_3_]]#1) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_5_]]) {axis = 2 : si64} : (tensor<?x1048576x2097152xf32>, tensor<?x1048576x2097152xf32>) -> tensor<?x1048576x4194304xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[VAR_3_]]#0) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[VAR_3_]]#1) : (tensor<?x1048576x768xf32>, tensor<768x2097152xf32>) -> tensor<?x1048576x2097152xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Concat"([[VAR_7_]], [[VAR_8_]]) {axis = 2 : si64} : (tensor<?x1048576x2097152xf32>, tensor<?x1048576x2097152xf32>) -> tensor<?x1048576x4194304xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_6_]], [[VAR_9_]]) {axis = 1 : si64} : (tensor<?x1048576x4194304xf32>, tensor<?x1048576x4194304xf32>) -> tensor<?x2097152x4194304xf32>
// CHECK:           return [[VAR_10_]] : tensor<?x2097152x4194304xf32>
// CHECK:         }
}

// -----

// Rewrite N-D QLinearMatMul into 3-D one.
  
func.func @test_nd_qlinearmatmul_nd_nd(%arg0: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, %arg1: tensor<?x?x64x384xf32> {onnx.dim_params = "0:bs,1:sl"}, %arg2: tensor<f32>, %arg3: tensor<i8>) -> tensor<?x?x384x384xf32> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg2, %arg3) : (tensor<?x?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x64xi8>
  %1 = "onnx.QuantizeLinear"(%arg1, %arg2, %arg3) : (tensor<?x?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x64x384xi8>
  %2 = "onnx.QLinearMatMul"(%0, %arg2, %arg3, %1, %arg2, %arg3, %arg2, %arg3) : (tensor<?x?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<?x?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xi8>
  %3 = "onnx.DequantizeLinear"(%2, %arg2, %arg3) : (tensor<?x?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xf32>
  return %3 : tensor<?x?x384x384xf32>

// CHECK-LABEL:  func.func @test_nd_qlinearmatmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, [[PARAM_1_:%.+]]: tensor<?x?x64x384xf32> {onnx.dim_params = "0:bs,1:sl"}, [[PARAM_2_:%.+]]: tensor<f32>, [[PARAM_3_:%.+]]: tensor<i8>) -> tensor<?x?x384x384xf32> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
  // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x384x64xf32>) -> tensor<4xi64>
  // CHECK:           [[VAR_7_:%.+]] = "onnx.Slice"([[VAR_6_]], [[VAR_3_]], [[VAR_2_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  // CHECK:           [[VAR_8_:%.+]] = "onnx.Concat"([[VAR_5_]], [[VAR_7_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
  // CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_8_]]) {allowzero = 0 : si64} : (tensor<?x?x384x64xf32>, tensor<3xi64>) -> tensor<?x384x64xf32>
  // CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.QuantizeLinear"([[VAR_9_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x384x64xi8>
  // CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<?x?x64x384xf32>) -> tensor<4xi64>
  // CHECK:           [[VAR_12_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_3_]], [[VAR_2_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  // CHECK:           [[VAR_13_:%.+]] = "onnx.Concat"([[VAR_5_]], [[VAR_12_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
  // CHECK:           [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_13_]]) {allowzero = 0 : si64} : (tensor<?x?x64x384xf32>, tensor<3xi64>) -> tensor<?x64x384xf32>
  // CHECK:           [[VAR_15_:%.+]] = "onnx.QuantizeLinear"([[VAR_14_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<?x64x384xi8>
  // CHECK:           [[VAR_16_:%.+]] = "onnx.QLinearMatMul"([[VAR_10_]], [[PARAM_2_]], [[PARAM_3_]], [[VAR_15_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_2_]], [[PARAM_3_]]) : (tensor<?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xi8>
  // CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.DequantizeLinear"([[VAR_16_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64} : (tensor<?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xf32>
  // CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x384x64xf32>) -> tensor<4xi64>
  // CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<?x?x64x384xf32>) -> tensor<4xi64>
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_18_]], [[VAR_4_]], [[VAR_0_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  // CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_0_]], [[VAR_2_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  // CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_20_]], [[VAR_21_]]) {axis = 0 : si64} : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
  // CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_17_]], [[VAR_22_]]) {allowzero = 0 : si64} : (tensor<?x384x384xf32>, tensor<4xi64>) -> tensor<?x?x384x384xf32>
  // CHECK:           return [[VAR_23_]] : tensor<?x?x384x384xf32>
  // CHECK:         }
}

func.func @test_nd_qlinearmatmul_nd_2d(%arg0: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, %arg1: tensor<64x384xf32>, %arg2: tensor<f32>, %arg3: tensor<i8>) -> tensor<?x?x384x384xf32> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg2, %arg3) : (tensor<?x?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x64xi8>
  %1 = "onnx.QuantizeLinear"(%arg1, %arg2, %arg3) : (tensor<64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<64x384xi8>
  %2 = "onnx.QLinearMatMul"(%0, %arg2, %arg3, %1, %arg2, %arg3, %arg2, %arg3) : (tensor<?x?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xi8>
  %3 = "onnx.DequantizeLinear"(%2, %arg2, %arg3) : (tensor<?x?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xf32>
  return %3 : tensor<?x?x384x384xf32>

// CHECK-LABEL:  func.func @test_nd_qlinearmatmul_nd_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, [[PARAM_1_:%.+]]: tensor<64x384xf32>, [[PARAM_2_:%.+]]: tensor<f32>, [[PARAM_3_:%.+]]: tensor<i8>) -> tensor<?x?x384x384xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[64, 384]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.QuantizeLinear"([[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<64x384xi8>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x384x64xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Slice"([[VAR_8_]], [[VAR_4_]], [[VAR_3_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_6_]], [[VAR_9_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_10_]]) {allowzero = 0 : si64} : (tensor<?x?x384x64xf32>, tensor<3xi64>) -> tensor<?x384x64xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.QuantizeLinear"([[VAR_11_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x384x64xi8>
// CHECK:           [[VAR_13_:%.+]] = "onnx.QLinearMatMul"([[VAR_12_]], [[PARAM_2_]], [[PARAM_3_]], [[VAR_7_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_2_]], [[PARAM_3_]]) : (tensor<?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xi8>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.DequantizeLinear"([[VAR_13_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64} : (tensor<?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x384x64xf32>) -> tensor<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_1_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_0_]], [[VAR_2_]], [[VAR_4_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]]) {axis = 0 : si64} : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Reshape"([[VAR_14_]], [[VAR_18_]]) {allowzero = 0 : si64} : (tensor<?x384x384xf32>, tensor<4xi64>) -> tensor<?x?x384x384xf32>
// CHECK:           return [[VAR_19_]] : tensor<?x?x384x384xf32>
// CHECK:         }
}

func.func @test_nd_qlinearmatmul_2d_nd(%arg0: tensor<384x64xf32>, %arg1: tensor<?x?x64x384xf32> {onnx.dim_params = "0:bs,1:sl"}, %arg2: tensor<f32>, %arg3: tensor<i8>) -> tensor<?x?x384x384xf32> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg2, %arg3) : (tensor<384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<384x64xi8>
  %1 = "onnx.QuantizeLinear"(%arg1, %arg2, %arg3) : (tensor<?x?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x64x384xi8>
  %2 = "onnx.QLinearMatMul"(%0, %arg2, %arg3, %1, %arg2, %arg3, %arg2, %arg3) : (tensor<384x64xi8>, tensor<f32>, tensor<i8>, tensor<?x?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xi8>
  %3 = "onnx.DequantizeLinear"(%2, %arg2, %arg3) : (tensor<?x?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xf32>
  return %3 : tensor<?x?x384x384xf32>

// CHECK-LABEL:  func.func @test_nd_qlinearmatmul_2d_nd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<384x64xf32>, [[PARAM_1_:%.+]]: tensor<?x?x64x384xf32> {onnx.dim_params = "0:bs,1:sl"}, [[PARAM_2_:%.+]]: tensor<f32>, [[PARAM_3_:%.+]]: tensor<i8>) -> tensor<?x?x384x384xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[384, 64]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<384x64xi8>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<?x?x64x384xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Slice"([[VAR_8_]], [[VAR_4_]], [[VAR_3_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_6_]], [[VAR_9_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_10_]]) {allowzero = 0 : si64} : (tensor<?x?x64x384xf32>, tensor<3xi64>) -> tensor<?x64x384xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.QuantizeLinear"([[VAR_11_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<?x64x384xi8>
// CHECK:           [[VAR_13_:%.+]] = "onnx.QLinearMatMul"([[VAR_7_]], [[PARAM_2_]], [[PARAM_3_]], [[VAR_12_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_2_]], [[PARAM_3_]]) : (tensor<384x64xi8>, tensor<f32>, tensor<i8>, tensor<?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xi8>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.DequantizeLinear"([[VAR_13_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64} : (tensor<?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x384x384xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<?x?x64x384xf32>) -> tensor<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_4_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_0_]], [[VAR_5_]], [[VAR_2_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]], [[VAR_18_]]) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Reshape"([[VAR_14_]], [[VAR_19_]]) {allowzero = 0 : si64} : (tensor<?x384x384xf32>, tensor<4xi64>) -> tensor<?x?x384x384xf32>
// CHECK:           return [[VAR_20_]] : tensor<?x?x384x384xf32>
// CHECK:         }
}

// Do not rewrite because of potential broadcasting.
func.func @test_nd_qlinearmatmul_nd_nd_not_rewriting(%arg0: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, %arg1: tensor<1x?x64x384xf32> {onnx.dim_params = "1:sl"}, %arg2: tensor<f32>, %arg3: tensor<i8>) -> tensor<?x?x384x384xf32> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg2, %arg3) : (tensor<?x?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x64xi8>
  %1 = "onnx.QuantizeLinear"(%arg1, %arg2, %arg3) : (tensor<1x?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<1x?x64x384xi8>
  %2 = "onnx.QLinearMatMul"(%0, %arg2, %arg3, %1, %arg2, %arg3, %arg2, %arg3) : (tensor<?x?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<1x?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xi8>
  %3 = "onnx.DequantizeLinear"(%2, %arg2, %arg3) : (tensor<?x?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xf32>
  return %3 : tensor<?x?x384x384xf32>

// CHECK-LABEL:  func.func @test_nd_qlinearmatmul_nd_nd_not_rewriting
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x384x64xf32> {onnx.dim_params = "0:bs,1:sl"}, [[PARAM_1_:%.+]]: tensor<1x?x64x384xf32> {onnx.dim_params = "1:sl"}, [[PARAM_2_:%.+]]: tensor<f32>, [[PARAM_3_:%.+]]: tensor<i8>) -> tensor<?x?x384x384xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?x?x384x64xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x64xi8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.QuantizeLinear"([[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x?x64x384xf32>, tensor<f32>, tensor<i8>) -> tensor<1x?x64x384xi8>
// CHECK:           [[VAR_2_:%.+]] = "onnx.QLinearMatMul"([[VAR_0_]], [[PARAM_2_]], [[PARAM_3_]], [[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_2_]], [[PARAM_3_]]) : (tensor<?x?x384x64xi8>, tensor<f32>, tensor<i8>, tensor<1x?x64x384xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xi8>
// CHECK:           [[VAR_3_:%.+]] = "onnx.DequantizeLinear"([[VAR_2_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = 1 : si64} : (tensor<?x?x384x384xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x384x384xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x384x384xf32>
// CHECK:         }
}
