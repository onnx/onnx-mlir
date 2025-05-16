// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file --mlir-print-debuginfo | FileCheck %s


func.func @test_reorder_relu_maxpool(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32> loc("Relu")
  %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_1", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc("MaxPool")
  return %1 : tensor<1x64x16x16xf32>

  // CHECK-LABEL: func @test_reorder_relu_maxpool
  // CHECK:           [[VAR_0_:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_MAX_POOL:#.+]])
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) : (tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_RELU:#.+]])
  // CHECK-DAG:       [[LOC_MAX_POOL:#.+]] = loc("MaxPool")
  // CHECK-DAG:       [[LOC_RELU:#.+]] = loc("Relu")
}

// -----

func.func @test_recompose_concat(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32> ) -> tensor<1x12x4xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x6x4xf32> loc("Concat1")
  %1 = "onnx.Concat"(%0, %arg0) {axis = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32> loc("Concat2")
  %2 = "onnx.Concat"(%1, %arg1) {axis = 1 : si64} : (tensor<1x9x4xf32>, tensor<1x3x4xf32>) -> tensor<1x12x4xf32> loc("Concat3")
  return %2 : tensor<1x12x4xf32>

  // CHECK-LABEL: func @test_recompose_concat
  // CHECK: "onnx.Concat"
  // CHECK-SAME: loc([[LOC_FUSED:#.+]])
  // CHECK-DAG:       [[LOC_C1:#.+]] = loc("Concat1")
  // CHECK-DAG:       [[LOC_C2:#.+]] = loc("Concat2")
  // CHECK-DAG:       [[LOC_C3:#.+]] = loc("Concat3")
  // CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_C3]], [[LOC_C2]], [[LOC_C1]]]) 
}

