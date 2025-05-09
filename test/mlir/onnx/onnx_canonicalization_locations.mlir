// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file --mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL:  func.func @layernorm_with_bias
func.func @layernorm_with_bias(%arg0: tensor<1x384x768xf32>, %arg1: tensor<768xf32>, %arg3: tensor<768xf32>) -> tensor<1x384x768xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %mean, %stddev = "onnx.LayerNormalization"(%arg0, %arg1, %none) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none) loc("LN")
  %ret = "onnx.Add"(%y, %arg3) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32> loc("Bias")
  return %ret : tensor<1x384x768xf32>
  // CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none) loc([[LOC_FUSED:#.+]])
  // CHECK:           return [[Y_]] : tensor<1x384x768xf32>
  // CHECK-DAG:       [[LOC_LN:#.+]] = loc("LN")
  // CHECK-DAG:       [[LOC_BIAS:#.+]] = loc("Bias")
  // CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_LN]], [[LOC_BIAS]]]) 
}


// -----

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

