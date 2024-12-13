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



