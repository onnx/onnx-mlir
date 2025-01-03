// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s --mlir-print-debuginfo -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func @layernorm_without_bias
func.func @layernorm_without_bias(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32> loc("mReduce")
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32> loc("sub")
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc("ddMul")
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32> loc("vReduce")
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32> loc("add")
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32> loc("sqrt")
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32> loc("div")
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32> loc("lnMul")
  return %Y : tensor<1x384x768xf32> loc("return")
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none) loc([[LOC_FUSED:#.+]])
// CHECK:           return [[VAR_Y_]] : tensor<1x384x768xf32>
// CHECK-DAG:       [[LOC_M_REDUCE:#.+]] = loc("mReduce")
// CHECK-DAG:       [[LOC_SUB:#.+]] = loc("sub")
// CHECK-DAG:       [[LOC_DD_MUL:#.+]] = loc("ddMul")
// CHECK-DAG:       [[LOC_V_REDUCE:#.+]] = loc("vReduce")
// CHECK-DAG:       [[LOC_ADD:#.+]] = loc("add")
// CHECK-DAG:       [[LOC_SQRT:#.+]] = loc("sqrt")
// CHECK-DAG:       [[LOC_DIV:#.+]] = loc("div")
// CHECK-DAG:       [[LOC_LN_MUL:#.+]] = loc("lnMul")
// CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_M_REDUCE]], [[LOC_SUB]], [[LOC_DD_MUL]], [[LOC_V_REDUCE]], [[LOC_ADD]], [[LOC_SQRT]], [[LOC_DIV]], [[LOC_LN_MUL]]]) 
}
