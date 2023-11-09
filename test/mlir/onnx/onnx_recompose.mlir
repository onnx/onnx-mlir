// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// Layernorm with bias (not recognized as need multiple passes).

func.func @layernorm_with_spurious_adds(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %x = "onnx.Add"(%input, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %output = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %output : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_with_spurious_adds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm without bias
func.func @layernorm_without_bias(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm, add/mul switched

func.func @layernorm_with_bias_swtiched(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_with_bias_swtiched
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Recognize the bias and fold into LayerNorm.
func.func @layernorm_without_bias(%arg0: tensor<1x384x768xf32>, %arg1: tensor<768xf32>, %bias: tensor<768xf32>) -> tensor<1x384x768xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %NormScaled, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Not a Layernorm as top sub has inputs switched
func.func @not_a_layer_norm(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%mean, %x) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @not_a_layer_norm
// CHECK-NOT:       "onnx.LayerNormalization"
// CHECK:         }
}

// -----
// Check alternative layer norm with reciprocal instead of div
func.func @layer_norm_with_reciprocal(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Reciprocal"(%StdDev) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_with_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div
func.func @layer_norm_with_div_by_one(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %one = onnx.Constant dense<1.0> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Div"(%one, %StdDev) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_with_div_by_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div, fail because it is 2 / x instead of 1 / x
func.func @not_a_layer_norm_with_div_by_two(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %one = onnx.Constant dense<2.0> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Div"(%one, %StdDev) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @not_a_layer_norm_with_div_by_two
// CHECK-NOT:       "onnx.LayerNormalization"
// CHECK:         }
}
