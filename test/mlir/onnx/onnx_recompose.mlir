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

// -----

// RMS Layer norm (sub switched)

func.func @rms_layer_norm_v1(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
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
// CHECK-LABEL:  func.func @rms_layer_norm_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.ReduceMeanV13"([[PARAM_0_]]) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Sub"([[VAR_0_]], [[PARAM_0_]]) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_1_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm

func.func @rms_layer_norm_v2(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %dd = "onnx.Mul"(%x, %x) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%x, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// COM: QLinearMatMul
func.func @qlinear_matmul(%arg0: tensor<?x?x768xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<768x768xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> (tensor<?x?x768xi8>) {
    %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xf32>
    %1 = "onnx.DequantizeLinear"(%arg3, %arg4, %arg5) {axis = 1 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
    %2 = "onnx.MatMul"(%0, %1) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %3 = "onnx.QuantizeLinear"(%2, %arg6, %arg7) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
    return %3: tensor<?x?x768xi8>

// CHECK-LABEL:  func.func @qlinear_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xi8>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<?x?x768xi8> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.QLinearMatMul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
// CHECK:           return [[VAR_0_]] : tensor<?x?x768xi8>
// CHECK:         }
}

// -----

// gelu(x) = [x * (erf(x/1.41421354) + 1)] * 0.5
func.func @test_gelu_erf_cst_1(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = [x * (1 + erf(x/1.41421354))] * 0.5
func.func @test_gelu_erf_cst_change_add_operand_order(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%one, %1) : (tensor<f32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_add_operand_order
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = [(erf(x/1.41421354) + 1) * x] * 0.5
func.func @test_gelu_erf_cst_change_mul_operand_order_1(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%2, %arg0) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_mul_operand_order_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) =  0.5 * [x * (erf(x/1.41421354) + 1) * x]
func.func @test_gelu_erf_cst_change_mul_operand_order_2(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%half, %3) : (tensor<f32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_mul_operand_order_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = x * (0.5 * (1 + tanh[0.797884583 * (x + 0.044715 * x^3)]))
func.func @test_gelu_tanh(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %three = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %sqrt2pi = onnx.Constant dense<0.797884583> : tensor<f32>
  %cst044715 = onnx.Constant dense<4.471500e-02> : tensor<f32>
  %0 = "onnx.Pow"(%arg0, %three) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
  %1 = "onnx.Mul"(%cst044715, %0) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.Add"(%arg0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.Mul"(%sqrt2pi, %2) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Tanh"(%3) : (tensor<*xf32>) -> tensor<*xf32>
  %5 = "onnx.Add"(%one, %4) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %6 = "onnx.Mul"(%half, %5) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %7 = "onnx.Mul"(%arg0, %6) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %7 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_gelu_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "tanh"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_gelu_erf_two_adds(%arg0: tensor<?x?x3072xf32>, %arg1: tensor<3072x768xf32>) -> tensor<?x?x768xf32> {
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %3 = onnx.Constant dense<3.000000e-01> : tensor<3072xf32>
  %4 = "onnx.Add"(%arg0, %3) : (tensor<?x?x3072xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
  %5 = "onnx.Div"(%4, %2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %6 = "onnx.Erf"(%5) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %7 = "onnx.Add"(%6, %1) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %8 = "onnx.Mul"(%4, %7) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %9 = "onnx.Mul"(%8, %0) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %10 = "onnx.MatMul"(%9, %arg1) : (tensor<?x?x3072xf32>, tensor<3072x768xf32>) -> tensor<?x?x768xf32>
  return %10 : tensor<?x?x768xf32>
}
// CHECK-LABEL:  func.func @test_gelu_erf_two_adds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>, [[PARAM_1_:%.+]]: tensor<3072x768xf32>) -> tensor<?x?x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.000000e-01> : tensor<3072xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<?x?x3072xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Gelu"([[VAR_1_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_2_]], [[PARAM_1_]]) : (tensor<?x?x3072xf32>, tensor<3072x768xf32>) -> tensor<?x?x768xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x768xf32>
// CHECK:         }
