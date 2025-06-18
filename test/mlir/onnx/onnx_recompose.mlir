// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// Layernorm with bias (not recognized as need multiple passes).

func.func @layernorm_with_spurious_adds(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %x = "onnx.Add"(%input, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm without bias
func.func @layernorm_without_bias(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_first_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_first_reduce_unsuitable_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.ReduceMeanV13"([[PARAM_0_]]) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_2_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_second_reduce_unsuitable_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.ReduceMeanV13"([[PARAM_0_]]) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMeanV13"([[VAR_3_]]) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_0_]]) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sqrt"([[VAR_5_]]) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_6_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Mul"([[VAR_7_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis = onnx.Constant dense<-1> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_dynamic_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>, %axis: tensor<?xi64>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMean"(%x, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_dynamic_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>, [[PARAM_3_:%.+]]: tensor<?xi64>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[PARAM_3_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[PARAM_3_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_0_]]) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sqrt"([[VAR_5_]]) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_6_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Mul"([[VAR_7_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_first_reduce_unsuitable_axis_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis1 = onnx.Constant dense<-2> : tensor<1xi64>
  %axis2 = onnx.Constant dense<-1> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis1) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis2) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_first_reduce_unsuitable_axis_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_3_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis1 = onnx.Constant dense<-1> : tensor<1xi64>
  %axis2 = onnx.Constant dense<-2> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis1) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis2) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_second_reduce_unsuitable_axis_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_3_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ReduceMean"([[VAR_5_]], [[VAR_2_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Add"([[VAR_6_]], [[VAR_0_]]) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Sqrt"([[VAR_7_]]) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Div"([[VAR_4_]], [[VAR_8_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Mul"([[VAR_9_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_10_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_noop(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %mean = "onnx.ReduceMean"(%x, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x768xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x768xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x768xf32>, tensor<f32>) -> tensor<1x384x768xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_0_]]) : (tensor<1x384x768xf32>, tensor<f32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sqrt"([[VAR_3_]]) : (tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Div"([[VAR_1_]], [[VAR_4_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_reduce_all(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %mean = "onnx.ReduceMean"(%x, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_reduce_all
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 0 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm, add/mul switched

func.func @layernorm_with_bias_switched(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK-LABEL:  func.func @layernorm_with_bias_switched
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Recognize the bias and fold into LayerNorm.
func.func @layernorm_without_bias(%arg0: tensor<1x384x768xf32>, %arg1: tensor<768xf32>, %bias: tensor<768xf32>) -> tensor<1x384x768xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %NormScaled, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Not a Layernorm as top sub has inputs switched
func.func @not_a_layer_norm(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div
func.func @layer_norm_with_div_by_one(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div, fail because it is 2 / x instead of 1 / x
func.func @not_a_layer_norm_with_div_by_two(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_1_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm

func.func @rms_layer_norm_v2(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm (containing pow(varEps, -0.5))

func.func @rms_layer_norm_v3(%x: tensor<1x384x768xf32>) -> (tensor<1x384x768xf32>) {
  %neg_half = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %xx = "onnx.Mul"(%x, %x) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%xx) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %invStdDev = "onnx.Pow"(%varEps, %neg_half) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %Y = "onnx.Mul"(%invStdDev, %x) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[PARAM_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<768xf32>
// CHECK:           [[PARAM_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm (containing pow(varEps, -0.5))

func.func @rms_layer_norm_v3_dyn_shape(%x: tensor<1x?x768xf32>) -> (tensor<1x?x768xf32>) {
  %neg_half = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %xx = "onnx.Mul"(%x, %x) : (tensor<1x?x768xf32>, tensor<1x?x768xf32>) -> tensor<1x?x768xf32>
  %var = "onnx.ReduceMeanV13"(%xx) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x?x768xf32>) -> tensor<1x?x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
  %invStdDev = "onnx.Pow"(%varEps, %neg_half) : (tensor<1x?x1xf32>, tensor<f32>) -> tensor<1x?x1xf32>
  %Y = "onnx.Mul"(%invStdDev, %x) : (tensor<1x?x1xf32>, tensor<1x?x768xf32>) -> tensor<1x?x768xf32>
  return %Y : tensor<1x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v3_dyn_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x768xf32>) -> tensor<1x?x768xf32> {
// CHECK:           [[PARAM_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<768xf32>
// CHECK:           [[PARAM_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x?x768xf32>, tensor<768xf32>, none) -> (tensor<1x?x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x?x768xf32>
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

// COM: AMD Disabled
// DISABLED-LABEL:  func.func @qlinear_matmul
// DISABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xi8>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<?x?x768xi8> {
// DISABLED:           [[VAR_0_:%.+]] = "onnx.QLinearMatMul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
// DISABLED:           return [[VAR_0_]] : tensor<?x?x768xi8>
// DISABLED:         }
}

// -----


func.func @qlinear_matmul_with_result_type(%arg0: tensor<?x?x768xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<768x768xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> (tensor<1x2x768xi8>) {
    %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xf32>
    %1 = "onnx.DequantizeLinear"(%arg3, %arg4, %arg5) {axis = 1 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
    %2 = "onnx.MatMul"(%0, %1) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %3 = "onnx.QuantizeLinear"(%2, %arg6, %arg7) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<1x2x768xi8>
    return %3: tensor<1x2x768xi8>
// COM: AMD Disabled
// DISABLED-LABEL:  func.func @qlinear_matmul_with_result_type
// DISABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xi8>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<1x2x768xi8> {
// DISABLED:           [[VAR_0_:%.+]] = "onnx.QLinearMatMul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<1x2x768xi8>
// DISABLED:           return [[VAR_0_]] : tensor<1x2x768xi8>
// DISABLED:         }
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


func.func @test_gelu_with_result_type(%arg0 : tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<1x2x3072xf32>
  "func.return"(%4) : (tensor<1x2x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_with_result_type
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<1x2x3072xf32>
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

// -----

func.func @test_depth_to_space_CRD(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-LABEL:func.func @test_depth_to_space_CRD
// CHECK-SAME:   (%[[PARAM_1:.+]]: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  %[[DTS:.+]] = "onnx.DepthToSpace"(%[[PARAM_1]]) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  return %[[DTS]] : tensor<1x32x1080x1920xf32>
//      CHECK:}

// -----

func.func @test_depth_to_space_CRD_missing_transpose_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_first_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x540x3840xf32> {
  %0 = onnx.Constant dense<[-1, 32, 1, 4, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 524, 3840]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x1x4x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x1x4x540x960xf32>) -> tensor<1x32x540x1x960x4xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x1x960x4xf32>, tensor<4xi64>) -> tensor<1x32x540x3840xf32>
  return %4 : tensor<1x32x540x3840xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 3, 5, 2]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_second_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 1, 32, 1080, 1920]> : tensor<5xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<5xi64>) -> tensor<1x1x32x1080x1920xf32>
  return %4 : tensor<1x1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_not_static_shapes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<6xi64>) -> tensor<*xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-LABEL:func.func @test_depth_to_space_DCR
// CHECK-SAME:   (%[[PARAM_1:.+]]: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  %[[DTS:.+]] = "onnx.DepthToSpace"(%[[PARAM_1]]) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  return %[[DTS]] : tensor<1x32x1080x1920xf32>
//      CHECK:}

// -----

func.func @test_depth_to_space_DCR_missing_transpose_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_first_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x2x17280x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 32, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x32x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x32x2x540x960xf32>) -> tensor<1x2x540x32x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x2x540x32x960x2xf32>, tensor<4xi64>) -> tensor<1x2x17280x1920xf32>
  return %4 : tensor<1x2x17280x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 2, 5, 1]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_second_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x540x3680xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 540, 3680]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x540x3680xf32>
  return %4 : tensor<1x32x540x3680xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_not_static_shapes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<6xi64>) -> tensor<*xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
// CHECK-NOT: onnx.DepthToSpace
