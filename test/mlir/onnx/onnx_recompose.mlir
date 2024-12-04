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

func.func @layernorm_without_bias_first_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_2_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_dynamic_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>, %axis: tensor<?xi64>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_3_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %mean = "onnx.ReduceMean"(%x, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_3_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReduceMean"([[VAR_4_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[VAR_5_]], [[VAR_0_]]) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Sqrt"([[VAR_6_]]) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Div"([[VAR_3_]], [[VAR_7_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Mul"([[VAR_8_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_9_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_reduce_all(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<9.99999974E-6> : tensor<f32>
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
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 0 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm, add/mul switched

func.func @layernorm_with_bias_switched(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
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
// CHECK-LABEL:  func.func @layernorm_with_bias_switched
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
