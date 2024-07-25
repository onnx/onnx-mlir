// RUN: onnx-mlir --printIR --EmitONNXIR %s | FileCheck %s

// COM: Check that Dequantize-MatMul-Quantize is always recomposed to QLinearMatMul before the removal of Quantize-Dequantize is applied. 
// COM: Otherwise, the recomposition of QLinearMatMul failed due to pattern mismatched (lack of DequantizeLinear).
module {
  func.func @qlinear_matmul(%arg0: tensor<?x?x768xf32>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<768x768xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> (tensor<?x?x768xi8>) {
      %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
      %1 = "onnx.DequantizeLinear"(%0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xf32>
      %2 = "onnx.DequantizeLinear"(%arg3, %arg4, %arg5) {axis = 1 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
      %3 = "onnx.MatMul"(%1, %2) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
      %4 = "onnx.QuantizeLinear"(%3, %arg6, %arg7) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
      return %4: tensor<?x?x768xi8>
  
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()

// CHECK-LABEL:  func.func @qlinear_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xf32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<?x?x768xi8> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64, onnx_node_name = "onnx.QuantizeLinear_0", saturate = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
// CHECK:           [[VAR_1_:%.+]] = "onnx.QLinearMatMul"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) {onnx_node_name = "onnx.QLinearMatMul_1"} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
// CHECK:           return [[VAR_1_]] : tensor<?x?x768xi8>
// CHECK:         }
// CHECK:         "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
