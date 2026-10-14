// RUN: onnx-mlir --append-decoding-strategy --EmitONNXIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length", onnx.name = "input_ids"}) -> (tensor<?x?x49155xf32> {onnx.dim_params = "0:batch_size,1:sequence_length", onnx.name = "logits"}) {
    %cst = onnx.Constant dense<0.5> : tensor<1x1x49155xf32>
    %cst_1 = onnx.Constant dense<1> : tensor<1xi64>
    %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<?x?xi64>) -> tensor<?x?xf32>
    %1 = "onnx.Unsqueeze"(%0, %cst_1) : (tensor<?x?xf32>, tensor<1xi64>) -> tensor<?x?x1xf32>
    %2 = "onnx.Add"(%1, %cst) : (tensor<?x?x1xf32>, tensor<1x1x49155xf32>) -> tensor<?x?x49155xf32>
    onnx.Return %2 : tensor<?x?x49155xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length", onnx.name = "input_ids"}) -> (tensor<?x1xi64> {onnx.name = "generated_ids"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2147483647> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x1x49155xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_0_]]) <{saturate = 1 : si64, to = f32}> {onnx_node_name = {{.*}}} : (tensor<?x?xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Unsqueeze"([[VAR_4_]], [[VAR_3_]]) {onnx_node_name = {{.*}}} : (tensor<?x?xf32>, tensor<1xi64>) -> tensor<?x1x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[VAR_5_]], [[VAR_2_]]) {onnx_node_name = {{.*}}} : (tensor<?x1x1xf32>, tensor<1x1x49155xf32>) -> tensor<?x1x49155xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Slice"([[VAR_6_]], [[VAR_1_]], [[VAR_0_]], [[VAR_3_]], [[VAR_3_]]) {onnx_node_name = {{.*}}} : (tensor<?x1x49155xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x1x49155xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.ArgMax"([[VAR_7_]]) <{axis = 2 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}> {onnx_node_name = {{.*}}} : (tensor<?x1x49155xf32>) -> tensor<?x1x1xi64>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Squeeze"([[VAR_8_]], [[VAR_3_]]) {onnx_node_name = {{.*}}} : (tensor<?x1x1xi64>, tensor<1xi64>) -> tensor<?x1xi64>
// CHECK:           return [[VAR_9_]] : tensor<?x1xi64>
// CHECK:         }
// CHECK:         "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
