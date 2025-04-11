// RUN: onnx-mlir  --useOnnxModelTypes=false --EmitONNXIR --printIR %s | FileCheck %s

func.func @test_recompose_concat(%arg0: tensor<1x3x6x6xf32>) -> tensor<1x12x6x6xf32> {
%0 = onnx.Constant dense<0.00999999977> : tensor<6x3x3x3xf32>
%1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
%2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_1", pads = [1, 1, 1, 1]} : (tensor<1x3x6x6xf32>, tensor<6x3x3x3xf32>, none) -> tensor<1x6x6x6xf32>
%3 = "onnx.Relu"(%2) {onnx_node_name = "onnx.Relu_2"} : (tensor<1x6x6x6xf32>) -> tensor<1x6x6x6xf32>
%4 = "onnx.Concat"(%arg0, %3) {axis = 1 : si64, onnx_node_name = "onnx.Concat_3"} : (tensor<1x3x6x6xf32>, tensor<1x6x6x6xf32>) -> tensor<1x9x6x6xf32>
%5 = "onnx.Concat"(%4, %arg0) {axis = 1 : si64, onnx_node_name = "onnx.Concat_4"} : (tensor<1x9x6x6xf32>, tensor<1x3x6x6xf32>) -> tensor<1x12x6x6xf32>
return %5 : tensor<1x12x6x6xf32>

  // CHECK-LABEL: func @test_recompose_concat
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x6x6xf32>) -> tensor<1x12x6x6xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<6x3x3x3xf32>
  
  // CHECK:      [[VAR_1_:%.+]] = "onnx.NoValue"()

  // CHECK:     [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     : (tensor<1x3x6x6xf32>, tensor<6x3x3x3xf32>, none) -> tensor<1x6x6x6xf32>
  // CHECK: [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) {onnx_node_name = "onnx.Relu_2"} : (tensor<1x6x6x6xf32>) -> tensor<1x6x6x6xf32>
  // CHECK: [[FINAL_OUT:%.+]] = "onnx.Concat"([[PARAM_0_]], [[VAR_3_]], [[PARAM_0_]]) {axis = 1 : si64, onnx_node_name = "onnx.Concat_0"} : (tensor<1x3x6x6xf32>, tensor<1x6x6x6xf32>, tensor<1x3x6x6xf32>) -> tensor<1x12x6x6xf32>
  // CHECK-NEXT:     return [[FINAL_OUT]] : tensor<1x12x6x6xf32>

}

func.func @test_recompose_concat_simple(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32> ) -> tensor<1x12x4xf32> {
%0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64, onnx_node_name = "onnx.Concat_0"} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x6x4xf32>
%1 = "onnx.Concat"(%0, %arg0) {axis = 1 : si64, onnx_node_name = "onnx.Concat_1"} : (tensor<1x6x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32>
%2 = "onnx.Concat"(%1, %arg1) {axis = 1 : si64, onnx_node_name = "onnx.Concat_2"} : (tensor<1x9x4xf32>, tensor<1x3x4xf32>) -> tensor<1x12x4xf32>
return %2 : tensor<1x12x4xf32>

  // CHECK-LABEL: func @test_recompose_concat_simple
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1x3x4xf32>) -> tensor<1x12x4xf32> {
  // CHECK: [[FINAL_OUT:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_0_]], [[PARAM_1_]]) 
  // CHECK-SAME: {axis = 1 : si64, onnx_node_name = "onnx.Concat_1"} 
  // CHECK-NEXT: return [[FINAL_OUT]] : tensor<1x12x4xf32>

}