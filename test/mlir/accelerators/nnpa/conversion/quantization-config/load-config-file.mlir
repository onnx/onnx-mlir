// RUN: cfg_file=$(dirname %s)/cfg.json && onnx-mlir-opt --nnpa-quant-ops-selection=load-config-file=$cfg_file --march=z17 --maccel=NNPA --split-input-file %s | FileCheck %s

func.func @test_load_config_file(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg0) {onnx_node_name = "MatMul_0"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.MatMul"(%0, %1) {onnx_node_name = "MatMul_2"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %3 : tensor<?x?xf32>

// CHECK-LABEL:  func.func @test_load_config_file
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_0_]]) {onnx_node_name = "MatMul_0", quantize = false} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_0_]]) {onnx_node_name = "MatMul_1", quantize = true} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_1_]]) {onnx_node_name = "MatMul_2", quantize = false} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<?x?xf32>
// CHECK:         }
}
