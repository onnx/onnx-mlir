// RUN: cfg_file=$(dirname %s)/load-cfg-all-on-cpu.json && onnx-mlir-opt --device-placement=load-config-file=$cfg_file --march=z16 --maccel=NNPA --split-input-file %s | FileCheck %s --check-prefix=ALL-ON-CPU

// RUN: cfg_file=$(dirname %s)/load-cfg-all-relu-on-cpu.json && onnx-mlir-opt --device-placement=load-config-file=$cfg_file --march=z16 --maccel=NNPA --split-input-file %s | FileCheck %s --check-prefix=ALL-RELU-ON-CPU

// RUN: cfg_file=$(dirname %s)/load-cfg-not-match-relu.json && onnx-mlir-opt --device-placement=load-config-file=$cfg_file --march=z16 --maccel=NNPA --split-input-file %s | FileCheck %s --check-prefix=NOT-MATCH-RELU

// RUN: cfg_file=$(dirname %s)/load-cfg-overlapping-condition.json && onnx-mlir-opt --device-placement=load-config-file=$cfg_file --march=z16 --maccel=NNPA --split-input-file %s | FileCheck %s --check-prefix=OVERLAPPING

func.func @test_load_config_file_all_on_cpu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// ALL-ON-CPU-LABEL:  func.func @test_load_config_file_all_on_cpu
// ALL-ON-CPU-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// ALL-ON-CPU:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "cpu", onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-ON-CPU:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-ON-CPU:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-ON-CPU:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {device = "cpu", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-ON-CPU:           onnx.Return [[VAR_3_]] : tensor<?x?x?xf32>
// ALL-ON-CPU:         }
}

// -----

func.func @test_load_config_file_all_relu_on_cpu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// ALL-RELU-ON-CPU-LABEL:  func.func @test_load_config_file_all_relu_on_cpu
// ALL-RELU-ON-CPU-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// ALL-RELU-ON-CPU:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "cpu", onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-RELU-ON-CPU:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-RELU-ON-CPU:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-RELU-ON-CPU:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {device = "nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// ALL-RELU-ON-CPU:           onnx.Return [[VAR_3_]] : tensor<?x?x?xf32>
// ALL-RELU-ON-CPU:         }
}

// -----

func.func @test_load_config_file_some_relu_on_cpu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// SOME-RELU-ON-CPU-LABEL:  func.func @test_load_config_file_some_relu_on_cpu
// SOME-RELU-ON-CPU-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// SOME-RELU-ON-CPU:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "nnpa", onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// SOME-RELU-ON-CPU:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// SOME-RELU-ON-CPU:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// SOME-RELU-ON-CPU:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {device = "nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// SOME-RELU-ON-CPU:           onnx.Return [[VAR_3_]] : tensor<?x?x?xf32>
// SOME-RELU-ON-CPU:         }
}

// -----

func.func @test_load_config_file_not_match_relu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// NOT-MATCH-RELU-LABEL:  func.func @test_load_config_file_not_match_relu
// NOT-MATCH-RELU-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// NOT-MATCH-RELU:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "nnpa", onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// NOT-MATCH-RELU:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "nnpa", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// NOT-MATCH-RELU:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "nnpa", onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// NOT-MATCH-RELU:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {device = "cpu", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// NOT-MATCH-RELU:           onnx.Return [[VAR_3_]] : tensor<?x?x?xf32>
// NOT-MATCH-RELU:         }
}

// -----

func.func @test_load_config_file_overlapping_condition(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// OVERLAPPING-LABEL:  func.func @test_load_config_file_overlapping_condition
// OVERLAPPING-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// OVERLAPPING:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "nnpa", onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// OVERLAPPING:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// OVERLAPPING:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// OVERLAPPING:           [[VAR_3_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]) {device = "nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// OVERLAPPING:           onnx.Return [[VAR_3_]] : tensor<?x?x?xf32>
// OVERLAPPING:         }
}
