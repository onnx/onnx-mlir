// RUN: cfg_file=$(dirname %s)/tensorinfo-config.json && onnx-mlir --EmitONNXIR --march=z17 --maccel=NNPA --config-file=$cfg_file --printIR %s | FileCheck %s

// COM: for the tests in this file, see tensorinfo-config.json for conditions.
// COM: tests are differentiated by onnx_node_name.

func.func @test_tensor_info_config(%arg0: tensor<8x8192xf32>) -> tensor<8x8192xf32> {
  %noconfig = "onnx.Relu"(%arg0) {onnx_node_name = "no_config"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %0 = "onnx.Relu"(%noconfig) {onnx_node_name = "test_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "test_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "test_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %3 = "onnx.Relu"(%2) {onnx_node_name = "test_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %4 = "onnx.Relu"(%3) {onnx_node_name = "test_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %5 = "onnx.Relu"(%4) {onnx_node_name = "test_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %6 = "onnx.Relu"(%5) {onnx_node_name = "test_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %7 = "onnx.Relu"(%6) {onnx_node_name = "test_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

  %8 = "onnx.Relu"(%7) {onnx_node_name = "test_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>


  %9 = "onnx.Relu"(%8) {onnx_node_name = "test_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %10 = "onnx.Relu"(%9) {onnx_node_name = "test_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %11 = "onnx.Relu"(%10) {onnx_node_name = "test_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %12 = "onnx.Relu"(%11) {onnx_node_name = "test_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %13 = "onnx.Relu"(%12) {onnx_node_name = "test_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %14 = "onnx.Relu"(%13) {onnx_node_name = "test_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %15 = "onnx.Relu"(%14) {onnx_node_name = "test_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %16 = "onnx.Relu"(%15) {onnx_node_name = "test_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

  onnx.Return %16: tensor<8x8192xf32>

// CHECK-LABEL:  func.func @test_tensor_info_config
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x8192xf32>) -> tensor<8x8192xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "nnpa", onnx_node_name = "no_config"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "test_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "test_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) {device = "cpu", onnx_node_name = "test_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Relu"([[VAR_3_]]) {device = "cpu", onnx_node_name = "test_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Relu"([[VAR_4_]]) {device = "cpu", onnx_node_name = "test_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Relu"([[VAR_5_]]) {device = "cpu", onnx_node_name = "test_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Relu"([[VAR_6_]]) {device = "cpu", onnx_node_name = "test_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Relu"([[VAR_7_]]) {device = "cpu", onnx_node_name = "test_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

// CHECK:           [[VAR_9_:%.+]] = "onnx.Relu"([[VAR_8_]]) {device = "cpu", onnx_node_name = "test_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

// CHECK:           [[VAR_10_:%.+]] = "onnx.Relu"([[VAR_9_]]) {device = "cpu", onnx_node_name = "test_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Relu"([[VAR_10_]]) {device = "cpu", onnx_node_name = "test_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Relu"([[VAR_11_]]) {device = "cpu", onnx_node_name = "test_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Relu"([[VAR_12_]]) {device = "cpu", onnx_node_name = "test_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Relu"([[VAR_13_]]) {device = "cpu", onnx_node_name = "test_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Relu"([[VAR_14_]]) {device = "cpu", onnx_node_name = "test_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Relu"([[VAR_15_]]) {device = "cpu", onnx_node_name = "test_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Relu"([[VAR_16_]]) {device = "cpu", onnx_node_name = "test_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

// CHECK:        }
}
