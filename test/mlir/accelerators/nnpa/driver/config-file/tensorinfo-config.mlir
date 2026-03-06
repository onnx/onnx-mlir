// RUN: cfg_file=$(dirname %s)/tensorinfo-config.json && onnx-mlir --EmitONNXIR --march=z17 --maccel=NNPA --config-file=$cfg_file --printIR %s | FileCheck %s

// COM: for the tests in this file, see tensorinfo-config.json for conditions.
// COM: tests are differentiated by onnx_node_name.

func.func @test_tensor_info_config(%arg0: tensor<8x8192xf32>) -> tensor<8x8192xf32> {
  %noconfig = "onnx.Relu"(%arg0) {onnx_node_name = "no_config"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

  // Input rank.
  %input0 = "onnx.Relu"(%noconfig) {onnx_node_name = "test_input_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input1 = "onnx.Relu"(%input0) {onnx_node_name = "test_input_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input2 = "onnx.Relu"(%input1) {onnx_node_name = "test_input_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input3 = "onnx.Relu"(%input2) {onnx_node_name = "test_input_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input4 = "onnx.Relu"(%input3) {onnx_node_name = "test_input_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input5 = "onnx.Relu"(%input4) {onnx_node_name = "test_input_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input6 = "onnx.Relu"(%input5) {onnx_node_name = "test_input_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input7 = "onnx.Relu"(%input6) {onnx_node_name = "test_input_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Input type. 
  %input8 = "onnx.Relu"(%input7) {onnx_node_name = "test_input_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Input dims.
  %input9 = "onnx.Relu"(%input8) {onnx_node_name = "test_input_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input10 = "onnx.Relu"(%input9) {onnx_node_name = "test_input_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input11 = "onnx.Relu"(%input10) {onnx_node_name = "test_input_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input12 = "onnx.Relu"(%input11) {onnx_node_name = "test_input_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input13 = "onnx.Relu"(%input12) {onnx_node_name = "test_input_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input14 = "onnx.Relu"(%input13) {onnx_node_name = "test_input_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input15 = "onnx.Relu"(%input14) {onnx_node_name = "test_input_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input16 = "onnx.Relu"(%input15) {onnx_node_name = "test_input_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %input17 = "onnx.Relu"(%input16) {onnx_node_name = "test_input_dim_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Input ID. 
  %input18 = "onnx.Relu"(%input17) {onnx_node_name = "test_input_id_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

  // Output rank.
  %output0 = "onnx.Relu"(%input18) {onnx_node_name = "test_output_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output1 = "onnx.Relu"(%output0) {onnx_node_name = "test_output_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output2 = "onnx.Relu"(%output1) {onnx_node_name = "test_output_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output3 = "onnx.Relu"(%output2) {onnx_node_name = "test_output_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output4 = "onnx.Relu"(%output3) {onnx_node_name = "test_output_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output5 = "onnx.Relu"(%output4) {onnx_node_name = "test_output_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output6 = "onnx.Relu"(%output5) {onnx_node_name = "test_output_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output7 = "onnx.Relu"(%output6) {onnx_node_name = "test_output_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Output type. 
  %output8 = "onnx.Relu"(%output7) {onnx_node_name = "test_output_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Output dims.
  %output9 = "onnx.Relu"(%output8) {onnx_node_name = "test_output_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output10 = "onnx.Relu"(%output9) {onnx_node_name = "test_output_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output11 = "onnx.Relu"(%output10) {onnx_node_name = "test_output_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output12 = "onnx.Relu"(%output11) {onnx_node_name = "test_output_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output13 = "onnx.Relu"(%output12) {onnx_node_name = "test_output_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output14 = "onnx.Relu"(%output13) {onnx_node_name = "test_output_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output15 = "onnx.Relu"(%output14) {onnx_node_name = "test_output_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output16 = "onnx.Relu"(%output15) {onnx_node_name = "test_output_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %output17 = "onnx.Relu"(%output16) {onnx_node_name = "test_output_dim_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // Output ID. 
  %output18 = "onnx.Relu"(%output17) {onnx_node_name = "test_output_id_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>

  onnx.Return %output18 : tensor<8x8192xf32>

  // CHECK-LABEL:  func.func @test_tensor_info_config
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x8192xf32>) -> tensor<8x8192xf32> {
  // CHECK:           [[VAR_0_:%.+]] = "onnx.Relu"([[PARAM_0_]]) {device = "nnpa", onnx_node_name = "no_config"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {device = "cpu", onnx_node_name = "test_input_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) {device = "cpu", onnx_node_name = "test_input_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) {device = "cpu", onnx_node_name = "test_input_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_4_:%.+]] = "onnx.Relu"([[VAR_3_]]) {device = "cpu", onnx_node_name = "test_input_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_5_:%.+]] = "onnx.Relu"([[VAR_4_]]) {device = "cpu", onnx_node_name = "test_input_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_6_:%.+]] = "onnx.Relu"([[VAR_5_]]) {device = "cpu", onnx_node_name = "test_input_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_7_:%.+]] = "onnx.Relu"([[VAR_6_]]) {device = "cpu", onnx_node_name = "test_input_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_8_:%.+]] = "onnx.Relu"([[VAR_7_]]) {device = "cpu", onnx_node_name = "test_input_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_9_:%.+]] = "onnx.Relu"([[VAR_8_]]) {device = "cpu", onnx_node_name = "test_input_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_10_:%.+]] = "onnx.Relu"([[VAR_9_]]) {device = "cpu", onnx_node_name = "test_input_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_11_:%.+]] = "onnx.Relu"([[VAR_10_]]) {device = "cpu", onnx_node_name = "test_input_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_12_:%.+]] = "onnx.Relu"([[VAR_11_]]) {device = "cpu", onnx_node_name = "test_input_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_13_:%.+]] = "onnx.Relu"([[VAR_12_]]) {device = "cpu", onnx_node_name = "test_input_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_14_:%.+]] = "onnx.Relu"([[VAR_13_]]) {device = "cpu", onnx_node_name = "test_input_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_15_:%.+]] = "onnx.Relu"([[VAR_14_]]) {device = "cpu", onnx_node_name = "test_input_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_16_:%.+]] = "onnx.Relu"([[VAR_15_]]) {device = "cpu", onnx_node_name = "test_input_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_17_:%.+]] = "onnx.Relu"([[VAR_16_]]) {device = "cpu", onnx_node_name = "test_input_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_18_:%.+]] = "onnx.Relu"([[VAR_17_]]) {device = "cpu", onnx_node_name = "test_input_dim_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_19_:%.+]] = "onnx.Relu"([[VAR_18_]]) {device = "cpu", onnx_node_name = "test_input_id_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_20_:%.+]] = "onnx.Relu"([[VAR_19_]]) {device = "cpu", onnx_node_name = "test_output_rank_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_21_:%.+]] = "onnx.Relu"([[VAR_20_]]) {device = "cpu", onnx_node_name = "test_output_rank_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_22_:%.+]] = "onnx.Relu"([[VAR_21_]]) {device = "cpu", onnx_node_name = "test_output_rank_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_23_:%.+]] = "onnx.Relu"([[VAR_22_]]) {device = "cpu", onnx_node_name = "test_output_rank_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_24_:%.+]] = "onnx.Relu"([[VAR_23_]]) {device = "cpu", onnx_node_name = "test_output_rank_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_25_:%.+]] = "onnx.Relu"([[VAR_24_]]) {device = "cpu", onnx_node_name = "test_output_rank_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_26_:%.+]] = "onnx.Relu"([[VAR_25_]]) {device = "cpu", onnx_node_name = "test_output_rank_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_27_:%.+]] = "onnx.Relu"([[VAR_26_]]) {device = "cpu", onnx_node_name = "test_output_rank_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_28_:%.+]] = "onnx.Relu"([[VAR_27_]]) {device = "cpu", onnx_node_name = "test_output_type"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_29_:%.+]] = "onnx.Relu"([[VAR_28_]]) {device = "cpu", onnx_node_name = "test_output_dim_eq_1"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_30_:%.+]] = "onnx.Relu"([[VAR_29_]]) {device = "cpu", onnx_node_name = "test_output_dim_eq_2"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_31_:%.+]] = "onnx.Relu"([[VAR_30_]]) {device = "cpu", onnx_node_name = "test_output_dim_lt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_32_:%.+]] = "onnx.Relu"([[VAR_31_]]) {device = "cpu", onnx_node_name = "test_output_dim_lt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_33_:%.+]] = "onnx.Relu"([[VAR_32_]]) {device = "cpu", onnx_node_name = "test_output_dim_gt"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_34_:%.+]] = "onnx.Relu"([[VAR_33_]]) {device = "cpu", onnx_node_name = "test_output_dim_gt_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_35_:%.+]] = "onnx.Relu"([[VAR_34_]]) {device = "cpu", onnx_node_name = "test_output_dim_not_eq"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_36_:%.+]] = "onnx.Relu"([[VAR_35_]]) {device = "cpu", onnx_node_name = "test_output_dim_modulo"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_37_:%.+]] = "onnx.Relu"([[VAR_36_]]) {device = "cpu", onnx_node_name = "test_output_dim_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:           [[VAR_38_:%.+]] = "onnx.Relu"([[VAR_37_]]) {device = "cpu", onnx_node_name = "test_output_id_negative"} : (tensor<8x8192xf32>) -> tensor<8x8192xf32>
  // CHECK:         }
}
