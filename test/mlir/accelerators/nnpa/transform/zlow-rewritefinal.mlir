// RUN: onnx-mlir-opt --maccel=NNPA --zlow-rewrite-final --canonicalize %s -split-input-file | FileCheck %s

// -----

func @test_prev_layer_lstm(
  %input_0 : memref<1x1x1x1x32x64xf16>, %h0_0 : memref<1x1x1x1x32x64xf16>, %c0_0 : memref<1x1x1x1x32x64xf16>, %input_weights_0 : memref<1x4x1x1x32x64xf16>, %input_bias_0 : memref<1x4x1x1x32x64xf16>, %hidden_weights_0 : memref<1x4x1x1x32x64xf16>, %hidden_bias_0 : memref<1x4x1x1x32x64xf16>, %work_area_0 : memref<40960xi8>, %shape_0 : memref<5xi64>, %hn_output_0 : memref<1x1x1x1x32x64xf16>, %cf_output_0 : memref<1x1x1x1x32x64xf16>,
  %input_1 : memref<1x1x1x1x32x64xf16>, %h0_1 : memref<2x1x1x1x32x64xf16>, %c0_1 : memref<2x1x1x1x32x64xf16>, %input_weights_1 : memref<1x8x2x1x32x64xf16>, %input_bias_1 : memref<1x8x1x1x32x64xf16>, %hidden_weights_1 : memref<1x8x2x1x32x64xf16>, %hidden_bias_1 : memref<1x8x1x1x32x64xf16>, %work_area_1 : memref<81920xi8>, %shape_1 : memref<5xi64>, %hn_output_1 : memref<1x3x2x1x32x64xf16>, %cf_output_1 : memref<1x3x2x1x32x64xf16>,
  %input_2 : memref<1x1x1x1x32x64xf16>, %h0_2 : memref<1x1x1x1x32x64xf16>, %c0_2 : memref<1x1x1x1x32x64xf16>, %input_weights_2 : memref<1x4x1x1x32x64xf16>, %input_bias_2 : memref<1x4x1x1x32x64xf16>, %hidden_weights_2 : memref<1x4x1x1x32x64xf16>, %hidden_bias_2 : memref<1x4x1x1x32x64xf16>, %work_area_2 : memref<40960xi8>, %shape_2 : memref<5xi64>, %hn_output_2 : memref<1x1x1x1x32x64xf16>, %cf_output_2 : memref<1x1x1x1x32x64xf16>
) -> (memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) {
  "zlow.lstm"(%input_0, %h0_0, %c0_0, %input_weights_0, %input_bias_0, %hidden_weights_0, %hidden_bias_0, %work_area_0, %shape_0, %hn_output_0, %cf_output_0) {direction = "forward", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) -> ()
  "zlow.lstm"(%hn_output_0, %h0_1, %c0_1, %input_weights_1, %input_bias_1, %hidden_weights_1, %hidden_bias_1, %work_area_1, %shape_1, %hn_output_1, %cf_output_1) {direction = "bidirectional", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<81920xi8>, memref<5xi64>, memref<1x3x2x1x32x64xf16>, memref<1x3x2x1x32x64xf16>) -> ()
  "zlow.lstm"(%hn_output_1, %h0_2, %c0_2, %input_weights_2, %input_bias_2, %hidden_weights_2, %hidden_bias_2, %work_area_2, %shape_2, %hn_output_2, %cf_output_2) {direction = "backward", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) -> ()
  return %hn_output_0, %hn_output_1, %hn_output_2 : memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>

    // CHECK-LABEL: test_prev_layer_lstm
    // CHECK: "zlow.lstm"([[input_0_:%.+]], [[h0_0_:%.+]], [[c0_0_:%.+]], [[input_weights_0_:%.+]], [[input_bias_0_:%.+]], [[hidden_weights_0_:%.+]], [[hidden_bias_0_:%.+]], [[work_area_0_:%.+]], [[shape_0_:%.+]], [[hn_output_0_:%.+]], [[cf_output_0_:%.+]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) -> ()
    // CHECK: "zlow.lstm"([[hn_output_0_:%.+]], [[h0_1_:%.+]], [[c0_1_:%.+]], [[input_weights_1_:%.+]], [[input_bias_1_:%.+]], [[hidden_weights_1_:%.+]], [[hidden_bias_1_:%.+]], [[work_area_1_:%.+]], [[shape_1_:%.+]], [[hn_output_1_:%.+]], [[cf_output_1_:%.+]]) {direction = "bidirectional", prev_layer = "uni", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<81920xi8>, memref<5xi64>, memref<1x3x2x1x32x64xf16>, memref<1x3x2x1x32x64xf16>) -> ()
    // CHECK: "zlow.lstm"([[hn_output_1_:%.+]], [[h0_2_:%.+]], [[c0_2_:%.+]], [[input_weights_2_:%.+]], [[input_bias_2_:%.+]], [[hidden_weights_2_:%.+]], [[hidden_bias_2_:%.+]], [[work_area_2_:%.+]], [[shape_2_:%.+]], [[hn_output_2_:%.+]], [[cf_output_2_:%.+]]) {direction = "backward", prev_layer = "bidir", return_all_steps = -1 : si64} : (memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) -> ()
    // CHECK: return [[hn_output_0_:%.+]], [[hn_output_1_:%.+]], [[hn_output_2_:%.+]] : memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>
}

// -----

func @test_prev_layer_gru(
  %input_0 : memref<1x1x1x1x32x64xf16>, %h0_0 : memref<1x1x1x1x32x64xf16>, %input_weights_0 : memref<1x4x1x1x32x64xf16>, %input_bias_0 : memref<1x4x1x1x32x64xf16>, %hidden_weights_0 : memref<1x4x1x1x32x64xf16>, %hidden_bias_0 : memref<1x4x1x1x32x64xf16>, %work_area_0 : memref<40960xi8>, %shape_0 : memref<5xi64>, %hn_output_0 : memref<1x1x1x1x32x64xf16>,
  %input_1 : memref<1x1x1x1x32x64xf16>, %h0_1 : memref<2x1x1x1x32x64xf16>, %input_weights_1 : memref<1x8x2x1x32x64xf16>, %input_bias_1 : memref<1x8x1x1x32x64xf16>, %hidden_weights_1 : memref<1x8x2x1x32x64xf16>, %hidden_bias_1 : memref<1x8x1x1x32x64xf16>, %work_area_1 : memref<81920xi8>, %shape_1 : memref<5xi64>, %hn_output_1 : memref<1x3x2x1x32x64xf16>,
  %input_2 : memref<1x1x1x1x32x64xf16>, %h0_2 : memref<1x1x1x1x32x64xf16>, %input_weights_2 : memref<1x4x1x1x32x64xf16>, %input_bias_2 : memref<1x4x1x1x32x64xf16>, %hidden_weights_2 : memref<1x4x1x1x32x64xf16>, %hidden_bias_2 : memref<1x4x1x1x32x64xf16>, %work_area_2 : memref<40960xi8>, %shape_2 : memref<5xi64>, %hn_output_2 : memref<1x1x1x1x32x64xf16>
) -> (memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>) {
  "zlow.gru"(%input_0, %h0_0, %input_weights_0, %input_bias_0, %hidden_weights_0, %hidden_bias_0, %work_area_0, %shape_0, %hn_output_0) {direction = "forward", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  "zlow.gru"(%hn_output_0, %h0_1, %input_weights_1, %input_bias_1, %hidden_weights_1, %hidden_bias_1, %work_area_1, %shape_1, %hn_output_1) {direction = "bidirectional", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<81920xi8>, memref<5xi64>, memref<1x3x2x1x32x64xf16>) -> ()
  "zlow.gru"(%hn_output_1, %h0_2, %input_weights_2, %input_bias_2, %hidden_weights_2, %hidden_bias_2, %work_area_2, %shape_2, %hn_output_2) {direction = "backward", prev_layer = "not_set", return_all_steps = -1 : si64} : (memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return %hn_output_0, %hn_output_1, %hn_output_2 : memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>

    // CHECK-LABEL: test_prev_layer_gru
    // CHECK: "zlow.gru"([[input_0_:%.+]], [[h0_0_:%.+]], [[input_weights_0_:%.+]], [[input_bias_0_:%.+]], [[hidden_weights_0_:%.+]], [[hidden_bias_0_:%.+]], [[work_area_0_:%.+]], [[shape_0_:%.+]], [[hn_output_0_:%.+]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>) -> ()
    // CHECK: "zlow.gru"([[hn_output_0_:%.+]], [[h0_1_:%.+]], [[input_weights_1_:%.+]], [[input_bias_1_:%.+]], [[hidden_weights_1_:%.+]], [[hidden_bias_1_:%.+]], [[work_area_1_:%.+]], [[shape_1_:%.+]], [[hn_output_1_:%.+]]) {direction = "bidirectional", prev_layer = "uni", return_all_steps = -1 : si64} : (memref<1x1x1x1x32x64xf16>, memref<2x1x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<1x8x2x1x32x64xf16>, memref<1x8x1x1x32x64xf16>, memref<81920xi8>, memref<5xi64>, memref<1x3x2x1x32x64xf16>) -> ()
    // CHECK: "zlow.gru"([[hn_output_1_:%.+]], [[h0_2_:%.+]], [[input_weights_2_:%.+]], [[input_bias_2_:%.+]], [[hidden_weights_2_:%.+]], [[hidden_bias_2_:%.+]], [[work_area_2_:%.+]], [[shape_2_:%.+]], [[hn_output_2_:%.+]]) {direction = "backward", prev_layer = "bidir", return_all_steps = -1 : si64} : (memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<1x4x1x1x32x64xf16>, memref<40960xi8>, memref<5xi64>, memref<1x1x1x1x32x64xf16>) -> ()
    // CHECK: return [[hn_output_0_:%.+]], [[hn_output_1_:%.+]], [[hn_output_2_:%.+]] : memref<1x1x1x1x32x64xf16>, memref<1x3x2x1x32x64xf16>, memref<1x1x1x1x32x64xf16>
}

