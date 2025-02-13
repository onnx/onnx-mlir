// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @gru_return_single_step(%input : tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = 0 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @gru_return_single_step
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_2_:%.+]]: tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_3_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_4_:%.+]]: tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_5_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<1x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.GRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]) {direction = "forward", hidden_size = 9 : si64, return_all_steps = 0 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<1x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:         }
}

// -----

func.func @gru_return_all_steps(%input : tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @gru_return_all_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_2_:%.+]]: tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_3_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_4_:%.+]]: tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_5_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<3x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.GRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<3x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x1x5x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:         }
}

// -----

// COM: Test unknown timesteps and batch size.
func.func @gru_unknown_dims(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_2_:%.+]]: tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_3_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_4_:%.+]]: tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_5_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.GRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:         }
}

// -----

func.func @gru_no_intial_h(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output = "zhigh.GRU"(%input, %cst, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @gru_no_intial_h
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_2_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_3_:%.+]]: tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_4_:%.+]]: tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.GRU"([[PARAM_0_]], [[VAR_cst_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]]) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:         }
}

// -----

func.func @gru_no_input_and_hidden_biases(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %cst, %hidden_weights, %cst) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @gru_no_input_and_hidden_biases
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_1_:%.+]]: tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[PARAM_2_:%.+]]: tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, [[PARAM_3_:%.+]]: tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.GRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_cst_]], [[PARAM_3_]], [[VAR_cst_]]) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none) -> tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x1x?x9xf16, #zhigh.layout<{dataLayout = "4DS"}>>
// CHECK:         }
}
