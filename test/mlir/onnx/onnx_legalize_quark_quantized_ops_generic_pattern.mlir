// RUN: onnx-mlir-opt -legalize-quark-quantized-ops --split-input-file %s | FileCheck %s

func.func @test_gather(%arg0 : tensor<3x3xbf16>, %arg1 : tensor<1x2xi64>) -> tensor<1x2x3xbf16> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<3x3xbf16>) -> tensor<3x3xf32>
    %1 = "onnx.Gather"(%0, %arg1) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
    %3 = "onnx.Cast"(%1) { saturate = 1 : si64, to = bf16} : (tensor<1x2x3xf32>) -> tensor<1x2x3xbf16>
    "onnx.Return"(%3) : (tensor<1x2x3xbf16>) -> ()
}
// CHECK-LABEL:   func.func @test_gather(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<3x3xbf16>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<1x2xi64>) -> tensor<1x2x3xbf16> {
// CHECK:           %[[VAL_2:.*]] = "onnx.Gather"(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : si64} : (tensor<3x3xbf16>, tensor<1x2xi64>) -> tensor<1x2x3xbf16>
// CHECK:           onnx.Return %[[VAL_2]] : tensor<1x2x3xbf16>
// CHECK:         }


// -----

func.func @test_gather2(%arg0 : tensor<3x3xbf16>, %arg1 : tensor<1x2xi64>) -> tensor<1x2x3xf32> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<3x3xbf16>) -> tensor<3x3xf32>
    %1 = "onnx.Gather"(%0, %arg1) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
    %3 = "onnx.Cast"(%1) { saturate = 1 : si64, to = bf16} : (tensor<1x2x3xf32>) -> tensor<1x2x3xbf16>
    %4 = "onnx.Cast"(%3) { saturate = 1 : si64, to = f32} : (tensor<1x2x3xbf16>) -> tensor<1x2x3xf32>
    "onnx.Return"(%4) : (tensor<1x2x3xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_gather2(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<3x3xbf16>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<1x2xi64>) -> tensor<1x2x3xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.Gather"(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : si64} : (tensor<3x3xbf16>, tensor<1x2xi64>) -> tensor<1x2x3xbf16>
// CHECK:           %[[VAL_3:.*]] = "onnx.Cast"(%[[VAL_2]]) {saturate = 1 : si64, to = f32} : (tensor<1x2x3xbf16>) -> tensor<1x2x3xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x3xf32>
// CHECK:         }

// -----

func.func @test_lstm_forward_mode(%arg0: tensor<7x2x3xbf16>, %arg1: tensor<1x16x3xbf16>, %arg2: tensor<1x16x4xbf16>, %arg3: tensor<1x32xbf16>, %arg4: tensor<1x2x4xbf16>, %arg5: tensor<1x2x4xbf16>, %arg6: tensor<1x12xbf16>) -> (tensor<7x1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>) {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<7x2x3xbf16>) -> tensor<7x2x3xf32>
    %2 = "onnx.Cast"(%arg1) {saturate = 1 : si64, to = f32} : (tensor<1x16x3xbf16>) -> tensor<1x16x3xf32>
    %3 = "onnx.Cast"(%arg2) {saturate = 1 : si64, to = f32} : (tensor<1x16x4xbf16>) -> tensor<1x16x4xf32>
    %4 = "onnx.Cast"(%arg3) {saturate = 1 : si64, to = f32} : (tensor<1x32xbf16>) -> tensor<1x32xf32>
    %5 = "onnx.Cast"(%arg4) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
    %6 = "onnx.Cast"(%arg5) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
    %7 = "onnx.Cast"(%arg6) {saturate = 1 : si64, to = f32} : (tensor<1x12xbf16>) -> tensor<1x12xf32>
    %Y, %Y_h, %Y_c = "onnx.LSTM"(%1, %2, %3, %4, %0, %5, %6, %7) {direction = "forward", hidden_size = 4 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (tensor<7x1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>)
    %8 = "onnx.Cast"(%Y) {saturate = 1 : si64, to = bf16} : (tensor<7x1x2x4xf32>) -> tensor<7x1x2x4xbf16>
    %9 = "onnx.Cast"(%Y_h) {saturate = 1 : si64, to = bf16} : (tensor<1x2x4xf32>) -> tensor<1x2x4xbf16>
    %10 = "onnx.Cast"(%Y_c) {saturate = 1 : si64, to = bf16} : (tensor<1x2x4xf32>) -> tensor<1x2x4xbf16>
    %11 = "onnx.Cast"(%8) {saturate = 1 : si64, to = f32} : (tensor<7x1x2x4xbf16>) -> tensor<7x1x2x4xf32>
    %12 = "onnx.Cast"(%9) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
    %13 = "onnx.Cast"(%10) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
    return %11, %12, %13 : tensor<7x1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>
}

// CHECK-LABEL:   func.func @test_lstm_forward_mode
// CHECK:           %[[VAL_7:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %[[VAL_7]], %arg4, %arg5, %arg6) {direction = "forward", hidden_size = 4 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<7x2x3xbf16>, tensor<1x16x3xbf16>, tensor<1x16x4xbf16>, tensor<1x32xbf16>, none, tensor<1x2x4xbf16>, tensor<1x2x4xbf16>, tensor<1x12xbf16>) -> (tensor<7x1x2x4xbf16>, tensor<1x2x4xbf16>, tensor<1x2x4xbf16>)
// CHECK:           %[[VAL_11:.*]] = "onnx.Cast"(%[[VAL_10]]) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<7x1x2x4xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.Cast"(%[[VAL_10]]) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Cast"(%[[VAL_10]]) {saturate = 1 : si64, to = f32} : (tensor<1x2x4xbf16>) -> tensor<1x2x4xf32>
// CHECK:           return %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : tensor<7x1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>
// CHECK:         }

// -----

func.func @onnx_add_static(%arg0: tensor<1x256x128xbf16>, %arg1: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<1x256x128xbf16>) -> tensor<1x256x128xf32>
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<1xbf16>) -> tensor<1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<1x256x128xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<1x256x128xf32>) -> tensor<1x256x128xbf16>
    return %3 : tensor<1x256x128xbf16>
}
// CHECK-LABEL:   func.func @onnx_add_static(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x256x128xbf16>,
// CHECK-SAME:                               %[[VAL_1:.*]]: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
// CHECK:           %[[VAL_2:.*]] = "onnx.Add"(%[[VAL_0]], %[[VAL_1]]) : (tensor<1x256x128xbf16>, tensor<1xbf16>) -> tensor<1x256x128xbf16>
// CHECK:           return %[[VAL_2]] : tensor<1x256x128xbf16>
// CHECK:         }

// -----

func.func @onnx_add_static_with_splat_constant(%arg0: tensor<1x256x128xbf16>, %arg1: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
    %cst = onnx.Constant {value = dense<-1.984375> : tensor<3xbf16>} : tensor<3xbf16>
    %0 = "onnx.Cast"(%cst) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<1xbf16>) -> tensor<1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<3xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<1x256x128xf32>) -> tensor<1x256x128xbf16>
    return %3 : tensor<1x256x128xbf16>
}
// CHECK-LABEL:   func.func @onnx_add_static_with_splat_constant(
// CHECK-SAME:                                             %[[VAL_0:.*]]: tensor<1x256x128xbf16>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<-1.984380e+00> : tensor<3xbf16>
// CHECK:           %[[VAL_3:.*]] = "onnx.Add"(%[[VAL_2]], %[[VAL_1]]) : (tensor<3xbf16>, tensor<1xbf16>) -> tensor<1x256x128xbf16>
// CHECK:           return %[[VAL_3]] : tensor<1x256x128xbf16>
// CHECK:         }

// -----

func.func @onnx_add_static_with_constants(%arg0: tensor<1x256x128xbf16>, %arg1: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
    %cst = onnx.Constant dense<[-8.192000e+03, -1.187500e+00, 1.187500e+00]> : tensor<3xbf16>
    %0 = "onnx.Cast"(%cst) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<1xbf16>) -> tensor<1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<3xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<1x256x128xf32>) -> tensor<1x256x128xbf16>
    return %3 : tensor<1x256x128xbf16>
}
// CHECK-LABEL:   func.func @onnx_add_static_with_constants(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<1x256x128xbf16>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[-8.192000e+03, -1.187500e+00, 1.187500e+00]> : tensor<3xbf16>
// CHECK:           %[[VAL_3:.*]] = "onnx.Add"(%[[VAL_2]], %[[VAL_1]]) : (tensor<3xbf16>, tensor<1xbf16>) -> tensor<1x256x128xbf16>
// CHECK:           return %[[VAL_3]] : tensor<1x256x128xbf16>
// CHECK:         }

// -----

func.func @onnx_add_dynamic(%arg0: tensor<?x?x?xbf16>, %arg1: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<?x?x?xbf16>) -> tensor<?x?x?xf32>
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<?x?x?xbf16>) -> tensor<?x?x?xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<?x?x?xf32>) -> tensor<?x?x?xbf16>
    return %3 : tensor<?x?x?xbf16>
}

// CHECK-LABEL:   func.func @onnx_add_dynamic(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<?x?x?xbf16>,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
// CHECK:           %[[VAL_2:.*]] = "onnx.Add"(%[[VAL_0]], %[[VAL_1]]) : (tensor<?x?x?xbf16>, tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
// CHECK:           return %[[VAL_2]] : tensor<?x?x?xbf16>
// CHECK:         }

// -----

func.func @onnx_add_dynamic_with_splat_constant(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
    %cst = onnx.Constant {value = dense<-1.984375> : tensor<3xf32>} : tensor<3xf32>
    %0 = "onnx.Cast"(%cst) { saturate = 1 : si64, to = bf16} : (tensor<3xf32>) -> tensor<3xbf16>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    %2 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<?x?x?xbf16>) -> tensor<?x?x?xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %4 = "onnx.Cast"(%3) { saturate = 1 : si64, to = bf16} : (tensor<?x?x?xf32>) -> tensor<?x?x?xbf16>
    return %4 : tensor<?x?x?xbf16>
}

// CHECK-LABEL:   func.func @onnx_add_dynamic_with_splat_constant(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<-1.984380e+00> : tensor<3xbf16>
// CHECK:           %[[VAL_3:.*]] = "onnx.Add"(%[[VAL_2]], %[[VAL_0]]) : (tensor<3xbf16>, tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
// CHECK:           return %[[VAL_3]] : tensor<?x?x?xbf16>
// CHECK:         }

// -----

func.func @onnx_add_dynamic_with_constants(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
    %cst = onnx.Constant {value = dense<[-8192.0, -1.1875, 1.1875]> : tensor<3xf32>} : tensor<3xf32>
    %0 = "onnx.Cast"(%cst) { saturate = 1 : si64, to = bf16} : (tensor<3xf32>) -> tensor<3xbf16>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    %2 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<?x?x?xbf16>) -> tensor<?x?x?xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %4 = "onnx.Cast"(%3) { saturate = 1 : si64, to = bf16} : (tensor<?x?x?xf32>) -> tensor<?x?x?xbf16>
    return %4 : tensor<?x?x?xbf16>
}

// CHECK-LABEL:   func.func @onnx_add_dynamic_with_constants(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[-8.192000e+03, -1.187500e+00, 1.187500e+00]> : tensor<3xbf16>
// CHECK:           %[[VAL_3:.*]] = "onnx.Add"(%[[VAL_2]], %[[VAL_0]]) : (tensor<3xbf16>, tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
// CHECK:           return %[[VAL_3]] : tensor<?x?x?xbf16>
// CHECK:         }

// -----

func.func @onnx_add_unknown_no_changes(%arg0: tensor<*xbf16>, %arg1: tensor<*xbf16>) -> tensor<*xbf16> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<*xbf16>) -> tensor<*xf32>
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<*xbf16>) -> tensor<*xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    return %3 : tensor<*xbf16>
}

// CHECK-LABEL:   func.func @onnx_add_unknown_no_changes(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<*xbf16>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<*xbf16>) -> tensor<*xbf16> {
// CHECK:           %[[VAL_2:.*]] = "onnx.Cast"(%[[VAL_0]]) {saturate = 1 : si64, to = f32} : (tensor<*xbf16>) -> tensor<*xf32>
// CHECK:           %[[VAL_3:.*]] = "onnx.Cast"(%[[VAL_1]]) {saturate = 1 : si64, to = f32} : (tensor<*xbf16>) -> tensor<*xf32>
// CHECK:           %[[VAL_4:.*]] = "onnx.Add"(%[[VAL_2]], %[[VAL_3]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_5:.*]] = "onnx.Cast"(%[[VAL_4]]) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
// CHECK:           return %[[VAL_5]] : tensor<*xbf16>
// CHECK:         }

// -----

func.func @LayerNorm_no_conversion_no_input_cast(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x256xf32>) -> (tensor<1x256x256xbf16>) {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg2, %arg1, %arg0) {axis = -1 : si64, epsilon = 9.99999996E-13 : f32, onnx_node_name = "LayerNorm_24", stash_type = 1 : si64} : (tensor<1x256x256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x256xf32>, none, none)
  %0 = "onnx.Cast"(%Y) {saturate = 1 : si64, to = bf16} : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
  onnx.Return %0 : tensor<1x256x256xbf16>
}
// CHECK-LABEL:   func.func @LayerNorm_no_conversion_no_input_cast
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]] = "onnx.LayerNormalization"(%arg2, %arg1, %arg0) {axis = -1 : si64, epsilon = 9.99999996E-13 : f32, onnx_node_name = "LayerNorm_24", stash_type = 1 : si64} : (tensor<1x256x256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x256xf32>, none, none)
// CHECK:           %[[VAL_6:.*]] = "onnx.Cast"(%[[VAL_3]]) {saturate = 1 : si64, to = bf16} : (tensor<1x256x256xf32>) -> tensor<1x256x256xbf16>
// CHECK:           onnx.Return %[[VAL_6]] : tensor<1x256x256xbf16>
// CHECK:         }
