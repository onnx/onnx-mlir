// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --disable-convtranspose-decompose %s -split-input-file | FileCheck %s


// -----

// Test unit strides. Only convert weight tensor

  func.func @test_convtrans_unitstrides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
    onnx.Return %1 : tensor<1x2x5x5xf32>
// CHECK-LABEL:   func.func @test_convtrans_unitstrides(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<1x1x3x3xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<1x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x5x5xf32>
// CHECK:         }
  }

// -----

// Test 1d input

  func.func @test_convtrans1d_unitstrides(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
    onnx.Return %1 : tensor<1x2x5xf32>
// CHECK-LABEL:   func.func @test_convtrans1d_unitstrides(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<1x1x3xf32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<1x2x3xf32>) -> tensor<1x2x5xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3xf32>, tensor<1x2x3xf32>, none) -> tensor<1x2x5xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x5xf32>
// CHECK:         }
  }

// -----

// Test 3d input

  func.func @test_convtrans3d_unitstrides(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
    onnx.Return %1 : tensor<1x2x5x6x7xf32>
// CHECK-LABEL:   func.func @test_convtrans3d_unitstrides(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<1x1x3x4x5xf32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<1x2x3x3x3xf32>) -> tensor<1x2x5x6x7xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x3x3xf32>, none) -> tensor<1x2x5x6x7xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x5x6x7xf32>
// CHECK:         }
  }

// -----

// Test non unit strides. Added pads between elements  in input data.

  func.func @test_convtrans_strides(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
    onnx.Return %1 : tensor<1x2x7x3xf32>
// CHECK-LABEL:   func.func @test_convtrans_strides(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x1x3x3xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 2, 1, 2], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x7x3xf32>
// CHECK:         }
  }

// -----

// Test output_padding. Additional pads are inserted after Conv op

  func.func @test_convtrans_outputpadding(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
    onnx.Return %1 : tensor<1x2x10x8xf32>
// CHECK-LABEL:   func.func @test_convtrans_outputpadding(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<1x1x3x3xf32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<1x2x3x3xf32>) -> tensor<1x2x10x8xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64, output_shape = [10, 8], strides = [3, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x10x8xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<1x2x10x8xf32>
// CHECK:         }
  }

// -----

// Test for unknown dimension in spatial dimensions

  func.func @test_convtranspose_unknown_spatial_dim(%arg0: tensor<?x?x3x3xf32>, %arg1: tensor<?x?x3x3xf32>) -> tensor<?x?x10x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "test", output_padding = [1, 1], output_shape = [10, 8], strides = [3, 2]} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x10x8xf32>
    onnx.Return %1 : tensor<?x?x10x8xf32>
// CHECK-LABEL:   func.func @test_convtranspose_unknown_spatial_dim(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<?x?x3x3xf32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: tensor<?x?x3x3xf32>) -> tensor<?x?x10x8xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "onnx.ConvTranspose"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "test", output_padding = [1, 1], output_shape = [10, 8], strides = [3, 2]} : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, none) -> tensor<?x?x10x8xf32>
// CHECK:           onnx.Return %[[VAL_3]] : tensor<?x?x10x8xf32>
// CHECK:         }
  }
