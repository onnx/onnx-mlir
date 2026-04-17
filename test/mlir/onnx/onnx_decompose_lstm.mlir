// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// Exercises DecomposeLSTMSeqUnrollPattern: sequence-length unrolling of an
// onnx.LSTM with seq_len>1 into a chain of seq_len=1 onnx.LSTM ops.

// RUN: onnx-mlir-opt --decompose-onnx="enable-lstm-seq-decomposition" %s -split-input-file | FileCheck %s

// -----

// Positive: seq_len=4 LSTM unrolls into 4 seq_len=1 LSTMs + Concat.
func.func @test_lstm_seq_unroll(
    %X: tensor<4x4x3xf32>,
    %W: tensor<1x32x3xf32>,
    %R: tensor<1x32x8xf32>,
    %B: tensor<1x64xf32>,
    %initial_h: tensor<1x4x8xf32>,
    %initial_c: tensor<1x4x8xf32>)
    -> (tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %none, %initial_h, %initial_c, %none)
      {direction = "forward", hidden_size = 8 : si64, layout = 0 : si64}
      : (tensor<4x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>,
         tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none)
      -> (tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %Y, %Y_h, %Y_c : tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>

// CHECK-LABEL:  func.func @test_lstm_seq_unroll
// CHECK-SAME:   (%[[X:.*]]: tensor<4x4x3xf32>, %[[W:.*]]: tensor<1x32x3xf32>, %[[R:.*]]: tensor<1x32x8xf32>, %[[B:.*]]: tensor<1x64xf32>, %[[IH:.*]]: tensor<1x4x8xf32>, %[[IC:.*]]: tensor<1x4x8xf32>)

// Per-timestep Slice range constants and the NoValue sentinel used for the
// (omitted) sequence_lens / peephole inputs of each unrolled LSTM.
// CHECK-DAG:    %[[C4:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:    %[[C3:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:    %[[C2:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:    %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:    %[[C0:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:    %[[C1:.*]] = onnx.Constant dense<1> : tensor<1xi64>

// Timestep 0: X[0:1] -> LSTM with the original initial_h / initial_c.
// CHECK:        %[[S0:.*]] = "onnx.Slice"(%[[X]], %[[C0]], %[[C1]], %[[C0]], %[[C1]]) : (tensor<4x4x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x3xf32>
// CHECK:        %[[Y0:[^,]+]], %[[YH0:[^,]+]], %[[YC0:[^ ]+]] = "onnx.LSTM"(%[[S0]], %[[W]], %[[R]], %[[B]], %[[NONE]], %[[IH]], %[[IC]], %[[NONE]]) {direction = "forward", hidden_size = 8 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<1x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>, tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none) -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)

// Timestep 1: X[1:2] -> LSTM chained from (YH0, YC0).
// CHECK:        %[[S1:.*]] = "onnx.Slice"(%[[X]], %[[C1]], %[[C2]], %[[C0]], %[[C1]]) : (tensor<4x4x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x3xf32>
// CHECK:        %[[Y1:[^,]+]], %[[YH1:[^,]+]], %[[YC1:[^ ]+]] = "onnx.LSTM"(%[[S1]], %[[W]], %[[R]], %[[B]], %[[NONE]], %[[YH0]], %[[YC0]], %[[NONE]]) {direction = "forward", hidden_size = 8 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<1x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>, tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none) -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)

// Timestep 2: X[2:3] -> LSTM chained from (YH1, YC1).
// CHECK:        %[[S2:.*]] = "onnx.Slice"(%[[X]], %[[C2]], %[[C3]], %[[C0]], %[[C1]]) : (tensor<4x4x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x3xf32>
// CHECK:        %[[Y2:[^,]+]], %[[YH2:[^,]+]], %[[YC2:[^ ]+]] = "onnx.LSTM"(%[[S2]], %[[W]], %[[R]], %[[B]], %[[NONE]], %[[YH1]], %[[YC1]], %[[NONE]]) {direction = "forward", hidden_size = 8 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<1x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>, tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none) -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)

// Timestep 3: X[3:4] -> LSTM chained from (YH2, YC2); its YH3/YC3 feed the return.
// CHECK:        %[[S3:.*]] = "onnx.Slice"(%[[X]], %[[C3]], %[[C4]], %[[C0]], %[[C1]]) : (tensor<4x4x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x3xf32>
// CHECK:        %[[Y3:[^,]+]], %[[YH3:[^,]+]], %[[YC3:[^ ]+]] = "onnx.LSTM"(%[[S3]], %[[W]], %[[R]], %[[B]], %[[NONE]], %[[YH2]], %[[YC2]], %[[NONE]]) {direction = "forward", hidden_size = 8 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<1x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>, tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none) -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)

// Concat the per-timestep Y's along the time axis; YH / YC come from the last step.
// CHECK:        %[[YCAT:.*]] = "onnx.Concat"(%[[Y0]], %[[Y1]], %[[Y2]], %[[Y3]]) {axis = 0 : si64} : (tensor<1x1x4x8xf32>, tensor<1x1x4x8xf32>, tensor<1x1x4x8xf32>, tensor<1x1x4x8xf32>) -> tensor<4x1x4x8xf32>
// CHECK:        return %[[YCAT]], %[[YH3]], %[[YC3]] : tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// -----

// Rejection 1: seq_len=1 -> pattern must not fire (no Slice/Concat inserted).
func.func @test_lstm_seq_unroll_seq1(
    %X: tensor<1x4x3xf32>,
    %W: tensor<1x32x3xf32>,
    %R: tensor<1x32x8xf32>,
    %B: tensor<1x64xf32>,
    %initial_h: tensor<1x4x8xf32>,
    %initial_c: tensor<1x4x8xf32>)
    -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %none, %initial_h, %initial_c, %none)
      {direction = "forward", hidden_size = 8 : si64, layout = 0 : si64}
      : (tensor<1x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>,
         tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none)
      -> (tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %Y, %Y_h, %Y_c : tensor<1x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>

// CHECK-LABEL:  func.func @test_lstm_seq_unroll_seq1
// CHECK:        "onnx.LSTM"
// CHECK-NOT:    "onnx.Slice"
// CHECK-NOT:    "onnx.Concat"
}

// -----

// Rejection 2: dynamic seq_len -> pattern must not fire.
func.func @test_lstm_seq_unroll_dynamic_seq(
    %X: tensor<?x4x3xf32>,
    %W: tensor<1x32x3xf32>,
    %R: tensor<1x32x8xf32>,
    %B: tensor<1x64xf32>,
    %initial_h: tensor<1x4x8xf32>,
    %initial_c: tensor<1x4x8xf32>)
    -> (tensor<?x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %none, %initial_h, %initial_c, %none)
      {direction = "forward", hidden_size = 8 : si64, layout = 0 : si64}
      : (tensor<?x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>,
         tensor<1x64xf32>, none, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none)
      -> (tensor<?x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %Y, %Y_h, %Y_c : tensor<?x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>

// CHECK-LABEL:  func.func @test_lstm_seq_unroll_dynamic_seq
// CHECK:        "onnx.LSTM"
// CHECK-NOT:    "onnx.Slice"
// CHECK-NOT:    "onnx.Concat"
}

// -----

// Rejection 3: sequence_lens input provided -> pattern must not fire.
func.func @test_lstm_seq_unroll_with_seqlens(
    %X: tensor<4x4x3xf32>,
    %W: tensor<1x32x3xf32>,
    %R: tensor<1x32x8xf32>,
    %B: tensor<1x64xf32>,
    %seq_lens: tensor<4xi32>,
    %initial_h: tensor<1x4x8xf32>,
    %initial_c: tensor<1x4x8xf32>)
    -> (tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %seq_lens, %initial_h, %initial_c, %none)
      {direction = "forward", hidden_size = 8 : si64, layout = 0 : si64}
      : (tensor<4x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>,
         tensor<1x64xf32>, tensor<4xi32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>, none)
      -> (tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %Y, %Y_h, %Y_c : tensor<4x1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>

// CHECK-LABEL:  func.func @test_lstm_seq_unroll_with_seqlens
// CHECK:        "onnx.LSTM"
// CHECK-NOT:    "onnx.Slice"
// CHECK-NOT:    "onnx.Concat"
}

// -----

// Rejection 4: layout=1 -> pattern must not fire (dim 0 is batch, not time).
func.func @test_lstm_seq_unroll_layout1(
    %X: tensor<4x4x3xf32>,
    %W: tensor<1x32x3xf32>,
    %R: tensor<1x32x8xf32>,
    %B: tensor<1x64xf32>,
    %initial_h: tensor<4x1x8xf32>,
    %initial_c: tensor<4x1x8xf32>)
    -> (tensor<4x4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%X, %W, %R, %B, %none, %initial_h, %initial_c, %none)
      {direction = "forward", hidden_size = 8 : si64, layout = 1 : si64}
      : (tensor<4x4x3xf32>, tensor<1x32x3xf32>, tensor<1x32x8xf32>,
         tensor<1x64xf32>, none, tensor<4x1x8xf32>, tensor<4x1x8xf32>, none)
      -> (tensor<4x4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>)
  return %Y, %Y_h, %Y_c : tensor<4x4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>

// CHECK-LABEL:  func.func @test_lstm_seq_unroll_layout1
// CHECK:        "onnx.LSTM"
// CHECK-NOT:    "onnx.Slice"
// CHECK-NOT:    "onnx.Concat"
}
