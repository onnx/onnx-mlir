// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Check that a small transformer-style inference block lowers through the
// normal ONNX-to-Krnl pipeline. This intentionally keeps the graph compact while
// covering attention MatMul/Softmax, residual Add, LayerNormalization, and an
// MLP projection.
func.func @test_transformer_block_lowering(
    %hidden: tensor<1x4x8xf32>,
    %w_q: tensor<8x8xf32>, %b_q: tensor<8xf32>,
    %w_k: tensor<8x8xf32>, %b_k: tensor<8xf32>,
    %w_v: tensor<8x8xf32>, %b_v: tensor<8xf32>,
    %w_o: tensor<8x8xf32>, %b_o: tensor<8xf32>,
    %ln1_scale: tensor<8xf32>, %ln1_bias: tensor<8xf32>,
    %w_ff1: tensor<8x16xf32>, %b_ff1: tensor<16xf32>,
    %w_ff2: tensor<16x8xf32>, %b_ff2: tensor<8xf32>,
    %ln2_scale: tensor<8xf32>, %ln2_bias: tensor<8xf32>)
    -> tensor<1x4x8xf32> {
  %q0 = "onnx.MatMul"(%hidden, %w_q) : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %q = "onnx.Add"(%q0, %b_q) : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %k0 = "onnx.MatMul"(%hidden, %w_k) : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %k = "onnx.Add"(%k0, %b_k) : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %v0 = "onnx.MatMul"(%hidden, %w_v) : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %v = "onnx.Add"(%v0, %b_v) : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %k_t = "onnx.Transpose"(%k) {perm = [0, 2, 1]} : (tensor<1x4x8xf32>) -> tensor<1x8x4xf32>
  %scores = "onnx.MatMul"(%q, %k_t) : (tensor<1x4x8xf32>, tensor<1x8x4xf32>) -> tensor<1x4x4xf32>
  %probs = "onnx.Softmax"(%scores) {axis = -1 : si64} : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %context = "onnx.MatMul"(%probs, %v) : (tensor<1x4x4xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %attn0 = "onnx.MatMul"(%context, %w_o) : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %attn = "onnx.Add"(%attn0, %b_o) : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %resid1 = "onnx.Add"(%hidden, %attn) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %ln1, %mean1, %invstd1 = "onnx.LayerNormalization"(%resid1, %ln1_scale, %ln1_bias) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<1x4x8xf32>, none, none)
  %ff1_0 = "onnx.MatMul"(%ln1, %w_ff1) : (tensor<1x4x8xf32>, tensor<8x16xf32>) -> tensor<1x4x16xf32>
  %ff1 = "onnx.Add"(%ff1_0, %b_ff1) : (tensor<1x4x16xf32>, tensor<16xf32>) -> tensor<1x4x16xf32>
  %act = "onnx.Relu"(%ff1) : (tensor<1x4x16xf32>) -> tensor<1x4x16xf32>
  %ff2_0 = "onnx.MatMul"(%act, %w_ff2) : (tensor<1x4x16xf32>, tensor<16x8xf32>) -> tensor<1x4x8xf32>
  %ff2 = "onnx.Add"(%ff2_0, %b_ff2) : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %resid2 = "onnx.Add"(%ln1, %ff2) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %ln2, %mean2, %invstd2 = "onnx.LayerNormalization"(%resid2, %ln2_scale, %ln2_bias) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<1x4x8xf32>, none, none)
  return %ln2 : tensor<1x4x8xf32>

// CHECK-LABEL: func.func @test_transformer_block_lowering
// CHECK:       memref<1x4x8xf32>
// CHECK:       krnl.define_loops
// CHECK:       math.exp
// CHECK:       math.sqrt
// CHECK:       return {{%.+}} : memref<1x4x8xf32>
}
