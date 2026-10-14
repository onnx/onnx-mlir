// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Attention with constant shapes (3D input).
//===----------------------------------------------------------------------===//

// COM: Basic 3D attention with all static shapes, no past KV cache.
func.func @test_attention_static_3d(%Q: tensor<2x10x64xf32>, %K: tensor<2x20x64xf32>, %V: tensor<2x20x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<2x10x64xf32>, tensor<2x20x64xf32>, tensor<2x20x64xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_static_3d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x10x64xf32>, none, none, none)
}

// -----

// COM: 3D attention with past KV cache, static shapes.
func.func @test_attention_static_3d_past_kv(%Q: tensor<2x1x64xf32>, %K: tensor<2x1x64xf32>, %V: tensor<2x1x64xf32>, %past_key: tensor<2x8x20x8xf32>, %past_value: tensor<2x8x20x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %past_key, %past_value, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<2x1x64xf32>, tensor<2x1x64xf32>, tensor<2x1x64xf32>, none, tensor<2x8x20x8xf32>, tensor<2x8x20x8xf32>, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, none)
  onnx.Return %Y, %present_key, %present_value : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

  // CHECK-LABEL: test_attention_static_3d_past_kv
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x1x64xf32>, tensor<2x8x21x8xf32>, tensor<2x8x21x8xf32>, none)
}

// -----

// COM: 4D attention with static shapes, no past KV cache.
func.func @test_attention_static_4d(%Q: tensor<2x8x10x8xf32>, %K: tensor<2x8x20x8xf32>, %V: tensor<2x8x20x8xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) : (tensor<2x8x10x8xf32>, tensor<2x8x20x8xf32>, tensor<2x8x20x8xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_static_4d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x8x10x8xf32>, none, none, none)
}

// -----

// COM: 3D attention with qk_matmul_output_mode = 1, static shapes.
func.func @test_attention_static_qk_output(%Q: tensor<2x10x64xf32>, %K: tensor<2x20x64xf32>, %V: tensor<2x20x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64, qk_matmul_output_mode = 1 : si64} : (tensor<2x10x64xf32>, tensor<2x20x64xf32>, tensor<2x20x64xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, tensor<*xf32>)
  onnx.Return %Y, %qk_out : tensor<*xf32>, tensor<*xf32>

  // CHECK-LABEL: test_attention_static_qk_output
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x10x64xf32>, none, none, tensor<2x8x10x20xf32>)
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Attention with mixed static/dynamic shapes.
//===----------------------------------------------------------------------===//

// COM: 3D attention with dynamic batch, static sequence lengths.
func.func @test_attention_dynamic_batch_static_seq_3d(%Q: tensor<?x10x64xf32>, %K: tensor<?x20x64xf32>, %V: tensor<?x20x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<?x10x64xf32>, tensor<?x20x64xf32>, tensor<?x20x64xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_batch_static_seq_3d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x10x64xf32>, none, none, none)
}

// -----

// COM: 3D attention with static batch, dynamic sequence lengths.
func.func @test_attention_static_batch_dynamic_seq_3d(%Q: tensor<2x?x64xf32>, %K: tensor<2x?x64xf32>, %V: tensor<2x?x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<2x?x64xf32>, tensor<2x?x64xf32>, tensor<2x?x64xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_static_batch_dynamic_seq_3d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x?x64xf32>, none, none, none)
}

// -----

// COM: 3D attention with dynamic batch, static sequence, past KV cache.
func.func @test_attention_dynamic_batch_static_seq_past_kv(%Q: tensor<?x1x64xf32>, %K: tensor<?x1x64xf32>, %V: tensor<?x1x64xf32>, %past_key: tensor<?x8x20x8xf32>, %past_value: tensor<?x8x20x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %past_key, %past_value, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<?x1x64xf32>, tensor<?x1x64xf32>, tensor<?x1x64xf32>, none, tensor<?x8x20x8xf32>, tensor<?x8x20x8xf32>, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, none)
  onnx.Return %Y, %present_key, %present_value : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_batch_static_seq_past_kv
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x1x64xf32>, tensor<?x8x21x8xf32>, tensor<?x8x21x8xf32>, none)
}

// -----

// COM: 3D attention with static batch, dynamic sequence, past KV cache.
func.func @test_attention_static_batch_dynamic_seq_past_kv(%Q: tensor<2x?x64xf32>, %K: tensor<2x?x64xf32>, %V: tensor<2x?x64xf32>, %past_key: tensor<2x8x?x8xf32>, %past_value: tensor<2x8x?x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %past_key, %past_value, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<2x?x64xf32>, tensor<2x?x64xf32>, tensor<2x?x64xf32>, none, tensor<2x8x?x8xf32>, tensor<2x8x?x8xf32>, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, none)
  onnx.Return %Y, %present_key, %present_value : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

  // CHECK-LABEL: test_attention_static_batch_dynamic_seq_past_kv
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x?x64xf32>, tensor<2x8x?x8xf32>, tensor<2x8x?x8xf32>, none)
}

// -----

// COM: 4D attention with dynamic batch, static sequence lengths.
func.func @test_attention_dynamic_batch_static_seq_4d(%Q: tensor<?x8x10x8xf32>, %K: tensor<?x8x20x8xf32>, %V: tensor<?x8x20x8xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) : (tensor<?x8x10x8xf32>, tensor<?x8x20x8xf32>, tensor<?x8x20x8xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_batch_static_seq_4d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x8x10x8xf32>, none, none, none)
}

// -----

// COM: 4D attention with static batch, dynamic sequence lengths.
func.func @test_attention_static_batch_dynamic_seq_4d(%Q: tensor<2x8x?x8xf32>, %K: tensor<2x8x?x8xf32>, %V: tensor<2x8x?x8xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) : (tensor<2x8x?x8xf32>, tensor<2x8x?x8xf32>, tensor<2x8x?x8xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_static_batch_dynamic_seq_4d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<2x8x?x8xf32>, none, none, none)
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Attention with fully dynamic shapes.
//===----------------------------------------------------------------------===//

// COM: 3D attention with dynamic batch and sequence lengths.
func.func @test_attention_dynamic_3d(%Q: tensor<?x?x64xf32>, %K: tensor<?x?x64xf32>, %V: tensor<?x?x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<?x?x64xf32>, tensor<?x?x64xf32>, tensor<?x?x64xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_3d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x?x64xf32>, none, none, none)
}

// -----

// COM: 3D attention with dynamic shapes and past KV cache.
func.func @test_attention_dynamic_3d_past_kv(%Q: tensor<?x?x64xf32>, %K: tensor<?x?x64xf32>, %V: tensor<?x?x64xf32>, %past_key: tensor<?x8x?x8xf32>, %past_value: tensor<?x8x?x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %past_key, %past_value, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<?x?x64xf32>, tensor<?x?x64xf32>, tensor<?x?x64xf32>, none, tensor<?x8x?x8xf32>, tensor<?x8x?x8xf32>, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, none)
  onnx.Return %Y, %present_key, %present_value : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_3d_past_kv
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x?x64xf32>, tensor<?x8x?x8xf32>, tensor<?x8x?x8xf32>, none)
}

// -----

// COM: 4D attention with dynamic shapes.
func.func @test_attention_dynamic_4d(%Q: tensor<?x8x?x8xf32>, %K: tensor<?x8x?x8xf32>, %V: tensor<?x8x?x8xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) : (tensor<?x8x?x8xf32>, tensor<?x8x?x8xf32>, tensor<?x8x?x8xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_dynamic_4d
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x8x?x8xf32>, none, none, none)
}

// -----

// COM: Fully dynamic 3D attention (all dimensions unknown).
func.func @test_attention_fully_dynamic(%Q: tensor<?x?x?xf32>, %K: tensor<?x?x?xf32>, %V: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %present_key, %present_value, %qk_out = "onnx.Attention"(%Q, %K, %V, %none, %none, %none, %none) {q_num_heads = 8 : si64, kv_num_heads = 8 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none, none) -> (tensor<*xf32>, none, none, none)
  onnx.Return %Y : tensor<*xf32>

  // CHECK-LABEL: test_attention_fully_dynamic
  // CHECK: [[Y:%.+]], [[PK:%.+]], [[PV:%.+]], [[QK:%.+]] = "onnx.Attention"
  // CHECK-SAME: -> (tensor<?x?x?xf32>, none, none, none)
}
