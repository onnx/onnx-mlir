// RUN: onnx-mlir-opt --decompose-onnx="enable-groupqueryattention-decompose=true enable-groupqueryattention-cache-slicing=false" %s -split-input-file | FileCheck %s

// When enable-groupqueryattention-cache-slicing is false, the cos/sin caches
// must be passed through to onnx.RotaryEmbedding without any Slice/Reshape/
// Expand, and the position_ids that the slicing would have implicitly
// selected must be materialized as a constant tensor of shape
// [batch_size, sequence_length] with values [past_seq_len .. total_seq_len-1]
// per batch row.

func.func @gqa_unsliced_cache_packed(
  %qkv: tensor<1x1x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x1x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}
// CHECK-LABEL:   func.func @gqa_unsliced_cache_packed(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<1x1x6144xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<1x16x256x96xf32>, %[[VAL_2:.*]]: tensor<1x16x256x96xf32>,
// CHECK-SAME:                                          %[[COS_CACHE:.*]]: tensor<4096x48xf32>,
// CHECK-SAME:                                          %[[SIN_CACHE:.*]]: tensor<4096x48xf32>)
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Reshape"
// CHECK-NOT: "onnx.Expand"
// CHECK-DAG:       %[[POS_IDS:.*]] = onnx.Constant dense<256> : tensor<1x1xi64>
// CHECK:           %[[VAL_8:.*]] = "onnx.RotaryEmbedding"({{.*}}, %[[COS_CACHE]], %[[SIN_CACHE]], %[[POS_IDS]])
// CHECK-SAME:          : (tensor<1x1x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x1xi64>) -> tensor<1x1x3072xf32>
// CHECK:           %[[VAL_9:.*]] = "onnx.RotaryEmbedding"({{.*}}, %[[COS_CACHE]], %[[SIN_CACHE]], %[[POS_IDS]])
// CHECK-SAME:          : (tensor<1x1x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x1xi64>) -> tensor<1x1x1536xf32>
// CHECK:           "onnx.Attention"(%[[VAL_8]], %[[VAL_9]],

// -----

// Multi-step prefill (seq_len > 1) with batch > 1 and a non-zero past length:
// the synthesized position_ids must be a 2D constant tensor with one row per
// batch containing [past_seq_len .. total_seq_len-1].

func.func @gqa_unsliced_cache_prefill_batched(
  %q: tensor<2x4x3072xf32>,
  %k: tensor<2x4x1536xf32>,
  %v: tensor<2x4x1536xf32>,
  %past_k: tensor<2x16x8x96xf32>,
  %past_v: tensor<2x16x8x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<11> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<2x4x3072xf32>, tensor<2x4x1536xf32>, tensor<2x4x1536xf32>, tensor<2x16x8x96xf32>, tensor<2x16x8x96xf32>, tensor<2x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>)
  return %out, %present_k, %present_v : tensor<2x4x3072xf32>, tensor<2x16x12x96xf32>, tensor<2x16x12x96xf32>
}
// CHECK-LABEL:   func.func @gqa_unsliced_cache_prefill_batched
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Reshape"
// CHECK-NOT: "onnx.Expand"
// CHECK-DAG:       %[[POS_IDS:.*]] = onnx.Constant dense<{{\[}}[8, 9, 10, 11], [8, 9, 10, 11]]> : tensor<2x4xi64>
// CHECK:           "onnx.RotaryEmbedding"({{.*}}, %[[POS_IDS]])
// CHECK-SAME:          : (tensor<2x4x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<2x4xi64>) -> tensor<2x4x3072xf32>
// CHECK:           "onnx.RotaryEmbedding"({{.*}}, %[[POS_IDS]])
// CHECK-SAME:          : (tensor<2x4x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<2x4xi64>) -> tensor<2x4x1536xf32>
// CHECK:           "onnx.Attention"

// -----

// When position_ids is provided by the caller, the synthesis path is not
// taken even with the flag set: the existing position_ids flows through
// directly and no Slice/Reshape/Expand is added.

func.func @gqa_unsliced_cache_with_user_position_ids(
  %q: tensor<1x4x3072xf32>,
  %k: tensor<1x4x1536xf32>,
  %v: tensor<1x4x1536xf32>,
  %past_k: tensor<1x16x8x96xf32>,
  %past_v: tensor<1x16x8x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>,
  %pos_ids: tensor<1x4xi64>
) -> (tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<11> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache, %pos_ids) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x4x3072xf32>, tensor<1x4x1536xf32>, tensor<1x4x1536xf32>, tensor<1x16x8x96xf32>, tensor<1x16x8x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x4xi64>) -> (tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>)
  return %out, %present_k, %present_v : tensor<1x4x3072xf32>, tensor<1x16x12x96xf32>, tensor<1x16x12x96xf32>
}
// CHECK-LABEL:   func.func @gqa_unsliced_cache_with_user_position_ids(
// CHECK-SAME:                                                          %[[Q:.*]]: tensor<1x4x3072xf32>,
// CHECK-SAME:                                                          %[[K:.*]]: tensor<1x4x1536xf32>, %[[V:.*]]: tensor<1x4x1536xf32>,
// CHECK-SAME:                                                          %[[PAST_K:.*]]: tensor<1x16x8x96xf32>, %[[PAST_V:.*]]: tensor<1x16x8x96xf32>,
// CHECK-SAME:                                                          %[[COS_CACHE:.*]]: tensor<4096x48xf32>, %[[SIN_CACHE:.*]]: tensor<4096x48xf32>,
// CHECK-SAME:                                                          %[[POS_IDS:.*]]: tensor<1x4xi64>)
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Reshape"
// CHECK-NOT: "onnx.Expand"
// CHECK:           %[[VAL_Q:.*]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COS_CACHE]], %[[SIN_CACHE]], %[[POS_IDS]])
// CHECK:           %[[VAL_K:.*]] = "onnx.RotaryEmbedding"(%[[K]], %[[COS_CACHE]], %[[SIN_CACHE]], %[[POS_IDS]])
// CHECK:           "onnx.Attention"(%[[VAL_Q]], %[[VAL_K]],
