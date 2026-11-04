module {
func.func @decoder_self_attn_step(
    %x      : tensor<?x?x768xf32>,          // [B, S, 768] hidden states of new token(s)
    %past_k : tensor<?x12x?x64xf32>,        // [B, 12, max_seq, 64] static K cache
    %past_v : tensor<?x12x?x64xf32>,        // [B, 12, max_seq, 64] static V cache
    %pos    : tensor<?xi64>)                // [B] write index = #tokens already cached
    -> (tensor<?x?x768xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>) {

  %none = "onnx.NoValue"() {value} : () -> none

  // Weights (splat placeholders for real initializers)
  %wq = onnx.Constant dense<0.01> : tensor<768x768xf32>
  %wk = onnx.Constant dense<0.01> : tensor<768x768xf32>
  %wv = onnx.Constant dense<0.01> : tensor<768x768xf32>
  %wo = onnx.Constant dense<0.01> : tensor<768x768xf32>
  %bq = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bk = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bv = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bo = onnx.Constant dense<0.0>  : tensor<768xf32>

  // 0 = "copy this dim from the input" (allowzero = 0), so B and S stay dynamic
  %split_shape = onnx.Constant dense<[0, 0, 12, 64]> : tensor<4xi64>
  %merge_shape = onnx.Constant dense<[0, 0, 768]>    : tensor<3xi64>

  // ---- Q/K/V projections: [B,S,768] ----
  %q0 = "onnx.MatMul"(%x, %wq) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
  %q1 = "onnx.Add"(%q0, %bq)   : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
  %k0 = "onnx.MatMul"(%x, %wk) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
  %k1 = "onnx.Add"(%k0, %bk)   : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
  %v0 = "onnx.MatMul"(%x, %wv) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
  %v1 = "onnx.Add"(%v0, %bv)   : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

  // ---- Split heads: [B,S,768] -> [B,S,12,64] -> [B,12,S,64] ----
  %q2 = "onnx.Reshape"(%q1, %split_shape) {allowzero = 0 : si64}
        : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  %q  = "onnx.Transpose"(%q2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
  %k2 = "onnx.Reshape"(%k1, %split_shape) {allowzero = 0 : si64}
        : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  %k  = "onnx.Transpose"(%k2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
  %v2 = "onnx.Reshape"(%v1, %split_shape) {allowzero = 0 : si64}
        : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  %v  = "onnx.Transpose"(%v2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

  // ---- KV-cache update: write S new tokens starting at %pos[b] ----
  %present_k = "onnx.TensorScatter"(%past_k, %k, %pos) {axis = -2 : si64, mode = "linear"}
      : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>, tensor<?xi64>) -> tensor<?x12x?x64xf32>
  %present_v = "onnx.TensorScatter"(%past_v, %v, %pos) {axis = -2 : si64, mode = "linear"}
      : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>, tensor<?xi64>) -> tensor<?x12x?x64xf32>

  // Valid KV length per batch = pos + S, with S taken from the input shape
  %seq_len = "onnx.Shape"(%x) {start = 1 : si64, end = 2 : si64}
      : (tensor<?x?x768xf32>) -> tensor<1xi64>
  %kv_len  = "onnx.Add"(%pos, %seq_len) : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>

  // ---- Attention over the full static cache ----
  // inputs: Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %present_k, %present_v, %none, %none, %none, %kv_len)
      {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64,
       scale = 1.250000e-01 : f32, softcap = 0.000000e+00 : f32}
      : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>,
         none, none, none, tensor<?xi64>)
      -> (tensor<?x12x?x64xf32>, none, none, none)

  // ---- Merge heads + output projection ----
  %y1  = "onnx.Transpose"(%y) {perm = [0, 2, 1, 3]}
         : (tensor<?x12x?x64xf32>) -> tensor<?x?x12x64xf32>
  %y2  = "onnx.Reshape"(%y1, %merge_shape) {allowzero = 0 : si64}
         : (tensor<?x?x12x64xf32>, tensor<3xi64>) -> tensor<?x?x768xf32>
  %o0  = "onnx.MatMul"(%y2, %wo) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
  %out = "onnx.Add"(%o0, %bo)    : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

  return %out, %present_k, %present_v
      : tensor<?x?x768xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>
}
"onnx.EntryPoint"() <{func = @decoder_self_attn_step}> : () -> ()
}
