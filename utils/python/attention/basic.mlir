module {
func.func @decoder_self_attn_step_concat(
    %x      : tensor<?x?x768xf32>,          // [B, S, 768] hidden states of new token(s)
    %past_k : tensor<?x12x?x64xf32>,        // [B, 12, P, 64] K cache (P = tokens so far)
    %past_v : tensor<?x12x?x64xf32>)        // [B, 12, P, 64] V cache
    -> (tensor<?x?x768xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>) {

  // Weights (splat placeholders for real initializers)
  // Keep the same with dynamic.mlir
  %wq = onnx.Constant dense<0.01> : tensor<768x768xf32>
  %wk = onnx.Constant dense<0.02> : tensor<768x768xf32>
  %wv = onnx.Constant dense<0.03> : tensor<768x768xf32>
  %wo = onnx.Constant dense<0.04> : tensor<768x768xf32>
  %bq = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bk = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bv = onnx.Constant dense<0.0>  : tensor<768xf32>
  %bo = onnx.Constant dense<0.0>  : tensor<768xf32>

  %split_shape = onnx.Constant dense<[0, 0, 12, 64]> : tensor<4xi64>
  %merge_shape = onnx.Constant dense<[0, 0, 768]>    : tensor<3xi64>
  %scale       = onnx.Constant dense<1.250000e-01>   : tensor<f32>      // 1/sqrt(64)
  %neg_inf     = onnx.Constant dense<0xFF800000>     : tensor<f32>      // -inf
  %axis0       = onnx.Constant dense<[0]>            : tensor<1xi64>
  %axis1       = onnx.Constant dense<[1]>            : tensor<1xi64>
  %zero        = onnx.Constant dense<0>              : tensor<i64>
  %one         = onnx.Constant dense<1>              : tensor<i64>

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
  %q3 = "onnx.Transpose"(%q2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
  %k2 = "onnx.Reshape"(%k1, %split_shape) {allowzero = 0 : si64}
        : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  %k  = "onnx.Transpose"(%k2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
  %v2 = "onnx.Reshape"(%v1, %split_shape) {allowzero = 0 : si64}
        : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>
  %v  = "onnx.Transpose"(%v2) {perm = [0, 2, 1, 3]}
        : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

  // ---- KV-cache update: append along the sequence axis -> [B,12,P+S,64] ----
  %present_k = "onnx.Concat"(%past_k, %k) {axis = 2 : si64}
      : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>) -> tensor<?x12x?x64xf32>
  %present_v = "onnx.Concat"(%past_v, %v) {axis = 2 : si64}
      : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>) -> tensor<?x12x?x64xf32>

  // ---- Scores: (Q * scale) @ K^T -> [B,12,S,P+S] ----
  %q    = "onnx.Mul"(%q3, %scale) : (tensor<?x12x?x64xf32>, tensor<f32>) -> tensor<?x12x?x64xf32>
  %kt   = "onnx.Transpose"(%present_k) {perm = [0, 1, 3, 2]}
          : (tensor<?x12x?x64xf32>) -> tensor<?x12x64x?xf32>
  %scores = "onnx.MatMul"(%q, %kt)
          : (tensor<?x12x?x64xf32>, tensor<?x12x64x?xf32>) -> tensor<?x12x?x?xf32>

  // ---- Causal mask: query i (absolute pos P+i) may see key j iff j <= P+i ----
  %P1 = "onnx.Shape"(%past_k) {start = 2 : si64, end = 3 : si64}
        : (tensor<?x12x?x64xf32>) -> tensor<1xi64>
  %S1 = "onnx.Shape"(%x) {start = 1 : si64, end = 2 : si64}
        : (tensor<?x?x768xf32>) -> tensor<1xi64>
  %T1 = "onnx.Add"(%P1, %S1) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %P  = "onnx.Squeeze"(%P1, %axis0) : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
  %T  = "onnx.Squeeze"(%T1, %axis0) : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>

  %key_pos = "onnx.Range"(%zero, %T, %one)                      // [T]  = 0 .. P+S-1
             : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %qry_pos = "onnx.Range"(%P, %T, %one)                         // [S]  = P .. P+S-1
             : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %qry_col = "onnx.Unsqueeze"(%qry_pos, %axis1)                 // [S,1]
             : (tensor<?xi64>, tensor<1xi64>) -> tensor<?x1xi64>
  %allowed = "onnx.LessOrEqual"(%key_pos, %qry_col)             // [S,T]
             : (tensor<?xi64>, tensor<?x1xi64>) -> tensor<?x?xi1>

  %masked = "onnx.Where"(%allowed, %scores, %neg_inf)           // broadcasts to [B,12,S,T]
            : (tensor<?x?xi1>, tensor<?x12x?x?xf32>, tensor<f32>) -> tensor<?x12x?x?xf32>

  // ---- Softmax over keys, then weight V -> [B,12,S,64] ----
  %probs = "onnx.Softmax"(%masked) {axis = -1 : si64}
           : (tensor<?x12x?x?xf32>) -> tensor<?x12x?x?xf32>
  %y     = "onnx.MatMul"(%probs, %present_v)
           : (tensor<?x12x?x?xf32>, tensor<?x12x?x64xf32>) -> tensor<?x12x?x64xf32>

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
"onnx.EntryPoint"() <{func = @decoder_self_attn_step_concat}> : () -> ()
}
