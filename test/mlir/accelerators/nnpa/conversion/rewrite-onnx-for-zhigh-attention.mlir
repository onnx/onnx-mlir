// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --shape-inference --rewrite-onnx-for-zhigh %s -split-input-file | FileCheck %s

// -----

// An attention that can be lowered to zhigh.SDPA3D is kept as is.
func.func @test_attention_kept(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x1x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x1x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_kept
  // CHECK: "onnx.Attention"
  // CHECK-NOT: "onnx.Softmax"
}

// -----

// An attention that cannot be lowered to zhigh.SDPA3D (scale is not 1.0) is
// decomposed into ONNX ops.
func.func @test_attention_decomposed(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x1x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 5.000000e-01 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x1x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_decomposed
  // CHECK-NOT: "onnx.Attention"
  // CHECK: "onnx.Softmax"
}
