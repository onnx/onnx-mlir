// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

// -----

// Non-broadcasting mask: mask dim 1 is the same as Q dim 1.
func.func @test_attention_non_broadcasting_mask(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x16x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_non_broadcasting_mask
  // CHECK-NOT: "onnx.Attention"
  // CHECK-DAG: "zhigh.Stick"(%{{.*}}) <{layout = "3DS"}>
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{broadcasting_mask = 0 : si64, transposeKT = 1 : si64}>
  // CHECK: "zhigh.Unstick"
  // CHECK: "onnx.Reshape"
  // CHECK: return
}

// -----

// Broadcasting mask: mask dim 1 is 1 while Q dim 1 is 16.
func.func @test_attention_broadcasting_mask(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x1x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x1x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_broadcasting_mask
  // CHECK-NOT: "onnx.Attention"
  // CHECK-DAG: "zhigh.Stick"(%{{.*}}) <{layout = "3DS"}>
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{broadcasting_mask = 1 : si64, transposeKT = 1 : si64}>
  // CHECK: "zhigh.Unstick"
  // CHECK: "onnx.Reshape"
  // CHECK: return
}

// -----

// Static shapes with a broadcasting mask.
func.func @test_attention_static_broadcasting_mask(%q: tensor<2x16x8x64xf32>, %k: tensor<2x16x32x64xf32>, %v: tensor<2x16x32x64xf32>, %mask: tensor<2x1x8x32xf32>) -> tensor<2x16x8x64xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<2x16x8x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x1x8x32xf32>, none, none, none) -> (tensor<2x16x8x64xf32>, none, none, none)
  return %y : tensor<2x16x8x64xf32>

  // CHECK-LABEL: func.func @test_attention_static_broadcasting_mask
  // CHECK-NOT: "onnx.Attention"
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{broadcasting_mask = 1 : si64, transposeKT = 1 : si64}> : (tensor<32x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<32x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<32x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<32x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  // CHECK: "zhigh.Unstick"
  // CHECK: return
}

// -----

// scale is not 1.0, so the op is not lowered.
func.func @test_attention_not_lowered_scale(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x1x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 5.000000e-01 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x1x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_not_lowered_scale
  // CHECK-NOT: zhigh.SDPA3D
  // CHECK: "onnx.Attention"
}

// -----

// is_causal is 1, so the op is not lowered.
func.func @test_attention_not_lowered_causal(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %mask: tensor<?x1x?x?xf32> {onnx.dim_params = "0:B,2:SQ,3:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 1 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x1x?x?xf32>, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_not_lowered_causal
  // CHECK-NOT: zhigh.SDPA3D
  // CHECK: "onnx.Attention"
}

// -----

// There is no mask, so the op is not lowered.
func.func @test_attention_not_lowered_no_mask(%q: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:SQ"}, %k: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}, %v: tensor<?x16x?x128xf32> {onnx.dim_params = "0:B,2:S"}) -> tensor<?x16x?x128xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %none, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, tensor<?x16x?x128xf32>, none, none, none, none) -> (tensor<?x16x?x128xf32>, none, none, none)
  return %y : tensor<?x16x?x128xf32>

  // CHECK-LABEL: func.func @test_attention_not_lowered_no_mask
  // CHECK-NOT: zhigh.SDPA3D
  // CHECK: "onnx.Attention"
}

// -----

// K and V have different shapes, so the op is not lowered.
func.func @test_attention_not_lowered_kv_shape(%q: tensor<2x16x8x64xf32>, %k: tensor<2x16x32x64xf32>, %v: tensor<2x16x32x32xf32>, %mask: tensor<2x1x8x32xf32>) -> tensor<2x16x8x32xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<2x16x8x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x32xf32>, tensor<2x1x8x32xf32>, none, none, none) -> (tensor<2x16x8x32xf32>, none, none, none)
  return %y : tensor<2x16x8x32xf32>

  // CHECK-LABEL: func.func @test_attention_not_lowered_kv_shape
  // CHECK-NOT: zhigh.SDPA3D
  // CHECK: "onnx.Attention"
}
