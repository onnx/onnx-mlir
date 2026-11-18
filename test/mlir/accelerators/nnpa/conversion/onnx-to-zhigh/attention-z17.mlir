// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

// -----

// On z17, K is not transposed explicitly since SDPA3D transposes it.
func.func @test_attention_hw_transpose(%q: tensor<2x16x8x64xf32>, %k: tensor<2x16x32x64xf32>, %v: tensor<2x16x32x64xf32>, %mask: tensor<2x1x8x32xf32>) -> tensor<2x16x8x64xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %pk, %pv, %qk = "onnx.Attention"(%q, %k, %v, %mask, %none, %none, %none) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 1.000000e+00 : f32, softcap = 0.000000e+00 : f32}> {onnx_node_name = "onnx.Attention_0"} : (tensor<2x16x8x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x1x8x32xf32>, none, none, none) -> (tensor<2x16x8x64xf32>, none, none, none)
  return %y : tensor<2x16x8x64xf32>

  // CHECK-LABEL: func.func @test_attention_hw_transpose
  // CHECK-NOT: "onnx.Attention"
  // CHECK-NOT: "onnx.Transpose"
  // CHECK: "zhigh.SDPA3D"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{broadcasting_mask = 1 : si64, transposeKT = 1 : si64}> : (tensor<32x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<32x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x8x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<32x32x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<32x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  // CHECK: "zhigh.Unstick"
  // CHECK: return
}
