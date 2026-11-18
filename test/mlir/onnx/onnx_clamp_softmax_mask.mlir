// RUN: onnx-mlir-opt --clamp-softmax-mask %s -split-input-file | FileCheck %s

// -----

// COM: Global padding mask and sliding-window mask sharing one -FLT_MAX fill
// COM: constant, each added to scores before a Softmax (as in ModernBERT /
// COM: granite-embedding-r2 exports). The fill becomes -1e9.
func.func @test_where_masks(%mask: tensor<2x4xf32>, %win: tensor<1x1x4x4xi1>, %s0: tensor<2x1x4x4xf32>, %s1: tensor<2x1x4x4xf32>) -> (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %shape = onnx.Constant dense<[2, 1, 4, 4]> : tensor<4xi64>
  %m4 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<2x4xf32>, tensor<2xi64>) -> tensor<2x1x1x4xf32>
  %inv = "onnx.Sub"(%one, %m4) : (tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
  %pad = "onnx.Cast"(%inv) {saturate = 1 : si64, to = i1} : (tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xi1>
  %g = "onnx.Where"(%pad, %min, %inv) : (tensor<2x1x1x4xi1>, tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
  %ge = "onnx.Expand"(%g, %shape) : (tensor<2x1x1x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
  %w = "onnx.Where"(%win, %ge, %min) : (tensor<1x1x4x4xi1>, tensor<2x1x4x4xf32>, tensor<f32>) -> tensor<2x1x4x4xf32>
  %a0 = "onnx.Add"(%s0, %ge) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %p0 = "onnx.Softmax"(%a0) {axis = -1 : si64} : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %a1 = "onnx.Add"(%s1, %w) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %p1 = "onnx.Softmax"(%a1) {axis = -1 : si64} : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  return %p0, %p1 : tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>
}
// CHECK-LABEL:  func.func @test_where_masks
// CHECK:          [[MIN:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-NOT:      -3.40282347E+38
// CHECK:          "onnx.Where"({{.*}}, [[MIN]], {{.*}})
// CHECK:          "onnx.Where"({{.*}}, {{.*}}, [[MIN]])
// CHECK-COUNT-2:  "onnx.Softmax"

// -----

// COM: A constant causal mask (with -inf) added to a padding mask: both
// COM: constants are clamped, element by element for the causal one.
func.func @test_added_masks(%pad: tensor<1x1x1x3xi1>, %s: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  %ninf = onnx.Constant dense<0xFF800000> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %causal = onnx.Constant dense<[[[[0.000000e+00, 0xFF800000, 0xFF800000], [0.000000e+00, 0.000000e+00, -3.40282347E+38], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]> : tensor<1x1x3x3xf32>
  %p = "onnx.Where"(%pad, %ninf, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %m = "onnx.Add"(%causal, %p) : (tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x3x3xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %y : tensor<1x1x3x3xf32>
}
// CHECK-LABEL:  func.func @test_added_masks
// CHECK:          [[NINF:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK:          [[CAUSAL:%.+]] = onnx.Constant dense<{{\[}}{{\[}}{{\[}}[0.000000e+00, -1.000000e+09, -1.000000e+09], [0.000000e+00, 0.000000e+00, -1.000000e+09], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]> : tensor<1x1x3x3xf32>
// CHECK:          "onnx.Where"({{.*}}, [[NINF]], {{.*}})
// CHECK:          "onnx.Add"([[CAUSAL]], {{.*}})

// -----

// COM: f16 mask: the fill becomes -1e4 (-1e9 does not fit in f16).
func.func @test_f16(%pad: tensor<1x4xi1>, %s: tensor<3x4xf16>) -> tensor<3x4xf16> {
  %min = onnx.Constant dense<-6.550400e+04> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f16>, tensor<f16>) -> tensor<1x4xf16>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf16>, tensor<1x4xf16>) -> tensor<3x4xf16>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf16>) -> tensor<3x4xf16>
  return %y : tensor<3x4xf16>
}
// CHECK-LABEL:  func.func @test_f16
// CHECK:          [[MIN:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f16>
// CHECK:          "onnx.Where"({{.*}}, [[MIN]], {{.*}})

// -----

// COM: Float attn_mask of onnx.Attention is a sink as well.
func.func @test_attention_mask(%Q: tensor<1x1x2x2xf32>, %K: tensor<1x1x3x2xf32>, %V: tensor<1x1x3x2xf32>, %pad: tensor<1x1x1x3xi1>) -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %Y, %pk, %pv, %qk = "onnx.Attention"(%Q, %K, %V, %m, %none, %none, %none) {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32} : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xf32>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
  return %Y : tensor<1x1x2x2xf32>
}
// CHECK-LABEL:  func.func @test_attention_mask
// CHECK:          [[MIN:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK:          [[MASK:%.+]] = "onnx.Where"({{.*}}, [[MIN]], {{.*}})
// CHECK:          "onnx.Attention"({{.*}}, {{.*}}, {{.*}}, [[MASK]], {{.*}})

// -----

// COM: Negative: the -FLT_MAX constant is also used as a sentinel outside the
// COM: mask (Max with the scores), so it is kept.
func.func @test_shared_sentinel(%pad: tensor<1x4xi1>, %s: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %r = "onnx.Max"(%s, %min) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
  return %y, %r : tensor<3x4xf32>, tensor<3x4xf32>
}
// CHECK-LABEL:  func.func @test_shared_sentinel
// CHECK:          [[MIN:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:      -1.000000e+09
// CHECK:          "onnx.Where"({{.*}}, [[MIN]], {{.*}})
// CHECK:          "onnx.Max"({{.*}}, [[MIN]])

// -----

// COM: Negative: the mask itself is also returned, so it is kept.
func.func @test_mask_escapes(%pad: tensor<1x4xi1>, %s: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<1x4xf32>) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y, %m : tensor<3x4xf32>, tensor<1x4xf32>
}
// CHECK-LABEL:  func.func @test_mask_escapes
// CHECK:          onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:      -1.000000e+09

// -----

// COM: Negative: a mask scaled by Mul ((1 - mask) * -FLT_MAX) is not
// COM: followed, so its constant is kept.
func.func @test_mul_mask(%mask: tensor<1x4xf32>, %s: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %inv = "onnx.Sub"(%one, %mask) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %m = "onnx.Mul"(%inv, %min) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>
}
// CHECK-LABEL:  func.func @test_mul_mask
// CHECK:          onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-NOT:      -1.000000e+09

// -----

// COM: Negative: a value of ordinary magnitude (-1e4) is left unchanged.
func.func @test_small_fill(%pad: tensor<1x4xi1>, %s: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %min = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>
}
// CHECK-LABEL:  func.func @test_small_fill
// CHECK:          onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK-NOT:      -1.000000e+09
