// COM: Baseline for rewrite-onnx-for-zhigh-clamp-mask.mlir (file-mode grounding with
// COM: utils/GroundLitTest.py): each function is the expected output of
// COM: the mask clamping of --rewrite-onnx-for-zhigh, i.e. the test function with its mask constants
// COM: below -1e9 (-1e4 for f16) set to that value, and nothing else changed.
// COM: Not a lit test (lit.cfg.py excludes .baseline files); no RUN line.
// -----

// CHECK-LABEL: func.func @test_where_masks
func.func @test_where_masks(%mask: tensor<2x4xf32> {onnx.name = "mask"}, %win: tensor<1x1x4x4xi1> {onnx.name = "win"}, %s0: tensor<2x1x4x4xf32> {onnx.name = "s0"}, %s1: tensor<2x1x4x4xf32> {onnx.name = "s1"}) -> (tensor<2x1x4x4xf32> {onnx.name = "p0"}, tensor<2x1x4x4xf32> {onnx.name = "p1"}) {
  %min = onnx.Constant dense<-1.000000e+09> : tensor<f32>
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
// -----

// CHECK-LABEL: func.func @test_where_masks_dyn
func.func @test_where_masks_dyn(%mask: tensor<?x?xf32> {onnx.name = "mask"}, %win: tensor<1x1x?x?xi1> {onnx.name = "win"}, %s0: tensor<?x1x?x?xf32> {onnx.name = "s0"}, %s1: tensor<?x1x?x?xf32> {onnx.name = "s1"}) -> (tensor<?x1x?x?xf32> {onnx.name = "p0"}, tensor<?x1x?x?xf32> {onnx.name = "p1"}) {
  %min = onnx.Constant dense<-1.000000e+09> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %shape = "onnx.Shape"(%s0) : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
  %m4 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x1x1x?xf32>
  %inv = "onnx.Sub"(%one, %m4) : (tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
  %pad = "onnx.Cast"(%inv) {saturate = 1 : si64, to = i1} : (tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xi1>
  %g = "onnx.Where"(%pad, %min, %inv) : (tensor<?x1x1x?xi1>, tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
  %ge = "onnx.Expand"(%g, %shape) : (tensor<?x1x1x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
  %w = "onnx.Where"(%win, %ge, %min) : (tensor<1x1x?x?xi1>, tensor<?x1x?x?xf32>, tensor<f32>) -> tensor<?x1x?x?xf32>
  %a0 = "onnx.Add"(%s0, %ge) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %p0 = "onnx.Softmax"(%a0) {axis = -1 : si64} : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %a1 = "onnx.Add"(%s1, %w) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %p1 = "onnx.Softmax"(%a1) {axis = -1 : si64} : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  return %p0, %p1 : tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>
}
// -----

// CHECK-LABEL: func.func @test_where_mask_mixed
func.func @test_where_mask_mixed(%mask: tensor<2x?xf32> {onnx.name = "mask"}, %s: tensor<2x3x?xf32> {onnx.name = "s"}) -> (tensor<2x3x?xf32> {onnx.name = "p"}) {
  %min = onnx.Constant dense<-1.000000e+09> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %m3 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<2x1x?xf32>
  %pad = "onnx.Cast"(%m3) {saturate = 1 : si64, to = i1} : (tensor<2x1x?xf32>) -> tensor<2x1x?xi1>
  %m = "onnx.Where"(%pad, %zero, %min) : (tensor<2x1x?xi1>, tensor<f32>, tensor<f32>) -> tensor<2x1x?xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<2x3x?xf32>, tensor<2x1x?xf32>) -> tensor<2x3x?xf32>
  %p = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  return %p : tensor<2x3x?xf32>
}
// -----

// CHECK-LABEL: func.func @test_added_masks
func.func @test_added_masks(%pad: tensor<1x1x1x3xi1> {onnx.name = "pad"}, %s: tensor<1x1x3x3xf32> {onnx.name = "s"}) -> (tensor<1x1x3x3xf32> {onnx.name = "y"}) {
  %ninf = onnx.Constant dense<-1.000000e+09> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %causal = onnx.Constant dense<[[[[0.000000e+00, -1.000000e+09, -1.000000e+09], [0.000000e+00, 0.000000e+00, -1.000000e+09], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]> : tensor<1x1x3x3xf32>
  %p = "onnx.Where"(%pad, %ninf, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %m = "onnx.Add"(%causal, %p) : (tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x3x3xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %y : tensor<1x1x3x3xf32>
}
// -----

// CHECK-LABEL: func.func @test_f16
func.func @test_f16(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf16> {onnx.name = "s"}) -> (tensor<3x4xf16> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-1.000000e+04> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f16>, tensor<f16>) -> tensor<1x4xf16>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf16>, tensor<1x4xf16>) -> tensor<3x4xf16>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf16>) -> tensor<3x4xf16>
  return %y : tensor<3x4xf16>
}
// -----

// CHECK-LABEL: func.func @test_attention_mask
func.func @test_attention_mask(%Q: tensor<1x1x2x2xf32> {onnx.name = "q"}, %K: tensor<1x1x3x2xf32> {onnx.name = "k"}, %V: tensor<1x1x3x2xf32> {onnx.name = "v"}, %pad: tensor<1x1x1x3xi1> {onnx.name = "pad"}) -> (tensor<1x1x2x2xf32> {onnx.name = "y"}) {
  %none = "onnx.NoValue"() {value} : () -> none
  %min = onnx.Constant dense<-1.000000e+09> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %Y, %pk, %pv, %qk = "onnx.Attention"(%Q, %K, %V, %m, %none, %none, %none) {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32} : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xf32>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
  return %Y : tensor<1x1x2x2xf32>
}
