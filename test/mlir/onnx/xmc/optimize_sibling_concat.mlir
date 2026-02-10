// RUN: onnx-mlir-opt --split-input-file --optimize-sibling-concat %s | FileCheck %s

// -----

// Two sibling concats share %arg0 and use it at the same index (0):
//   c0 = Concat(%arg0, %arg1)
//   c1 = Concat(%arg0, %arg2)
// In this MLIR pipeline we assume NCHW, so "channel concat" is axis=1.
//
// When the pass picks one concat as the target, it rewrites it into:
//   swapped = Concat(%other, %shared)
//   slice_tail = Slice(swapped, begin=[0,sz_other,0,0], end=[...])  (-> %shared part)
//   slice_head = Slice(swapped, begin=[0,0,0,0], end=[...,sz_other]) (-> %other part)
//   reconcat = Concat(slice_tail, slice_head)  (same type as original)
//
// Note: the target selection requires an "elimination opportunity"
// (Concat->InstanceNormalization->Conv). We construct that consumer chain
// for %c0 so the pass rewrites %c0.

func.func @test_optimize_sibling_concat(%arg0: tensor<1x12x3x4xf32>,
                                       %arg1: tensor<1x12x3x4xf32>,
                                       %arg2: tensor<1x12x3x4xf32>,
                                       %scale: tensor<24xf32>,
                                       %bias: tensor<24xf32>,
                                       %w: tensor<8x24x1x1xf32>) -> (tensor<1x8x3x4xf32>, tensor<1x24x3x4xf32>) {
  %c0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
      : (tensor<1x12x3x4xf32>, tensor<1x12x3x4xf32>) -> tensor<1x24x3x4xf32>
  %c1 = "onnx.Concat"(%arg0, %arg2) {axis = 1 : si64}
      : (tensor<1x12x3x4xf32>, tensor<1x12x3x4xf32>) -> tensor<1x24x3x4xf32>

  // Consumer chain to trigger target selection on %c0.
  %in = "onnx.InstanceNormalization"(%c0, %scale, %bias) {epsilon = 1.0e-5 : f32}
      : (tensor<1x24x3x4xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x24x3x4xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %out = "onnx.Conv"(%in, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                  kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
      : (tensor<1x24x3x4xf32>, tensor<8x24x1x1xf32>, none) -> tensor<1x8x3x4xf32>

  // Keep %c1 live so the sibling-concat pattern matches deterministically.
  return %out, %c1 : tensor<1x8x3x4xf32>, tensor<1x24x3x4xf32>
}

// CHECK-LABEL: func.func @test_optimize_sibling_concat
// NOTE: MLIR may print `onnx.Constant` without quotes and may compress splats.
// CHECK-DAG: %[[B1:.*]] = onnx.Constant dense<[0, 12, 0, 0]> : tensor<4xi64>
// CHECK-DAG: %[[E1:.*]] = onnx.Constant dense<[1, 24, 3, 4]> : tensor<4xi64>
// CHECK-DAG: %[[AX:.*]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK-DAG: %[[ST:.*]] = onnx.Constant dense<1> : tensor<4xi64>
// CHECK-DAG: %[[B2:.*]] = onnx.Constant dense<0> : tensor<4xi64>
// CHECK-DAG: %[[E2:.*]] = onnx.Constant dense<[1, 12, 3, 4]> : tensor<4xi64>
// CHECK: %[[SWAP:.*]] = "onnx.Concat"(%arg1, %arg0) {axis = 1 : si64}
// CHECK: %[[S1:.*]] = "onnx.Slice"(%[[SWAP]], %[[B1]], %[[E1]], %[[AX]], %[[ST]])
// CHECK: %[[S2:.*]] = "onnx.Slice"(%[[SWAP]], %[[B2]], %[[E2]], %[[AX]], %[[ST]])
// CHECK: %[[RC:.*]] = "onnx.Concat"(%[[S1]], %[[S2]]) {axis = 1 : si64}
// CHECK: %[[C1:.*]] = "onnx.Concat"(%arg0, %arg2) {axis = 1 : si64}
// CHECK: "onnx.InstanceNormalization"(%[[RC]]
// CHECK: return %{{.*}}, %[[C1]]

// -----

// Negative test: asymmetric share indices, so the pass must NOT rewrite.
func.func @test_optimize_sibling_concat_no_match_asymmetric_share(
    %arg0: tensor<1x12x3x4xf32>,
    %arg1: tensor<1x12x3x4xf32>,
    %arg2: tensor<1x12x3x4xf32>,
    %scale: tensor<24xf32>,
    %bias: tensor<24xf32>,
    %w: tensor<8x24x1x1xf32>) -> (tensor<1x8x3x4xf32>, tensor<1x24x3x4xf32>) {
  %c0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
      : (tensor<1x12x3x4xf32>, tensor<1x12x3x4xf32>) -> tensor<1x24x3x4xf32>
  %c1 = "onnx.Concat"(%arg1, %arg2) {axis = 1 : si64}
      : (tensor<1x12x3x4xf32>, tensor<1x12x3x4xf32>) -> tensor<1x24x3x4xf32>

  %in = "onnx.InstanceNormalization"(%c0, %scale, %bias) {epsilon = 1.0e-5 : f32}
      : (tensor<1x24x3x4xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x24x3x4xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %out = "onnx.Conv"(%in, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                  kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
      : (tensor<1x24x3x4xf32>, tensor<8x24x1x1xf32>, none) -> tensor<1x8x3x4xf32>

  return %out, %c1 : tensor<1x8x3x4xf32>, tensor<1x24x3x4xf32>
}

// CHECK-LABEL: func.func @test_optimize_sibling_concat_no_match_asymmetric_share
// CHECK-NOT: "onnx.Slice"
// CHECK: %[[C0:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
// CHECK: %[[C1:.*]] = "onnx.Concat"(%arg1, %arg2) {axis = 1 : si64}
// CHECK: "onnx.InstanceNormalization"(%[[C0]]
// CHECK: return %{{.*}}, %[[C1]]

