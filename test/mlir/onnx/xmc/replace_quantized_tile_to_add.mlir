// RUN: onnx-mlir-opt --replace-quantized-tile-to-add %s | FileCheck %s
// NOTE: Assumes quant-types style IR: ranked tensors with !quant.uniform element
// types. Replaces Tile with onnx.Add and a zp (or zero) splat only when NumPy
// broadcast of input vs splat matches the tile output shape (e.g. repeat on a
// size-1 axis).

// -----
// Broadcastable tile: [1x4] * repeats [3,1] -> [3x4]; 1 broadcasts to 3.
// CHECK-LABEL: func.func @tile_to_fused_eltwise_broadcastable
func.func @tile_to_fused_eltwise_broadcastable(
    %arg0: tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<3x4x!quant.uniform<u8:f32, 0.1:128>> {
  %repeats = "onnx.Constant"() {
    value = dense<[3, 1]> : tensor<2xi64>
  } : () -> tensor<2xi64>
  %0 = "onnx.Tile"(%arg0, %repeats) :
      (tensor<1x4x!quant.uniform<u8:f32, 0.1:128>>, tensor<2xi64>)
      -> tensor<3x4x!quant.uniform<u8:f32, 0.1:128>>
  return %0 : tensor<3x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-DAG: onnx.Constant
  // CHECK: "onnx.Add"(%arg0,
  // CHECK-NOT: "onnx.Tile"
}

// -----
// Not broadcastable: [2x4] * repeats [3,1] -> [6x4]; 2 vs 6 is invalid for Add.
// CHECK-LABEL: func.func @tile_stays_when_not_broadcastable
func.func @tile_stays_when_not_broadcastable(
    %arg0: tensor<2x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<6x4x!quant.uniform<u8:f32, 0.1:128>> {
  %repeats = "onnx.Constant"() {
    value = dense<[3, 1]> : tensor<2xi64>
  } : () -> tensor<2xi64>
  %0 = "onnx.Tile"(%arg0, %repeats) :
      (tensor<2x4x!quant.uniform<u8:f32, 0.1:128>>, tensor<2xi64>)
      -> tensor<6x4x!quant.uniform<u8:f32, 0.1:128>>
  return %0 : tensor<6x4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: "onnx.Tile"(%arg0,
  // CHECK-NOT: "onnx.Add"
}

// -----

func.func @topk_tile(%arg0: tensor<1x3600x!quant.uniform<u16:f32, 2.3757114831823856E-4:51533>>, %arg1: tensor<1x3600x4x!quant.uniform<u16:f32, 2.1844083676114678E-5:2948>>) -> tensor<1x300x4x!quant.uniform<u16:f32, 2.1844083676114678E-5:2948>> {
  %0 = onnx.Constant dense<300> : tensor<1xi64>
  %1 = onnx.Constant dense<[1, 300, 1]> : tensor<3xi64>
  %2 = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  %Values, %Indices = "onnx.TopK"(%arg0, %0) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<1x3600x!quant.uniform<u16:f32, 2.3757114831823856E-4:51533>>, tensor<1xi64>) -> (tensor<1x300xf32>, tensor<1x300xi64>)
  %3 = "onnx.Cast"(%Indices) {saturate = 1 : si64, to = i32} : (tensor<1x300xi64>) -> tensor<1x300xi32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x300xi32>, tensor<3xi64>) -> tensor<1x300x1xi32>
  %5 = "onnx.Tile"(%4, %2) : (tensor<1x300x1xi32>, tensor<3xi64>) -> tensor<1x300x4xi32>
  %6 = "onnx.GatherElements"(%arg1, %5) {axis = 1 : si64} : (tensor<1x3600x4x!quant.uniform<u16:f32, 2.1844083676114678E-5:2948>>, tensor<1x300x4xi32>) -> tensor<1x300x4x!quant.uniform<u16:f32, 2.1844083676114678E-5:2948>>
  return %6 : tensor<1x300x4x!quant.uniform<u16:f32, 2.1844083676114678E-5:2948>>
}

// CHECK-LABEL: @topk_tile
// CHECK-NOT: "onnx.Tile"
// CHECK-DAG: dense<[1, 3600, 1]>
// CHECK: "onnx.Reshape"(%arg0
// CHECK: "onnx.Add"
// CHECK-SAME: (tensor<1x3600x1x!quant.uniform<u16:f32, 2.3757114831823856E-4:51533>>,
// CHECK-SAME: tensor<1x3600x4x!quant.uniform<u16:f32, 2.3757114831823856E-4:51533>>)
// CHECK-SAME: -> tensor<1x3600x4x!quant.uniform<u16:f32, 2.3757114831823856E-4:51533>>
// CHECK: "onnx.TopK"
// CHECK: "onnx.Cast"
// CHECK: "onnx.GatherElements"(%arg1
