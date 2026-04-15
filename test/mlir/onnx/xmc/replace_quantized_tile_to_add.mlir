// RUN: onnx-mlir-opt --replace-quantized-tile-to-add %s | FileCheck %s
// NOTE: Assumes quant-types style IR: ranked tensors with !quant.uniform element
// types. Replaces Tile only when NumPy broadcast of input vs tiled zp-const
// matches the tile output shape (e.g. repeat on a size-1 axis).

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

  // CHECK-DAG: "onnx.Constant"
  // CHECK: "onnx.XCOMPILERFusedEltwise"(%arg0,
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
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
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}
