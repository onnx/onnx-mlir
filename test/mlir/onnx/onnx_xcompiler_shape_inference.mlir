// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Shape inference tests for XCOMPILER Operations  
/// Domain: com.amd.xcompiler
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
/// XCOMPILER FusedEltwise Tests (Quantized Element-wise Operations)
//===----------------------------------------------------------------------===//

// COM: Test basic element-wise add with same shapes (no broadcast needed)
func.func @test_XCOMPILER_fused_eltwise_same_shape(%arg0: tensor<1x64x28x28xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<1x64x28x28xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_same_shape
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with scalar-like tensor [1] x [N,C,H,W]
func.func @test_XCOMPILER_fused_eltwise_broadcast_scalar(%arg0: tensor<1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "MUL",
    nonlinear = "NONE"
  } : (tensor<1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_scalar
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with channel dimension [1,C,1,1] x [N,C,H,W]
func.func @test_XCOMPILER_fused_eltwise_broadcast_channel(%arg0: tensor<1x64x1x1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "RELU"
  } : (tensor<1x64x1x1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_channel
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting with different ranks [1,C,1,1] x [N,C,H,W] -> channel-wise broadcast
func.func @test_XCOMPILER_fused_eltwise_broadcast_rank_diff(%arg0: tensor<1x64x1x1xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "SUB",
    nonlinear = "NONE"
  } : (tensor<1x64x1x1xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_rank_diff
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test broadcasting [1,4,1] x [3,1,5] -> [3,4,5]
func.func @test_XCOMPILER_fused_eltwise_broadcast_numpy(%arg0: tensor<1x4x1xi8>, %arg1: tensor<3x1x5xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "DIV",
    nonlinear = "NONE"
  } : (tensor<1x4x1xi8>, tensor<3x1x5xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_broadcast_numpy
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<3x4x5xi8>
  // CHECK: onnx.Return [[RES]] : tensor<3x4x5xi8>
}

// -----

// COM: Test with LeakyReLU activation
func.func @test_XCOMPILER_fused_eltwise_leaky_relu(%arg0: tensor<1x64x28x28xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "LEAKYRELU",
    leakyrelu_alpha = 0.01 : f32,
    prelu_in = 2621 : si64,
    prelu_shift = 18 : si64
  } : (tensor<1x64x28x28xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_leaky_relu
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

// -----

// COM: Test with dynamic dimensions - broadcasting resolves known dims
func.func @test_XCOMPILER_fused_eltwise_dynamic(%arg0: tensor<?x64x?x?xi8>, %arg1: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {
    type = "ADD",
    nonlinear = "NONE"
  } : (tensor<?x64x?x?xi8>, tensor<1x64x28x28xi8>) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_dynamic
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {{.*}} -> tensor<?x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<?x64x28x28xi8>
}

// -----

// COM: Test with single input (B is optional, using none)
func.func @test_XCOMPILER_fused_eltwise_single_input(%arg0: tensor<1x64x28x28xi8>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %none) {
    type = "ADD",
    nonlinear = "RELU"
  } : (tensor<1x64x28x28xi8>, none) -> tensor<*xi8>
  onnx.Return %0 : tensor<*xi8>

  // CHECK-LABEL: test_XCOMPILER_fused_eltwise_single_input
  // CHECK: [[RES:%.+]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %0) {{.*}} -> tensor<1x64x28x28xi8>
  // CHECK: onnx.Return [[RES]] : tensor<1x64x28x28xi8>
}

