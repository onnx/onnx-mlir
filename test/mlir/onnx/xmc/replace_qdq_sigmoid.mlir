// RUN: onnx-mlir-opt --replace-qdq-sigmoid %s | FileCheck %s
//
// Test replace-qdq-sigmoid pass: quantized Sigmoid and Sigmoid+Mul(const)
// are replaced with onnx.XCOMPILERFusedEltwise type="QLINEARSIGMOID".
// Assumes quantized types (e.g. after quant-types); no explicit Q/DQ ops.

// -----
// Quantized Sigmoid -> XCOMPILERFusedEltwise QLINEARSIGMOID (no mul_y)
// CHECK-LABEL: func.func @test_quantized_sigmoid
func.func @test_quantized_sigmoid(%arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.00392157:0>> {

  %0 = "onnx.Sigmoid"(%arg0) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.00392157:0>>

  return %0 : tensor<1x16x32x32x!quant.uniform<u8:f32, 0.00392157:0>>

  // CHECK-NOT: "onnx.Sigmoid"
  // CHECK: %[[NONE:.*]] = "onnx.NoValue"
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NONE]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "QLINEARSIGMOID"
  // CHECK: return %[[FUSED]]
}

// -----
// Sigmoid -> Mul(const) with quantized types -> QLINEARSIGMOID with mul_y
// CHECK-LABEL: func.func @test_quantized_sigmoid_mul
func.func @test_quantized_sigmoid_mul(%arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {

  %sigmoid = "onnx.Sigmoid"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.01:128>>

  %cst = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %mul = "onnx.Mul"(%sigmoid, %cst) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.01:128>>, tensor<f32>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  return %mul : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK-NOT: "onnx.Sigmoid"
  // CHECK-NOT: "onnx.Mul"
  // CHECK: %[[NONE:.*]] = "onnx.NoValue"
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NONE]])
  // CHECK-SAME: mul_y = 2.000000e+00
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "QLINEARSIGMOID"
  // CHECK: return %[[FUSED]]
}

// -----
// With enable-lut-sigmoid=true the fused op has enable_lut_sigmoid = true
// RUN: onnx-mlir-opt --replace-qdq-sigmoid="enable-lut-sigmoid=true" %s | FileCheck %s --check-prefix=LUT
// Default run: this function is also replaced (enable_lut_sigmoid false).
// CHECK-LABEL: func.func @test_quantized_sigmoid_lut
// LUT-LABEL: func.func @test_quantized_sigmoid_lut
func.func @test_quantized_sigmoid_lut(%arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x4x4x4x!quant.uniform<u8:f32, 0.01:0>> {

  %0 = "onnx.Sigmoid"(%arg0) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.05:128>>)
      -> tensor<1x4x4x4x!quant.uniform<u8:f32, 0.01:0>>

  return %0 : tensor<1x4x4x4x!quant.uniform<u8:f32, 0.01:0>>

  // CHECK-NOT: "onnx.Sigmoid"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: type = "QLINEARSIGMOID"
  // LUT-NOT: "onnx.Sigmoid"
  // LUT: "onnx.XCOMPILERFusedEltwise"
  // LUT-SAME: enable_lut_sigmoid = true
  // LUT-SAME: type = "QLINEARSIGMOID"
}

// -----
// Negative: non-quantized Sigmoid is not replaced
// CHECK-LABEL: func.func @test_sigmoid_float_not_replaced
func.func @test_sigmoid_float_not_replaced(%arg0: tensor<1x8x8xf32>) -> tensor<1x8x8xf32> {

  %0 = "onnx.Sigmoid"(%arg0) :
      (tensor<1x8x8xf32>) -> tensor<1x8x8xf32>

  return %0 : tensor<1x8x8xf32>

  // CHECK: "onnx.Sigmoid"(%arg0)
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Sigmoid + Mul(non-const): only Sigmoid is replaced (Sigmoid+Mul pattern needs constant).
// So we get QLINEARSIGMOID feeding into Mul.
// CHECK-LABEL: func.func @test_sigmoid_mul_non_const_not_replaced
func.func @test_sigmoid_mul_non_const_not_replaced(
    %arg0: tensor<1x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x!quant.uniform<u8:f32, 0.02:128>> {

  %sigmoid = "onnx.Sigmoid"(%arg0) :
      (tensor<1x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x!quant.uniform<u8:f32, 0.01:128>>
  %mul = "onnx.Mul"(%sigmoid, %arg1) :
      (tensor<1x4x4x!quant.uniform<u8:f32, 0.01:128>>,
       tensor<1x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x!quant.uniform<u8:f32, 0.02:128>>

  return %mul : tensor<1x4x4x!quant.uniform<u8:f32, 0.02:128>>

  // Sigmoid is replaced by QLINEARSIGMOID; Mul stays (other operand not constant).
  // CHECK-NOT: "onnx.Sigmoid"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: type = "QLINEARSIGMOID"
  // CHECK: "onnx.Mul"
}
