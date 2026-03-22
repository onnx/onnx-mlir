// Verify that quant-types followed by constprop-onnx does not crash or
// miscompute. The IsIntOrFloatType constraint on element-wise const-prop
// patterns causes them to skip quantized-typed constants, leaving the ops
// intact. Data-movement patterns (e.g. Transpose) are NOT guarded because
// rearranging storage values preserves quantized semantics.
// RUN: onnx-mlir-opt %s --quant-types --constprop-onnx | FileCheck %s

func.func @add_two_quantized_constants() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %b = onnx.Constant dense<[6, 12]> : tensor<2xui8>
  %b_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %add = "onnx.Add"(%a_dq, %b_dq) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%add, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  return %final : tensor<2xf32>
}

// CHECK-LABEL: @add_two_quantized_constants
// CHECK:         onnx.Constant {value = dense<[10, 20]> : tensor<2xui8>} : tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>
// CHECK:         onnx.Constant {value = dense<[6, 12]> : tensor<2xui8>} : tensor<2x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK:         onnx.Add
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<2x!quant.uniform<u8:f32, 2.500000e-01>>)


func.func @mul_two_quantized_constants() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %b = onnx.Constant dense<[8, 4]> : tensor<2xui8>
  %b_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %mul = "onnx.Mul"(%a_dq, %b_dq) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%mul, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  return %final : tensor<2xf32>
}

// CHECK-LABEL: @mul_two_quantized_constants
// CHECK:         onnx.Constant {value = dense<[10, 20]> : tensor<2xui8>} : tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>
// CHECK:         onnx.Constant {value = dense<[8, 4]> : tensor<2xui8>} : tensor<2x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK:         onnx.Mul
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<2x!quant.uniform<u8:f32, 2.500000e-01>>)


func.func @sub_two_quantized_constants() -> tensor<2xf32> {
  %a = onnx.Constant dense<[20, 40]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %b = onnx.Constant dense<[6, 12]> : tensor<2xui8>
  %b_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %sub = "onnx.Sub"(%a_dq, %b_dq) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%sub, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  return %final : tensor<2xf32>
}

// CHECK-LABEL: @sub_two_quantized_constants
// CHECK:         onnx.Sub
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<2x!quant.uniform<u8:f32, 2.500000e-01>>)


func.func @neg_quantized_constant() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %neg = "onnx.Neg"(%a_dq) : (tensor<2xf32>) -> tensor<2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%neg, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  return %final : tensor<2xf32>
}

// CHECK-LABEL: @neg_quantized_constant
// CHECK:         onnx.Neg
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>)
// CHECK:         quant.scast


func.func @relu_quantized_constant() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %relu = "onnx.Relu"(%a_dq) : (tensor<2xf32>) -> tensor<2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%relu, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  return %final : tensor<2xf32>
}

// CHECK-LABEL: @relu_quantized_constant
// CHECK:         onnx.Relu
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>)
// CHECK:         quant.scast


func.func @matmul_two_quantized_constants() -> tensor<2x2xf32> {
  %a = onnx.Constant dense<[[10, 20], [30, 40]]> : tensor<2x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %b = onnx.Constant dense<[[2, 4], [6, 8]]> : tensor<2x2xui8>
  %b_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  %matmul = "onnx.MatMul"(%a_dq, %b_dq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%matmul, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  return %final : tensor<2x2xf32>
}

// CHECK-LABEL: @matmul_two_quantized_constants
// CHECK:         onnx.Constant {value = dense<{{.*}}> : tensor<2x2xui8>} : tensor<2x2x!quant.uniform<u8:f32, 5.000000e-01>>
// CHECK:         onnx.Constant {value = dense<{{.*}}> : tensor<2x2xui8>} : tensor<2x2x!quant.uniform<u8:f32, 2.500000e-01>>
// CHECK:         onnx.MatMul


func.func @reduce_sum_quantized_constant() -> tensor<1xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %reduce = "onnx.ReduceSum"(%a_dq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2xf32>, tensor<1xi64>) -> tensor<1xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%reduce, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1xf32>, tensor<f32>, tensor<ui8>) -> tensor<1xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1xui8>, tensor<f32>, tensor<ui8>) -> tensor<1xf32>

  return %final : tensor<1xf32>
}

// CHECK-LABEL: @reduce_sum_quantized_constant
// CHECK:         onnx.ReduceSum
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>
// CHECK:         quant.scast


// Transpose — should still be folded
func.func @transpose_quantized_constant() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>

  %t = "onnx.Transpose"(%a_dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%t, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x1xf32>

  return %final : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_quantized_constant
// CHECK-NOT:     onnx.Transpose
// CHECK:         onnx.Constant
// CHECK:         quant.scast
