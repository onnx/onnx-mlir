// Verify that quant-types followed by constprop-onnx does not crash or
// miscompute. The IsIntOrFloatType constraint on element-wise const-prop
// patterns causes them to skip quantized-typed constants. For rearrangement
// ops (like Transpose), the ValuesHaveSameDType constraint ensures const-prop
// is skipped when input/output element types differ (e.g. one side is
// quantized and the other is not, or different quantization parameters).

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


// Unary ops with quantized input (from DQL) but no Q on output — should NOT
// be folded because IsIntOrFloatType blocks quantized inputs.

func.func @neg_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Neg"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @neg_quantized_input_float_output
// CHECK:         onnx.Neg
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @relu_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Relu"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @relu_quantized_input_float_output
// CHECK:         onnx.Relu
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @abs_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Abs"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @abs_quantized_input_float_output
// CHECK:         onnx.Abs
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @ceil_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Ceil"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @ceil_quantized_input_float_output
// CHECK:         onnx.Ceil
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @floor_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Floor"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @floor_quantized_input_float_output
// CHECK:         onnx.Floor
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @round_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Round"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @round_quantized_input_float_output
// CHECK:         onnx.Round
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @cos_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Cos"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @cos_quantized_input_float_output
// CHECK:         onnx.Cos
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @sin_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Sin"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @sin_quantized_input_float_output
// CHECK:         onnx.Sin
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @erf_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Erf"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @erf_quantized_input_float_output
// CHECK:         onnx.Erf
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @exp_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Exp"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @exp_quantized_input_float_output
// CHECK:         onnx.Exp
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @log_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Log"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @log_quantized_input_float_output
// CHECK:         onnx.Log
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @sigmoid_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Sigmoid"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @sigmoid_quantized_input_float_output
// CHECK:         onnx.Sigmoid
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @sqrt_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Sqrt"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @sqrt_quantized_input_float_output
// CHECK:         onnx.Sqrt
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


func.func @reciprocal_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %s = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %s, %z) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  %r = "onnx.Reciprocal"(%dq) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @reciprocal_quantized_input_float_output
// CHECK:         onnx.Reciprocal
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2xf32>


// Transpose with quantized input but no Q on output — should NOT be folded
func.func @transpose_quantized_input_float_output() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>

  %t = "onnx.Transpose"(%a_dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>

  return %t : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_quantized_input_float_output
// CHECK:         onnx.Transpose
// CHECK-SAME:      (tensor<1x2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2x1xf32>


// Transpose with different input/output quantized types — should NOT be folded
func.func @transpose_quantized_dtype_mismatch() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>

  %t = "onnx.Transpose"(%a_dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>

  %out_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%t, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x1xf32>

  return %final : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_quantized_dtype_mismatch
// CHECK:         onnx.Transpose
// CHECK-SAME:      (tensor<1x2x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<2x1x!quant.uniform<u8:f32, 2.500000e-01>>


// Transpose with matching input/output quantized types — should still be folded
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


// Per-axis Q along input axis 1; after perm=[1,0] the quant axis becomes 0.
// Input/output use the same per-axis scales — fold transpose into the constant.
func.func @transpose_per_axis_quant_constant() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %a_zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<1x2xf32>

  %t = "onnx.Transpose"(%a_dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>

  %out_scale = onnx.Constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %out_zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>
  %q = "onnx.QuantizeLinear"(%t, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<2xf32>, tensor<2xui8>) -> tensor<2x1xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<2x1xf32>

  return %final : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_per_axis_quant_constant
// CHECK-NOT:     onnx.Transpose
// CHECK:         !quant.uniform<u8:f32:0, {5.000000e-01,5.000000e-01}>
// CHECK:         quant.scast


// Same per-axis input path, but output Q uses different per-axis scales than
// the remapped input quant type — transpose const-prop must not fold.
func.func @transpose_per_axis_quant_mismatch_scales() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %a_zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<1x2xf32>

  %t = "onnx.Transpose"(%a_dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>

  %out_scale = onnx.Constant dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>
  %out_zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>
  %q = "onnx.QuantizeLinear"(%t, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<2xf32>, tensor<2xui8>) -> tensor<2x1xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<2x1xf32>

  return %final : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_per_axis_quant_mismatch_scales
// CHECK:         onnx.Transpose
// CHECK-SAME:      perm = [1, 0]


// Reshape with quantized input but float output — should NOT be folded
func.func @reshape_quantized_input_float_output() -> tensor<1x2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %shape = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %r = "onnx.Reshape"(%a_dq, %shape) {allowzero = 0 : si64} : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>

  return %r : tensor<1x2xf32>
}

// CHECK-LABEL: @reshape_quantized_input_float_output
// CHECK:         onnx.Reshape
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<1x2xf32>


// Reshape with matching input/output quantized types — should still be folded
func.func @reshape_quantized_constant() -> tensor<1x2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %shape = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %r = "onnx.Reshape"(%a_dq, %shape) {allowzero = 0 : si64} : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>

  %out_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %out_zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%r, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
  %final = "onnx.DequantizeLinear"(%q, %out_scale, %out_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>

  return %final : tensor<1x2xf32>
}

// CHECK-LABEL: @reshape_quantized_constant
// CHECK-NOT:     onnx.Reshape
// CHECK:         onnx.Constant
// CHECK:         quant.scast


// Unsqueeze with quantized input but float output — should NOT be folded
func.func @unsqueeze_quantized_input_float_output() -> tensor<1x2xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %r = "onnx.Unsqueeze"(%a_dq, %axes) : (tensor<2xf32>, tensor<1xi64>) -> tensor<1x2xf32>

  return %r : tensor<1x2xf32>
}

// CHECK-LABEL: @unsqueeze_quantized_input_float_output
// CHECK:         onnx.Unsqueeze
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<1x2xf32>


// Squeeze with quantized input but float output — should NOT be folded
func.func @squeeze_quantized_input_float_output() -> tensor<2xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>

  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %r = "onnx.Squeeze"(%a_dq, %axes) : (tensor<1x2xf32>, tensor<1xi64>) -> tensor<2xf32>

  return %r : tensor<2xf32>
}

// CHECK-LABEL: @squeeze_quantized_input_float_output
// CHECK:         onnx.Squeeze
// CHECK-SAME:      (tensor<1x2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<2xf32>


// Gather with quantized input but float output — should NOT be folded
func.func @gather_quantized_input_float_output() -> tensor<1xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %indices = onnx.Constant dense<[0]> : tensor<1xi64>
  %r = "onnx.Gather"(%a_dq, %indices) {axis = 0 : si64} : (tensor<2xf32>, tensor<1xi64>) -> tensor<1xf32>

  return %r : tensor<1xf32>
}

// CHECK-LABEL: @gather_quantized_input_float_output
// CHECK:         onnx.Gather
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<1xf32>


// Slice with quantized input but float output — should NOT be folded
func.func @slice_quantized_input_float_output() -> tensor<1xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %starts = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends = onnx.Constant dense<[1]> : tensor<1xi64>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %r = "onnx.Slice"(%a_dq, %starts, %ends, %axes, %axes) : (tensor<2xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>

  return %r : tensor<1xf32>
}

// CHECK-LABEL: @slice_quantized_input_float_output
// CHECK:         onnx.Slice
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>


// ReduceMean noop fold with quantized input — should NOT be folded.
// The fold method returns getData() when there are no reduction axes and
// noop_with_empty_axes is set; the IsIntOrFloatType guard prevents a type
// mismatch between the quantized input and the float result.
func.func @reduce_mean_noop_quantized_input_float_output() -> tensor<1xf32> {
  %a = onnx.Constant dense<[10, 20]> : tensor<2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>

  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>

  %none = "onnx.NoValue"() {value} : () -> none
  %r = "onnx.ReduceMean"(%a_dq, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2xf32>, none) -> tensor<1xf32>

  return %r : tensor<1xf32>
}

// CHECK-LABEL: @reduce_mean_noop_quantized_input_float_output
// CHECK:         onnx.ReduceMean
// CHECK-SAME:      (tensor<2x!quant.uniform<u8:f32, 5.000000e-01>>