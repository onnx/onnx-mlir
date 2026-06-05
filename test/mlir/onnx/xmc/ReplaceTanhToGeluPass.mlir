// RUN: onnx-mlir-opt --split-input-file --replace-tanh-to-gelu %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// ============================================================================
// PASS CASES: tanh-approximation GELU with quantized types
//
// The matcher dequantizes scalar constants as f = (raw - zp) * scale and
// expects the canonical tanh-GELU constants:
//   0.044715        (inner cubic coefficient)
//   0.797884583     (sqrt(2/pi))
//   1.0             (added after Tanh)
//   0.5             (final outer multiplier)
//   3.0             (Pow exponent)
// All match to within 1e-2.
// ============================================================================

// Variant A: the final 0.5 multiply trails the (x * (1+tanh)) Mul.
//   pow  = Pow(x, 3.0)
//   mul1 = Mul(c_0.044715, pow)
//   add1 = Add(x, mul1)
//   mul2 = Mul(c_0.79788, add1)
//   tanh = Tanh(mul2)
//   add2 = Add(c_1, tanh)
//   mul3 = Mul(x, add2)              <-- variant A: x is the other operand
//   mul4 = Mul(mul3, c_0.5)          <-- variant A: trailing 0.5 multiply
// CHECK-LABEL: @tanh_gelu_variant_a_pass
func.func @tanh_gelu_variant_a_pass(%arg0: tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>> {
  %c_three   = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011764706112444401>>
  %c_044715  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>
  %c_07978   = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>
  %c_one     = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>
  %c_half    = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>

  %pow  = "onnx.Pow"(%arg0, %c_three) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<!quant.uniform<u8:f32, 0.011764706112444401>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>
  %mul1 = "onnx.Mul"(%pow, %c_044715) : (tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>, tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>
  %add1 = "onnx.Add"(%mul1, %arg0) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>
  %mul2 = "onnx.Mul"(%add1, %c_07978) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>, tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>
  %tanh = "onnx.Tanh"(%mul2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>
  %add2 = "onnx.Add"(%tanh, %c_one) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>, tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>
  %mul3 = "onnx.Mul"(%arg0, %add2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>
  %mul4 = "onnx.Mul"(%mul3, %c_half) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>, tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
  return %mul4 : tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
}
// CHECK-NOT: "onnx.Pow"
// CHECK-NOT: "onnx.Tanh"
// CHECK: %[[GELU:.*]] = "onnx.Gelu"(%arg0) {approximate = "tanh"}
// CHECK-SAME: (tensor<1x4x8x!quant.uniform<u8:f32, {{.+}}:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, {{.+}}:7>>
// CHECK: return %[[GELU]]

// -----

// Variant B: the 0.5 * x multiply is precomputed, then multiplied by (1+tanh).
// This is the form the PSA2/Gelu-stacked models actually emit.
//   pre = Mul(x, c_0.5)              <-- variant B: precomputed 0.5 * x
//   mul = Mul(add2, pre)             <-- the outermost Mul IS the Gelu output
// CHECK-LABEL: @tanh_gelu_variant_b_pass
func.func @tanh_gelu_variant_b_pass(%arg0: tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>> {
  %c_three  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011764706112444401>>
  %c_044715 = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>
  %c_07978  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>
  %c_one    = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>
  %c_half   = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>

  %pow  = "onnx.Pow"(%arg0, %c_three) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<!quant.uniform<u8:f32, 0.011764706112444401>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>
  %mul1 = "onnx.Mul"(%pow, %c_044715) : (tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>, tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>
  %add1 = "onnx.Add"(%mul1, %arg0) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>
  %mul2 = "onnx.Mul"(%add1, %c_07978) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>, tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>
  %tanh = "onnx.Tanh"(%mul2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>
  %add2 = "onnx.Add"(%tanh, %c_one) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>, tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>
  %pre = "onnx.Mul"(%arg0, %c_half) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0215903651:123>>
  %out = "onnx.Mul"(%add2, %pre) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0215903651:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
  return %out : tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
}
// CHECK-NOT: "onnx.Pow"
// CHECK-NOT: "onnx.Tanh"
// CHECK: %[[GELU:.*]] = "onnx.Gelu"(%arg0) {approximate = "tanh"}
// CHECK-SAME: (tensor<1x4x8x!quant.uniform<u8:f32, {{.+}}:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, {{.+}}:7>>
// CHECK: return %[[GELU]]

// ============================================================================
// FAIL CASES: patterns that should NOT be transformed
// ============================================================================

// -----

// Non-quantized types: matcher's quantized-type guard fires; no rewrite.
// CHECK-LABEL: @tanh_gelu_non_quantized_fail
func.func @tanh_gelu_non_quantized_fail(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %c_three  = onnx.Constant dense<3.0> : tensor<f32>
  %c_044715 = onnx.Constant dense<0.044715> : tensor<f32>
  %c_07978  = onnx.Constant dense<0.797884583> : tensor<f32>
  %c_one    = onnx.Constant dense<1.0> : tensor<f32>
  %c_half   = onnx.Constant dense<0.5> : tensor<f32>

  %pow  = "onnx.Pow"(%arg0, %c_three) : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
  %mul1 = "onnx.Mul"(%pow, %c_044715) : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
  %add1 = "onnx.Add"(%mul1, %arg0) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %mul2 = "onnx.Mul"(%add1, %c_07978) : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
  %tanh = "onnx.Tanh"(%mul2) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %add2 = "onnx.Add"(%tanh, %c_one) : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
  %mul3 = "onnx.Mul"(%arg0, %add2) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %mul4 = "onnx.Mul"(%mul3, %c_half) : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4x8xf32>
  return %mul4 : tensor<1x4x8xf32>
}
// CHECK: "onnx.Tanh"
// CHECK-NOT: "onnx.Gelu"

// -----

// Pow exponent != 3: matcher rejects (constant-value check).
// CHECK-LABEL: @tanh_gelu_wrong_pow_exp_fail
func.func @tanh_gelu_wrong_pow_exp_fail(%arg0: tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>> {
  // exponent stores raw=170 with scale=0.011765 -> 170 * 0.011765 = 2.0  (NOT 3)
  %c_two    = onnx.Constant {value = dense<170> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011764706112444401>>
  %c_044715 = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>
  %c_07978  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>
  %c_one    = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>
  %c_half   = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>

  %pow  = "onnx.Pow"(%arg0, %c_two) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<!quant.uniform<u8:f32, 0.011764706112444401>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>
  %mul1 = "onnx.Mul"(%pow, %c_044715) : (tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>, tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>
  %add1 = "onnx.Add"(%mul1, %arg0) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>
  %mul2 = "onnx.Mul"(%add1, %c_07978) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>, tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>
  %tanh = "onnx.Tanh"(%mul2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>
  %add2 = "onnx.Add"(%tanh, %c_one) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>, tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>
  %mul3 = "onnx.Mul"(%arg0, %add2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>
  %mul4 = "onnx.Mul"(%mul3, %c_half) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>, tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
  return %mul4 : tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
}
// CHECK: "onnx.Tanh"
// CHECK-NOT: "onnx.Gelu"

// -----

// Tanh has multiple uses: matcher's hasOneUse guard fires; no rewrite.
// CHECK-LABEL: @tanh_gelu_multi_use_fail
func.func @tanh_gelu_multi_use_fail(%arg0: tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> (tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>) {
  %c_three  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.011764706112444401>>
  %c_044715 = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>
  %c_07978  = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>
  %c_one    = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>
  %c_half   = onnx.Constant {value = dense<255> : tensor<ui8>} : tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>

  %pow  = "onnx.Pow"(%arg0, %c_three) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<!quant.uniform<u8:f32, 0.011764706112444401>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>
  %mul1 = "onnx.Mul"(%pow, %c_044715) : (tensor<1x4x8x!quant.uniform<u8:f32, 1.3129487037658691:115>>, tensor<!quant.uniform<u8:f32, 1.7535293591208756E-4>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>
  %add1 = "onnx.Add"(%mul1, %arg0) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0587084964:115>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>
  %mul2 = "onnx.Mul"(%add1, %c_07978) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.1018892303:119>>, tensor<!quant.uniform<u8:f32, 0.0031289590988308191>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>
  %tanh = "onnx.Tanh"(%mul2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0812958479:119>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>
  %add2 = "onnx.Add"(%tanh, %c_one) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>, tensor<!quant.uniform<u8:f32, 0.0039215688593685627>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>
  %mul3 = "onnx.Mul"(%arg0, %add2) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0431807302:123>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>
  %mul4 = "onnx.Mul"(%mul3, %c_half) : (tensor<1x4x8x!quant.uniform<u8:f32, 0.0459092372:14>>, tensor<!quant.uniform<u8:f32, 0.0019607844296842813>>) -> tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>
  return %mul4, %tanh : tensor<1x4x8x!quant.uniform<u8:f32, 0.0229546186:7>>, tensor<1x4x8x!quant.uniform<u8:f32, 0.0078431377:128>>
}
// CHECK: "onnx.Tanh"
// CHECK-NOT: "onnx.Gelu"
