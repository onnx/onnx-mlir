// RUN: onnx-mlir-opt --replace-qdq-eltwise %s | FileCheck %s
// NOTE: This pass assumes quant-types has already run, so ops use native
// `!quant.uniform` types (no explicit Q/DQ ops).

// -----
// Test Pattern 2: Element-wise with ReLU Activation Fusion - Add + ReLU
// Both operations already have quantized types (post quant-types pass)
// Should create single onnx.XCOMPILERFusedEltwise op with type="ADD", nonlinear="RELU"
// CHECK-LABEL: func.func @test_quantized_add_relu
func.func @test_quantized_add_relu(%arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
                                    %arg1: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>> {

  // Add operation with quantized inputs and output (after quant-types)
  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
       tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // ReLU on quantized data
  %relu = "onnx.Relu"(%add) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: nonlinear = "RELU"
  // CHECK-SAME: type = "ADD"
  // CHECK-NOT: onnx.Add
  // CHECK-NOT: onnx.Relu
}

// -----
// Test Pattern 2: Mul + PReLU with quantized types
// NOTE: XCOMPILERFusedEltwise does not model PReLU slope, so this is NOT fused.
// CHECK-LABEL: func.func @test_quantized_mul_prelu
func.func @test_quantized_mul_prelu(%arg0: tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>,
                                     %arg1: tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>,
                                     %slope: tensor<8xf32>)
    -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>> {

  %mul = "onnx.Mul"(%arg0, %arg1) :
      (tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>,
       tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>)
      -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>

  %prelu = "onnx.PRelu"(%mul, %slope) :
      (tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>, tensor<8xf32>)
      -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>

  return %prelu : tensor<1x8x16x16x!quant.uniform<i8:f32, 0.03:0>>

  // CHECK: "onnx.Mul"(%arg0, %arg1)
  // CHECK: "onnx.PRelu"(%{{.*}}, %{{.*}})
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Test Pattern 2: Sub + ReLU with quantized types
// CHECK-LABEL: func.func @test_quantized_sub_relu
func.func @test_quantized_sub_relu(%arg0: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
                                    %arg1: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>> {

  %sub = "onnx.Sub"(%arg0, %arg1) :
      (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  %relu = "onnx.Relu"(%sub) :
      (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  return %relu : tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: nonlinear = "RELU"
  // CHECK-SAME: type = "SUB"
  // CHECK-NOT: "onnx.Sub"
  // CHECK-NOT: "onnx.Relu"
}

// -----
// Test Pattern 2: Add + LeakyReLU with quantized types
// CHECK-LABEL: func.func @test_quantized_add_leakyrelu
func.func @test_quantized_add_leakyrelu(%arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
                                         %arg1: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>> {

  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
       tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  %leaky = "onnx.LeakyRelu"(%add) {alpha = 0.1 : f32} :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  return %leaky : tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: leakyrelu_alpha = 1.000000e-01
  // CHECK-SAME: nonlinear = "LEAKYRELU"
  // CHECK-SAME: prelu_in = 26 : si64
  // CHECK-SAME: prelu_shift = 8 : si64
  // CHECK-SAME: type = "ADD"
  // CHECK-NOT: "onnx.Add"
  // CHECK-NOT: "onnx.LeakyRelu"
}

// -----
// Test Pattern 2: Div + PReLU with quantized types
// NOTE: not fused (PReLU slope not modeled).
// CHECK-LABEL: func.func @test_quantized_div_prelu
func.func @test_quantized_div_prelu(%arg0: tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>,
                                     %arg1: tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>,
                                     %slope: tensor<8xf32>)
    -> tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>> {

  %div = "onnx.Div"(%arg0, %arg1) :
      (tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>,
       tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>)
      -> tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>

  %prelu = "onnx.PRelu"(%div, %slope) :
      (tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>, tensor<8xf32>)
      -> tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>

  return %prelu : tensor<1x8x8x8x!quant.uniform<i8:f32, 0.05:-5>>

  // CHECK: "onnx.Div"(%arg0, %arg1)
  // CHECK: "onnx.PRelu"(%{{.*}}, %{{.*}})
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Test Pattern 2: Tanh + ReLU with quantized types
// CHECK-LABEL: func.func @test_quantized_tanh_relu
func.func @test_quantized_tanh_relu(%arg0: tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>> {

  %tanh = "onnx.Tanh"(%arg0) :
      (tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>

  %relu = "onnx.Relu"(%tanh) :
      (tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>

  return %relu : tensor<1x16x16x16x!quant.uniform<u8:f32, 0.02:128>>

  // Pattern doesn't apply to unary ops (Tanh requires 2 operands for QLinearEltwise)
  // CHECK: %[[TANH:.*]] = "onnx.Tanh"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[TANH]])
  // CHECK: return %[[RELU]]
}

// -----
// Test Pattern 2: Sqrt + ReLU with quantized types
// CHECK-LABEL: func.func @test_quantized_sqrt_relu
func.func @test_quantized_sqrt_relu(%arg0: tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>)
    -> tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>> {

  %sqrt = "onnx.Sqrt"(%arg0) :
      (tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>)
      -> tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>

  %relu = "onnx.Relu"(%sqrt) :
      (tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>)
      -> tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>

  return %relu : tensor<1x32x32x32x!quant.uniform<i8:f32, 0.01:0>>

  // Pattern doesn't apply to unary ops (Sqrt requires 2 operands for QLinearEltwise)
  // CHECK: %[[SQRT:.*]] = "onnx.Sqrt"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[SQRT]])
  // CHECK: return %[[RELU]]
}

// -----
// Test Pattern 3: BFloat16 Intermediate with ReLU
// Add with BF16 intermediate, then ReLU with INT8 output
// CHECK-LABEL: func.func @test_bf16_add_relu
func.func @test_bf16_add_relu(%arg0: tensor<1x32x64x64xf32>,
                               %arg1: tensor<1x32x64x64xf32>)
    -> tensor<1x32x64x64x!quant.uniform<i8:f32, 0.015:0>> {

  // Add outputs BF16 (intermediate representation)
  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x32x64x64xf32>, tensor<1x32x64x64xf32>)
      -> tensor<1x32x64x64xbf16>

  // ReLU produces quantized INT8 output
  %relu = "onnx.Relu"(%add) :
      (tensor<1x32x64x64xbf16>)
      -> tensor<1x32x64x64x!quant.uniform<i8:f32, 0.015:0>>

  return %relu : tensor<1x32x64x64x!quant.uniform<i8:f32, 0.015:0>>

  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) {{.*}} -> tensor<1x32x64x64xf32>
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[ADD]])
  // CHECK: return %[[RELU]]
}

// -----
// Test Pattern 3: BFloat16 with PReLU
// CHECK-LABEL: func.func @test_bf16_add_prelu
func.func @test_bf16_add_prelu(%arg0: tensor<1x16x32x32xf32>,
                                %arg1: tensor<1x16x32x32xf32>,
                                %slope: tensor<16xf32>)
    -> tensor<1x16x32x32x!quant.uniform<i8:f32, 0.02:0>> {

  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x16x32x32xf32>, tensor<1x16x32x32xf32>)
      -> tensor<1x16x32x32xbf16>

  %prelu = "onnx.PRelu"(%add, %slope) :
      (tensor<1x16x32x32xbf16>, tensor<16xf32>)
      -> tensor<1x16x32x32x!quant.uniform<i8:f32, 0.02:0>>

  return %prelu : tensor<1x16x32x32x!quant.uniform<i8:f32, 0.02:0>>

  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) {{.*}} -> tensor<1x16x32x32xf32>
  // CHECK: %[[PRELU:.*]] = "onnx.PRelu"(%[[ADD]], %{{.*}})
  // CHECK: return %[[PRELU]]
}

// -----
// Test Pattern 3: BFloat16 with LeakyReLU
// CHECK-LABEL: func.func @test_bf16_add_leakyrelu
func.func @test_bf16_add_leakyrelu(%arg0: tensor<1x8x16x16xf32>,
                                    %arg1: tensor<1x8x16x16xf32>)
    -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.01:0>> {

  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>)
      -> tensor<1x8x16x16xbf16>

  %leaky = "onnx.LeakyRelu"(%add) {alpha = 0.1 : f32} :
      (tensor<1x8x16x16xbf16>)
      -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.01:0>>

  return %leaky : tensor<1x8x16x16x!quant.uniform<i8:f32, 0.01:0>>

  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) {{.*}} -> tensor<1x8x16x16xf32>
  // CHECK: %[[LEAKY:.*]] = "onnx.LeakyRelu"(%[[ADD]])
  // CHECK: return %[[LEAKY]]
}

// -----
// Test Pattern 4: Post-Quantized ReLU (IPU Strix) - Add -> ReLU
// RUN: onnx-mlir-opt --replace-qdq-eltwise="enable-ipu-strix=true" %s | FileCheck %s --check-prefix=STRIX
// Both operations are float but should stay quantized
// STRIX-LABEL: func.func @test_strix_add_relu
func.func @test_strix_add_relu(%arg0: tensor<1x16x28x28xf32>,
                                %arg1: tensor<1x16x28x28xf32>)
    -> tensor<1x16x28x28xf32> {

  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x16x28x28xf32>, tensor<1x16x28x28xf32>)
      -> tensor<1x16x28x28xf32>

  %relu = "onnx.Relu"(%add) :
      (tensor<1x16x28x28xf32>)
      -> tensor<1x16x28x28xf32>

  return %relu : tensor<1x16x28x28xf32>

  // STRIX: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) {strix_keep_quantized = true}
  // STRIX: %[[RELU:.*]] = "onnx.Relu"(%[[ADD]]) {strix_keep_quantized = true}
  // STRIX: return %[[RELU]]
}

// -----
// Test Pattern 4: Post-Quantized ReLU (IPU Strix) - Mul -> ReLU
// STRIX-LABEL: func.func @test_strix_mul_relu
func.func @test_strix_mul_relu(%arg0: tensor<1x8x14x14xf32>,
                                %arg1: tensor<1x8x14x14xf32>)
    -> tensor<1x8x14x14xf32> {

  %mul = "onnx.Mul"(%arg0, %arg1) :
      (tensor<1x8x14x14xf32>, tensor<1x8x14x14xf32>)
      -> tensor<1x8x14x14xf32>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x8x14x14xf32>)
      -> tensor<1x8x14x14xf32>

  return %relu : tensor<1x8x14x14xf32>

  // STRIX: %[[MUL:.*]] = "onnx.Mul"(%arg0, %arg1) {strix_keep_quantized = true}
  // STRIX: %[[RELU:.*]] = "onnx.Relu"(%[[MUL]]) {strix_keep_quantized = true}
  // STRIX: return %[[RELU]]
}

// -----
// Test: Operations with matching quantization parameters
// CHECK-LABEL: func.func @test_matching_quantization
func.func @test_matching_quantization(%arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>,
                                       %arg1: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>> {

  // All inputs and output have matching quantization parameters
  %add = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>

  return %add : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Clip (no activation) -> fused CLIP
// CHECK-LABEL: func.func @test_quantized_clip_no_activation
func.func @test_quantized_clip_no_activation(
    %arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>)
    -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>> {
  %cmin = "onnx.Constant"() {value = dense<-128> : tensor<i64>} : () -> tensor<i64>
  %cmax = "onnx.Constant"() {value = dense<127> : tensor<i64>} : () -> tensor<i64>
  %clip = "onnx.Clip"(%arg0, %cmin, %cmax) :
      (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>, tensor<i64>, tensor<i64>)
      -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>
  return %clip : tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]){{.*}}clip_max = 127 : si64{{.*}}clip_min = -128 : si64{{.*}}nonlinear = "NONE"{{.*}}type = "CLIP"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Exp (no activation) -> fused EXP
// CHECK-LABEL: func.func @test_quantized_exp_no_activation
func.func @test_quantized_exp_no_activation(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %exp = "onnx.Exp"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %exp : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "EXP"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Elu (no activation) -> fused ELU
// CHECK-LABEL: func.func @test_quantized_elu_no_activation
func.func @test_quantized_elu_no_activation(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %elu = "onnx.Elu"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %elu : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ELU"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Sin (no activation) -> fused SIN
// CHECK-LABEL: func.func @test_quantized_sin_no_activation
func.func @test_quantized_sin_no_activation(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %sin = "onnx.Sin"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %sin : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "SIN"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Cos (no activation) -> fused COS
// CHECK-LABEL: func.func @test_quantized_cos_no_activation
func.func @test_quantized_cos_no_activation(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %cos = "onnx.Cos"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %cos : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "COS"
  // CHECK: return %[[FUSED]]
}

// -----
// Test: Pattern 1 + Pattern 2 in same function
//  - Exp (no activation) should fuse with nonlinear="NONE", type="EXP"
//  - Add + Relu should fuse with nonlinear="RELU", type="ADD"
// CHECK-LABEL: func.func @test_pattern1_and_pattern2
func.func @test_pattern1_and_pattern2(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>,
    %arg1: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
    %arg2: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>,
        tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>) {
  %exp = "onnx.Exp"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  %add = "onnx.Add"(%arg1, %arg2) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
       tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>
  %relu = "onnx.Relu"(%add) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  return %exp, %relu :
      tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>,
      tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK-DAG: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK-DAG: %[[EXP:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]){{.*}}nonlinear = "NONE"{{.*}}type = "EXP"
  // CHECK-DAG: %[[ADDRELU:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg1, %arg2){{.*}}nonlinear = "RELU"{{.*}}type = "ADD"
  // CHECK: return %[[EXP]], %[[ADDRELU]]
}

// -----
// Test: Pattern 1 (CLIP) + Pattern 2 in same function
// CHECK-LABEL: func.func @test_pattern1_clip_and_pattern2
func.func @test_pattern1_clip_and_pattern2(
    %arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>,
    %arg1: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
    %arg2: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
    -> (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>,
        tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>) {
  %cmin = "onnx.Constant"() {value = dense<-128> : tensor<i64>} : () -> tensor<i64>
  %cmax = "onnx.Constant"() {value = dense<127> : tensor<i64>} : () -> tensor<i64>
  %clip = "onnx.Clip"(%arg0, %cmin, %cmax) :
      (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>, tensor<i64>, tensor<i64>)
      -> tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>

  %sub = "onnx.Sub"(%arg1, %arg2) :
      (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>
  %relu = "onnx.Relu"(%sub) :
      (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  return %clip, %relu :
      tensor<1x4x8x8x!quant.uniform<i8:f32, 0.01:0>>,
      tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-DAG: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK-DAG: %[[CLIP:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]){{.*}}clip_max = 127 : si64{{.*}}clip_min = -128 : si64{{.*}}nonlinear = "NONE"{{.*}}type = "CLIP"
  // CHECK-DAG: %[[SUBRELU:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg1, %arg2){{.*}}nonlinear = "RELU"{{.*}}type = "SUB"
  // CHECK: return %[[CLIP]], %[[SUBRELU]]
}

// -----
// Test: Multiple fusions in same function
// CHECK-LABEL: func.func @test_multiple_fusions
func.func @test_multiple_fusions(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
    %arg1: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
    %arg2: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
        tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>) {

  // First fusion: Add + ReLU
  %add1 = "onnx.Add"(%arg0, %arg1) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
       tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  %relu1 = "onnx.Relu"(%add1) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // Second fusion: Mul + ReLU
  %mul = "onnx.Mul"(%arg1, %arg2) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
       tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  %relu2 = "onnx.Relu"(%mul) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  return %relu1, %relu2 :
      tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
      tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: %[[FUSED_ADD:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: nonlinear = "RELU"
  // CHECK-SAME: type = "ADD"
  // CHECK: %[[FUSED_MUL:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg1, %arg2)
  // CHECK-SAME: nonlinear = "RELU"
  // CHECK-SAME: type = "MUL"
  // CHECK: return %[[FUSED_ADD]], %[[FUSED_MUL]]
}