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

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]) {{.*}}nonlinear = "RELU"{{.*}}type = "TANH"
  // CHECK: return %[[FUSED]]
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

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]) {{.*}}nonlinear = "RELU"{{.*}}type = "SQRT"
  // CHECK: return %[[FUSED]]
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
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]){{.*}}max = 127 : i32{{.*}}min = -128 : i32{{.*}}nonlinear = "NONE"{{.*}}type = "CLAMP"
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
// Test Pattern 1: Mod (no activation) -> fused MOD
// CHECK-LABEL: func.func @test_quantized_mod_no_activation
func.func @test_quantized_mod_no_activation(
    %arg0: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>> {
  %mod = "onnx.Mod"(%arg0, %arg1) :
      (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>
  return %mod : tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1)
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "MOD"
  // CHECK-NOT: onnx.Mod
}

// -----
// Test Pattern 1: Standalone Relu (u8, zp!=0) -> fused RELU
// CHECK-LABEL: func.func @test_quantized_relu_standalone
func.func @test_quantized_relu_standalone(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %relu = "onnx.Relu"(%arg0) :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %relu : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "RELU"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Standalone LeakyRelu -> fused LEAKYRELU with alpha attrs
// CHECK-LABEL: func.func @test_quantized_leakyrelu_standalone
func.func @test_quantized_leakyrelu_standalone(
    %arg0: tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
    -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>> {
  %leaky = "onnx.LeakyRelu"(%arg0) {alpha = 0.01 : f32} :
      (tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>)
      -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>
  return %leaky : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.02:128>>

  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: leakyrelu_alpha = {{[0-9.e+-]+}} : f32
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: prelu_in = 3 : si64
  // CHECK-SAME: prelu_shift = 8 : si64
  // CHECK-SAME: type = "LEAKYRELU"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 1: Quantized Sigmoid  -> fused QLINEARSIGMOID
// (moved from replace-qdq-sigmoid; simple case handled by replace-qdq-eltwise.)
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
  // CHECK-SAME: type = "SIGMOID"
  // CHECK: return %[[FUSED]]
}

// -----
// Negative: non-quantized Sigmoid is not replaced by replace-qdq-eltwise
// CHECK-LABEL: func.func @test_sigmoid_float_not_replaced
func.func @test_sigmoid_float_not_replaced(%arg0: tensor<1x8x8xf32>) -> tensor<1x8x8xf32> {

  %0 = "onnx.Sigmoid"(%arg0) :
      (tensor<1x8x8xf32>) -> tensor<1x8x8xf32>

  return %0 : tensor<1x8x8xf32>

  // CHECK: "onnx.Sigmoid"(%arg0)
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Test Pattern 1: Equal - quantized inputs, bool output (standard ONNX).
// Pass does not fuse (result not quantized); op remains.
// CHECK-LABEL: func.func @test_quantized_equal_no_activation
func.func @test_quantized_equal_no_activation(
    %arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x4xi1> {
  %eq = "onnx.Equal"(%arg0, %arg1) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x4xi1>
  return %eq : tensor<1x4x4x4xi1>

  // Result is bool, so pass does not fuse (requires quantized result).
  // CHECK: "onnx.Equal"(%arg0, %arg1)
  // CHECK: return
}

// -----
// Test Pattern 1: Less - quantized inputs, bool output (standard ONNX).
// Pass does not fuse (result not quantized); op remains.
// CHECK-LABEL: func.func @test_quantized_less_no_activation
func.func @test_quantized_less_no_activation(
    %arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x4xi1> {
  %less = "onnx.Less"(%arg0, %arg1) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x4xi1>
  return %less : tensor<1x4x4x4xi1>

  // CHECK: "onnx.Less"(%arg0, %arg1)
  // CHECK: return
}

// -----
// Test Pattern 1: Greater - quantized inputs, bool output (standard ONNX).
// Pass does not fuse (result not quantized); op remains.
// CHECK-LABEL: func.func @test_quantized_greater_no_activation
func.func @test_quantized_greater_no_activation(
    %arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x4xi1> {
  %greater = "onnx.Greater"(%arg0, %arg1) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x4xi1>
  return %greater : tensor<1x4x4x4xi1>

  // CHECK: "onnx.Greater"(%arg0, %arg1)
  // CHECK: return
}

// -----
// Test Pattern 1: LessOrEqual - quantized inputs, bool output (standard ONNX).
// Pass does not fuse (result not quantized); op remains.
// CHECK-LABEL: func.func @test_quantized_less_or_equal_no_activation
func.func @test_quantized_less_or_equal_no_activation(
    %arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x4xi1> {
  %le = "onnx.LessOrEqual"(%arg0, %arg1) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x4xi1>
  return %le : tensor<1x4x4x4xi1>

  // CHECK: "onnx.LessOrEqual"(%arg0, %arg1)
  // CHECK: return
}

// -----
// Test Pattern 1: GreaterOrEqual - quantized inputs, bool output (standard ONNX).
// Pass does not fuse (result not quantized); op remains.
// CHECK-LABEL: func.func @test_quantized_greater_or_equal_no_activation
func.func @test_quantized_greater_or_equal_no_activation(
    %arg0: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
    %arg1: tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x4x4xi1> {
  %ge = "onnx.GreaterOrEqual"(%arg0, %arg1) :
      (tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>,
       tensor<1x4x4x4x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x4x4xi1>
  return %ge : tensor<1x4x4x4xi1>

  // CHECK: "onnx.GreaterOrEqual"(%arg0, %arg1)
  // CHECK: return
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
  // CHECK-DAG: %[[CLIP:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]]){{.*}}max = 127 : i32{{.*}}min = -128 : i32{{.*}}nonlinear = "NONE"{{.*}}type = "CLAMP"
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

// -----
// Test Pattern 5: Replace quantized Expand with Eltwise ADD (4D, C=1, W=1)
// Input shape [1,1,1,1] -> Expand to [1,1,32,32]
// Should create XCOMPILERFusedEltwise(input, zeros, type="ADD")
// CHECK-LABEL: func.func @test_expand_to_eltwise_4d
func.func @test_expand_to_eltwise_4d(
    %arg0: tensor<1x1x1x1x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x1x32x32x!quant.uniform<u8:f32, 0.05:128>> {
  %shape = "onnx.Constant"() {value = dense<[1, 1, 32, 32]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x1x1x1x!quant.uniform<u8:f32, 0.05:128>>, tensor<4xi64>)
      -> tensor<1x1x32x32x!quant.uniform<u8:f32, 0.05:128>>
  return %expand : tensor<1x1x32x32x!quant.uniform<u8:f32, 0.05:128>>

  // CHECK: %[[ZEROS:.*]] = onnx.Constant {value = dense<0> : tensor<1x1x32x32xui8>}
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[ZEROS]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
// Test Pattern 5: Replace quantized Expand with i8 quantized type (4D, C=1, W=1)
// CHECK-LABEL: func.func @test_expand_to_eltwise_i8
func.func @test_expand_to_eltwise_i8(
    %arg0: tensor<1x1x1x1x!quant.uniform<i8:f32, 0.03:0>>)
    -> tensor<1x1x16x16x!quant.uniform<i8:f32, 0.03:0>> {
  %shape = "onnx.Constant"() {value = dense<[1, 1, 16, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x1x1x1x!quant.uniform<i8:f32, 0.03:0>>, tensor<4xi64>)
      -> tensor<1x1x16x16x!quant.uniform<i8:f32, 0.03:0>>
  return %expand : tensor<1x1x16x16x!quant.uniform<i8:f32, 0.03:0>>

  // CHECK: %[[ZEROS:.*]] = onnx.Constant {value = dense<0> : tensor<1x1x16x16xi8>}
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[ZEROS]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
// Pattern 5: 4D input with dim[2] == 1 should fire even if W (dim 3) != 1
// (gate is now only on dim[2]; matches xcompiler's ReplaceQDQExpandToEltwisePass)
// CHECK-LABEL: func.func @test_expand_match_w_not_one
func.func @test_expand_match_w_not_one(
    %arg0: tensor<1x1x1x4x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x1x8x4x!quant.uniform<u8:f32, 0.05:128>> {
  %shape = "onnx.Constant"() {value = dense<[1, 1, 8, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x1x1x4x!quant.uniform<u8:f32, 0.05:128>>, tensor<4xi64>)
      -> tensor<1x1x8x4x!quant.uniform<u8:f32, 0.05:128>>
  return %expand : tensor<1x1x8x4x!quant.uniform<u8:f32, 0.05:128>>

  // CHECK: %[[ZEROS:.*]] = onnx.Constant {value = dense<0> : tensor<1x1x8x4xui8>}
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[ZEROS]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
// Pattern 5: 4D input with dim[2] == 1 should fire even if C (dim 1) != 1
// (gate is now only on dim[2]; matches xcompiler's ReplaceQDQExpandToEltwisePass)
// CHECK-LABEL: func.func @test_expand_match_c_not_one
func.func @test_expand_match_c_not_one(
    %arg0: tensor<1x16x1x1x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.05:128>> {
  %shape = "onnx.Constant"() {value = dense<[1, 16, 32, 32]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x16x1x1x!quant.uniform<u8:f32, 0.05:128>>, tensor<4xi64>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.05:128>>
  return %expand : tensor<1x16x32x32x!quant.uniform<u8:f32, 0.05:128>>

  // CHECK: %[[ZEROS:.*]] = onnx.Constant {value = dense<0> : tensor<1x16x32x32xui8>}
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[ZEROS]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
// Negative: Pattern 5 should NOT fire when 4D input has dim[2] != 1
// (only dim[2] is gated under the new behavior)
// CHECK-LABEL: func.func @test_expand_no_match_dim2_not_one
func.func @test_expand_no_match_dim2_not_one(
    %arg0: tensor<1x1x4x1x!quant.uniform<u8:f32, 0.05:128>>)
    -> tensor<1x1x4x8x!quant.uniform<u8:f32, 0.05:128>> {
  %shape = "onnx.Constant"() {value = dense<[1, 1, 4, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x1x4x1x!quant.uniform<u8:f32, 0.05:128>>, tensor<4xi64>)
      -> tensor<1x1x4x8x!quant.uniform<u8:f32, 0.05:128>>
  return %expand : tensor<1x1x4x8x!quant.uniform<u8:f32, 0.05:128>>

  // CHECK: "onnx.Expand"
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Negative: Pattern 5 should NOT fire on non-quantized Expand
// CHECK-LABEL: func.func @test_expand_float_not_replaced
func.func @test_expand_float_not_replaced(
    %arg0: tensor<1x16x1x1xf32>)
    -> tensor<1x16x32x32xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, 16, 32, 32]> : tensor<4xi64>} : () -> tensor<4xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x16x1x1xf32>, tensor<4xi64>)
      -> tensor<1x16x32x32xf32>
  return %expand : tensor<1x16x32x32xf32>

  // CHECK: "onnx.Expand"
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
}

// -----
// Test Pattern 5: Non-4D tensor (3D) should also be replaced
// CHECK-LABEL: func.func @test_expand_to_eltwise_3d
func.func @test_expand_to_eltwise_3d(
    %arg0: tensor<1x1x8x!quant.uniform<i8:f32, 0.02:0>>)
    -> tensor<4x16x8x!quant.uniform<i8:f32, 0.02:0>> {
  %shape = "onnx.Constant"() {value = dense<[4, 16, 8]> : tensor<3xi64>} : () -> tensor<3xi64>
  %expand = "onnx.Expand"(%arg0, %shape) :
      (tensor<1x1x8x!quant.uniform<i8:f32, 0.02:0>>, tensor<3xi64>)
      -> tensor<4x16x8x!quant.uniform<i8:f32, 0.02:0>>
  return %expand : tensor<4x16x8x!quant.uniform<i8:f32, 0.02:0>>

  // CHECK: %[[ZEROS:.*]] = onnx.Constant {value = dense<0> : tensor<4x16x8xi8>}
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[ZEROS]])
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "ADD"
  // CHECK: return %[[FUSED]]
}

// -----
//===----------------------------------------------------------------------===//
// maybeWidenNarrowConstOperand: i8/u8 constant operand widened to match the
// i16/u16 activation in an arithmetic ADD/MUL/SUB/DIV fusion. Scale and
// zero_point are preserved on the widened type; the original narrow constant
// becomes dead and is removed.
//===----------------------------------------------------------------------===//

// -----
// Signed i16 activation x signed i8 single-use constant in MUL fusion.
// Helper sign-extends the constant to i16, emits a new wider onnx.Constant,
// and the XCOMPILERFusedEltwise consumes the i16 operand. No i8 storage
// quant type remains in the fused op's operand list.
// CHECK-LABEL: func.func @widen_signed_i8_const_to_i16_in_mul
func.func @widen_signed_i8_const_to_i16_in_mul(
    %arg0: tensor<1x16x32x32x!quant.uniform<i16:f32, 0.05:0>>)
    -> tensor<1x16x32x32x!quant.uniform<i16:f32, 0.05:0>> {
  %c = "onnx.Constant"() {value = dense<5> : tensor<1x16x1x1xi8>} :
      () -> tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>
  %m = "onnx.Mul"(%arg0, %c) :
      (tensor<1x16x32x32x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>)
      -> tensor<1x16x32x32x!quant.uniform<i16:f32, 0.05:0>>
  return %m : tensor<1x16x32x32x!quant.uniform<i16:f32, 0.05:0>>

  // CHECK: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 2.500000e-01>>
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: type = "MUL"
  // CHECK-NOT: !quant.uniform<i8
}

// -----
// Unsigned u16 activation + unsigned u8 single-use constant in ADD fusion.
// Helper zero-extends the constant to u16 and rewires the fused op.
// CHECK-LABEL: func.func @widen_unsigned_u8_const_to_u16_in_add
func.func @widen_unsigned_u8_const_to_u16_in_add(
    %arg0: tensor<1x8x4x4x!quant.uniform<u16:f32, 0.04:32768>>)
    -> tensor<1x8x4x4x!quant.uniform<u16:f32, 0.04:32768>> {
  %c = "onnx.Constant"() {value = dense<200> : tensor<1x8x1x1xui8>} :
      () -> tensor<1x8x1x1x!quant.uniform<u8:f32, 0.04:128>>
  %a = "onnx.Add"(%arg0, %c) :
      (tensor<1x8x4x4x!quant.uniform<u16:f32, 0.04:32768>>,
       tensor<1x8x1x1x!quant.uniform<u8:f32, 0.04:128>>)
      -> tensor<1x8x4x4x!quant.uniform<u16:f32, 0.04:32768>>
  return %a : tensor<1x8x4x4x!quant.uniform<u16:f32, 0.04:32768>>

  // CHECK: onnx.Constant {{.*}} : tensor<1x8x1x1x!quant.uniform<u16:f32, 4.000000e-02:128>>
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: type = "ADD"
  // CHECK-NOT: !quant.uniform<u8
}

// -----
// Both operands are already i16: helper is a no-op (aW == bW). Fusion still
// proceeds; the constant tensor in the fused op stays at i16 storage.
// CHECK-LABEL: func.func @widen_skipped_same_width_i16
func.func @widen_skipped_same_width_i16(
    %arg0: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>)
    -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>> {
  %c = "onnx.Constant"() {value = dense<300> : tensor<1x16x1x1xi16>} :
      () -> tensor<1x16x1x1x!quant.uniform<i16:f32, 0.25:0>>
  %m = "onnx.Mul"(%arg0, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i16:f32, 0.25:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  return %m : tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>

  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: type = "MUL"
  // CHECK-NOT: !quant.uniform<i8
  // CHECK-NOT: !quant.uniform<u8
}

// -----
// Multi-use narrow constant: a single widened i16 constant is created and
// shared by all users. The original i8 constant is erased.
// CHECK-LABEL: func.func @widen_multi_use_const
func.func @widen_multi_use_const(
    %arg0: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
    %arg1: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>)
    -> (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
        tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>) {
  %c = "onnx.Constant"() {value = dense<5> : tensor<1x16x1x1xi8>} :
      () -> tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>
  %m0 = "onnx.Mul"(%arg0, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  %m1 = "onnx.Mul"(%arg1, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  return %m0, %m1 : tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
                    tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>

  // No i8 constant should remain (original erased).
  // CHECK-NOT: !quant.uniform<i8
  // Exactly one i16 constant created and reused by both fused ops.
  // CHECK: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 2.500000e-01>>
  // CHECK-NOT: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 2.500000e-01>>
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-NOT: !quant.uniform<i8
}

// -----
// Three users sharing the same i8 constant, all with i16 on the other side.
// A single widened i16 constant is created and reused by all three.
// CHECK-LABEL: func.func @widen_reuse_three_users
func.func @widen_reuse_three_users(
    %arg0: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
    %arg1: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
    %arg2: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>)
    -> (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
        tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
        tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>) {
  %c = "onnx.Constant"() {value = dense<42> : tensor<1x16x1x1xi8>} :
      () -> tensor<1x16x1x1x!quant.uniform<i8:f32, 0.1:0>>
  %m0 = "onnx.Mul"(%arg0, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.1:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  %m1 = "onnx.Add"(%arg1, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.1:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  %m2 = "onnx.Sub"(%arg2, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.1:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  return %m0, %m1, %m2 :
      tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
      tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
      tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>

  // No i8 remaining.
  // CHECK-NOT: !quant.uniform<i8
  // One i16 constant, reused by all three fused ops.
  // CHECK: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 1.000000e-01>>
  // CHECK-NOT: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 1.000000e-01>>
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-NOT: !quant.uniform<i8
}

// -----
// Mixed-width users: shared i8 constant has one user with i16 activation
// and another with i8 activation. Only the i16 user gets a widened copy;
// the i8 user keeps the original constant.
// CHECK-LABEL: func.func @widen_per_user_mixed_width
func.func @widen_per_user_mixed_width(
    %arg0: tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
    %arg1: tensor<1x16x8x8x!quant.uniform<i8:f32, 0.03:0>>)
    -> (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
        tensor<1x16x8x8x!quant.uniform<i8:f32, 0.03:0>>) {
  %c = "onnx.Constant"() {value = dense<5> : tensor<1x16x1x1xi8>} :
      () -> tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>
  %m0 = "onnx.Mul"(%arg0, %c) :
      (tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>
  %m1 = "onnx.Mul"(%arg1, %c) :
      (tensor<1x16x8x8x!quant.uniform<i8:f32, 0.03:0>>,
       tensor<1x16x1x1x!quant.uniform<i8:f32, 0.25:0>>)
      -> tensor<1x16x8x8x!quant.uniform<i8:f32, 0.03:0>>
  return %m0, %m1 : tensor<1x16x8x8x!quant.uniform<i16:f32, 0.05:0>>,
                    tensor<1x16x8x8x!quant.uniform<i8:f32, 0.03:0>>

  // i16 user gets widened constant, i8 user keeps original
  // CHECK-DAG: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i16:f32, 2.500000e-01>>
  // CHECK-DAG: onnx.Constant {{.*}} : tensor<1x16x1x1x!quant.uniform<i8:f32, 2.500000e-01>>
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
}