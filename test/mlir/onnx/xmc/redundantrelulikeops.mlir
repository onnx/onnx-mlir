// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --remove-redundant-relu-like-ops %s | FileCheck %s

// The chain collapses to a single Relu, and the surviving Relu keeps the
// chain's final (output) ResultNames "r4".
func.func @test_relu_chain(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %1 = "onnx.Relu"(%arg0) {ResultNames = ["r1"]} : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "onnx.Relu"(%1) {ResultNames = ["r2"]} : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "onnx.Relu"(%2) {ResultNames = ["r3"]} : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "onnx.Relu"(%3) {ResultNames = ["r4"]} : (tensor<1xf32>) -> tensor<1xf32>

  return %4 : tensor<1xf32>
}

// CHECK-LABEL: func.func @test_relu_chain
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0) {{.*}}ResultNames = ["r4"]
// CHECK-NOT: "onnx.Relu"(%[[R]])
// CHECK: return %[[R]]

// -----

// Inner Relu fans out: %0 feeds both the outer Relu (%1) and an independent
// consumer (Sigmoid). Because the inner Relu has more than one use, the pass
// skips it and leaves the chain untouched.
func.func @test_relu_inner_fanout(%arg0: tensor<1xf32>)
    -> (tensor<1xf32>, tensor<1xf32>) {
  %0 = "onnx.Relu"(%arg0) {ResultNames = ["inner"]} : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "onnx.Relu"(%0) {ResultNames = ["outer"]} : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "onnx.Sigmoid"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  return %1, %2 : tensor<1xf32>, tensor<1xf32>
}

// CHECK-LABEL: func.func @test_relu_inner_fanout
// CHECK: %[[INNER:.*]] = "onnx.Relu"(%arg0) {{.*}}ResultNames = ["inner"]
// CHECK: %[[OUTER:.*]] = "onnx.Relu"(%[[INNER]]) {{.*}}ResultNames = ["outer"]
// CHECK: %[[SIG:.*]] = "onnx.Sigmoid"(%[[INNER]])
// CHECK: return %[[OUTER]], %[[SIG]]

// -----

// Relu after Clip(min=0, max=6) (i.e. ReLU6): the Clip output is >= 0, so the
// Relu is a no-op and is removed; uses are rerouted to the Clip result. The
// Clip has a single use, so the Relu's output ResultNames "relu_out" transfers
// onto the surviving Clip.
func.func @test_relu_after_relu6(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %min = onnx.Constant dense<0.0> : tensor<f32>
  %max = onnx.Constant dense<6.0> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %min, %max) {ResultNames = ["clip_out"]} : (tensor<1x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8xf32>
  %1 = "onnx.Relu"(%0) {ResultNames = ["relu_out"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_after_relu6
// CHECK: %[[CLIP:.*]] = "onnx.Clip"{{.*}}ResultNames = ["relu_out"]
// CHECK-NOT: "onnx.Relu"
// CHECK: return %[[CLIP]]

// -----

// Relu after a Clip with min < 0: the Clip output can be negative, so the
// Relu is NOT redundant and must be kept.
func.func @test_relu_after_clip_negmin(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %min = onnx.Constant dense<-1.0> : tensor<f32>
  %max = onnx.Constant dense<6.0> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<1x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_after_clip_negmin
// CHECK: %[[CLIP:.*]] = "onnx.Clip"
// CHECK: %[[R:.*]] = "onnx.Relu"(%[[CLIP]])
// CHECK: return %[[R]]

// -----

// Quantized Relu that also requantizes (input scale/zp differ from output):
// even though the input (a Relu) is non-negative, removing this Relu would
// drop the requant, so it is kept.
func.func @test_relu_requant_kept(%arg0: tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.25:64>> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  %1 = "onnx.Relu"(%0) : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.25:64>>
  return %1 : tensor<1x8x!quant.uniform<u8:f32, 0.25:64>>
}
// CHECK-LABEL: func.func @test_relu_requant_kept
// CHECK: %[[R0:.*]] = "onnx.Relu"(%arg0)
// CHECK: %[[R1:.*]] = "onnx.Relu"(%[[R0]])
// CHECK: return %[[R1]]

// -----

// Quantized Relu chain with identical scale/zero-point on every edge: neither
// Relu requantizes, so the chain collapses to a single Relu.
func.func @test_relu_quant_chain_same_qparams(%arg0: tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  %1 = "onnx.Relu"(%0) : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  return %1 : tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
}
// CHECK-LABEL: func.func @test_relu_quant_chain_same_qparams
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0)
// CHECK-NOT: "onnx.Relu"(%[[R]])
// CHECK: return %[[R]]

// -----

// Relu after a Clip with min > 0 (>= 0): Clip output >= 2 > 0, so the Relu is
// redundant and removed.
func.func @test_relu_after_clip_posmin(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %min = onnx.Constant dense<2.0> : tensor<f32>
  %max = onnx.Constant dense<6.0> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<1x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_after_clip_posmin
// CHECK: %[[CLIP:.*]] = "onnx.Clip"
// CHECK-NOT: "onnx.Relu"
// CHECK: return %[[CLIP]]

// -----

// Standalone Relu whose input (a block argument) is not provably non-negative:
// the Relu is a real op and must be kept.
func.func @test_relu_standalone(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_standalone
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0)
// CHECK: return %[[R]]

// -----

// Relu after a Clip with no min operand (no lower bound): non-negativity
// cannot be proven, so the Relu is kept.
func.func @test_relu_after_clip_no_min(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %max = onnx.Constant dense<6.0> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %none, %max) : (tensor<1x8xf32>, none, tensor<f32>) -> tensor<1x8xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_after_clip_no_min
// CHECK: %[[CLIP:.*]] = "onnx.Clip"
// CHECK: %[[R:.*]] = "onnx.Relu"(%[[CLIP]])
// CHECK: return %[[R]]

// -----

// Clip(min=0) fans out to both the Relu and a Sigmoid. Unlike the Relu->Relu
// fan-out case, the redundant Relu is still removed because it is replaced by
// its (already non-negative) Clip input; the Clip stays for the other use.
func.func @test_relu_after_clip_fanout(%arg0: tensor<1x8xf32>)
    -> (tensor<1x8xf32>, tensor<1x8xf32>) {
  %min = onnx.Constant dense<0.0> : tensor<f32>
  %max = onnx.Constant dense<6.0> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<1x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %2 = "onnx.Sigmoid"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1, %2 : tensor<1x8xf32>, tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_relu_after_clip_fanout
// CHECK: %[[CLIP:.*]] = "onnx.Clip"
// CHECK-NOT: "onnx.Relu"
// CHECK: %[[SIG:.*]] = "onnx.Sigmoid"(%[[CLIP]])
// CHECK: return %[[CLIP]], %[[SIG]]

// -----

// Adjacent LeakyReLU fold: leaky(0.2, leaky(0.1, x)) -> leaky(0.1*0.2, x).
// The chain collapses to a single LeakyReLU with the product slope (0.02, in
// f32 0.0200000014). The folded op keeps the outer's output name "l1".
func.func @test_leaky_adjacent_fold(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32, ResultNames = ["l0"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32, ResultNames = ["l1"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_adjacent_fold
// CHECK: %[[L:.*]] = "onnx.LeakyRelu"(%arg0) {ResultNames = ["l1"], alpha = 0.0200000014 : f32}
// CHECK-NOT: "onnx.LeakyRelu"(%[[L]])
// CHECK: return %[[L]]

// -----

// A 3-deep LeakyReLU chain folds to one: 0.5 * 0.5 * 0.5 = 0.125. The folded
// op keeps the outermost (output) name "c2".
func.func @test_leaky_chain_fold(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.5 : f32, ResultNames = ["c0"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.5 : f32, ResultNames = ["c1"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 0.5 : f32, ResultNames = ["c2"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %2 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_chain_fold
// CHECK: %[[L:.*]] = "onnx.LeakyRelu"(%arg0) {ResultNames = ["c2"], alpha = 1.250000e-01 : f32}
// CHECK-NOT: "onnx.LeakyRelu"(%[[L]])
// CHECK: return %[[L]]

// -----

// LeakyReLU after a Relu: input is non-negative, so leaky(x) = x and the
// LeakyReLU is removed. Its single-use producer (the Relu) inherits the
// LeakyReLU's output name "lk".
func.func @test_leaky_after_relu(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.Relu"(%arg0) {ResultNames = ["r0"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.1 : f32, ResultNames = ["lk"]} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_after_relu
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0) {{.*}}ResultNames = ["lk"]
// CHECK-NOT: "onnx.LeakyRelu"
// CHECK: return %[[R]]

// -----

// Adjacent LeakyReLU separated only by a requant (different scale/zp on the
// inner input vs this op's output): the single-quant-domain guard fails, so
// the fold is NOT performed (collapsing across a requant is an accuracy
// approximation and is intentionally skipped).
func.func @test_leaky_requant_not_folded(%arg0: tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.25:64>> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.25:64>>
  return %1 : tensor<1x8x!quant.uniform<u8:f32, 0.25:64>>
}
// CHECK-LABEL: func.func @test_leaky_requant_not_folded
// CHECK: %[[L0:.*]] = "onnx.LeakyRelu"(%arg0)
// CHECK: %[[L1:.*]] = "onnx.LeakyRelu"(%[[L0]])
// CHECK: return %[[L1]]

// -----

// Two adjacent quantized LeakyReLUs in the SAME quant domain (identical
// scale/zero-point on the inner input and this op's output): the type-equality
// guard is satisfied, so they fold into a single LeakyReLU with the product
// slope (0.1*0.2 = 0.02, f32 0.0200000014).
func.func @test_leaky_quant_same_domain_fold(%arg0: tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
  return %1 : tensor<1x8x!quant.uniform<u8:f32, 0.5:128>>
}
// CHECK-LABEL: func.func @test_leaky_quant_same_domain_fold
// CHECK: %[[L:.*]] = "onnx.LeakyRelu"(%arg0) {alpha = 0.0200000014 : f32}
// CHECK-NOT: "onnx.LeakyRelu"(%[[L]])
// CHECK: return %[[L]]

// -----

// Negative-range guard (constraint 3): same-domain quantized LeakyReLUs whose
// intermediate is unsigned with zero-point at the min code (uint8 zp=0). The
// negative branch clamps to 0, so the effective slope is 0 -> the fold
// canonicalizes to ReLU, NOT LeakyReLU(product).
func.func @test_leaky_negrange_to_relu(%arg0: tensor<1x8x!quant.uniform<u8:f32, 0.5>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5>> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5>>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32} : (tensor<1x8x!quant.uniform<u8:f32, 0.5>>) -> tensor<1x8x!quant.uniform<u8:f32, 0.5>>
  return %1 : tensor<1x8x!quant.uniform<u8:f32, 0.5>>
}
// CHECK-LABEL: func.func @test_leaky_negrange_to_relu
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0)
// CHECK-NOT: "onnx.LeakyRelu"
// CHECK: return %[[R]]

// -----

// Numeric underflow guard (constraint 6): merged slope 1e-4 * 1e-4 = 1e-8 is
// below the flush-to-ReLU threshold, so the fold canonicalizes to ReLU.
func.func @test_leaky_underflow_to_relu(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.0001 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.0001 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_underflow_to_relu
// CHECK: %[[R:.*]] = "onnx.Relu"(%arg0)
// CHECK-NOT: "onnx.LeakyRelu"
// CHECK: return %[[R]]

// -----

// Fanout guard (constraint 5): the inner LeakyReLU feeds both the outer
// LeakyReLU and a Sigmoid, so it is not folded (removing it would duplicate).
func.func @test_leaky_fold_fanout_skip(%arg0: tensor<1x8xf32>)
    -> (tensor<1x8xf32>, tensor<1x8xf32>) {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %2 = "onnx.Sigmoid"(%0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1, %2 : tensor<1x8xf32>, tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_fold_fanout_skip
// CHECK: %[[L0:.*]] = "onnx.LeakyRelu"(%arg0)
// CHECK: %[[L1:.*]] = "onnx.LeakyRelu"(%[[L0]])
// CHECK: %[[S:.*]] = "onnx.Sigmoid"(%[[L0]])
// CHECK: return %[[L1]], %[[S]]

// -----

// Activation-identity guard (constraint 1): the inner LeakyReLU has a negative
// slope, which would break the composition proof, so no fold is performed.
func.func @test_leaky_negative_slope_skip(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = -0.1 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 0.2 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %1 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_negative_slope_skip
// CHECK: %[[L0:.*]] = "onnx.LeakyRelu"(%arg0)
// CHECK: %[[L1:.*]] = "onnx.LeakyRelu"(%[[L0]])
// CHECK: return %[[L1]]

// -----

// Standalone LeakyReLU on a block argument: not foldable, not removable; kept.
func.func @test_leaky_standalone(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
// CHECK-LABEL: func.func @test_leaky_standalone
// CHECK: %[[L:.*]] = "onnx.LeakyRelu"(%arg0)
// CHECK: return %[[L]]
