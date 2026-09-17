// RUN: onnx-mlir-opt --march=arm64 --shape-inference --fusion-op-transform --convert-onnx-to-krnl="enable-simd=true" %s -split-input-file | FileCheck %s

// Tests for the Krnl lowering of onnx.Fused(kind="simd-split-op-gather"),
// see src/Conversion/ONNXToKrnl/Tensor/FusedSplitOpGather.cpp. Checks for
// exactly one memref.alloc (no intermediate Slice/Concat buffers) and
// vector.load/vector.store (real SIMD, not scalar krnl.load/krnl.store) on
// the aligned portion of each half.

// -----

// The RoPE rotate_half idiom, exactly as seen in Granite-4 -- including the
// INT64_MAX "slice to the end of the axis" sentinel. Neg is on the high
// half; Concat reorders the halves (high first), so the negated high half
// is written at output offset 0 and the untouched low half at offset 64.

func.func @rotate_half_int64_max_sentinel(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
  return %3 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @rotate_half_int64_max_sentinel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4x8x128xf32>) -> memref<1x4x8x128xf32> {
// Exactly one allocation -- no intermediate Slice/Concat buffers.
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<1x4x8x128xf32>
// CHECK:           krnl.iterate
// Low half: plain SIMD copy from [0,64) to output offset 64. Each 64-element
// half is a multiple of the register-pressure-driven unroll factor (totVL =
// archVL(4) * unrollVL(8) = 32 for this cheap, single-op-mix kernel on
// arm64/f32), so it is one single, fully-SIMD 32-wide block, no remainder.
// CHECK:             [[LOOP_LOW_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_LOW_:%.+]], {{.*}} = krnl.block [[LOOP_LOW_]] 32
// CHECK:             krnl.iterate([[BLOCK_LOW_]])
// CHECK:               [[LOW_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:               vector.store [[LOW_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// High half: SIMD load from offset 64, negate, store to output offset 0.
// CHECK:             [[LOOP_HIGH_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_HIGH_:%.+]], {{.*}} = krnl.block [[LOOP_HIGH_]] 32
// CHECK:             krnl.iterate([[BLOCK_HIGH_]])
// CHECK:               [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[HIGH_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Binary per-half op with an external, same-shape operand -- also confirms
// the external operand's memref (not just the shared source) is correctly
// SIMD-loaded, and that the multiply produces a vector-typed (not
// mistakenly scalar-typed) result. Halves are 64 elements (rather than a
// bare archVL of 4) so the register-pressure unroll model actually kicks in
// -- see the "too small to unroll" fallback exercised deliberately by
// small_half_falls_back_to_scalar below.

func.func @scaled_half_with_external_operand(%arg0: tensor<2x128xf32>, %scale: tensor<2x64xf32>) -> tensor<2x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c1axis, %c1) : (tensor<2x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c1axis, %c1) : (tensor<2x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x64xf32>
  %2 = "onnx.Mul"(%1, %scale) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x128xf32>
  return %3 : tensor<2x128xf32>

// CHECK-LABEL:  func.func @scaled_half_with_external_operand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x128xf32>, [[PARAM_1_:%.+]]: memref<2x64xf32>) -> memref<2x128xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<2x128xf32>
// CHECK:           krnl.iterate
// Low half: plain SIMD copy.
// CHECK:               vector.load [[PARAM_0_]][{{.*}}] : memref<2x128xf32>, vector<32xf32>
// High half: SIMD load of both the shared source and the external operand,
// vector-typed multiply, SIMD store.
// CHECK:               [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<2x128xf32>, vector<32xf32>
// CHECK:               [[SCALE_VAL_:%.+]] = vector.load [[PARAM_1_]][{{.*}}] : memref<2x64xf32>, vector<32xf32>
// CHECK:               [[MUL_VAL_:%.+]] = arith.mulf [[HIGH_VAL_]], [[SCALE_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[MUL_VAL_]], [[ALLOC_]][{{.*}}] : memref<2x128xf32>, vector<32xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Halves of exactly one hardware vector width (archVL=4 on arm64/f32) with
// no room to safely overcompute (the two halves share one output buffer, so
// canOverCompute is always false here) fall below the register-pressure
// model's "worth it" threshold and cleanly fall back to plain scalar
// krnl.load/krnl.store -- exactly what a standalone elementwise op of the
// same size on a function-argument input (also not overcompute-safe) would
// do under the same shared cost model.

func.func @small_half_falls_back_to_scalar(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Neg"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @small_half_falls_back_to_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x8xf32>) -> memref<2x8xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<2x8xf32>
// CHECK:           krnl.iterate
// Low half: plain scalar copy, no vector.load/krnl.block.
// CHECK-NOT:       vector.load
// CHECK:               [[LOW_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<2x8xf32>
// CHECK:               krnl.store [[LOW_VAL_]], [[ALLOC_]][{{.*}}] : memref<2x8xf32>
// High half: scalar load, negate, store.
// CHECK:               [[HIGH_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<2x8xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[HIGH_VAL_]] : f32
// CHECK:               krnl.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<2x8xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Half length (50) is a multiple of archVL (4) but not of the full
// register-pressure unroll factor (totVL = 32) -- exercises simdIterateIE's
// own scalar-remainder tail: one 32-wide SIMD block over the aligned
// prefix, then a plain scalar loop for the last 18 elements. No new
// remainder-handling code was needed for this: simdIterateIE (shared with
// plain elementwise codegen) already does this whenever `fullySimd` is
// false.

func.func @half_len_not_multiple_of_unroll_factor(%arg0: tensor<1x100xf32>) -> tensor<1x100xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c50 = onnx.Constant dense<50> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c50, %c1axis, %c1) : (tensor<1x100xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x50xf32>
  %1 = "onnx.Slice"(%arg0, %c50, %cmax, %c1axis, %c1) : (tensor<1x100xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x50xf32>
  %2 = "onnx.Neg"(%1) : (tensor<1x50xf32>) -> tensor<1x50xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<1x50xf32>, tensor<1x50xf32>) -> tensor<1x100xf32>
  return %3 : tensor<1x100xf32>

// CHECK-LABEL:  func.func @half_len_not_multiple_of_unroll_factor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x100xf32>) -> memref<1x100xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<1x100xf32>
// CHECK:           krnl.iterate
// Low half (plain copy): SIMD main block over the aligned 32-element
// prefix, then a plain scalar krnl.load/krnl.store loop over the remaining
// 18 elements -- simdIterateIE's own remainder handling, shared with plain
// elementwise codegen, kicking in because 50 is not a multiple of totVL(32).
// CHECK:               [[LOOP_LOW_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_LOW_:%.+]], {{.*}} = krnl.block [[LOOP_LOW_]] 32
// CHECK:               krnl.iterate([[BLOCK_LOW_]])
// CHECK:                 [[LOW_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x100xf32>, vector<32xf32>
// CHECK:                 vector.store [[LOW_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x100xf32>, vector<32xf32>
// CHECK:               [[LOOP_LOW_REM_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_LOW_REM_]])
// CHECK:                 [[LOW_REM_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<1x100xf32>
// CHECK:                 krnl.store [[LOW_REM_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x100xf32>
// High half (negate): same SIMD-main-block + scalar-remainder split, with
// the negate applied in both the vector and scalar tail.
// CHECK:               [[LOOP_HIGH_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_HIGH_:%.+]], {{.*}} = krnl.block [[LOOP_HIGH_]] 32
// CHECK:               krnl.iterate([[BLOCK_HIGH_]])
// CHECK:                 [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x100xf32>, vector<32xf32>
// CHECK:                 [[NEG_VAL_:%.+]] = arith.negf [[HIGH_VAL_]] : vector<32xf32>
// CHECK:                 vector.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x100xf32>, vector<32xf32>
// CHECK:               [[LOOP_HIGH_REM_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_HIGH_REM_]])
// CHECK:                 [[HIGH_REM_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<1x100xf32>
// CHECK:                 [[NEG_REM_VAL_:%.+]] = arith.negf [[HIGH_REM_VAL_]] : f32
// CHECK:                 krnl.store [[NEG_REM_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x100xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Op on the low half only -- high half is a plain SIMD copy, low half gets
// the negate. Exercises hasOpForSplitLow (every other case above only
// exercises hasOpForSplitHigh).

func.func @op_on_low_half_only(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %2 = "onnx.Neg"(%0) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
  %3 = "onnx.Concat"(%2, %1) {axis = 3 : si64} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
  return %3 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @op_on_low_half_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4x8x128xf32>) -> memref<1x4x8x128xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<1x4x8x128xf32>
// CHECK:           krnl.iterate
// Low half: SIMD load, negate, store -- at output offset 0 (unchanged).
// CHECK:               [[LOW_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[LOW_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// High half: plain SIMD copy from offset 64 to offset 64.
// CHECK:               [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:               vector.store [[HIGH_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>, vector<32xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Both halves transformed by different unary ops (Neg on low, Relu on
// high) -- exercises hasOpForSplitLow and hasOpForSplitHigh together, with
// no external operand on either side.

func.func @both_halves_transformed(%arg0: tensor<3x4x8x128xf32>) -> tensor<3x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<3x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<3x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x4x8x64xf32>
  %2 = "onnx.Neg"(%0) : (tensor<3x4x8x64xf32>) -> tensor<3x4x8x64xf32>
  %3 = "onnx.Relu"(%1) : (tensor<3x4x8x64xf32>) -> tensor<3x4x8x64xf32>
  %4 = "onnx.Concat"(%2, %3) {axis = 3 : si64} : (tensor<3x4x8x64xf32>, tensor<3x4x8x64xf32>) -> tensor<3x4x8x128xf32>
  return %4 : tensor<3x4x8x128xf32>

// CHECK-LABEL:  func.func @both_halves_transformed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x8x128xf32>) -> memref<3x4x8x128xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<3x4x8x128xf32>
// CHECK:           krnl.iterate
// Low half: SIMD load, negate, store.
// CHECK:               [[LOW_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<3x4x8x128xf32>, vector<32xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[LOW_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<3x4x8x128xf32>, vector<32xf32>
// High half: SIMD load, relu (max with a zero splat), store.
// CHECK:               [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<3x4x8x128xf32>, vector<32xf32>
// CHECK:               [[RELU_VAL_:%.+]] = arith.maxnumf {{.*}}, [[HIGH_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[RELU_VAL_]], [[ALLOC_]][{{.*}}] : memref<3x4x8x128xf32>, vector<32xf32>
// CHECK:           return [[ALLOC_]]
}

// -----

// Dynamic batch and sequence-length dims (the exact tensor<?x4x?x128xf32>
// shape from the real Granite-4 IR), static split axis/point -- the outer
// (non-split) loop bound is now a dynamic memref.dim read instead of a
// literal, but the split axis itself is still handled by literal-offset
// SIMD loops exactly as in the fully-static cases above.

func.func @dynamic_non_split_dims(%arg0: tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<?x4x?x64xf32>) -> tensor<?x4x?x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<?x4x?x64xf32>, tensor<?x4x?x64xf32>) -> tensor<?x4x?x128xf32>
  return %3 : tensor<?x4x?x128xf32>

// CHECK-LABEL:  func.func @dynamic_non_split_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x?x128xf32>) -> memref<?x4x?x128xf32> {
// Output alloc is sized from dynamic dims read off the input, not literals.
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]]
// CHECK:           [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]]
// CHECK:           [[ALLOC_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) {{.*}} : memref<?x4x?x128xf32>
// CHECK:           krnl.iterate
// Low half: plain SIMD copy to output offset 64.
// CHECK:               [[LOW_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<?x4x?x128xf32>, vector<32xf32>
// CHECK:               vector.store [[LOW_VAL_]], [[ALLOC_]][{{.*}}] : memref<?x4x?x128xf32>, vector<32xf32>
// High half: SIMD load, negate, store to output offset 0.
// CHECK:               [[HIGH_VAL_:%.+]] = vector.load [[PARAM_0_]][{{.*}}] : memref<?x4x?x128xf32>, vector<32xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[HIGH_VAL_]] : vector<32xf32>
// CHECK:               vector.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<?x4x?x128xf32>, vector<32xf32>
// CHECK:           return [[ALLOC_]]
}
