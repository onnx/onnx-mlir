// RUN: onnx-mlir-opt --qdq-canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --qdq-canonicalize="max-round-trip-diff=2" %s -split-input-file | FileCheck %s --check-prefix=TOL

func.func @test_qdq_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %3 : tensor<1x128x768xui16>

}

// CHECK-LABEL: func.func @test_qdq_pattern1(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: return %arg0 : tensor<1x128x768xui16>
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.QuantizeLinear

func.func @test_qdq_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = onnx.Constant dense<6.57987776E-5> : tensor<f32>
%3 = onnx.Constant dense<45664> : tensor<ui16>
%4 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%5 = "onnx.QuantizeLinear"(%4, %2, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %5 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern2(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

func.func @test_qdq_pattern3(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 2 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %3 : tensor<1x128x768xui16>

}

// CHECK-LABEL: func.func @test_qdq_pattern3(
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128x768xui16>
// CHECK-SAME: ) -> tensor<1x128x768xui16>
// CHECK-NEXT:   return %[[ARG0]] : tensor<1x128x768xui16>
// CHECK-NEXT: }

func.func @test_qdq_pattern4(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 1 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %3 : tensor<1x128x768xui16>

}

// CHECK-LABEL: func.func @test_qdq_pattern4(
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128x768xui16>
// CHECK-SAME: ) -> tensor<1x128x768xui16>
// CHECK-NEXT:   return %[[ARG0]] : tensor<1x128x768xui16>
// CHECK-NEXT: }
func.func @test_qdq_pattern6(%arg0: tensor<1x128x768xui16>, %arg1: tensor<f32>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<39664> : tensor<ui16>
%1 = "onnx.DequantizeLinear"(%arg0, %arg1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%2 = "onnx.QuantizeLinear"(%1, %arg1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %2 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern6(%arg0: tensor<1x128x768xui16>, %arg1: tensor<f32>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

func.func @test_qdq_pattern7(%arg0: tensor<1x128x768xui16>, %arg1: tensor<ui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = "onnx.DequantizeLinear"(%arg0, %0, %arg1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%2 = "onnx.QuantizeLinear"(%1, %0, %arg1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %2 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern7(%arg0: tensor<1x128x768xui16>, %arg1: tensor<ui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

func.func @test_qdq_pattern8(%arg0: tensor<1x128x768xi16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %3 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern8(%arg0: tensor<1x128x768xi16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

func.func @test_qdq_pattern9_per_axis(%arg0: tensor<1x3x1x2xui16>) -> tensor<1x3x1x2xui16> {
%0 = onnx.Constant dense<[0.01, 0.1, 0.02]> : tensor<3xf32>
%1 = onnx.Constant dense<[0, 0, 0]> : tensor<3xui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x1x2xui16>, tensor<3xf32>, tensor<3xui16>) -> tensor<1x3x1x2xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x1x2xf32>, tensor<3xf32>, tensor<3xui16>) -> tensor<1x3x1x2xui16>
return %3 : tensor<1x3x1x2xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern9_per_axis(
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x3x1x2xui16>
// CHECK-SAME: ) -> tensor<1x3x1x2xui16>
// CHECK-NEXT:   return %[[ARG0]] : tensor<1x3x1x2xui16>
// CHECK-NEXT: }



func.func @test_qdq_pattern10_per_block(%arg0: tensor<1x128xui16>) -> tensor<1x128xui16> {
%0 = onnx.Constant dense<[[0.02, 0.025, 0.03, 0.04]]> : tensor<1x4xf32>
%1 = onnx.Constant dense<[[0, 0, 0, 0]]> : tensor<1x4xui16>
%2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 32 : si64} : (tensor<1x128xui16>, tensor<1x4xf32>, tensor<1x4xui16>) -> tensor<1x128xf32>
%3 = "onnx.QuantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 32 : si64} : (tensor<1x128xf32>, tensor<1x4xf32>, tensor<1x4xui16>) -> tensor<1x128xui16>
return %3 : tensor<1x128xui16>
}

// CHECK-LABEL: func.func @test_qdq_pattern10_per_block(
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128xui16>
// CHECK-SAME: ) -> tensor<1x128xui16>
// CHECK-NEXT:   return %[[ARG0]] : tensor<1x128xui16>
// CHECK-NEXT: }

// -----

// Tolerance round-trip: DQ and Q have slightly different per-tensor scales but
// the same zero-point. Over the full ui16 range the DQ->Q round-trip differs by
// at most 2 codes (max_diff == 2). With the default exact mode (max-round-trip-
// diff=0) the pair must NOT fold; with max-round-trip-diff=2 it folds away.
func.func @test_qdq_tol_within(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = onnx.Constant dense<2.580006754E-5> : tensor<f32>
%3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%4 = "onnx.QuantizeLinear"(%3, %2, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %4 : tensor<1x128x768xui16>
}

// Default exact mode: scales differ bit-for-bit, so the pair is kept.
// CHECK-LABEL: func.func @test_qdq_tol_within(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

// Tolerance mode (max-round-trip-diff=2): round-trip diff <= 2, so it folds.
// TOL-LABEL: func.func @test_qdq_tol_within(
// TOL-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128x768xui16>
// TOL-SAME: ) -> tensor<1x128x768xui16>
// TOL:   return %[[ARG0]] : tensor<1x128x768xui16>
// TOL-NOT: onnx.DequantizeLinear
// TOL-NOT: onnx.QuantizeLinear

// -----

// Tolerance round-trip beyond the allowance: the scales differ enough that the
// DQ->Q round-trip differs by up to 20 codes (max_diff == 20). Neither the
// exact mode nor max-round-trip-diff=2 may fold this pair.
func.func @test_qdq_tol_beyond(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = onnx.Constant dense<2.581167699E-5> : tensor<f32>
%3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%4 = "onnx.QuantizeLinear"(%3, %2, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %4 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_tol_beyond(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

// TOL-LABEL: func.func @test_qdq_tol_beyond(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// TOL: onnx.DequantizeLinear
// TOL: onnx.QuantizeLinear

// -----

// Zero-points differ by 1 (39664 vs 39665) with identical scales. In exact
// mode (default) the zp mismatch is caught before the round-trip and the pair
// is kept. In tolerant mode the round-trip judges zp and scale jointly: the
// diff is 1 <= 2, so the pair folds.
func.func @test_qdq_tol_zp_mismatch(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%1 = onnx.Constant dense<39664> : tensor<ui16>
%2 = onnx.Constant dense<39665> : tensor<ui16>
%3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%4 = "onnx.QuantizeLinear"(%3, %0, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %4 : tensor<1x128x768xui16>
}

// CHECK-LABEL: func.func @test_qdq_tol_zp_mismatch(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

// TOL-LABEL: func.func @test_qdq_tol_zp_mismatch(
// TOL-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128x768xui16>
// TOL-SAME: ) -> tensor<1x128x768xui16>
// TOL:   return %[[ARG0]] : tensor<1x128x768xui16>
// TOL-NOT: onnx.DequantizeLinear
// TOL-NOT: onnx.QuantizeLinear

// -----

// The tolerance path only applies to per-tensor (scalar) qparams. Per-axis
// scales that differ must NOT fold, even under max-round-trip-diff=2.
func.func @test_qdq_tol_per_axis(%arg0: tensor<1x3x1x2xui16>) -> tensor<1x3x1x2xui16> {
%0 = onnx.Constant dense<[0.01, 0.1, 0.02]> : tensor<3xf32>
%1 = onnx.Constant dense<[0, 0, 0]> : tensor<3xui16>
%2 = onnx.Constant dense<[0.0100001, 0.1, 0.02]> : tensor<3xf32>
%3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x1x2xui16>, tensor<3xf32>, tensor<3xui16>) -> tensor<1x3x1x2xf32>
%4 = "onnx.QuantizeLinear"(%3, %2, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x1x2xf32>, tensor<3xf32>, tensor<3xui16>) -> tensor<1x3x1x2xui16>
return %4 : tensor<1x3x1x2xui16>
}

// CHECK-LABEL: func.func @test_qdq_tol_per_axis(%arg0: tensor<1x3x1x2xui16>) -> tensor<1x3x1x2xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

// TOL-LABEL: func.func @test_qdq_tol_per_axis(%arg0: tensor<1x3x1x2xui16>) -> tensor<1x3x1x2xui16>
// TOL: onnx.DequantizeLinear
// TOL: onnx.QuantizeLinear

// -----

// Real-world near-equal scales: the DQ scale (4.0574639569967985e-05) and the Q
// scale (4.057464320794679e-05) differ only in the last few f32 mantissa bits
// (relative diff ~9e-8), with identical zero-points. Over the full ui16 range
// the DQ->Q round-trip is exactly the identity (max diff == 0). It is bit-
// different, so exact mode keeps it; any non-zero tolerance folds it.
func.func @test_qdq_tol_near_equal_scale(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
%0 = onnx.Constant dense<4.0574639569967985e-05> : tensor<f32>
%1 = onnx.Constant dense<32681> : tensor<ui16>
%2 = onnx.Constant dense<4.057464320794679e-05> : tensor<f32>
%3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
%4 = "onnx.QuantizeLinear"(%3, %2, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
return %4 : tensor<1x128x768xui16>
}

// Exact mode: scales differ bit-for-bit, so the pair is kept.
// CHECK-LABEL: func.func @test_qdq_tol_near_equal_scale(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16>
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.QuantizeLinear

// Tolerance mode: round-trip diff == 0 <= 2, so it folds away.
// TOL-LABEL: func.func @test_qdq_tol_near_equal_scale(
// TOL-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<1x128x768xui16>
// TOL-SAME: ) -> tensor<1x128x768xui16>
// TOL:   return %[[ARG0]] : tensor<1x128x768xui16>
// TOL-NOT: onnx.DequantizeLinear
// TOL-NOT: onnx.QuantizeLinear
