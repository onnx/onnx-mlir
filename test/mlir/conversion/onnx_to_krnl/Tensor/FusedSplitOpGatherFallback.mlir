// RUN: onnx-mlir-opt --march=arm64 --shape-inference --convert-onnx-to-krnl="enable-simd=true" %s -split-input-file | FileCheck %s

// Hand-constructed onnx.Fused(kind="simd-split-op-gather") with a stale
// hasOpForSplitLow attribute (claims true, but the body still only has the
// 4 chain ops from a hasOpForSplitLow=false detection -- no low-half op node
// actually present). No real detection run ever produces this; it exists to
// confirm ONNXFusedSplitOpGatherLowering::lowerVerified's own verify() check
// still catches an inconsistent fused op and falls back to
// FusedOpInlineFallback -- i.e. the dedicated lowering existing at all
// doesn't weaken the safety net. Distinguishing signal: the dedicated
// lowering always produces exactly one memref.alloc of the output type;
// the inlined fallback instead materializes each original Slice/Concat
// operand separately.

func.func @stale_attrs_fall_back(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %0 = "onnx.Fused"(%arg0) <{kind = "simd-split-op-gather"}> ({
  ^bb0(%arg1: tensor<1x4x8x128xf32>):
    %1 = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
    %2 = onnx.Constant dense<0> : tensor<1xi64>
    %3 = onnx.Constant dense<64> : tensor<1xi64>
    %4 = onnx.Constant dense<3> : tensor<1xi64>
    %5 = onnx.Constant dense<1> : tensor<1xi64>
    %6 = "onnx.Slice"(%arg1, %2, %3, %4, %5) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
    %7 = "onnx.Slice"(%arg1, %3, %1, %4, %5) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
    %8 = "onnx.Neg"(%7) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
    %9 = "onnx.Concat"(%8, %6) <{axis = 3 : si64}> : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    onnx.Yield %9 : tensor<1x4x8x128xf32>
  }) {axis = 3 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = true, onnx_node_name = "onnx.Slice-onnx.Slice-onnx.Neg-onnx.Concat", outputOffsetForSplitHigh = 0 : i64, outputOffsetForSplitLow = 64 : i64, splitPoint = 64 : i64} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
  return %0 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @stale_attrs_fall_back
// Fallback inlines the original Slice/Neg/Concat lowering: 3 separate
// tensor<1x4x8x64xf32> allocations (both slices' outputs, and Neg's
// output) plus the final tensor<1x4x8x128xf32> Concat result -- never the
// single fused-lowering allocation the well-formed cases above produce.
// CHECK:           memref.alloc() {{.*}} : memref<1x4x8x64xf32>
// CHECK:           memref.alloc() {{.*}} : memref<1x4x8x64xf32>
// CHECK:           memref.alloc() {{.*}} : memref<1x4x8x64xf32>
// CHECK:           memref.alloc() {{.*}} : memref<1x4x8x128xf32>
}
