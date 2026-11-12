// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Covers the optional trailing scalar Mul of the
// zhigh.extended_layout_transform FusedOp kind (LayoutTransform -> Reshape ->
// Transpose -> DLF16ToF32 -> Mul). The loop structure is the same as without
// the Mul; the only difference is one multiply by mulScalar applied to each
// 4xf32 vector right after the DLF16 conversion.

// -----

func.func @fused_elt_mul(%arg0: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<4xi64>) -> tensor<3x8x?x64xf32> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.extended_layout_transform"}> ({
  ^bb0(%arg2: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg3: tensor<4xi64>):
    %1 = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
    %2 = "onnx.LayoutTransform"(%arg2) : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x512xf16>
    %3 = "onnx.Reshape"(%2, %arg3) <{allowzero = 0 : si64}> : (tensor<3x?x512xf16>, tensor<4xi64>) -> tensor<3x?x8x64xf16>
    %4 = "onnx.Transpose"(%3) <{perm = [0, 2, 1, 3]}> : (tensor<3x?x8x64xf16>) -> tensor<3x8x?x64xf16>
    %5 = "zhigh.DLF16ToF32"(%4) : (tensor<3x8x?x64xf16>) -> tensor<3x8x?x64xf32>
    %6 = "onnx.Mul"(%5, %1) : (tensor<3x8x?x64xf32>, tensor<1xf32>) -> tensor<3x8x?x64xf32>
    onnx.Yield %6 : tensor<3x8x?x64xf32>
  }) {dlf16ToF32 = true, mulScalar = 1.250000e-01 : f32, reshapeMergeAxis = -1 : i64, reshapeSplitAxis = 2 : i64, reshapeSplitFactor = 64 : i64, transposePattern = [0, 2, 1, 3]} : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4xi64>) -> tensor<3x8x?x64xf32>
  return %0 : tensor<3x8x?x64xf32>

// CHECK-LABEL:  func.func @fused_elt_mul
// CHECK-DAG:       [[SCALAR_:%.+]] = arith.constant dense<1.250000e-01> : vector<4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc({{.*}}) {{.*}}: memref<3x8x?x64xf32>
// CHECK:           krnl.iterate
// CHECK-COUNT-4:       [[HIGH_:%.+]], [[LOW_:%.+]] = "zlow.vec_dlf16_to_f32"({{.*}}) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-NEXT:          [[HIGH_SCALED_:%.+]] = arith.mulf [[HIGH_]], [[SCALAR_]] : vector<4xf32>
// CHECK-NEXT:          [[LOW_SCALED_:%.+]] = arith.mulf [[LOW_]], [[SCALAR_]] : vector<4xf32>
// CHECK:               vector.store [[HIGH_SCALED_]], [[RES_]]
// CHECK:               vector.store [[LOW_SCALED_]], [[RES_]]
// CHECK-NOT:       onnx.
// CHECK:           return [[RES_]] : memref<3x8x?x64xf32>
}

// -----

// A fused op with no mulScalar attr (and no Mul in its body) is still lowered
// by the fused pattern, with no multiply.

func.func @fused_elt_no_mul_attr(%arg0: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<4xi64>) -> tensor<3x8x?x64xf32> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.extended_layout_transform"}> ({
  ^bb0(%arg2: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg3: tensor<4xi64>):
    %2 = "onnx.LayoutTransform"(%arg2) : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x512xf16>
    %3 = "onnx.Reshape"(%2, %arg3) <{allowzero = 0 : si64}> : (tensor<3x?x512xf16>, tensor<4xi64>) -> tensor<3x?x8x64xf16>
    %4 = "onnx.Transpose"(%3) <{perm = [0, 2, 1, 3]}> : (tensor<3x?x8x64xf16>) -> tensor<3x8x?x64xf16>
    %5 = "zhigh.DLF16ToF32"(%4) : (tensor<3x8x?x64xf16>) -> tensor<3x8x?x64xf32>
    onnx.Yield %5 : tensor<3x8x?x64xf32>
  }) {dlf16ToF32 = true, reshapeMergeAxis = -1 : i64, reshapeSplitAxis = 2 : i64, reshapeSplitFactor = 64 : i64, transposePattern = [0, 2, 1, 3]} : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4xi64>) -> tensor<3x8x?x64xf32>
  return %0 : tensor<3x8x?x64xf32>

// CHECK-LABEL:  func.func @fused_elt_no_mul_attr
// CHECK-NOT:       krnl.memcpy
// CHECK:           "zlow.vec_dlf16_to_f32"
// CHECK-NOT:       arith.mulf
// CHECK:           return
}

// -----

// Body altered after fusion: the Mul was removed but mulScalar still says
// 0.125. verify() must reject it, so the body is inlined and lowered op by op
// (the layout transform and transpose then show up as krnl.memcpy copies),
// with no multiply.

func.func @fused_elt_mul_tampered(%arg0: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<4xi64>) -> tensor<3x8x?x64xf32> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.extended_layout_transform"}> ({
  ^bb0(%arg2: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg3: tensor<4xi64>):
    %2 = "onnx.LayoutTransform"(%arg2) : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x512xf16>
    %3 = "onnx.Reshape"(%2, %arg3) <{allowzero = 0 : si64}> : (tensor<3x?x512xf16>, tensor<4xi64>) -> tensor<3x?x8x64xf16>
    %4 = "onnx.Transpose"(%3) <{perm = [0, 2, 1, 3]}> : (tensor<3x?x8x64xf16>) -> tensor<3x8x?x64xf16>
    %5 = "zhigh.DLF16ToF32"(%4) : (tensor<3x8x?x64xf16>) -> tensor<3x8x?x64xf32>
    onnx.Yield %5 : tensor<3x8x?x64xf32>
  }) {dlf16ToF32 = true, mulScalar = 1.250000e-01 : f32, reshapeMergeAxis = -1 : i64, reshapeSplitAxis = 2 : i64, reshapeSplitFactor = 64 : i64, transposePattern = [0, 2, 1, 3]} : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4xi64>) -> tensor<3x8x?x64xf32>
  return %0 : tensor<3x8x?x64xf32>

// CHECK-LABEL:  func.func @fused_elt_mul_tampered
// CHECK:           krnl.memcpy
// CHECK-NOT:       arith.mulf
// CHECK:           return
}
