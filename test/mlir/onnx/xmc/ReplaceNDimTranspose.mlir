// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: flexml-opt %configPassStx -annotate-config="library-metadata-dirs=%S" %s -replace-ndim-transpose -o - | FileCheck %s
func.func @replace_ndim_transpose_positive(%arg0: tensor<1x4x320x16xf32>) -> tensor<320x1x64xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<1x4x320x16xf32>) -> tensor<320x1x16x4xf32>

  // Reshape operation (user of transpose)
  %shape = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<320x1x16x4xf32>, tensor<3xi64>) -> tensor<320x1x64xf32>

  return %1 : tensor<320x1x64xf32>
}
// Original transpose with perm {2, 0, 3, 1} -> output shape {320, 1, 16, 4}
// CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0)
// CHECK-SAME: perm = [0, 2, 1, 3]
// CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[T1]])
// CHECK-SAME: perm = [0, 1, 3, 2]
