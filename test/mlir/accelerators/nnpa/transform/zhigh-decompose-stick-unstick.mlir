// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zhigh-decompose-stick-unstick --split-input-file %s | FileCheck %s

// COM: Decompose when there are only data movement ops between unstick and stick.
func.func @data_movement_and_stick_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %4 = "zhigh.Stick"(%3) {layout = "4D"} : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %4 : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @data_movement_and_stick_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf16>) -> tensor<1x5x3x?xf16>
// CHECK:           [[VAR_2_:%.+]] = "onnx.LayoutTransform"([[VAR_1_]]) {target_layout = "4D"} : (tensor<1x5x3x?xf16>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_2_]] : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

// COM: Unstick's output is returned and used for other computations.
// COM: Decompose because we can reduce one F32ToDLF16 from the stick op.
func.func @data_movement_and_stick_and_return_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) {
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %4 = "zhigh.Stick"(%3) {layout = "4D"} : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %2, %4 : tensor<1x3x5x?xf32>, tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-LABEL:  func.func @data_movement_and_stick_and_return_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.DLF16ToF32"([[VAR_0_]]) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf16>) -> tensor<1x5x3x?xf16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.LayoutTransform"([[VAR_2_]]) {target_layout = "4D"} : (tensor<1x5x3x?xf16>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:           return [[VAR_1_]], [[VAR_3_]] : tensor<1x3x5x?xf32>, tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK:         }
}

// -----

// COM: Unstick's output is used by data movement and comput ops.
// COM: Decompose because we can reduce one F32ToDLF16 from the stick op.
func.func @data_movement_and_stick_and_compute_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?xf32>) -> (tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x3x5x?xf32>) {
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %4 = "zhigh.Stick"(%3) {layout = "4D"} : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
  %5 = "onnx.Add"(%2, %arg1) : (tensor<1x3x5x?xf32>, tensor<?xf32>) -> tensor<1x3x5x?xf32>
  return %4, %5 : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x3x5x?xf32>

// CHECK-LABEL:  func.func @data_movement_and_stick_and_compute_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, [[PARAM_1_:%.+]]: tensor<?xf32>) -> (tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x3x5x?xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.DLF16ToF32"([[VAR_0_]]) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf16>) -> tensor<1x5x3x?xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.LayoutTransform"([[VAR_2_]]) {target_layout = "4D"} : (tensor<1x5x3x?xf16>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_1_]]) : (tensor<1x3x5x?xf32>, tensor<?xf32>) -> tensor<1x3x5x?xf32>
// CHECK:           return [[VAR_3_]], [[VAR_4_]] : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<1x3x5x?xf32>
// CHECK:         }
}

// -----

// COM: Decompose when only view ops are between unstick and stick.
func.func @view_and_stick_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %shape = onnx.Constant dense<[15, -1]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Reshape"(%2, %shape) : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
  %4 = "onnx.Unsqueeze"(%3, %axes) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
  %5 = "zhigh.Stick"(%4) {layout = "3DS"} : (tensor<15x1x?xf32>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %5 : tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @view_and_stick_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[15, -1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x3x5x?xf16>, tensor<2xi64>) -> tensor<15x?xf16>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[VAR_3_]], [[VAR_1_]]) : (tensor<15x?xf16>, tensor<1xi64>) -> tensor<15x1x?xf16>
// CHECK:           [[VAR_5_:%.+]] = "onnx.LayoutTransform"([[VAR_4_]]) {target_layout = "3DS"} : (tensor<15x1x?xf16>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_5_]] : tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----


// COM: Decompose when only data movement ops are between unstick and compute ops.
// COM: We want to move data on smaller data type.
func.func @data_movement_and_compute_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x5x3x?xf32> {
  %shape = onnx.Constant dense<[15, -1]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %4 = "onnx.Relu"(%3) : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf32>
  return %4 : tensor<1x5x3x?xf32>

// CHECK-LABEL:  func.func @data_movement_and_compute_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x5x3x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf16>) -> tensor<1x5x3x?xf16>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.DLF16ToF32"([[VAR_1_]]) : (tensor<1x5x3x?xf16>) -> tensor<1x5x3x?xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x5x3x?xf32>
// CHECK:         }
}

// -----

// COM: Do not decompose when only view ops are between unstick and compute ops.
// COM: Reshape and unsqueeze are views, there is no benefit for data movement on smaller data type.
func.func @view_and_compute_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<15x1x?xf32> {
  %shape = onnx.Constant dense<[15, -1]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Reshape"(%2, %shape) : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
  %4 = "onnx.Unsqueeze"(%3, %axes) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
  %5 = "onnx.Relu"(%4) : (tensor<15x1x?xf32>) -> tensor<15x1x?xf32>
  return %5 : tensor<15x1x?xf32>

// CHECK-LABEL:  func.func @view_and_compute_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<15x1x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[15, -1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[VAR_3_]], [[VAR_1_]]) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Relu"([[VAR_4_]]) : (tensor<15x1x?xf32>) -> tensor<15x1x?xf32>
// CHECK:           return [[VAR_5_]] : tensor<15x1x?xf32>
// CHECK:         }
}

// -----

// COM: Multiple branches: do not decompose since there would have redundant DLF16ToF32 ops.
func.func @branch_compute_and_return_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<15x1x?xf32>) {
  %shape = onnx.Constant dense<[15, -1]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Reshape"(%2, %shape) : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
  %4 = "onnx.Unsqueeze"(%3, %axes) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
  %5 = "onnx.Relu"(%4) : (tensor<15x1x?xf32>) -> tensor<15x1x?xf32>
  return %2, %5 : tensor<1x3x5x?xf32>, tensor<15x1x?xf32>

// CHECK-LABEL:  func.func @branch_compute_and_return_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<15x1x?xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[15, -1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[VAR_3_]], [[VAR_1_]]) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Relu"([[VAR_4_]]) : (tensor<15x1x?xf32>) -> tensor<15x1x?xf32>
// CHECK:           return [[VAR_2_]], [[VAR_5_]] : tensor<1x3x5x?xf32>, tensor<15x1x?xf32>
// CHECK:         }
}

// -----

// COM: Multiple branches: return branch and stick branch. Decompose because we can reduce more DLF16ToF32.
func.func @branch_stick_and_return_ops(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) {
  %shape1 = onnx.Constant dense<[15, -1]> : tensor<2xi64>
  %shape2 = onnx.Constant dense<[15, 1, -1]> : tensor<3xi64>
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf32>

  %3 = "onnx.Reshape"(%2, %shape1) : (tensor<1x3x5x?xf32>, tensor<2xi64>) -> tensor<15x?xf32>
  %4 = "onnx.Unsqueeze"(%3, %axes) : (tensor<15x?xf32>, tensor<1xi64>) -> tensor<15x1x?xf32>
  %5 = "zhigh.Stick"(%4) {layout = "3DS"} : (tensor<15x1x?xf32>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  %6 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %7 = "onnx.Reshape"(%6, %shape2) : (tensor<1x5x3x?xf32>, tensor<3xi64>) -> tensor<15x1x?xf32>
  %8 = "zhigh.Stick"(%7) {layout = "3DS"} : (tensor<15x1x?xf32>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  %9 = "zhigh.Add"(%5, %8) : (tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

  return %2, %9 : tensor<1x3x5x?xf32>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @branch_stick_and_return_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<1x3x5x?xf32>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[15, -1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[15, 1, -1]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.LayoutTransform"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x3x5x?xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.DLF16ToF32"([[VAR_3_]]) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x3x5x?xf16>, tensor<2xi64>) -> tensor<15x?xf16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[VAR_5_]], [[VAR_2_]]) : (tensor<15x?xf16>, tensor<1xi64>) -> tensor<15x1x?xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.LayoutTransform"([[VAR_6_]]) {target_layout = "3DS"} : (tensor<15x1x?xf16>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf16>) -> tensor<1x5x3x?xf16>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x5x3x?xf16>, tensor<3xi64>) -> tensor<15x1x?xf16>
// CHECK:           [[VAR_10_:%.+]] = "onnx.LayoutTransform"([[VAR_9_]]) {target_layout = "3DS"} : (tensor<15x1x?xf16>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Add"([[VAR_7_]], [[VAR_10_]]) : (tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_4_]], [[VAR_11_]] : tensor<1x3x5x?xf32>, tensor<15x1x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// COM: Do not decompose when the layout is NHWC.
func.func @test_nhwc(%arg0: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
  %2 = "zhigh.Unstick"(%arg0) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x?xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 1, 3] } : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
  %4 = "zhigh.Stick"(%3) {layout = "NHWC"} : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
  return %4 : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>

// CHECK-LABEL:  func.func @test_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<1x3x5x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x3x5x?xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x5x?xf32>) -> tensor<1x5x3x?xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "NHWC"} : (tensor<1x5x3x?xf32>) -> tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_2_]] : tensor<1x5x3x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

