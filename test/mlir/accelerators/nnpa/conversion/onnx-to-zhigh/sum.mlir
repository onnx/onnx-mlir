// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

// COM: Check the singleton case of lowering ONNXSumOp to ZHighAddOp,
// COM: where ONNXSumOp has two inputs and is lowered to a single ZHighAddOp.

func.func @test_sum_2_operands(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_sum_2_operands
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

// COM: Check the recursion case of lowering ONNXSumOp to ZHighAddOp,
// COM: where ONNXSumOp has N inputs (N > 2), and is recursively unfolded to
// COM: multiple ZHighAddOp(s).

func.func @test_sum_4_operands(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>, %arg2 : tensor<10x10xf32>, %arg3 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2, %arg3) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_sum_4_operands
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>, [[PARAM_2_:%.+]]: tensor<10x10xf32>, [[PARAM_3_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[PARAM_3_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Add"([[VAR_2_]], [[VAR_3_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Add"([[VAR_1_]], [[VAR_6_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Unstick"([[VAR_7_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           [[VAR_9_:%.+]] = "zhigh.Stick"([[VAR_8_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.Add"([[VAR_0_]], [[VAR_9_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_11_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

/// Do not lower broadcasting onnx.Sum to zHigh.
func.func @test_sum_not_lowered_diff_shape(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum_not_lowered_diff_shape
}

// -----

/// Do not lower onnx.Sum to zHigh if inputs have different shape including unknown dimensions.
/// Need to check whether input tensors is really different at runtime.
func.func @test_sum_not_lowered_unknown_dims(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum_not_lowered_unknown_dims
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_sum(%arg0 : tensor<32769x10xf32>, %arg1 : tensor<32769x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<32769x10xf32>, tensor<32769x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_sum
// CHECK:        "onnx.Sum"
}
