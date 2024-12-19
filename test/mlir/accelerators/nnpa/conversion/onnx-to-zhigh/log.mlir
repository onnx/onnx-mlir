// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_log(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Log"([[VAR_0_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

// COM:  Do not lower broadcasting onnx.Log to zHigh.
func.func @test_log_not_lowered_diff_shape(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_log_not_lowered_diff_shape
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_log(%arg0 : tensor<32769x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<32769x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_log
// CHECK:        "onnx.Log"
}
