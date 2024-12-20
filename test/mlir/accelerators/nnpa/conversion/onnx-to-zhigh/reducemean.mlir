// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @should_lower_to_zhigh(%arg0 : tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) { axes = [2, 3] }: (tensor<1x3x5x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @should_lower_to_zhigh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x7xf32>) -> tensor<1x3x1x1xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MeanReduce2d"([[VAR_1_]]) : (tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x3x1x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x1x1xf32>
// CHECK:         }
}

// -----

func.func @should_not_lower_noaxes(%arg0 : tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) : (tensor<1x3x5x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL: should_not_lower_noaxes
// CHECK: [[RES0:%.+]] = "onnx.ReduceMeanV13"(%arg0) {keepdims = 1 : si64} : (tensor<1x3x5x7xf32>) -> tensor<1x1x1x1xf32>
// CHECK: return [[RES0]] : tensor<1x1x1x1xf32>
}

// -----

func.func @should_not_lower_keepdim0(%arg0 : tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) { axes = [2, 3], keepdims = 0 : si64 } : (tensor<1x3x5x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL: should_not_lower_keepdim0
// CHECK: [[RES0:%.+]] = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 0 : si64} : (tensor<1x3x5x7xf32>) -> tensor<1x3xf32>
// CHECK: return [[RES0]] : tensor<1x3xf32>
}

// -----

func.func @should_not_lower_too_large_data(%arg0 : tensor<1x3x5x2048xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) { axes = [2, 3] } : (tensor<1x3x5x2048xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL: should_not_lower_too_large_data
// CHECK: [[RES0:%.+]] = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x5x2048xf32>) -> tensor<1x3x1x1xf32>
// CHECK: return [[RES0]] : tensor<1x3x1x1xf32>
}

// -----

func.func @should_not_lower_5D(%arg0 : tensor<1x3x5x7x9xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) { axes = [2, 3] } : (tensor<1x3x5x7x9xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL: should_not_lower_5D
// CHECK: [[RES0:%.+]] = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x5x7x9xf32>) -> tensor<1x3x1x1x9xf32>
// CHECK: return [[RES0]] : tensor<1x3x1x1x9xf32>
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_reducemean_v13(%arg0 : tensor<32769x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) { axes = [2, 3] }: (tensor<32769x3x5x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_reducemean_v13
// CHECK:        "onnx.ReduceMeanV13"
}
