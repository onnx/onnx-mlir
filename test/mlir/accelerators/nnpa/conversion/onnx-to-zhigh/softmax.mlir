// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_softmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_softmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.UnsqueezeV11"([[PARAM_0_]]) {axes = [0]} : (tensor<10x10xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3DS"} : (tensor<*xf32>) -> tensor<*xf16>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Softmax"([[VAR_1_]]) {act_func = "ACT_NONE"} : (tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<*xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_4_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func.func @test_softmax_3D(%arg0 : tensor<10x10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<10x10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_softmax_3D
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<10x10x10xf32>) -> tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Softmax"([[VAR_0_]]) {act_func = "ACT_NONE"} : (tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<10x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<10x10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10x10xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_logsoftmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) : (tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Log"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_logsoftmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.UnsqueezeV11"([[PARAM_0_]]) {axes = [0]} : (tensor<10x10xf32>) -> tensor<1x10x10xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3DS"} : (tensor<1x10x10xf32>) -> tensor<1x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Softmax"([[VAR_1_]]) {act_func = "ACT_LOG"} : (tensor<1x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x10x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x10x10xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<1x10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_4_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_logsoftmax_dyn(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) : (tensor<?x?xf32>) -> tensor<*xf32>
  %1 = "onnx.Log"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_onnx_logsoftmax_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.UnsqueezeV11"([[PARAM_0_]]) {axes = [0]} : (tensor<?x?xf32>) -> tensor<1x?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[VAR_0_]]) {layout = "3DS"} : (tensor<1x?x?xf32>) -> tensor<1x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Softmax"([[VAR_1_]]) {act_func = "ACT_LOG"} : (tensor<1x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<1x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<1x?x?xf32>) -> tensor<?x?xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x?xf32>
// CHECK:         }
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.


func.func @test_exceed_limit_softmax(%arg0 : tensor<32769x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<32769x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_softmax
// CHECK:        "onnx.Softmax"
}
