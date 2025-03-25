// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s | FileCheck %s

func.func @test_gelu_erf_arch15(%arg0 : tensor<1x2xf32>) -> tensor<1x2xf32>{
  %0 ="onnx.Gelu"(%arg0) {approximate = "none"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
 "func.return"(%0) : (tensor<1x2xf32>) -> ()


// CHECK-LABEL:  func @test_gelu_erf_arch15
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2xf32>) -> tensor<1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<1x2xf32>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Gelu"(%0) {approximate = "none"} : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_gelu_tanh_arch15(%arg0 : tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 ="onnx.Gelu"(%arg0) {approximate = "tanh"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
 "func.return"(%0) : (tensor<1x2xf32>) -> ()

// CHECK-LABEL:  func @test_gelu_tanh_arch15
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2xf32>) -> tensor<1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<1x2xf32>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Gelu"(%0) {approximate = "tanh"} : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2xf32>
// CHECK:         }
}
