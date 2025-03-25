// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s | FileCheck %s

func.func @test_invsqrt_reciprocal(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %a = "onnx.Sqrt"(%arg0) : (tensor<10x10xf32>) -> tensor<*xf32>
  %y = "onnx.Reciprocal"(%a) : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%y) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_invsqrt_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.InvSqrt"([[VAR_0_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10xf32>
// CHECK:         }
}

func.func @test_invsqrt_div(%arg0 : tensor<1x2xf32>) -> tensor<1x2xf32> {
  %x = onnx.Constant dense<[[1.0, 1.0]]> : tensor<1x2xf32>
  %a = "onnx.Sqrt"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %y = "onnx.Div"(%x, %a) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  "func.return"(%y) : (tensor<1x2xf32>) -> ()

// CHECK-LABEL:  func @test_invsqrt_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2xf32>) -> tensor<1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<1x2xf32>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.InvSqrt"([[VAR_0_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2xf32>
// CHECK:         }
}

func.func @test_invsqrt_div2(%arg0 : tensor<1x2xf32>) -> tensor<*xf32> {
  %x = onnx.Constant dense<[[1.0, 1.0]]> : tensor<1x2xf32>
  %a = "onnx.Sqrt"(%arg0) : (tensor<1x2xf32>) -> tensor<*xf32>
  %y = "onnx.Div"(%x, %a) : (tensor<1x2xf32>, tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%y) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_invsqrt_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2xf32>) -> tensor<1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<1x2xf32>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.InvSqrt"([[VAR_0_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Unstick"([[VAR_1_]]) : (tensor<1x2xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2xf32>
// CHECK:         }
}
