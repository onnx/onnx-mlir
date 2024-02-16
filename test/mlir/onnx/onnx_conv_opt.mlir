// RUN: onnx-mlir-opt --conv-opt-onnx='simd-data-layout' %s -split-input-file | FileCheck %s

// Simple intro of layout transform
module {
  func.func @test_onnx_conv_simple_pattern(%arg0: tensor<5x3x32x32xf32>, %arg1: tensor<?x3x2x2xf32>) -> tensor<5x?x31x31xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {auto_pad = "NOTSET", kernel_shape = [2, 2], pads = [0, 0, 0, 0]} : (tensor<5x3x32x32xf32>, tensor<?x3x2x2xf32>, none) -> tensor<5x?x31x31xf32>
    onnx.Return %1 : tensor<5x?x31x31xf32>
  }
}
//  use arg names: ['image', 'filter']
// mlir2FileCheck.py -a'["image", "filter"]'
// CHECK-LABEL:  func.func @test_onnx_conv_simple_pattern
// CHECK-SAME:   ([[IMAGE_:%.+]]: tensor<5x3x32x32xf32>, [[FILTER_:%.+]]: tensor<?x3x2x2xf32>) -> tensor<5x?x31x31xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.LayoutTransform"([[IMAGE_]]) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.LayoutTransform"([[FILTER_]]) {target_layout = #onnx.layout<{dataLayout = "KCMN4C4K"}>} : (tensor<?x3x2x2xf32>) -> tensor<?x3x2x2xf32, #onnx.layout<{dataLayout = "KCMN4C4K"}>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0]} : (tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>, tensor<?x3x2x2xf32, #onnx.layout<{dataLayout = "KCMN4C4K"}>>, none) -> tensor<5x?x31x31xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
// CHECK:           [[VAR_4_:%.+]] = "onnx.LayoutTransform"([[VAR_3_]]) {target_layout = "STANDARD"} : (tensor<5x?x31x31xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x?x31x31xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<5x?x31x31xf32>
// CHECK:         }
