// RUN: onnx-mlir-opt -onnx-hybrid-transform="canonicalization=true constant-propagation=true decomposition=true enable-convtranspose=false enable-convtranspose-1d-phased=false enable-convtranspose-phased=true enable-instancenorm-decompose=true enable-split-to-slice=false max-num-rewrites-multiplier=2.000000e-01 max-num-rewrites-offset=20 quark-quantized-ops-legalization=true recomposition=true shape-inference=true" %s | FileCheck %s

// Illustrates the interaction between 

func.func @resize_test_hybrid_transform(%arg0: tensor<1x256x20x20xf32> {onnx.name = "input"}) -> (tensor<*xbf16> {onnx.name = "output"}) {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
    %2 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<1x256x20x20xf32>) -> tensor<*xbf16>
    %3 = "onnx.Cast"(%2) {saturate = 1 : si64, to = f32} : (tensor<*xbf16>) -> tensor<*xf32>
    %4 = "onnx.Resize"(%3, %0, %1, %0) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor", onnx_node_name = "/fpn/up/Resize"} : (tensor<*xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
    %5 = "onnx.Cast"(%4) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    onnx.Return %5 : tensor<*xbf16>
}
// CHECK-LABEL:  func.func @resize_test_hybrid_transform
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x256x20x20xf32> {onnx.name = "input"}) -> (tensor<1x256x40x40xbf16> {onnx.name = "output"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = bf16} : (tensor<1x256x20x20xf32>) -> tensor<1x256x20x20xbf16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Resize"([[VAR_2_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor", onnx_node_name = "/fpn/up/Resize"} : (tensor<1x256x20x20xbf16>, none, tensor<4xf32>, none) -> tensor<1x256x40x40xbf16>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<1x256x40x40xbf16>
// CHECK:         }
