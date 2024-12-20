// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @test_zhigh_quantized_stick_dlfloat16(%arg0: tensor<1x3x5xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  onnx.Return %0#0: tensor<*xf16>

// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_dlfloat16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5xf32>) -> tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>> {
// CHECK:           [[NONE:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[NONE]], [[NONE]]) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           onnx.Return [[Out_]] : tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>
// CHECK:         }
}

// -----

func.func @test_zhigh_quantized_stick_int8(%arg0: tensor<1x3x5xf32>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "int8", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  onnx.Return %0#0: tensor<*xi8>

// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_int8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5xf32>) -> tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>> {
// CHECK:           [[NONE:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[NONE]], [[NONE]]) {layout = "3DS", quantized_type = "int8", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           onnx.Return [[Out_]] : tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>
// CHECK:         }
}

// -----

func.func @test_zhigh_quantized_stick_weights(%arg0: tensor<1x3x5xf32>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "weights", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  onnx.Return %0#0: tensor<*xi8>

// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_weights
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5xf32>) -> tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "WEIGHTS"}>> {
// CHECK:           [[NONE:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[NONE]], [[NONE]]) {layout = "3DS", quantized_type = "weights", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           onnx.Return [[Out_]] : tensor<1x3x5xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "WEIGHTS"}>>
// CHECK:         }
}
