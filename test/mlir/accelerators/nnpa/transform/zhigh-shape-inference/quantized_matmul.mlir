// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @test_zhigh_quantized_matmul(%arg0: tensor<1x3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<7xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %x:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16"} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  %y:3 = "zhigh.QuantizedStick"(%arg1, %none, %none) {layout = "2D", quantized_type = "weights"} : (tensor<5x7xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  %b:3 = "zhigh.QuantizedStick"(%arg2, %none, %none) {layout = "1D", quantized_type = "int8"} : (tensor<7xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  %m:3 = "zhigh.QuantizedMatMul"(%x#0, %x#1, %x#2, %y#0, %y#1, %y#2, %b#0, %b#1, %b#2, %none, %none) {DequantizeOutput = 0 : si64} : (tensor<*xf16>, tensor<f32>, tensor<f32>, tensor<*xi8>, tensor<f32>, tensor<f32>, tensor<*xi8>, tensor<f32>, tensor<f32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>) 
  onnx.Return %m#0: tensor<*xf16>

// CHECK-LABEL:  func.func @test_zhigh_quantized_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5xf32>, [[PARAM_1_:%.+]]: tensor<5x7xf32>, [[PARAM_2_:%.+]]: tensor<7xf32>) -> tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_0_]], [[VAR_0_]]) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_0_:%.+]], [[RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[PARAM_1_]], [[VAR_0_]], [[VAR_0_]]) {layout = "2D", quantized_type = "weights", sym_mode = 0 : i64} : (tensor<5x7xf32>, none, none) -> (tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_3_:%.+]], [[RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[PARAM_2_]], [[VAR_0_]], [[VAR_0_]]) {layout = "1D", quantized_type = "int8", sym_mode = 0 : i64} : (tensor<7xf32>, none, none) -> (tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_6_:%.+]], [[OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[Out_]], [[RecScale_]], [[VAR_Offset_]], [[Out_]]_0, [[RecScale_]]_1, [[VAR_Offset_]]_2, [[Out_]]_3, [[RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_0_]], [[VAR_0_]]) {DequantizeOutput = 0 : si64, DisableClipping = 0 : si64, PreComputedBias = 0 : si64} : (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, none, none) -> (tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           onnx.Return [[Out_6_]] : tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>
// CHECK:         }
}

// -----

func.func @test_zhigh_quantized_matmul_dequantized(%arg0: tensor<1x3x5xf32>, %arg1: tensor<5x7xf32>, %arg2: tensor<7xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %x:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16"} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  %y:3 = "zhigh.QuantizedStick"(%arg1, %none, %none) {layout = "2D", quantized_type = "weights"} : (tensor<5x7xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  %b:3 = "zhigh.QuantizedStick"(%arg2, %none, %none) {layout = "1D", quantized_type = "int8"} : (tensor<7xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  %m:3 = "zhigh.QuantizedMatMul"(%x#0, %x#1, %x#2, %y#0, %y#1, %y#2, %b#0, %b#1, %b#2, %none, %none) {DequantizeOutput = -1 : si64} : (tensor<*xf16>, tensor<f32>, tensor<f32>, tensor<*xi8>, tensor<f32>, tensor<f32>, tensor<*xi8>, tensor<f32>, tensor<f32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>) 
  onnx.Return %m#0: tensor<*xf16>

// CHECK-LABEL:  func.func @test_zhigh_quantized_matmul_dequantized
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5xf32>, [[PARAM_1_:%.+]]: tensor<5x7xf32>, [[PARAM_2_:%.+]]: tensor<7xf32>) -> tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_0_]], [[VAR_0_]]) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_0_:%.+]], [[RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[PARAM_1_]], [[VAR_0_]], [[VAR_0_]]) {layout = "2D", quantized_type = "weights", sym_mode = 0 : i64} : (tensor<5x7xf32>, none, none) -> (tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_3_:%.+]], [[RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[PARAM_2_]], [[VAR_0_]], [[VAR_0_]]) {layout = "1D", quantized_type = "int8", sym_mode = 0 : i64} : (tensor<7xf32>, none, none) -> (tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_6_:%.+]], [[OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[Out_]], [[RecScale_]], [[VAR_Offset_]], [[Out_]]_0, [[RecScale_]]_1, [[VAR_Offset_]]_2, [[Out_]]_3, [[RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_0_]], [[VAR_0_]]) {DequantizeOutput = -1 : si64, DisableClipping = 0 : si64, PreComputedBias = 0 : si64} : (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, none, none) -> (tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           onnx.Return [[Out_6_]] : tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}
