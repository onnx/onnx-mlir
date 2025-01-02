// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s

func.func @qlinearmatmul_i8_f32(%arg0: tensor<2x4xi8> {onnx.name = "a"}, %arg1: tensor<f32> {onnx.name = "a_scale"}, %arg2: tensor<i8> {onnx.name = "a_zero_point"}, %arg3: tensor<4x3xi8> {onnx.name = "b"}, %arg4: tensor<f32> {onnx.name = "b_scale"}, %arg5: tensor<i8> {onnx.name = "b_zero_point"}, %arg6: tensor<f32> {onnx.name = "y_scale"}, %arg7: tensor<i8> {onnx.name = "y_zero_point"}) -> (tensor<2x3xi8> {onnx.name = "y"}) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<2x4xi8>, tensor<f32>, tensor<i8>, tensor<4x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<2x3xi8>
    onnx.Return %0 : tensor<2x3xi8>

// CHECK-LABEL:  func.func @qlinearmatmul_i8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xi8> {onnx.name = "a"}, [[PARAM_1_:%.+]]: tensor<f32> {onnx.name = "a_scale"}, [[PARAM_2_:%.+]]: tensor<i8> {onnx.name = "a_zero_point"}, [[PARAM_3_:%.+]]: tensor<4x3xi8> {onnx.name = "b"}, [[PARAM_4_:%.+]]: tensor<f32> {onnx.name = "b_scale"}, [[PARAM_5_:%.+]]: tensor<i8> {onnx.name = "b_zero_point"}, [[PARAM_6_:%.+]]: tensor<f32> {onnx.name = "y_scale"}, [[PARAM_7_:%.+]]: tensor<i8> {onnx.name = "y_zero_point"}) -> (tensor<2x3xi8> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Reciprocal"([[PARAM_1_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Cast"([[PARAM_2_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reciprocal"([[PARAM_4_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_5_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Reciprocal"([[PARAM_6_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Cast"([[PARAM_7_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[Out_:%.+]], [[RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) {layout = "2D", quantized_type = "INT8", sym_mode = 0 : i64} : (tensor<2x4xi8>, tensor<f32>, tensor<f32>) -> (tensor<2x4xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_0_:%.+]], [[RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[PARAM_3_]], [[VAR_3_]], [[VAR_4_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<4x3xi8>, tensor<f32>, tensor<f32>) -> (tensor<4x3xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[Out_3_:%.+]], [[OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[Out_]], [[RecScale_]], [[VAR_Offset_]], [[Out_0_]], [[RecScale_1_]], [[VAR_Offset_2_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_5_]], [[VAR_6_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = 0 : si64} : (tensor<2x4xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, tensor<4x3xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[Out_3_]]) : (tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>) -> tensor<2x3xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_7_]]) {saturate = 1 : si64, to = i8} : (tensor<2x3xf32>) -> tensor<2x3xi8>
// CHECK:           onnx.Return [[VAR_8_]] : tensor<2x3xi8>
// CHECK:         }
}

// -----

func.func @qlinearmatmul_ui8_f32(%arg0: tensor<2x4xui8> {onnx.name = "a"}, %arg1: tensor<f32> {onnx.name = "a_scale"}, %arg2: tensor<ui8> {onnx.name = "a_zero_point"}, %arg3: tensor<4x3xui8> {onnx.name = "b"}, %arg4: tensor<f32> {onnx.name = "b_scale"}, %arg5: tensor<ui8> {onnx.name = "b_zero_point"}, %arg6: tensor<f32> {onnx.name = "y_scale"}, %arg7: tensor<ui8> {onnx.name = "y_zero_point"}) -> (tensor<2x3xui8> {onnx.name = "y"}) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<2x4xui8>, tensor<f32>, tensor<ui8>, tensor<4x3xui8>, tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>) -> tensor<2x3xui8>
    onnx.Return %0 : tensor<2x3xui8>

// CHECK-LABEL:  func.func @qlinearmatmul_ui8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xui8> {onnx.name = "a"}, [[PARAM_1_:%.+]]: tensor<f32> {onnx.name = "a_scale"}, [[PARAM_2_:%.+]]: tensor<ui8> {onnx.name = "a_zero_point"}, [[PARAM_3_:%.+]]: tensor<4x3xui8> {onnx.name = "b"}, [[PARAM_4_:%.+]]: tensor<f32> {onnx.name = "b_scale"}, [[PARAM_5_:%.+]]: tensor<ui8> {onnx.name = "b_zero_point"}, [[PARAM_6_:%.+]]: tensor<f32> {onnx.name = "y_scale"}, [[PARAM_7_:%.+]]: tensor<ui8> {onnx.name = "y_zero_point"}) -> (tensor<2x3xui8> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<128> : tensor<i16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = i16} : (tensor<2x4xui8>) -> tensor<2x4xi16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_0_]]) : (tensor<2x4xi16>, tensor<i16>) -> tensor<2x4xi16>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[VAR_3_]]) {saturate = 1 : si64, to = i8} : (tensor<2x4xi16>) -> tensor<2x4xi8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[PARAM_3_]]) {saturate = 1 : si64, to = i16} : (tensor<4x3xui8>) -> tensor<4x3xi16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_5_]], [[VAR_0_]]) : (tensor<4x3xi16>, tensor<i16>) -> tensor<4x3xi16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_6_]]) {saturate = 1 : si64, to = i8} : (tensor<4x3xi16>) -> tensor<4x3xi8>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Reciprocal"([[PARAM_1_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Cast"([[PARAM_2_]]) {saturate = 1 : si64, to = i16} : (tensor<ui8>) -> tensor<i16>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Sub"([[VAR_9_]], [[VAR_0_]]) : (tensor<i16>, tensor<i16>) -> tensor<i16>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Cast"([[VAR_10_]]) {saturate = 1 : si64, to = i8} : (tensor<i16>) -> tensor<i8>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Cast"([[VAR_11_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Reciprocal"([[PARAM_4_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Cast"([[PARAM_5_]]) {saturate = 1 : si64, to = i16} : (tensor<ui8>) -> tensor<i16>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Sub"([[VAR_14_]], [[VAR_0_]]) : (tensor<i16>, tensor<i16>) -> tensor<i16>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Cast"([[VAR_15_]]) {saturate = 1 : si64, to = i8} : (tensor<i16>) -> tensor<i8>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Cast"([[VAR_16_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Reciprocal"([[PARAM_6_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Cast"([[PARAM_7_]]) {saturate = 1 : si64, to = i16} : (tensor<ui8>) -> tensor<i16>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Sub"([[VAR_19_]], [[VAR_0_]]) : (tensor<i16>, tensor<i16>) -> tensor<i16>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Cast"([[VAR_20_]]) {saturate = 1 : si64, to = i8} : (tensor<i16>) -> tensor<i8>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Cast"([[VAR_21_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[VAR_4_]], [[VAR_8_]], [[VAR_12_]]) {layout = "2D", quantized_type = "INT8", sym_mode = 0 : i64} : (tensor<2x4xi8>, tensor<f32>, tensor<f32>) -> (tensor<2x4xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_7_]], [[VAR_13_]], [[VAR_17_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<4x3xi8>, tensor<f32>, tensor<f32>) -> (tensor<4x3xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_RecScale_]]_1, [[VAR_Offset_]]_2, [[VAR_1_]], [[VAR_1_]], [[VAR_1_]], [[VAR_1_]]8, [[VAR_22_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = 0 : si64} : (tensor<2x4xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, tensor<4x3xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_23_:%.+]] = "zhigh.Unstick"([[VAR_Out_3_]]) : (tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>) -> tensor<2x3xf32>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Cast"([[VAR_23_]]) {saturate = 1 : si64, to = i16} : (tensor<2x3xf32>) -> tensor<2x3xi16>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Add"([[VAR_24_]], [[VAR_0_]]) : (tensor<2x3xi16>, tensor<i16>) -> tensor<2x3xi16>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Cast"([[VAR_25_]]) {saturate = 1 : si64, to = ui16} : (tensor<2x3xi16>) -> tensor<2x3xui16>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Cast"([[VAR_26_]]) {saturate = 1 : si64, to = ui8} : (tensor<2x3xui16>) -> tensor<2x3xui8>
// CHECK:           onnx.Return [[VAR_27_]] : tensor<2x3xui8>
// CHECK:         }
}
