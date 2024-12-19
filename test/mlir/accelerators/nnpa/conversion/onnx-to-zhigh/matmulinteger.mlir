// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh --canonicalize --convert-zhigh-to-onnx %s -split-input-file | FileCheck %s --check-prefix=CHECK-FUSION

func.func @matmulinteger(%arg0: tensor<?x?x768xui8>, %arg1: tensor<768x768xi8>, %arg2: tensor<ui8>, %arg3: tensor<i8>) -> tensor<?x?x768xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<?x?x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<?x?x768xi32>
  return %0 : tensor<?x?x768xi32>

// CHECK-LABEL:  func.func @matmulinteger
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xui8>, [[PARAM_1_:%.+]]: tensor<768x768xi8>, [[PARAM_2_:%.+]]: tensor<ui8>, [[PARAM_3_:%.+]]: tensor<i8>) -> tensor<?x?x768xi32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = i8} : (tensor<?x?x768xui8>) -> tensor<?x?x768xi8>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_2_]]) {saturate = 1 : si64, to = i8} : (tensor<ui8>) -> tensor<i8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[VAR_4_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Cast"([[PARAM_3_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[VAR_3_]], [[VAR_2_]], [[VAR_5_]]) {layout = "3DS", quantized_type = "INT8", sym_mode = 0 : i64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[PARAM_1_]], [[VAR_2_]], [[VAR_6_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_RecScale_]]_1, [[VAR_Offset_]]_2, [[VAR_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_2_]], [[VAR_1_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = 0 : si64} : (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Unstick"([[VAR_Out_3_]]) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_7_]]) {saturate = 1 : si64, to = i32} : (tensor<?x?x768xf32>) -> tensor<?x?x768xi32>
// CHECK:           return [[VAR_8_]] : tensor<?x?x768xi32>
// CHECK:         }
}

// -----

// Do not do pre_compute when B is not a constant.
func.func @matmulinteger_no_precompute_bias(%arg0: tensor<?x?x768xui8>, %arg1: tensor<768x768xi8>, %arg2: tensor<ui8>) -> tensor<?x?x768xi32> {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %0) : (tensor<?x?x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<?x?x768xi32>
  return %1 : tensor<?x?x768xi32>

// CHECK-LABEL:  func.func @matmulinteger_no_precompute_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xui8>, [[PARAM_1_:%.+]]: tensor<768x768xi8>, [[PARAM_2_:%.+]]: tensor<ui8>) -> tensor<?x?x768xi32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = i8} : (tensor<?x?x768xui8>) -> tensor<?x?x768xi8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[PARAM_2_]]) {saturate = 1 : si64, to = i8} : (tensor<ui8>) -> tensor<i8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Cast"([[VAR_5_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_0_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[VAR_4_]], [[VAR_3_]], [[VAR_6_]]) {layout = "3DS", quantized_type = "INT8", sym_mode = 0 : i64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[PARAM_1_]], [[VAR_3_]], [[VAR_7_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_RecScale_]]_1, [[VAR_Offset_]]_2, [[VAR_1_]], [[VAR_1_]], [[VAR_1_]], [[VAR_3_]], [[VAR_2_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = 0 : si64} : (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Unstick"([[VAR_Out_3_]]) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_8_]]) {saturate = 1 : si64, to = i32} : (tensor<?x?x768xf32>) -> tensor<?x?x768xi32>
// CHECK:           return [[VAR_9_]] : tensor<?x?x768xi32>
// CHECK:         }
}

// -----

func.func @matmulinteger_precompute_bias(%arg0: tensor<?x?x768xui8>, %arg1: tensor<ui8>) -> tensor<?x?x768xi32> {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %B = onnx.Constant dense<0> : tensor<768x768xi8>
  %1 = "onnx.MatMulInteger"(%arg0, %B, %arg1, %0) : (tensor<?x?x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<?x?x768xi32>
  return %1 : tensor<?x?x768xi32>

// CHECK-LABEL:  func.func @matmulinteger_precompute_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xui8>, [[PARAM_1_:%.+]]: tensor<ui8>) -> tensor<?x?x768xi32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<768x768xi8>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = i8} : (tensor<?x?x768xui8>) -> tensor<?x?x768xi8>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Cast"([[PARAM_1_]]) {saturate = 1 : si64, to = i8} : (tensor<ui8>) -> tensor<i8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_6_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_1_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[VAR_5_]], [[VAR_4_]], [[VAR_7_]]) {layout = "3DS", quantized_type = "INT8", sym_mode = 0 : i64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_2_]], [[VAR_4_]], [[VAR_8_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_2_]]) {saturate = 1 : si64, to = f32} : (tensor<768x768xi8>) -> tensor<768x768xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.ReduceSum"([[VAR_9_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<768x768xf32>, tensor<i64>) -> tensor<768xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Div"([[VAR_4_]], [[VAR_4_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Div"([[VAR_11_]], [[VAR_4_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Mul"([[VAR_12_]], [[VAR_7_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_13_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Mul"([[VAR_14_]], [[VAR_10_]]) : (tensor<f32>, tensor<768xf32>) -> tensor<768xf32>
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_15_]], [[VAR_4_]], [[VAR_3_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<768xf32>, tensor<f32>, tensor<f32>) -> (tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_RecScale_]]_1, [[VAR_Offset_]]_2, [[VAR_Out_]]_3, [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_4_]], [[VAR_3_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x768xi8, #zhigh.layout<{dataLayout = "3DS", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_16_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Cast"([[VAR_16_]]) {saturate = 1 : si64, to = i32} : (tensor<?x?x768xf32>) -> tensor<?x?x768xi32>
// CHECK:           return [[VAR_17_]] : tensor<?x?x768xi32>
// CHECK:         }
}

// -----

func.func @matmulinteger_rewrite_from_mul_pattern_in_bert(%arg0: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
  %0 = onnx.Constant dense<5> : tensor<768x768xi8>
  %1 = onnx.Constant dense<0.00656270096> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<?x?x768xf32>) -> (tensor<?x?x768xui8>, tensor<f32>, tensor<ui8>)
  %3 = "onnx.MatMulInteger"(%y, %0, %y_zero_point, %2) : (tensor<?x?x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<?x?x768xi32>
  %4 = "onnx.Cast"(%3) {saturate = 1 : si64, to = f32} : (tensor<?x?x768xi32>) -> tensor<?x?x768xf32>
  %5 = "onnx.Mul"(%4, %y_scale) : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
  %6 = "onnx.Mul"(%5, %1) : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
  return %6 : tensor<?x?x768xf32>

// CHECK-LABEL:  func.func @matmulinteger_rewrite_from_mul_pattern_in_bert
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<5> : tensor<768x768xi8>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.00656270096> : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_6_]], [[VAR_6_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x?x768xf32>, none, none) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Reciprocal"([[VAR_4_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_5_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_3_]], [[VAR_7_]], [[VAR_8_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_3_]]) {saturate = 1 : si64, to = f32} : (tensor<768x768xi8>) -> tensor<768x768xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.ReduceSum"([[VAR_9_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<768x768xf32>, tensor<i64>) -> tensor<768xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_RecScale_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Div"([[VAR_11_]], [[VAR_7_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Mul"([[VAR_12_]], [[VAR_Offset_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_1_]]3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Mul"([[VAR_14_]], [[VAR_10_]]) : (tensor<f32>, tensor<768xf32>) -> tensor<768xf32>
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_15_]], [[VAR_2_]], [[VAR_1_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<768xf32>, tensor<f32>, tensor<f32>) -> (tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_7_]], [[VAR_8_]], [[VAR_Out_]]_3, [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_2_]], [[VAR_1_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_16_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
// CHECK:           return [[VAR_16_]] : tensor<?x?x768xf32>
// CHECK:         }
}

// -----

func.func @matmulinteger_fuse_add_pattern_in_bert(%arg0: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
   %0 = onnx.Constant dense<-2> : tensor<i64>
   %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
   %2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
   %3 = onnx.Constant dense<5.000000e+00> : tensor<768xf32>
   %4 = onnx.Constant dense<5> : tensor<768x768xi8>
   %5 = onnx.Constant dense<0.00656270096> : tensor<f32>
   %6 = onnx.Constant dense<0> : tensor<i8>
   %7 = "onnx.NoValue"() {value} : () -> none
   %Out, %RecScale, %Offset = "zhigh.QuantizedStick"(%arg0, %7, %7) {layout = "3DS", quantized_type = "DLFLOAT16"} : (tensor<?x?x768xf32>, none, none) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
   %8 = "onnx.Reciprocal"(%5) : (tensor<f32>) -> tensor<f32>
   %9 = "onnx.Cast"(%6) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
   %Out_0, %RecScale_1, %Offset_2 = "zhigh.QuantizedStick"(%4, %8, %9) {layout = "2D", quantized_type = "WEIGHTS"} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
   %10 = "onnx.Cast"(%4) {saturate = 1 : si64, to = f32} : (tensor<768x768xi8>) -> tensor<768x768xf32>
   %11 = "onnx.ReduceSum"(%10, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<768x768xf32>, tensor<i64>) -> tensor<768xf32>
   %12 = "onnx.Div"(%2, %RecScale) : (tensor<f32>, tensor<f32>) -> tensor<f32>
   %13 = "onnx.Div"(%12, %8) : (tensor<f32>, tensor<f32>) -> tensor<f32>
   %14 = "onnx.Mul"(%13, %Offset) : (tensor<f32>, tensor<f32>) -> tensor<f32>
   %15 = "onnx.Sub"(%1, %14) : (tensor<f32>, tensor<f32>) -> tensor<f32>
   %16 = "onnx.Mul"(%15, %11) : (tensor<f32>, tensor<768xf32>) -> tensor<768xf32>
   %17 = "onnx.Add"(%3, %16) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
   %Out_3, %RecScale_4, %Offset_5 = "zhigh.QuantizedStick"(%17, %2, %1) {layout = "1D", quantized_type = "DLFLOAT16"} : (tensor<768xf32>, tensor<f32>, tensor<f32>) -> (tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
   %Out_6, %OutRecScale, %OutOffset = "zhigh.QuantizedMatMul"(%Out, %RecScale, %Offset, %Out_0, %8, %9, %Out_3, %RecScale_4, %Offset_5, %2, %1) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
   %18 = "zhigh.Unstick"(%Out_6) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
   return %18 : tensor<?x?x768xf32>

// CHECK-FUSION-LABEL:  func.func @matmulinteger_fuse_add_pattern_in_bert
// CHECK-FUSION-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
// CHECK-FUSION-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2> : tensor<i64>
// CHECK-FUSION-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-FUSION-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-FUSION-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<768xf32>
// CHECK-FUSION-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<5> : tensor<768x768xi8>
// CHECK-FUSION-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0.00656270096> : tensor<f32>
// CHECK-FUSION-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-FUSION-DAG:       [[VAR_7_:%.+]] = "onnx.NoValue"() {value} : () -> none
   // CHECK-FUSION:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_7_]], [[VAR_7_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x?x768xf32>, none, none) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK-FUSION-DAG:       [[VAR_8_:%.+]] = "onnx.Reciprocal"([[VAR_5_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-FUSION-DAG:       [[VAR_9_:%.+]] = "onnx.Cast"([[VAR_6_]]) {saturate = 1 : si64, to = f32} : (tensor<i8>) -> tensor<f32>
   // CHECK-FUSION:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_4_]], [[VAR_8_]], [[VAR_9_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<768x768xi8>, tensor<f32>, tensor<f32>) -> (tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK-FUSION:           [[VAR_10_:%.+]] = "onnx.Cast"([[VAR_4_]]) {saturate = 1 : si64, to = f32} : (tensor<768x768xi8>) -> tensor<768x768xf32>
// CHECK-FUSION-DAG:       [[VAR_11_:%.+]] = "onnx.ReduceSum"([[VAR_10_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<768x768xf32>, tensor<i64>) -> tensor<768xf32>
// CHECK-FUSION-DAG:       [[VAR_12_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_RecScale_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-FUSION:           [[VAR_13_:%.+]] = "onnx.Div"([[VAR_12_]], [[VAR_8_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-FUSION:           [[VAR_14_:%.+]] = "onnx.Mul"([[VAR_13_]], [[VAR_Offset_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-FUSION:           [[VAR_15_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_1_]]4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-FUSION:           [[VAR_16_:%.+]] = "onnx.Mul"([[VAR_15_]], [[VAR_11_]]) : (tensor<f32>, tensor<768xf32>) -> tensor<768xf32>
// CHECK-FUSION:           [[VAR_17_:%.+]] = "onnx.Add"([[VAR_3_]], [[VAR_16_]]) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
   // CHECK-FUSION:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_17_]], [[VAR_2_]], [[VAR_1_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<768xf32>, tensor<f32>, tensor<f32>) -> (tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK-FUSION:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_8_]], [[VAR_9_]], [[VAR_Out_]]_3, [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_2_]], [[VAR_1_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<768x768xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<768xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK-FUSION:           [[VAR_18_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x768xf32>
// CHECK-FUSION:           return [[VAR_18_]] : tensor<?x?x768xf32>
// CHECK-FUSION:         }
}
