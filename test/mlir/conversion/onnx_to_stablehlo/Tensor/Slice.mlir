// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x2xf32>
// CHECK:         }

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x3xf32>
// CHECK:         }

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x2xf32>
// CHECK:         }

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[2, -1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x2xf32>
// CHECK:         }

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[5, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x2xf32>
// CHECK:         }

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 4] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<[1, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<[1, -2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[VAR_5_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.slice [[VAR_4_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_11_:%.+]] = stablehlo.compare  LT, [[VAR_9_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.negate [[VAR_9_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_10_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.add [[VAR_8_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_14_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_10_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_13_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_12_]], [[VAR_16_]], [[PARAM_0_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.compare  GT, [[VAR_18_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.select [[VAR_21_]], [[VAR_2_]], [[VAR_18_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.compare  LT, [[VAR_22_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.add [[VAR_22_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_24_]], [[VAR_22_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_17_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_17_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[VAR_5_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.slice [[VAR_4_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  LT, [[VAR_30_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_32_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<2x4xi1>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.negate [[VAR_30_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_31_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.add [[VAR_29_]], [[VAR_7_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.reverse [[VAR_20_]], dims = [1] : tensor<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_35_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_31_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_34_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_33_]], [[VAR_37_]], [[VAR_20_]] : tensor<2x4xi1>, tensor<2x4xf32>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.compare  GT, [[VAR_39_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_42_]], [[VAR_1_]], [[VAR_39_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.compare  LT, [[VAR_43_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.add [[VAR_43_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.select [[VAR_44_]], [[VAR_45_]], [[VAR_43_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_38_]], [[VAR_6_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_38_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_25_]], [[VAR_46_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.concatenate [[VAR_19_]], [[VAR_40_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_53_:%.+]] = stablehlo.real_dynamic_slice [[VAR_41_]], [[VAR_50_]], [[VAR_51_]], [[VAR_52_]] : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_53_]] : tensor<1x2xf32>
// CHECK:         }

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x5xf32>) -> tensor<?x2x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<-1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK:           [[VAR_7_:%.+]] = shape.get_extent [[VAR_6_]], [[CST_0_]] : tensor<3xindex>, index -> index
// CHECK:           [[VAR_8_:%.+]] = shape.from_extents [[VAR_7_]] : index
// CHECK:           [[VAR_9_:%.+]] = shape.to_extent_tensor [[VAR_8_]] : !shape.shape -> tensor<1xindex>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : tensor<1xindex> to tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.slice [[VAR_2_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.slice [[VAR_2_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.slice [[VAR_3_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_14_:%.+]] = stablehlo.compare  LT, [[VAR_12_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_14_]], [[VAR_6_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.negate [[VAR_12_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.add [[VAR_13_]], [[VAR_5_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.add [[VAR_11_]], [[VAR_5_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [1] : tensor<?x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.select [[VAR_14_]], [[VAR_17_]], [[VAR_11_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.select [[VAR_14_]], [[VAR_18_]], [[VAR_13_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.select [[VAR_14_]], [[VAR_16_]], [[VAR_12_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.select [[VAR_15_]], [[VAR_19_]], [[PARAM_0_]] : tensor<?x4x5xi1>, tensor<?x4x5xf32>
// CHECK:           [[VAR_24_:%.+]] = stablehlo.compare  GT, [[VAR_21_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_25_:%.+]] = stablehlo.select [[VAR_24_]], [[VAR_1_]], [[VAR_21_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_25_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.add [[VAR_25_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_27_]], [[VAR_25_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.compare  LT, [[VAR_20_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.add [[VAR_20_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.select [[VAR_29_]], [[VAR_30_]], [[VAR_20_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.slice [[VAR_2_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.slice [[VAR_2_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.slice [[VAR_3_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_35_:%.+]] = stablehlo.compare  LT, [[VAR_33_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_35_]], [[VAR_6_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.negate [[VAR_33_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.add [[VAR_34_]], [[VAR_5_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.add [[VAR_32_]], [[VAR_5_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.reverse [[VAR_23_]], dims = [2] : tensor<?x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.select [[VAR_35_]], [[VAR_38_]], [[VAR_32_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.select [[VAR_35_]], [[VAR_39_]], [[VAR_34_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_43_:%.+]] = stablehlo.select [[VAR_35_]], [[VAR_37_]], [[VAR_33_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.select [[VAR_36_]], [[VAR_40_]], [[VAR_23_]] : tensor<?x4x5xi1>, tensor<?x4x5xf32>
// CHECK:           [[VAR_45_:%.+]] = stablehlo.compare  GT, [[VAR_42_]], [[VAR_0_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_46_:%.+]] = stablehlo.select [[VAR_45_]], [[VAR_0_]], [[VAR_42_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.compare  LT, [[VAR_46_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.add [[VAR_46_]], [[VAR_0_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.select [[VAR_47_]], [[VAR_48_]], [[VAR_46_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.compare  LT, [[VAR_41_]], [[VAR_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.add [[VAR_41_]], [[VAR_0_]] : tensor<1xi64>
// CHECK:           [[VAR_52_:%.+]] = stablehlo.select [[VAR_50_]], [[VAR_51_]], [[VAR_41_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_53_:%.+]] = stablehlo.concatenate [[VAR_4_]], [[VAR_31_]], [[VAR_52_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_54_:%.+]] = stablehlo.concatenate [[VAR_10_]], [[VAR_28_]], [[VAR_49_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_55_:%.+]] = stablehlo.concatenate [[VAR_5_]], [[VAR_22_]], [[VAR_43_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_56_:%.+]] = stablehlo.real_dynamic_slice [[VAR_44_]], [[VAR_53_]], [[VAR_54_]], [[VAR_55_]] : (tensor<?x4x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x2x3xf32>
// CHECK:           return [[VAR_56_]] : tensor<?x2x3xf32>
// CHECK:         }

// -----

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<*xi64> {
  %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%res) : (tensor<*xi64>) -> ()
}

// CHECK-LABEL:  func.func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2xi64>, [[PARAM_1_:%.+]]: tensor<2xi64>, [[PARAM_2_:%.+]]: tensor<2xi64>) -> tensor<3x?x?xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [3, 4, 5] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.constant dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.slice [[PARAM_0_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.compare  LT, [[VAR_8_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_10_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<3x4x5xi1>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.negate [[VAR_8_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.add [[VAR_9_]], [[VAR_6_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.add [[VAR_7_]], [[VAR_6_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.reverse [[VAR_4_]], dims = [1] : tensor<3x4x5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.select [[VAR_10_]], [[VAR_13_]], [[VAR_7_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.select [[VAR_10_]], [[VAR_14_]], [[VAR_9_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.select [[VAR_10_]], [[VAR_12_]], [[VAR_8_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.select [[VAR_11_]], [[VAR_15_]], [[VAR_4_]] : tensor<3x4x5xi1>, tensor<3x4x5xi64>
// CHECK:           [[VAR_20_:%.+]] = stablehlo.compare  GT, [[VAR_17_]], [[VAR_2_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_21_:%.+]] = stablehlo.select [[VAR_20_]], [[VAR_2_]], [[VAR_17_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.compare  LT, [[VAR_21_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.add [[VAR_21_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.select [[VAR_22_]], [[VAR_23_]], [[VAR_21_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.compare  LT, [[VAR_16_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.add [[VAR_16_]], [[VAR_2_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.select [[VAR_25_]], [[VAR_26_]], [[VAR_16_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.slice [[PARAM_0_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_31_:%.+]] = stablehlo.compare  LT, [[VAR_29_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_31_]], [[VAR_0_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<3x4x5xi1>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.negate [[VAR_29_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.add [[VAR_30_]], [[VAR_6_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_28_]], [[VAR_6_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.reverse [[VAR_19_]], dims = [2] : tensor<3x4x5xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.select [[VAR_31_]], [[VAR_34_]], [[VAR_28_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.select [[VAR_31_]], [[VAR_35_]], [[VAR_30_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.select [[VAR_31_]], [[VAR_33_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_36_]], [[VAR_19_]] : tensor<3x4x5xi1>, tensor<3x4x5xi64>
// CHECK:           [[VAR_41_:%.+]] = stablehlo.compare  GT, [[VAR_38_]], [[VAR_1_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_42_:%.+]] = stablehlo.select [[VAR_41_]], [[VAR_1_]], [[VAR_38_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_43_:%.+]] = stablehlo.compare  LT, [[VAR_42_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.add [[VAR_42_]], [[VAR_1_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.select [[VAR_43_]], [[VAR_44_]], [[VAR_42_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.compare  LT, [[VAR_37_]], [[VAR_5_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.add [[VAR_37_]], [[VAR_1_]] : tensor<1xi64>
// CHECK:           [[VAR_48_:%.+]] = stablehlo.select [[VAR_46_]], [[VAR_47_]], [[VAR_37_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.concatenate [[VAR_5_]], [[VAR_27_]], [[VAR_48_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.concatenate [[VAR_3_]], [[VAR_24_]], [[VAR_45_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_6_]], [[VAR_18_]], [[VAR_39_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_52_:%.+]] = stablehlo.real_dynamic_slice [[VAR_40_]], [[VAR_49_]], [[VAR_50_]], [[VAR_51_]] : (tensor<3x4x5xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x?x?xi64>
// CHECK:           return [[VAR_52_]] : tensor<3x?x?xi64>
// CHECK:         }
