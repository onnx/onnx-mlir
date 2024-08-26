// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo="enable-unroll=false" --canonicalize -split-input-file %s | FileCheck %s
func.func @test_lstm_loop(%arg0 : tensor<128x16x512xf32>, %arg1 : tensor<2x2048xf32>, %arg2 : tensor<2x1024x512xf32>, %arg3 : tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<2x16x256xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg2, %arg3, %arg1, %1, %0, %0, %1) {direction = "bidirectional", hidden_size = 256 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<128x16x512xf32>, tensor<2x1024x512xf32>, tensor<2x1024x256xf32>, tensor<2x2048xf32>, none, tensor<2x16x256xf32>, tensor<2x16x256xf32>, none) -> (tensor<128x2x16x256xf32>, tensor<2x16x256xf32>, tensor<2x16x256xf32>)
  return %Y : tensor<128x2x16x256xf32>
// CHECK-LABEL:  func.func @test_lstm_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x16x512xf32>, [[PARAM_1_:%.+]]: tensor<2x2048xf32>, [[PARAM_2_:%.+]]: tensor<2x1024x512xf32>, [[PARAM_3_:%.+]]: tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [1] : tensor<1xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [1, 1] : tensor<2xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.const_shape [1, 1, 16, 256] : tensor<4xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.const_shape [16, 1024] : tensor<2xindex>
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.const_shape [16, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.const_shape [2048] : tensor<1xindex>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.const_shape [1024, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.const_shape [1024, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = shape.const_shape [16, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_c_:%.+]] = stablehlo.constant dense<127> : tensor<1xi64>
// CHECK-DAG:       [[VAR_c_0_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_c_1_:%.+]] = stablehlo.constant dense<768> : tensor<i64>
// CHECK-DAG:       [[VAR_c_2_:%.+]] = stablehlo.constant dense<512> : tensor<i64>
// CHECK-DAG:       [[VAR_c_3_:%.+]] = stablehlo.constant dense<256> : tensor<i64>
// CHECK-DAG:       [[VAR_c_4_:%.+]] = stablehlo.constant dense<128> : tensor<1xi64>
// CHECK-DAG:       [[VAR_cst_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16x256xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<128x1x16x256xf32>
// CHECK-DAG:       [[VAR_c_6_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[VAR_c_7_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_c_8_:%.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.dynamic_slice [[VAR_cst_]], [[VAR_c_6_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.dynamic_reshape [[VAR_9_]], [[VAR_8_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.dynamic_slice [[VAR_cst_]], [[VAR_c_6_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_reshape [[VAR_11_]], [[VAR_8_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.dynamic_slice [[VAR_cst_]], [[VAR_c_8_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.dynamic_reshape [[VAR_13_]], [[VAR_8_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.dynamic_slice [[VAR_cst_]], [[VAR_c_8_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.dynamic_reshape [[VAR_15_]], [[VAR_8_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.dynamic_reshape [[VAR_17_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.dynamic_reshape [[VAR_18_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.slice [[PARAM_3_]] [0:1, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.slice [[PARAM_3_]] [1:2, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.dynamic_reshape [[VAR_21_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.dynamic_reshape [[VAR_22_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.transpose [[VAR_19_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.transpose [[VAR_23_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.transpose [[VAR_20_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.transpose [[VAR_24_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.dynamic_reshape [[VAR_29_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.dynamic_reshape [[VAR_30_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.slice [[VAR_31_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.slice [[VAR_31_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.slice [[VAR_31_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.slice [[VAR_31_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.slice [[VAR_31_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.slice [[VAR_31_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.slice [[VAR_31_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.slice [[VAR_31_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.slice [[VAR_32_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.slice [[VAR_32_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = stablehlo.slice [[VAR_32_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.slice [[VAR_32_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.slice [[VAR_32_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.slice [[VAR_32_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.slice [[VAR_32_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.slice [[VAR_32_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_49_:%.+]]:4 = stablehlo.while([[VAR_iterArg_:%.+]] = [[VAR_c_7_]], [[VAR_iterArg_9_:%.+]] = [[VAR_cst_5_]], [[VAR_iterArg_10_:%.+]] = [[VAR_10_]], [[VAR_iterArg_11_:%.+]] = [[VAR_12_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_52_:%.+]] = stablehlo.compare  LT, [[VAR_iterArg_]], [[VAR_c_4_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_53_:%.+]] = stablehlo.reshape [[VAR_52_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_53_]] : tensor<i1>
// CHECK:           } do {
// CHECK:             [[VAR_52_1_:%.+]] = stablehlo.reshape [[VAR_iterArg_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:             [[VAR_53_1_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_52_1_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 512] : (tensor<128x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_54_:%.+]] = stablehlo.dynamic_reshape [[VAR_53_1_]], [[VAR_4_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = stablehlo.broadcast_in_dim [[VAR_54_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_56_:%.+]] = stablehlo.broadcast_in_dim [[VAR_25_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = stablehlo.dot [[VAR_55_]], [[VAR_56_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_58_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_10_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_59_:%.+]] = stablehlo.broadcast_in_dim [[VAR_26_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_60_:%.+]] = stablehlo.dot [[VAR_58_]], [[VAR_59_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_61_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_57_]], [[VAR_3_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_62_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_60_]], [[VAR_3_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_63_:%.+]] = stablehlo.add [[VAR_61_]], [[VAR_62_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_64_:%.+]] = stablehlo.dynamic_slice [[VAR_63_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_65_:%.+]] = stablehlo.dynamic_slice [[VAR_63_]], [[VAR_c_6_]], [[VAR_c_3_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_66_:%.+]] = stablehlo.dynamic_slice [[VAR_63_]], [[VAR_c_6_]], [[VAR_c_2_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_67_:%.+]] = stablehlo.dynamic_slice [[VAR_63_]], [[VAR_c_6_]], [[VAR_c_1_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_68_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_64_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_69_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_33_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_70_:%.+]] = stablehlo.add [[VAR_68_]], [[VAR_69_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_71_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_70_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_72_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_37_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_73_:%.+]] = stablehlo.add [[VAR_71_]], [[VAR_72_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_74_:%.+]] = stablehlo.logistic [[VAR_73_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_66_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_76_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_35_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_77_:%.+]] = stablehlo.add [[VAR_75_]], [[VAR_76_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_78_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_77_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_79_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_39_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_80_:%.+]] = stablehlo.add [[VAR_78_]], [[VAR_79_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_81_:%.+]] = stablehlo.logistic [[VAR_80_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_82_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_67_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_83_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_36_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_84_:%.+]] = stablehlo.add [[VAR_82_]], [[VAR_83_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_85_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_84_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_86_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_40_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_87_:%.+]] = stablehlo.add [[VAR_85_]], [[VAR_86_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_88_:%.+]] = stablehlo.tanh [[VAR_87_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_89_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_81_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_90_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_11_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_91_:%.+]] = stablehlo.multiply [[VAR_89_]], [[VAR_90_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_92_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_74_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_93_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_88_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_94_:%.+]] = stablehlo.multiply [[VAR_92_]], [[VAR_93_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_95_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_91_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_96_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_94_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_97_:%.+]] = stablehlo.add [[VAR_95_]], [[VAR_96_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_98_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_65_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_99_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_34_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_100_:%.+]] = stablehlo.add [[VAR_98_]], [[VAR_99_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_101_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_100_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_102_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_38_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_103_:%.+]] = stablehlo.add [[VAR_101_]], [[VAR_102_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_104_:%.+]] = stablehlo.logistic [[VAR_103_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_105_:%.+]] = stablehlo.tanh [[VAR_97_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_106_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_104_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_107_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_105_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_108_:%.+]] = stablehlo.multiply [[VAR_106_]], [[VAR_107_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_109_:%.+]] = stablehlo.dynamic_reshape [[VAR_108_]], [[VAR_2_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_110_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_]], [[VAR_1_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_111_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_9_]], [[VAR_110_]], [[VAR_109_]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:             ^bb0([[arg4_:%.+]]: tensor<f32>, [[arg5_:%.+]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_]] : tensor<f32>
// CHECK:             }) : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_112_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_]], [[VAR_0_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_113_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_c_0_]], [[VAR_0_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_114_:%.+]] = stablehlo.add [[VAR_112_]], [[VAR_113_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_114_]], [[VAR_111_]], [[VAR_108_]], [[VAR_97_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_50_:%.+]]:4 = stablehlo.while([[VAR_iterArg_1_:%.+]] = [[VAR_c_]], [[VAR_iterArg_9_1_:%.+]] = [[VAR_c_]]st_5, [[VAR_iterArg_10_1_:%.+]] = [[VAR_14_]], [[VAR_iterArg_11_1_:%.+]] = [[VAR_16_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_52_2_:%.+]] = stablehlo.compare  GE, [[VAR_iterArg_1_]], [[VAR_c_7_]] : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_53_2_:%.+]] = stablehlo.reshape [[VAR_52_2_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_53_2_]] : tensor<i1>
// CHECK:           } do {
// CHECK:             [[VAR_52_3_:%.+]] = stablehlo.reshape [[VAR_iterArg_1_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:             [[VAR_53_3_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_52_3_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [1, 16, 512] : (tensor<128x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_54_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_53_3_]], [[VAR_4_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_55_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_54_1_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_56_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_27_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_1_:%.+]] = stablehlo.dot [[VAR_55_1_]], [[VAR_56_1_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_58_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_10_1_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_59_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_28_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_60_1_:%.+]] = stablehlo.dot [[VAR_58_1_]], [[VAR_59_1_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_61_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_57_1_]], [[VAR_3_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_62_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_60_1_]], [[VAR_3_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_63_1_:%.+]] = stablehlo.add [[VAR_61_1_]], [[VAR_62_1_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_64_1_:%.+]] = stablehlo.dynamic_slice [[VAR_63_1_]], [[VAR_c_6_]], [[VAR_c_6_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_65_1_:%.+]] = stablehlo.dynamic_slice [[VAR_63_1_]], [[VAR_c_6_]], [[VAR_c_3_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_66_1_:%.+]] = stablehlo.dynamic_slice [[VAR_63_1_]], [[VAR_c_6_]], [[VAR_c_2_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_67_1_:%.+]] = stablehlo.dynamic_slice [[VAR_63_1_]], [[VAR_c_6_]], [[VAR_c_1_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_68_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_64_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_69_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_41_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_70_1_:%.+]] = stablehlo.add [[VAR_68_1_]], [[VAR_69_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_71_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_70_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_72_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_45_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_73_1_:%.+]] = stablehlo.add [[VAR_71_1_]], [[VAR_72_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_74_1_:%.+]] = stablehlo.logistic [[VAR_73_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_75_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_66_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_76_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_43_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_77_1_:%.+]] = stablehlo.add [[VAR_75_1_]], [[VAR_76_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_78_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_77_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_79_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_47_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_80_1_:%.+]] = stablehlo.add [[VAR_78_1_]], [[VAR_79_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_81_1_:%.+]] = stablehlo.logistic [[VAR_80_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_82_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_67_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_83_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_44_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_84_1_:%.+]] = stablehlo.add [[VAR_82_1_]], [[VAR_83_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_85_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_84_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_86_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_48_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_87_1_:%.+]] = stablehlo.add [[VAR_85_1_]], [[VAR_86_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_88_1_:%.+]] = stablehlo.tanh [[VAR_87_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_89_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_81_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_90_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_11_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_91_1_:%.+]] = stablehlo.multiply [[VAR_89_1_]], [[VAR_90_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_92_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_74_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_93_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_88_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_94_1_:%.+]] = stablehlo.multiply [[VAR_92_1_]], [[VAR_93_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_95_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_91_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_96_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_94_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_97_1_:%.+]] = stablehlo.add [[VAR_95_1_]], [[VAR_96_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_98_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_65_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_99_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_42_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_100_1_:%.+]] = stablehlo.add [[VAR_98_1_]], [[VAR_99_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_101_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_100_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_102_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_46_]], [[VAR_8_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_103_1_:%.+]] = stablehlo.add [[VAR_101_1_]], [[VAR_102_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_104_1_:%.+]] = stablehlo.logistic [[VAR_103_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_105_1_:%.+]] = stablehlo.tanh [[VAR_97_1_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_106_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_104_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_107_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_105_1_]], [[VAR_8_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_108_1_:%.+]] = stablehlo.multiply [[VAR_106_1_]], [[VAR_107_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_109_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_108_1_]], [[VAR_2_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_110_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_1_]], [[VAR_1_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_111_1_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_9_1_]], [[VAR_110_1_]], [[VAR_109_1_]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK:             ^bb0([[arg4_]]: tensor<f32>, [[arg5_]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_]] : tensor<f32>
// CHECK:             }) : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_112_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_1_]], [[VAR_0_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_113_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_c_0_]], [[VAR_0_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_114_1_:%.+]] = stablehlo.subtract [[VAR_112_1_]], [[VAR_113_1_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_114_1_]], [[VAR_111_1_]], [[VAR_108_1_]], [[VAR_97_1_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_51_:%.+]] = stablehlo.concatenate [[VAR_49_]]#1, [[VAR_50_]]#1, dim = 1 : (tensor<128x1x16x256xf32>, tensor<128x1x16x256xf32>) -> tensor<128x2x16x256xf32>
// CHECK:           return [[VAR_51_]] : tensor<128x2x16x256xf32>
// CHECK:         }
}