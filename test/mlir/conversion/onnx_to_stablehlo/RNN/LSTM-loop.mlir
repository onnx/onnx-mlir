// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo="enable-unroll=false" --canonicalize -split-input-file %s | FileCheck %s
func.func @test_lstm_loop(%arg0 : tensor<128x16x512xf32>, %arg1 : tensor<2x2048xf32>, %arg2 : tensor<2x1024x512xf32>, %arg3 : tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<2x16x256xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg2, %arg3, %arg1, %1, %0, %0, %1) {direction = "bidirectional", hidden_size = 256 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<128x16x512xf32>, tensor<2x1024x512xf32>, tensor<2x1024x256xf32>, tensor<2x2048xf32>, none, tensor<2x16x256xf32>, tensor<2x16x256xf32>, none) -> (tensor<128x2x16x256xf32>, tensor<2x16x256xf32>, tensor<2x16x256xf32>)
  return %Y : tensor<128x2x16x256xf32>
// CHECK-LABEL:  func.func @test_lstm_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x16x512xf32>, [[PARAM_1_:%.+]]: tensor<2x2048xf32>, [[PARAM_2_:%.+]]: tensor<2x1024x512xf32>, [[PARAM_3_:%.+]]: tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [16, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [1] : tensor<1xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.const_shape [1, 1] : tensor<2xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.const_shape [1, 1, 16, 256] : tensor<4xindex>
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.const_shape [16, 1024] : tensor<2xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.const_shape [16, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.const_shape [2048] : tensor<1xindex>
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.const_shape [1024, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = shape.const_shape [1024, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.constant dense<127> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.constant dense<768> : tensor<i64>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.constant dense<512> : tensor<i64>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.constant dense<256> : tensor<i64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.constant dense<128> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16x256xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<128x1x16x256xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           [[VAR_20_:%.+]] = stablehlo.dynamic_slice [[VAR_15_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.dynamic_reshape [[VAR_20_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.dynamic_slice [[VAR_15_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.dynamic_reshape [[VAR_22_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.dynamic_slice [[VAR_15_]], [[VAR_19_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.dynamic_reshape [[VAR_24_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.dynamic_slice [[VAR_15_]], [[VAR_19_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.dynamic_reshape [[VAR_26_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.dynamic_reshape [[VAR_28_]], [[VAR_8_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.dynamic_reshape [[VAR_29_]], [[VAR_8_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.slice [[PARAM_3_]] [0:1, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.slice [[PARAM_3_]] [1:2, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.dynamic_reshape [[VAR_32_]], [[VAR_7_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.dynamic_reshape [[VAR_33_]], [[VAR_7_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.transpose [[VAR_30_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.transpose [[VAR_34_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.transpose [[VAR_31_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.transpose [[VAR_35_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.dynamic_reshape [[VAR_40_]], [[VAR_6_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = stablehlo.dynamic_reshape [[VAR_41_]], [[VAR_6_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.slice [[VAR_42_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.slice [[VAR_42_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.slice [[VAR_42_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.slice [[VAR_42_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.slice [[VAR_42_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.slice [[VAR_42_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.slice [[VAR_42_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.slice [[VAR_42_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.slice [[VAR_43_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = stablehlo.slice [[VAR_43_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = stablehlo.slice [[VAR_43_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_55_:%.+]] = stablehlo.slice [[VAR_43_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = stablehlo.slice [[VAR_43_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = stablehlo.slice [[VAR_43_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = stablehlo.slice [[VAR_43_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = stablehlo.slice [[VAR_43_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_60_:%.+]]:4 = stablehlo.while([[VAR_iterArg_:%.+]] = [[VAR_18_]], [[VAR_iterArg_0_:%.+]] = [[VAR_16_]], [[VAR_iterArg_1_:%.+]] = [[VAR_21_]], [[VAR_iterArg_2_:%.+]] = [[VAR_23_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_63_:%.+]] = stablehlo.compare  LT, [[VAR_iterArg_]], [[VAR_14_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_64_:%.+]] = stablehlo.reshape [[VAR_63_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_64_]] : tensor<i1>
// CHECK:           } do {
// CHECK:             [[VAR_63_1_:%.+]] = stablehlo.reshape [[VAR_iterArg_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:             [[VAR_64_1_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_63_1_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 512] : (tensor<128x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_65_:%.+]] = stablehlo.dynamic_reshape [[VAR_64_1_]], [[VAR_5_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_66_:%.+]] = stablehlo.broadcast_in_dim [[VAR_65_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_67_:%.+]] = stablehlo.broadcast_in_dim [[VAR_36_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_68_:%.+]] = stablehlo.dot [[VAR_66_]], [[VAR_67_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_69_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_1_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_70_:%.+]] = stablehlo.broadcast_in_dim [[VAR_37_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_71_:%.+]] = stablehlo.dot [[VAR_69_]], [[VAR_70_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_72_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_68_]], [[VAR_4_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_73_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_71_]], [[VAR_4_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_74_:%.+]] = stablehlo.add [[VAR_72_]], [[VAR_73_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = stablehlo.dynamic_slice [[VAR_74_]], [[VAR_17_]], [[VAR_17_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_76_:%.+]] = stablehlo.dynamic_slice [[VAR_74_]], [[VAR_17_]], [[VAR_13_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_77_:%.+]] = stablehlo.dynamic_slice [[VAR_74_]], [[VAR_17_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_78_:%.+]] = stablehlo.dynamic_slice [[VAR_74_]], [[VAR_17_]], [[VAR_11_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_79_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_75_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_80_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_44_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_81_:%.+]] = stablehlo.add [[VAR_79_]], [[VAR_80_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_82_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_81_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_83_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_48_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_84_:%.+]] = stablehlo.add [[VAR_82_]], [[VAR_83_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_85_:%.+]] = stablehlo.logistic [[VAR_84_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_86_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_77_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_87_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_46_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_88_:%.+]] = stablehlo.add [[VAR_86_]], [[VAR_87_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_89_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_88_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_90_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_50_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_91_:%.+]] = stablehlo.add [[VAR_89_]], [[VAR_90_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_92_:%.+]] = stablehlo.logistic [[VAR_91_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_93_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_78_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_94_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_47_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_95_:%.+]] = stablehlo.add [[VAR_93_]], [[VAR_94_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_96_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_95_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_97_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_51_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_98_:%.+]] = stablehlo.add [[VAR_96_]], [[VAR_97_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_99_:%.+]] = stablehlo.tanh [[VAR_98_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_100_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_92_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_101_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_2_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_102_:%.+]] = stablehlo.multiply [[VAR_100_]], [[VAR_101_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_103_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_85_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_104_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_99_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_105_:%.+]] = stablehlo.multiply [[VAR_103_]], [[VAR_104_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_106_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_102_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_107_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_105_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_108_:%.+]] = stablehlo.add [[VAR_106_]], [[VAR_107_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_109_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_76_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_110_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_45_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_111_:%.+]] = stablehlo.add [[VAR_109_]], [[VAR_110_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_112_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_111_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_113_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_49_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_114_:%.+]] = stablehlo.add [[VAR_112_]], [[VAR_113_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_115_:%.+]] = stablehlo.logistic [[VAR_114_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_116_:%.+]] = stablehlo.tanh [[VAR_108_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_117_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_115_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_118_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_116_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_119_:%.+]] = stablehlo.multiply [[VAR_117_]], [[VAR_118_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_120_:%.+]] = stablehlo.dynamic_reshape [[VAR_119_]], [[VAR_3_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_121_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_]], [[VAR_2_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_122_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_0_]], [[VAR_121_]], [[VAR_120_]]) ({
// CHECK:             ^bb0([[arg4_:%.+]]: tensor<f32>, [[arg5_:%.+]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_]] : tensor<f32>
// CHECK:             }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_123_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_]], [[VAR_1_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_124_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_10_]], [[VAR_1_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_125_:%.+]] = stablehlo.add [[VAR_123_]], [[VAR_124_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_125_]], [[VAR_122_]], [[VAR_119_]], [[VAR_108_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_61_:%.+]]:4 = stablehlo.while([[VAR_iterArg_1_:%.+]] = [[VAR_9_]], [[VAR_iterArg_0_1_:%.+]] = [[VAR_16_]], [[VAR_iterArg_1_1_:%.+]] = [[VAR_25_]], [[VAR_iterArg_2_1_:%.+]] = [[VAR_27_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_63_2_:%.+]] = stablehlo.compare  GE, [[VAR_iterArg_1_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_64_2_:%.+]] = stablehlo.reshape [[VAR_63_2_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_64_2_]] : tensor<i1>
// CHECK:           } do {
// CHECK:             [[VAR_63_3_:%.+]] = stablehlo.reshape [[VAR_iterArg_1_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:             [[VAR_64_3_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_63_3_]], [[VAR_17_]], [[VAR_17_]], sizes = [1, 16, 512] : (tensor<128x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_65_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_64_3_]], [[VAR_5_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_66_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_65_1_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_67_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_38_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_68_1_:%.+]] = stablehlo.dot [[VAR_66_1_]], [[VAR_67_1_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_69_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_1_1_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_70_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_39_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_71_1_:%.+]] = stablehlo.dot [[VAR_69_1_]], [[VAR_70_1_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_72_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_68_1_]], [[VAR_4_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_73_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_71_1_]], [[VAR_4_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_74_1_:%.+]] = stablehlo.add [[VAR_72_1_]], [[VAR_73_1_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_75_1_:%.+]] = stablehlo.dynamic_slice [[VAR_74_1_]], [[VAR_17_]], [[VAR_17_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_76_1_:%.+]] = stablehlo.dynamic_slice [[VAR_74_1_]], [[VAR_17_]], [[VAR_13_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_77_1_:%.+]] = stablehlo.dynamic_slice [[VAR_74_1_]], [[VAR_17_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_78_1_:%.+]] = stablehlo.dynamic_slice [[VAR_74_1_]], [[VAR_17_]], [[VAR_11_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_79_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_75_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_80_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_52_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_81_1_:%.+]] = stablehlo.add [[VAR_79_1_]], [[VAR_80_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_82_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_81_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_83_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_56_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_84_1_:%.+]] = stablehlo.add [[VAR_82_1_]], [[VAR_83_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_85_1_:%.+]] = stablehlo.logistic [[VAR_84_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_86_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_77_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_87_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_54_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_88_1_:%.+]] = stablehlo.add [[VAR_86_1_]], [[VAR_87_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_89_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_88_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_90_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_58_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_91_1_:%.+]] = stablehlo.add [[VAR_89_1_]], [[VAR_90_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_92_1_:%.+]] = stablehlo.logistic [[VAR_91_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_93_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_78_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_94_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_55_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_95_1_:%.+]] = stablehlo.add [[VAR_93_1_]], [[VAR_94_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_96_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_95_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_97_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_59_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_98_1_:%.+]] = stablehlo.add [[VAR_96_1_]], [[VAR_97_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_99_1_:%.+]] = stablehlo.tanh [[VAR_98_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_100_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_92_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_101_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_2_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_102_1_:%.+]] = stablehlo.multiply [[VAR_100_1_]], [[VAR_101_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_103_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_85_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_104_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_99_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_105_1_:%.+]] = stablehlo.multiply [[VAR_103_1_]], [[VAR_104_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_106_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_102_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_107_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_105_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_108_1_:%.+]] = stablehlo.add [[VAR_106_1_]], [[VAR_107_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_109_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_76_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_110_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_53_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_111_1_:%.+]] = stablehlo.add [[VAR_109_1_]], [[VAR_110_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_112_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_111_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_113_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_57_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_114_1_:%.+]] = stablehlo.add [[VAR_112_1_]], [[VAR_113_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_115_1_:%.+]] = stablehlo.logistic [[VAR_114_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_116_1_:%.+]] = stablehlo.tanh [[VAR_108_1_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_117_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_115_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_118_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_116_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_119_1_:%.+]] = stablehlo.multiply [[VAR_117_1_]], [[VAR_118_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_120_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_119_1_]], [[VAR_3_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_121_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_1_]], [[VAR_2_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_122_1_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_0_1_]], [[VAR_121_1_]], [[VAR_120_1_]]) ({
// CHECK:             ^bb0([[arg4_:%.+]]: tensor<f32>, [[arg5_:%.+]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_]] : tensor<f32>
// CHECK:             }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_123_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_1_]], [[VAR_1_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_124_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_10_]], [[VAR_1_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_125_1_:%.+]] = stablehlo.subtract [[VAR_123_1_]], [[VAR_124_1_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_125_1_]], [[VAR_122_1_]], [[VAR_119_1_]], [[VAR_108_1_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_62_:%.+]] = stablehlo.concatenate [[VAR_60_]]#1, [[VAR_61_]]#1, dim = 1 : (tensor<128x1x16x256xf32>, tensor<128x1x16x256xf32>) -> tensor<128x2x16x256xf32>
// CHECK:           return [[VAR_62_]] : tensor<128x2x16x256xf32>
// CHECK:         }
}