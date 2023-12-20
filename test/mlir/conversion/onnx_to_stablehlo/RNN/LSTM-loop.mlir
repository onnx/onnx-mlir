// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo="enable-unroll=false" --canonicalize -split-input-file %s | FileCheck %s
func.func @test_lstm_loop(%arg0 : tensor<128x16x512xf32>, %arg1 : tensor<2x2048xf32>, %arg2 : tensor<2x1024x512xf32>, %arg3 : tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<2x16x256xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg2, %arg3, %arg1, %1, %0, %0, %1) {direction = "bidirectional", hidden_size = 256 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<128x16x512xf32>, tensor<2x1024x512xf32>, tensor<2x1024x256xf32>, tensor<2x2048xf32>, none, tensor<2x16x256xf32>, tensor<2x16x256xf32>, none) -> (tensor<128x2x16x256xf32>, tensor<2x16x256xf32>, tensor<2x16x256xf32>)
  return %Y : tensor<128x2x16x256xf32>
// CHECK-LABEL:  func.func @test_lstm_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x16x512xf32>, [[PARAM_1_:%.+]]: tensor<2x2048xf32>, [[PARAM_2_:%.+]]: tensor<2x1024x512xf32>, [[PARAM_3_:%.+]]: tensor<2x1024x256xf32>) -> tensor<128x2x16x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [16, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [1, 1] : tensor<2xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.const_shape [1, 1, 16, 256] : tensor<4xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.const_shape [16, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.const_shape [128, 16, 512] : tensor<3xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.const_shape [2048] : tensor<1xindex>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.const_shape [1024, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.const_shape [1024, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = shape.const_shape [2, 16, 256] : tensor<3xindex>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.constant dense<127> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.constant dense<1024> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.constant dense<768> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = shape.const_shape [16, 1024] : tensor<2xindex>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.constant dense<512> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = shape.const_shape [1] : tensor<1xindex>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.constant dense<128> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.constant dense<16> : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16x256xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<128x1x16x256xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.slice [[VAR_20_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_26_:%.+]] = stablehlo.compare  LT, [[VAR_24_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_26_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.negate [[VAR_24_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.add [[VAR_25_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.add [[VAR_23_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.reverse [[VAR_18_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_29_]], [[VAR_23_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_30_]], [[VAR_25_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.select [[VAR_26_]], [[VAR_28_]], [[VAR_24_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.select [[VAR_27_]], [[VAR_31_]], [[VAR_18_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_36_:%.+]] = stablehlo.compare  GT, [[VAR_33_]], [[VAR_22_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_37_:%.+]] = stablehlo.select [[VAR_36_]], [[VAR_22_]], [[VAR_33_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.compare  LT, [[VAR_37_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.add [[VAR_37_]], [[VAR_22_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.select [[VAR_38_]], [[VAR_39_]], [[VAR_37_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.compare  LT, [[VAR_32_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.add [[VAR_32_]], [[VAR_22_]] : tensor<1xi64>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.select [[VAR_41_]], [[VAR_42_]], [[VAR_32_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.concatenate [[VAR_43_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.concatenate [[VAR_40_]], [[VAR_17_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.concatenate [[VAR_34_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_47_:%.+]] = stablehlo.real_dynamic_slice [[VAR_35_]], [[VAR_44_]], [[VAR_45_]], [[VAR_46_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.dynamic_reshape [[VAR_47_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.slice [[VAR_20_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_52_:%.+]] = stablehlo.compare  LT, [[VAR_50_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_53_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_52_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_54_:%.+]] = stablehlo.negate [[VAR_50_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_55_:%.+]] = stablehlo.add [[VAR_51_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_56_:%.+]] = stablehlo.add [[VAR_49_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_57_:%.+]] = stablehlo.reverse [[VAR_18_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_58_:%.+]] = stablehlo.select [[VAR_52_]], [[VAR_55_]], [[VAR_49_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_59_:%.+]] = stablehlo.select [[VAR_52_]], [[VAR_56_]], [[VAR_51_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_60_:%.+]] = stablehlo.select [[VAR_52_]], [[VAR_54_]], [[VAR_50_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_61_:%.+]] = stablehlo.select [[VAR_53_]], [[VAR_57_]], [[VAR_18_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_62_:%.+]] = stablehlo.compare  GT, [[VAR_59_]], [[VAR_22_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_63_:%.+]] = stablehlo.select [[VAR_62_]], [[VAR_22_]], [[VAR_59_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_64_:%.+]] = stablehlo.compare  LT, [[VAR_63_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_65_:%.+]] = stablehlo.add [[VAR_63_]], [[VAR_22_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_66_:%.+]] = stablehlo.select [[VAR_64_]], [[VAR_65_]], [[VAR_63_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_67_:%.+]] = stablehlo.compare  LT, [[VAR_58_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_68_:%.+]] = stablehlo.add [[VAR_58_]], [[VAR_22_]] : tensor<1xi64>
// CHECK:           [[VAR_69_:%.+]] = stablehlo.select [[VAR_67_]], [[VAR_68_]], [[VAR_58_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_70_:%.+]] = stablehlo.concatenate [[VAR_69_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_71_:%.+]] = stablehlo.concatenate [[VAR_66_]], [[VAR_17_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_72_:%.+]] = stablehlo.concatenate [[VAR_60_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_73_:%.+]] = stablehlo.real_dynamic_slice [[VAR_61_]], [[VAR_70_]], [[VAR_71_]], [[VAR_72_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_74_:%.+]] = stablehlo.dynamic_reshape [[VAR_73_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_75_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_76_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_77_:%.+]] = stablehlo.slice [[VAR_22_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_78_:%.+]] = stablehlo.compare  LT, [[VAR_76_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_79_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_78_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_80_:%.+]] = stablehlo.negate [[VAR_76_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_81_:%.+]] = stablehlo.add [[VAR_77_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_82_:%.+]] = stablehlo.add [[VAR_75_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_83_:%.+]] = stablehlo.reverse [[VAR_18_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_84_:%.+]] = stablehlo.select [[VAR_78_]], [[VAR_81_]], [[VAR_75_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_85_:%.+]] = stablehlo.select [[VAR_78_]], [[VAR_82_]], [[VAR_77_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_86_:%.+]] = stablehlo.select [[VAR_78_]], [[VAR_80_]], [[VAR_76_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_87_:%.+]] = stablehlo.select [[VAR_79_]], [[VAR_83_]], [[VAR_18_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_88_:%.+]] = stablehlo.compare  GT, [[VAR_85_]], [[VAR_22_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_89_:%.+]] = stablehlo.select [[VAR_88_]], [[VAR_22_]], [[VAR_85_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_90_:%.+]] = stablehlo.compare  LT, [[VAR_89_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_91_:%.+]] = stablehlo.add [[VAR_89_]], [[VAR_22_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_92_:%.+]] = stablehlo.select [[VAR_90_]], [[VAR_91_]], [[VAR_89_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_93_:%.+]] = stablehlo.compare  LT, [[VAR_84_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_94_:%.+]] = stablehlo.add [[VAR_84_]], [[VAR_22_]] : tensor<1xi64>
// CHECK:           [[VAR_95_:%.+]] = stablehlo.select [[VAR_93_]], [[VAR_94_]], [[VAR_84_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_96_:%.+]] = stablehlo.concatenate [[VAR_95_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_97_:%.+]] = stablehlo.concatenate [[VAR_92_]], [[VAR_17_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_98_:%.+]] = stablehlo.concatenate [[VAR_86_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_99_:%.+]] = stablehlo.real_dynamic_slice [[VAR_87_]], [[VAR_96_]], [[VAR_97_]], [[VAR_98_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_100_:%.+]] = stablehlo.dynamic_reshape [[VAR_99_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_101_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_102_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_103_:%.+]] = stablehlo.slice [[VAR_22_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_104_:%.+]] = stablehlo.compare  LT, [[VAR_102_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_105_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_104_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_106_:%.+]] = stablehlo.negate [[VAR_102_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_107_:%.+]] = stablehlo.add [[VAR_103_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_108_:%.+]] = stablehlo.add [[VAR_101_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_109_:%.+]] = stablehlo.reverse [[VAR_18_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_110_:%.+]] = stablehlo.select [[VAR_104_]], [[VAR_107_]], [[VAR_101_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_111_:%.+]] = stablehlo.select [[VAR_104_]], [[VAR_108_]], [[VAR_103_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_112_:%.+]] = stablehlo.select [[VAR_104_]], [[VAR_106_]], [[VAR_102_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_113_:%.+]] = stablehlo.select [[VAR_105_]], [[VAR_109_]], [[VAR_18_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_114_:%.+]] = stablehlo.compare  GT, [[VAR_111_]], [[VAR_22_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_115_:%.+]] = stablehlo.select [[VAR_114_]], [[VAR_22_]], [[VAR_111_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_116_:%.+]] = stablehlo.compare  LT, [[VAR_115_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_117_:%.+]] = stablehlo.add [[VAR_115_]], [[VAR_22_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_118_:%.+]] = stablehlo.select [[VAR_116_]], [[VAR_117_]], [[VAR_115_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_119_:%.+]] = stablehlo.compare  LT, [[VAR_110_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_120_:%.+]] = stablehlo.add [[VAR_110_]], [[VAR_22_]] : tensor<1xi64>
// CHECK:           [[VAR_121_:%.+]] = stablehlo.select [[VAR_119_]], [[VAR_120_]], [[VAR_110_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_122_:%.+]] = stablehlo.concatenate [[VAR_121_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_123_:%.+]] = stablehlo.concatenate [[VAR_118_]], [[VAR_17_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_124_:%.+]] = stablehlo.concatenate [[VAR_112_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_125_:%.+]] = stablehlo.real_dynamic_slice [[VAR_113_]], [[VAR_122_]], [[VAR_123_]], [[VAR_124_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_126_:%.+]] = stablehlo.dynamic_reshape [[VAR_125_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_127_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-DAG:       [[VAR_128_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_129_:%.+]] = stablehlo.dynamic_reshape [[VAR_127_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_130_:%.+]] = stablehlo.dynamic_reshape [[VAR_128_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_131_:%.+]] = stablehlo.slice [[PARAM_3_]] [0:1, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-DAG:       [[VAR_132_:%.+]] = stablehlo.slice [[PARAM_3_]] [1:2, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_133_:%.+]] = stablehlo.dynamic_reshape [[VAR_131_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_134_:%.+]] = stablehlo.dynamic_reshape [[VAR_132_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_135_:%.+]] = stablehlo.transpose [[VAR_129_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_136_:%.+]] = stablehlo.transpose [[VAR_133_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_137_:%.+]] = stablehlo.transpose [[VAR_130_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-DAG:       [[VAR_138_:%.+]] = stablehlo.transpose [[VAR_134_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_139_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-DAG:       [[VAR_140_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_141_:%.+]] = stablehlo.dynamic_reshape [[VAR_139_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_142_:%.+]] = stablehlo.dynamic_reshape [[VAR_140_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_143_:%.+]] = stablehlo.slice [[VAR_141_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_144_:%.+]] = stablehlo.slice [[VAR_141_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_145_:%.+]] = stablehlo.slice [[VAR_141_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_146_:%.+]] = stablehlo.slice [[VAR_141_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_147_:%.+]] = stablehlo.slice [[VAR_141_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_148_:%.+]] = stablehlo.slice [[VAR_141_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_149_:%.+]] = stablehlo.slice [[VAR_141_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_150_:%.+]] = stablehlo.slice [[VAR_141_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_151_:%.+]] = stablehlo.slice [[VAR_142_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_152_:%.+]] = stablehlo.slice [[VAR_142_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_153_:%.+]] = stablehlo.slice [[VAR_142_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_154_:%.+]] = stablehlo.slice [[VAR_142_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_155_:%.+]] = stablehlo.slice [[VAR_142_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_156_:%.+]] = stablehlo.slice [[VAR_142_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_157_:%.+]] = stablehlo.slice [[VAR_142_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_158_:%.+]] = stablehlo.slice [[VAR_142_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_159_:%.+]]:4 = stablehlo.while([[VAR_iterArg_:%.+]] = [[VAR_20_]], [[VAR_iterArg_0_:%.+]] = [[VAR_19_]], [[VAR_iterArg_1_:%.+]] = [[VAR_48_]], [[VAR_iterArg_2_:%.+]] = [[VAR_74_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_162_:%.+]] = stablehlo.compare  LT, [[VAR_iterArg_]], [[VAR_15_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_163_:%.+]] = stablehlo.reshape [[VAR_162_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_163_]] : tensor<i1>
// CHECK:           } do {
// CHECK-DAG:         [[VAR_162_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_163_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_21_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_164_:%.+]] = stablehlo.add [[VAR_162_1_]], [[VAR_163_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_165_:%.+]] = stablehlo.slice [[VAR_iterArg_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_166_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_167_:%.+]] = stablehlo.slice [[VAR_164_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_168_:%.+]] = stablehlo.compare  LT, [[VAR_166_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_169_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_168_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<128x16x512xi1>
// CHECK-DAG:         [[VAR_170_:%.+]] = stablehlo.negate [[VAR_166_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_171_:%.+]] = stablehlo.add [[VAR_167_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_172_:%.+]] = stablehlo.add [[VAR_165_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_173_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<128x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_174_:%.+]] = stablehlo.select [[VAR_168_]], [[VAR_171_]], [[VAR_165_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_175_:%.+]] = stablehlo.select [[VAR_168_]], [[VAR_172_]], [[VAR_167_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_176_:%.+]] = stablehlo.select [[VAR_168_]], [[VAR_170_]], [[VAR_166_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_177_:%.+]] = stablehlo.select [[VAR_169_]], [[VAR_173_]], [[PARAM_0_]] : tensor<128x16x512xi1>, tensor<128x16x512xf32>
// CHECK:             [[VAR_178_:%.+]] = stablehlo.compare  GT, [[VAR_175_]], [[VAR_15_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_179_:%.+]] = stablehlo.select [[VAR_178_]], [[VAR_15_]], [[VAR_175_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_180_:%.+]] = stablehlo.compare  LT, [[VAR_179_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_181_:%.+]] = stablehlo.add [[VAR_179_]], [[VAR_15_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_182_:%.+]] = stablehlo.select [[VAR_180_]], [[VAR_181_]], [[VAR_179_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_183_:%.+]] = stablehlo.compare  LT, [[VAR_174_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_184_:%.+]] = stablehlo.add [[VAR_174_]], [[VAR_15_]] : tensor<1xi64>
// CHECK:             [[VAR_185_:%.+]] = stablehlo.select [[VAR_183_]], [[VAR_184_]], [[VAR_174_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_186_:%.+]] = stablehlo.concatenate [[VAR_185_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:         [[VAR_187_:%.+]] = stablehlo.concatenate [[VAR_182_]], [[VAR_17_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:         [[VAR_188_:%.+]] = stablehlo.concatenate [[VAR_176_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:             [[VAR_189_:%.+]] = stablehlo.real_dynamic_slice [[VAR_177_]], [[VAR_186_]], [[VAR_187_]], [[VAR_188_]] : (tensor<128x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_190_:%.+]] = stablehlo.dynamic_reshape [[VAR_189_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_191_:%.+]] = stablehlo.broadcast_in_dim [[VAR_190_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_192_:%.+]] = stablehlo.broadcast_in_dim [[VAR_135_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_193_:%.+]] = stablehlo.dot [[VAR_191_]], [[VAR_192_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_194_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_1_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_195_:%.+]] = stablehlo.broadcast_in_dim [[VAR_136_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_196_:%.+]] = stablehlo.dot [[VAR_194_]], [[VAR_195_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_197_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_193_]], [[VAR_12_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_198_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_196_]], [[VAR_12_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_199_:%.+]] = stablehlo.add [[VAR_197_]], [[VAR_198_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_200_:%.+]] = stablehlo.slice [[VAR_20_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_201_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_202_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_203_:%.+]] = stablehlo.compare  LT, [[VAR_201_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_204_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_203_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_205_:%.+]] = stablehlo.negate [[VAR_201_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_206_:%.+]] = stablehlo.add [[VAR_202_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_207_:%.+]] = stablehlo.add [[VAR_200_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_208_:%.+]] = stablehlo.reverse [[VAR_199_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_209_:%.+]] = stablehlo.select [[VAR_203_]], [[VAR_206_]], [[VAR_200_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_210_:%.+]] = stablehlo.select [[VAR_203_]], [[VAR_207_]], [[VAR_202_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_211_:%.+]] = stablehlo.select [[VAR_203_]], [[VAR_205_]], [[VAR_201_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_212_:%.+]] = stablehlo.select [[VAR_204_]], [[VAR_208_]], [[VAR_199_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_213_:%.+]] = stablehlo.compare  GT, [[VAR_210_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_214_:%.+]] = stablehlo.select [[VAR_213_]], [[VAR_10_]], [[VAR_210_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_215_:%.+]] = stablehlo.compare  LT, [[VAR_214_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_216_:%.+]] = stablehlo.add [[VAR_214_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_217_:%.+]] = stablehlo.select [[VAR_215_]], [[VAR_216_]], [[VAR_214_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_218_:%.+]] = stablehlo.compare  LT, [[VAR_209_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_219_:%.+]] = stablehlo.add [[VAR_209_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_220_:%.+]] = stablehlo.select [[VAR_218_]], [[VAR_219_]], [[VAR_209_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_221_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_220_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_222_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_217_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_223_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_21_]]1, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_224_:%.+]] = stablehlo.real_dynamic_slice [[VAR_212_]], [[VAR_221_]], [[VAR_222_]], [[VAR_223_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_225_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_226_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_227_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_228_:%.+]] = stablehlo.compare  LT, [[VAR_226_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_229_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_228_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_230_:%.+]] = stablehlo.negate [[VAR_226_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_231_:%.+]] = stablehlo.add [[VAR_227_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_232_:%.+]] = stablehlo.add [[VAR_225_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_233_:%.+]] = stablehlo.reverse [[VAR_199_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_234_:%.+]] = stablehlo.select [[VAR_228_]], [[VAR_231_]], [[VAR_225_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_235_:%.+]] = stablehlo.select [[VAR_228_]], [[VAR_232_]], [[VAR_227_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_236_:%.+]] = stablehlo.select [[VAR_228_]], [[VAR_230_]], [[VAR_226_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_237_:%.+]] = stablehlo.select [[VAR_229_]], [[VAR_233_]], [[VAR_199_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_238_:%.+]] = stablehlo.compare  GT, [[VAR_235_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_239_:%.+]] = stablehlo.select [[VAR_238_]], [[VAR_10_]], [[VAR_235_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_240_:%.+]] = stablehlo.compare  LT, [[VAR_239_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_241_:%.+]] = stablehlo.add [[VAR_239_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_242_:%.+]] = stablehlo.select [[VAR_240_]], [[VAR_241_]], [[VAR_239_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_243_:%.+]] = stablehlo.compare  LT, [[VAR_234_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_244_:%.+]] = stablehlo.add [[VAR_234_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_245_:%.+]] = stablehlo.select [[VAR_243_]], [[VAR_244_]], [[VAR_234_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_246_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_245_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_247_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_242_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_248_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_236_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_249_:%.+]] = stablehlo.real_dynamic_slice [[VAR_237_]], [[VAR_246_]], [[VAR_247_]], [[VAR_248_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_250_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_251_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_252_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_253_:%.+]] = stablehlo.compare  LT, [[VAR_251_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_254_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_253_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_255_:%.+]] = stablehlo.negate [[VAR_251_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_256_:%.+]] = stablehlo.add [[VAR_252_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_257_:%.+]] = stablehlo.add [[VAR_250_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_258_:%.+]] = stablehlo.reverse [[VAR_199_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_259_:%.+]] = stablehlo.select [[VAR_253_]], [[VAR_256_]], [[VAR_250_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_260_:%.+]] = stablehlo.select [[VAR_253_]], [[VAR_257_]], [[VAR_252_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_261_:%.+]] = stablehlo.select [[VAR_253_]], [[VAR_255_]], [[VAR_251_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_262_:%.+]] = stablehlo.select [[VAR_254_]], [[VAR_258_]], [[VAR_199_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_263_:%.+]] = stablehlo.compare  GT, [[VAR_260_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_264_:%.+]] = stablehlo.select [[VAR_263_]], [[VAR_10_]], [[VAR_260_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_265_:%.+]] = stablehlo.compare  LT, [[VAR_264_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_266_:%.+]] = stablehlo.add [[VAR_264_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_267_:%.+]] = stablehlo.select [[VAR_265_]], [[VAR_266_]], [[VAR_264_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_268_:%.+]] = stablehlo.compare  LT, [[VAR_259_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_269_:%.+]] = stablehlo.add [[VAR_259_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_270_:%.+]] = stablehlo.select [[VAR_268_]], [[VAR_269_]], [[VAR_259_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_271_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_270_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_272_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_267_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_273_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_261_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_274_:%.+]] = stablehlo.real_dynamic_slice [[VAR_262_]], [[VAR_271_]], [[VAR_272_]], [[VAR_273_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_275_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_276_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_277_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_278_:%.+]] = stablehlo.compare  LT, [[VAR_276_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_279_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_278_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_280_:%.+]] = stablehlo.negate [[VAR_276_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_281_:%.+]] = stablehlo.add [[VAR_277_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_282_:%.+]] = stablehlo.add [[VAR_275_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_283_:%.+]] = stablehlo.reverse [[VAR_199_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_284_:%.+]] = stablehlo.select [[VAR_278_]], [[VAR_281_]], [[VAR_275_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_285_:%.+]] = stablehlo.select [[VAR_278_]], [[VAR_282_]], [[VAR_277_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_286_:%.+]] = stablehlo.select [[VAR_278_]], [[VAR_280_]], [[VAR_276_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_287_:%.+]] = stablehlo.select [[VAR_279_]], [[VAR_283_]], [[VAR_199_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_288_:%.+]] = stablehlo.compare  GT, [[VAR_285_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_289_:%.+]] = stablehlo.select [[VAR_288_]], [[VAR_10_]], [[VAR_285_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_290_:%.+]] = stablehlo.compare  LT, [[VAR_289_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_291_:%.+]] = stablehlo.add [[VAR_289_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_292_:%.+]] = stablehlo.select [[VAR_290_]], [[VAR_291_]], [[VAR_289_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_293_:%.+]] = stablehlo.compare  LT, [[VAR_284_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_294_:%.+]] = stablehlo.add [[VAR_284_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_295_:%.+]] = stablehlo.select [[VAR_293_]], [[VAR_294_]], [[VAR_284_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_296_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_295_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_297_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_292_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_298_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_286_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_299_:%.+]] = stablehlo.real_dynamic_slice [[VAR_287_]], [[VAR_296_]], [[VAR_297_]], [[VAR_298_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_300_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_224_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_301_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_143_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_302_:%.+]] = stablehlo.add [[VAR_300_]], [[VAR_301_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_303_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_302_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_304_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_147_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_305_:%.+]] = stablehlo.add [[VAR_303_]], [[VAR_304_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_306_:%.+]] = stablehlo.logistic [[VAR_305_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_307_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_274_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_308_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_145_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_309_:%.+]] = stablehlo.add [[VAR_307_]], [[VAR_308_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_310_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_309_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_311_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_149_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_312_:%.+]] = stablehlo.add [[VAR_310_]], [[VAR_311_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_313_:%.+]] = stablehlo.logistic [[VAR_312_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_314_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_299_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_315_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_146_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_316_:%.+]] = stablehlo.add [[VAR_314_]], [[VAR_315_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_317_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_316_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_318_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_150_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_319_:%.+]] = stablehlo.add [[VAR_317_]], [[VAR_318_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_320_:%.+]] = stablehlo.tanh [[VAR_319_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_321_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_313_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_322_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_2_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_323_:%.+]] = stablehlo.multiply [[VAR_321_]], [[VAR_322_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_324_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_306_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_325_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_320_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_326_:%.+]] = stablehlo.multiply [[VAR_324_]], [[VAR_325_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_327_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_323_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_328_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_326_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_329_:%.+]] = stablehlo.add [[VAR_327_]], [[VAR_328_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_330_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_249_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_331_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_144_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_332_:%.+]] = stablehlo.add [[VAR_330_]], [[VAR_331_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_333_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_332_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_334_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_148_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_335_:%.+]] = stablehlo.add [[VAR_333_]], [[VAR_334_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_336_:%.+]] = stablehlo.logistic [[VAR_335_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_337_:%.+]] = stablehlo.tanh [[VAR_329_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_338_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_336_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_339_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_337_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_340_:%.+]] = stablehlo.multiply [[VAR_338_]], [[VAR_339_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_341_:%.+]] = stablehlo.dynamic_reshape [[VAR_340_]], [[VAR_2_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_342_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_]], [[VAR_1_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_343_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_0_]], [[VAR_342_]], [[VAR_341_]]) ({
// CHECK:             ^bb0([[arg4_:%.+]]: tensor<f32>, [[arg5_:%.+]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_]] : tensor<f32>
// CHECK:             }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_344_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_345_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_21_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_346_:%.+]] = stablehlo.add [[VAR_344_]], [[VAR_345_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_346_]], [[VAR_343_]], [[VAR_340_]], [[VAR_329_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_160_:%.+]]:4 = stablehlo.while([[VAR_iterArg_1_:%.+]] = [[VAR_9_]], [[VAR_iterArg_0_1_:%.+]] = [[VAR_19_]], [[VAR_iterArg_1_1_:%.+]] = [[VAR_100_]], [[VAR_iterArg_2_1_:%.+]] = [[VAR_126_]]) : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:            cond {
// CHECK:             [[VAR_162_2_:%.+]] = stablehlo.compare  GE, [[VAR_iterArg_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_163_2_:%.+]] = stablehlo.reshape [[VAR_162_2_]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:             stablehlo.return [[VAR_163_2_]] : tensor<i1>
// CHECK:           } do {
// CHECK-DAG:         [[VAR_162_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_1_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_163_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_21_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_164_1_:%.+]] = stablehlo.add [[VAR_162_3_]], [[VAR_163_3_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_165_1_:%.+]] = stablehlo.slice [[VAR_iterArg_1_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_166_1_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_167_1_:%.+]] = stablehlo.slice [[VAR_164_1_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_168_1_:%.+]] = stablehlo.compare  LT, [[VAR_166_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_169_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_168_1_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<128x16x512xi1>
// CHECK-DAG:         [[VAR_170_1_:%.+]] = stablehlo.negate [[VAR_166_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_171_1_:%.+]] = stablehlo.add [[VAR_167_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_172_1_:%.+]] = stablehlo.add [[VAR_165_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_173_1_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<128x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_174_1_:%.+]] = stablehlo.select [[VAR_168_1_]], [[VAR_171_1_]], [[VAR_165_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_175_1_:%.+]] = stablehlo.select [[VAR_168_1_]], [[VAR_172_1_]], [[VAR_167_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_176_1_:%.+]] = stablehlo.select [[VAR_168_1_]], [[VAR_170_1_]], [[VAR_166_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_177_1_:%.+]] = stablehlo.select [[VAR_169_1_]], [[VAR_173_1_]], [[PARAM_0_]] : tensor<128x16x512xi1>, tensor<128x16x512xf32>
// CHECK:             [[VAR_178_1_:%.+]] = stablehlo.compare  GT, [[VAR_175_1_]], [[VAR_15_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_179_1_:%.+]] = stablehlo.select [[VAR_178_1_]], [[VAR_15_]], [[VAR_175_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_180_1_:%.+]] = stablehlo.compare  LT, [[VAR_179_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_181_1_:%.+]] = stablehlo.add [[VAR_179_1_]], [[VAR_15_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_182_1_:%.+]] = stablehlo.select [[VAR_180_1_]], [[VAR_181_1_]], [[VAR_179_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_183_1_:%.+]] = stablehlo.compare  LT, [[VAR_174_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_184_1_:%.+]] = stablehlo.add [[VAR_174_1_]], [[VAR_15_]] : tensor<1xi64>
// CHECK:             [[VAR_185_1_:%.+]] = stablehlo.select [[VAR_183_1_]], [[VAR_184_1_]], [[VAR_174_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_186_1_:%.+]] = stablehlo.concatenate [[VAR_185_1_]], [[VAR_20_]], [[VAR_20_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:         [[VAR_187_1_:%.+]] = stablehlo.concatenate [[VAR_182_1_]], [[VAR_17_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:         [[VAR_188_1_:%.+]] = stablehlo.concatenate [[VAR_176_1_]], [[VAR_21_]], [[VAR_21_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:             [[VAR_189_1_:%.+]] = stablehlo.real_dynamic_slice [[VAR_177_1_]], [[VAR_186_1_]], [[VAR_187_1_]], [[VAR_188_1_]] : (tensor<128x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:             [[VAR_190_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_189_1_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_191_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_190_1_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:         [[VAR_192_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_137_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_193_1_:%.+]] = stablehlo.dot [[VAR_191_1_]], [[VAR_192_1_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_194_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_iterArg_1_1_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_195_1_:%.+]] = stablehlo.broadcast_in_dim [[VAR_138_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_196_1_:%.+]] = stablehlo.dot [[VAR_194_1_]], [[VAR_195_1_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_197_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_193_1_]], [[VAR_12_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:             [[VAR_198_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_196_1_]], [[VAR_12_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_199_1_:%.+]] = stablehlo.add [[VAR_197_1_]], [[VAR_198_1_]] : tensor<16x1024xf32>
// CHECK-DAG:         [[VAR_200_1_:%.+]] = stablehlo.slice [[VAR_20_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_201_1_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_202_1_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_203_1_:%.+]] = stablehlo.compare  LT, [[VAR_201_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_204_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_203_1_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_205_1_:%.+]] = stablehlo.negate [[VAR_201_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_206_1_:%.+]] = stablehlo.add [[VAR_202_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_207_1_:%.+]] = stablehlo.add [[VAR_200_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_208_1_:%.+]] = stablehlo.reverse [[VAR_199_1_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_209_1_:%.+]] = stablehlo.select [[VAR_203_1_]], [[VAR_206_1_]], [[VAR_200_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_210_1_:%.+]] = stablehlo.select [[VAR_203_1_]], [[VAR_207_1_]], [[VAR_202_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_211_1_:%.+]] = stablehlo.select [[VAR_203_1_]], [[VAR_205_1_]], [[VAR_201_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_212_1_:%.+]] = stablehlo.select [[VAR_204_1_]], [[VAR_208_1_]], [[VAR_199_1_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_213_1_:%.+]] = stablehlo.compare  GT, [[VAR_210_1_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_214_1_:%.+]] = stablehlo.select [[VAR_213_1_]], [[VAR_10_]], [[VAR_210_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_215_1_:%.+]] = stablehlo.compare  LT, [[VAR_214_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_216_1_:%.+]] = stablehlo.add [[VAR_214_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_217_1_:%.+]] = stablehlo.select [[VAR_215_1_]], [[VAR_216_1_]], [[VAR_214_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_218_1_:%.+]] = stablehlo.compare  LT, [[VAR_209_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_219_1_:%.+]] = stablehlo.add [[VAR_209_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_220_1_:%.+]] = stablehlo.select [[VAR_218_1_]], [[VAR_219_1_]], [[VAR_209_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_221_1_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_220_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_222_1_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_217_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_223_1_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_21_]]1, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_224_1_:%.+]] = stablehlo.real_dynamic_slice [[VAR_212_1_]], [[VAR_221_1_]], [[VAR_222_1_]], [[VAR_223_1_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_225_1_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_226_1_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_227_1_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_228_1_:%.+]] = stablehlo.compare  LT, [[VAR_226_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_229_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_228_1_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_230_1_:%.+]] = stablehlo.negate [[VAR_226_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_231_1_:%.+]] = stablehlo.add [[VAR_227_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_232_1_:%.+]] = stablehlo.add [[VAR_225_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_233_1_:%.+]] = stablehlo.reverse [[VAR_199_1_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_234_1_:%.+]] = stablehlo.select [[VAR_228_1_]], [[VAR_231_1_]], [[VAR_225_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_235_1_:%.+]] = stablehlo.select [[VAR_228_1_]], [[VAR_232_1_]], [[VAR_227_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_236_1_:%.+]] = stablehlo.select [[VAR_228_1_]], [[VAR_230_1_]], [[VAR_226_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_237_1_:%.+]] = stablehlo.select [[VAR_229_1_]], [[VAR_233_1_]], [[VAR_199_1_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_238_1_:%.+]] = stablehlo.compare  GT, [[VAR_235_1_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_239_1_:%.+]] = stablehlo.select [[VAR_238_1_]], [[VAR_10_]], [[VAR_235_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_240_1_:%.+]] = stablehlo.compare  LT, [[VAR_239_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_241_1_:%.+]] = stablehlo.add [[VAR_239_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_242_1_:%.+]] = stablehlo.select [[VAR_240_1_]], [[VAR_241_1_]], [[VAR_239_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_243_1_:%.+]] = stablehlo.compare  LT, [[VAR_234_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_244_1_:%.+]] = stablehlo.add [[VAR_234_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_245_1_:%.+]] = stablehlo.select [[VAR_243_1_]], [[VAR_244_1_]], [[VAR_234_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_246_1_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_245_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_247_1_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_242_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_248_1_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_236_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_249_1_:%.+]] = stablehlo.real_dynamic_slice [[VAR_237_1_]], [[VAR_246_1_]], [[VAR_247_1_]], [[VAR_248_1_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_250_1_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_251_1_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_252_1_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_253_1_:%.+]] = stablehlo.compare  LT, [[VAR_251_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_254_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_253_1_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_255_1_:%.+]] = stablehlo.negate [[VAR_251_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_256_1_:%.+]] = stablehlo.add [[VAR_252_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_257_1_:%.+]] = stablehlo.add [[VAR_250_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_258_1_:%.+]] = stablehlo.reverse [[VAR_199_1_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_259_1_:%.+]] = stablehlo.select [[VAR_253_1_]], [[VAR_256_1_]], [[VAR_250_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_260_1_:%.+]] = stablehlo.select [[VAR_253_1_]], [[VAR_257_1_]], [[VAR_252_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_261_1_:%.+]] = stablehlo.select [[VAR_253_1_]], [[VAR_255_1_]], [[VAR_251_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_262_1_:%.+]] = stablehlo.select [[VAR_254_1_]], [[VAR_258_1_]], [[VAR_199_1_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_263_1_:%.+]] = stablehlo.compare  GT, [[VAR_260_1_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_264_1_:%.+]] = stablehlo.select [[VAR_263_1_]], [[VAR_10_]], [[VAR_260_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_265_1_:%.+]] = stablehlo.compare  LT, [[VAR_264_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_266_1_:%.+]] = stablehlo.add [[VAR_264_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_267_1_:%.+]] = stablehlo.select [[VAR_265_1_]], [[VAR_266_1_]], [[VAR_264_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_268_1_:%.+]] = stablehlo.compare  LT, [[VAR_259_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_269_1_:%.+]] = stablehlo.add [[VAR_259_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_270_1_:%.+]] = stablehlo.select [[VAR_268_1_]], [[VAR_269_1_]], [[VAR_259_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_271_1_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_270_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_272_1_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_267_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_273_1_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_261_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_274_1_:%.+]] = stablehlo.real_dynamic_slice [[VAR_262_1_]], [[VAR_271_1_]], [[VAR_272_1_]], [[VAR_273_1_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_275_1_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_276_1_:%.+]] = stablehlo.slice [[VAR_21_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_277_1_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:             [[VAR_278_1_:%.+]] = stablehlo.compare  LT, [[VAR_276_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_279_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_278_1_]], [[VAR_12_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:         [[VAR_280_1_:%.+]] = stablehlo.negate [[VAR_276_1_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_281_1_:%.+]] = stablehlo.add [[VAR_277_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_282_1_:%.+]] = stablehlo.add [[VAR_275_1_]], [[VAR_21_]] : tensor<1xi64>
// CHECK-DAG:         [[VAR_283_1_:%.+]] = stablehlo.reverse [[VAR_199_1_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_284_1_:%.+]] = stablehlo.select [[VAR_278_1_]], [[VAR_281_1_]], [[VAR_275_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_285_1_:%.+]] = stablehlo.select [[VAR_278_1_]], [[VAR_282_1_]], [[VAR_277_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_286_1_:%.+]] = stablehlo.select [[VAR_278_1_]], [[VAR_280_1_]], [[VAR_276_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_287_1_:%.+]] = stablehlo.select [[VAR_279_1_]], [[VAR_283_1_]], [[VAR_199_1_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:             [[VAR_288_1_:%.+]] = stablehlo.compare  GT, [[VAR_285_1_]], [[VAR_10_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:             [[VAR_289_1_:%.+]] = stablehlo.select [[VAR_288_1_]], [[VAR_10_]], [[VAR_285_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_290_1_:%.+]] = stablehlo.compare  LT, [[VAR_289_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_291_1_:%.+]] = stablehlo.add [[VAR_289_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_292_1_:%.+]] = stablehlo.select [[VAR_290_1_]], [[VAR_291_1_]], [[VAR_289_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_293_1_:%.+]] = stablehlo.compare  LT, [[VAR_284_1_]], [[VAR_20_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:         [[VAR_294_1_:%.+]] = stablehlo.add [[VAR_284_1_]], [[VAR_10_]] : tensor<1xi64>
// CHECK:             [[VAR_295_1_:%.+]] = stablehlo.select [[VAR_293_1_]], [[VAR_294_1_]], [[VAR_284_1_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:         [[VAR_296_1_:%.+]] = stablehlo.concatenate [[VAR_20_]], [[VAR_295_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_297_1_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_292_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:         [[VAR_298_1_:%.+]] = stablehlo.concatenate [[VAR_21_]], [[VAR_286_1_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_299_1_:%.+]] = stablehlo.real_dynamic_slice [[VAR_287_1_]], [[VAR_296_1_]], [[VAR_297_1_]], [[VAR_298_1_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_300_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_224_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_301_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_151_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_302_1_:%.+]] = stablehlo.add [[VAR_300_1_]], [[VAR_301_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_303_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_302_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_304_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_155_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_305_1_:%.+]] = stablehlo.add [[VAR_303_1_]], [[VAR_304_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_306_1_:%.+]] = stablehlo.logistic [[VAR_305_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_307_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_274_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_308_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_153_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_309_1_:%.+]] = stablehlo.add [[VAR_307_1_]], [[VAR_308_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_310_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_309_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_311_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_157_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_312_1_:%.+]] = stablehlo.add [[VAR_310_1_]], [[VAR_311_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_313_1_:%.+]] = stablehlo.logistic [[VAR_312_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_314_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_299_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_315_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_154_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_316_1_:%.+]] = stablehlo.add [[VAR_314_1_]], [[VAR_315_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_317_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_316_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_318_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_158_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_319_1_:%.+]] = stablehlo.add [[VAR_317_1_]], [[VAR_318_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_320_1_:%.+]] = stablehlo.tanh [[VAR_319_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_321_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_313_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_322_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_2_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_323_1_:%.+]] = stablehlo.multiply [[VAR_321_1_]], [[VAR_322_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_324_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_306_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_325_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_320_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_326_1_:%.+]] = stablehlo.multiply [[VAR_324_1_]], [[VAR_325_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_327_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_323_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_328_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_326_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_329_1_:%.+]] = stablehlo.add [[VAR_327_1_]], [[VAR_328_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_330_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_249_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_331_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_152_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_332_1_:%.+]] = stablehlo.add [[VAR_330_1_]], [[VAR_331_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_333_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_332_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_334_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_156_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_335_1_:%.+]] = stablehlo.add [[VAR_333_1_]], [[VAR_334_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_336_1_:%.+]] = stablehlo.logistic [[VAR_335_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_337_1_:%.+]] = stablehlo.tanh [[VAR_329_1_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_338_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_336_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:         [[VAR_339_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_337_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:             [[VAR_340_1_:%.+]] = stablehlo.multiply [[VAR_338_1_]], [[VAR_339_1_]] : tensor<16x256xf32>
// CHECK-DAG:         [[VAR_341_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_340_1_]], [[VAR_2_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:         [[VAR_342_1_:%.+]] = stablehlo.dynamic_reshape [[VAR_iterArg_1_]], [[VAR_1_]] : (tensor<1xi64>, tensor<2xindex>) -> tensor<1x1xi64>
// CHECK:             [[VAR_343_1_:%.+]] = "stablehlo.scatter"([[VAR_iterArg_0_1_]], [[VAR_342_1_]], [[VAR_341_1_]]) ({
// CHECK:             ^bb0([[arg4_1_:%.+]]: tensor<f32>, [[arg5_1_:%.+]]: tensor<f32>):
// CHECK:               stablehlo.return [[arg5_1_]] : tensor<f32>
// CHECK:             }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<128x1x16x256xf32>, tensor<1x1xi64>, tensor<1x1x16x256xf32>) -> tensor<128x1x16x256xf32>
// CHECK-DAG:         [[VAR_344_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_iterArg_1_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:         [[VAR_345_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_21_]], [[VAR_14_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK:             [[VAR_346_1_:%.+]] = stablehlo.subtract [[VAR_344_1_]], [[VAR_345_1_]] : tensor<1xi64>
// CHECK:             stablehlo.return [[VAR_346_1_]], [[VAR_343_1_]], [[VAR_340_1_]], [[VAR_329_1_]] : tensor<1xi64>, tensor<128x1x16x256xf32>, tensor<16x256xf32>, tensor<16x256xf32>
// CHECK:           }
// CHECK:           [[VAR_161_:%.+]] = stablehlo.concatenate [[VAR_159_]]#1, [[VAR_160_]]#1, dim = 1 : (tensor<128x1x16x256xf32>, tensor<128x1x16x256xf32>) -> tensor<128x2x16x256xf32>
// CHECK:           return [[VAR_161_]] : tensor<128x2x16x256xf32>
// CHECK:         }
}