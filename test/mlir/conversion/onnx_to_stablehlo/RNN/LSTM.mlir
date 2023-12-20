// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize -split-input-file %s | FileCheck %s
func.func @test_lstm(%arg0 : tensor<2x16x512xf32>, %arg1 : tensor<2x2048xf32>, %arg2 : tensor<2x1024x512xf32>, %arg3 : tensor<2x1024x256xf32>) -> tensor<2x2x16x256xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<2x16x256xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg2, %arg3, %arg1, %1, %0, %0, %1) {direction = "bidirectional", hidden_size = 256 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<2x16x512xf32>, tensor<2x1024x512xf32>, tensor<2x1024x256xf32>, tensor<2x2048xf32>, none, tensor<2x16x256xf32>, tensor<2x16x256xf32>, none) -> (tensor<2x2x16x256xf32>, tensor<2x16x256xf32>, tensor<2x16x256xf32>)
  return %Y : tensor<2x2x16x256xf32>
// CHECK-LABEL:  func.func @test_lstm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x16x512xf32>, [[PARAM_1_:%.+]]: tensor<2x2048xf32>, [[PARAM_2_:%.+]]: tensor<2x1024x512xf32>, [[PARAM_3_:%.+]]: tensor<2x1024x256xf32>) -> tensor<2x2x16x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [16, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [1, 1, 16, 256] : tensor<4xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.const_shape [16, 1024] : tensor<2xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.const_shape [16, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.const_shape [2, 16, 512] : tensor<3xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.const_shape [2048] : tensor<1xindex>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.const_shape [1024, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.const_shape [1024, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = shape.const_shape [2, 16, 256] : tensor<3xindex>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.constant dense<1024> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.constant dense<768> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.constant dense<512> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = shape.const_shape [1] : tensor<1xindex>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.constant dense<256> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.constant dense<16> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16x256xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_22_:%.+]] = stablehlo.compare  LT, [[VAR_20_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_22_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.negate [[VAR_20_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.add [[VAR_21_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.add [[VAR_19_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.reverse [[VAR_15_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.select [[VAR_22_]], [[VAR_25_]], [[VAR_19_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.select [[VAR_22_]], [[VAR_26_]], [[VAR_21_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.select [[VAR_22_]], [[VAR_24_]], [[VAR_20_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.select [[VAR_23_]], [[VAR_27_]], [[VAR_15_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_32_:%.+]] = stablehlo.compare  GT, [[VAR_29_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_33_:%.+]] = stablehlo.select [[VAR_32_]], [[VAR_18_]], [[VAR_29_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.compare  LT, [[VAR_33_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.add [[VAR_33_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.select [[VAR_34_]], [[VAR_35_]], [[VAR_33_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.compare  LT, [[VAR_28_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.add [[VAR_28_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_39_:%.+]] = stablehlo.select [[VAR_37_]], [[VAR_38_]], [[VAR_28_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.concatenate [[VAR_39_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.concatenate [[VAR_36_]], [[VAR_14_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.concatenate [[VAR_30_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_43_:%.+]] = stablehlo.real_dynamic_slice [[VAR_31_]], [[VAR_40_]], [[VAR_41_]], [[VAR_42_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.dynamic_reshape [[VAR_43_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_48_:%.+]] = stablehlo.compare  LT, [[VAR_46_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_48_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.negate [[VAR_46_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.add [[VAR_47_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.add [[VAR_45_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_53_:%.+]] = stablehlo.reverse [[VAR_15_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_54_:%.+]] = stablehlo.select [[VAR_48_]], [[VAR_51_]], [[VAR_45_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_55_:%.+]] = stablehlo.select [[VAR_48_]], [[VAR_52_]], [[VAR_47_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_56_:%.+]] = stablehlo.select [[VAR_48_]], [[VAR_50_]], [[VAR_46_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_57_:%.+]] = stablehlo.select [[VAR_49_]], [[VAR_53_]], [[VAR_15_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_58_:%.+]] = stablehlo.compare  GT, [[VAR_55_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_59_:%.+]] = stablehlo.select [[VAR_58_]], [[VAR_18_]], [[VAR_55_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_60_:%.+]] = stablehlo.compare  LT, [[VAR_59_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_61_:%.+]] = stablehlo.add [[VAR_59_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_62_:%.+]] = stablehlo.select [[VAR_60_]], [[VAR_61_]], [[VAR_59_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_63_:%.+]] = stablehlo.compare  LT, [[VAR_54_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_64_:%.+]] = stablehlo.add [[VAR_54_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_65_:%.+]] = stablehlo.select [[VAR_63_]], [[VAR_64_]], [[VAR_54_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_66_:%.+]] = stablehlo.concatenate [[VAR_65_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_67_:%.+]] = stablehlo.concatenate [[VAR_62_]], [[VAR_14_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_68_:%.+]] = stablehlo.concatenate [[VAR_56_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_69_:%.+]] = stablehlo.real_dynamic_slice [[VAR_57_]], [[VAR_66_]], [[VAR_67_]], [[VAR_68_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_70_:%.+]] = stablehlo.dynamic_reshape [[VAR_69_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_71_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_72_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_73_:%.+]] = stablehlo.slice [[VAR_18_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_74_:%.+]] = stablehlo.compare  LT, [[VAR_72_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_75_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_74_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_76_:%.+]] = stablehlo.negate [[VAR_72_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_77_:%.+]] = stablehlo.add [[VAR_73_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_78_:%.+]] = stablehlo.add [[VAR_71_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_79_:%.+]] = stablehlo.reverse [[VAR_15_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_80_:%.+]] = stablehlo.select [[VAR_74_]], [[VAR_77_]], [[VAR_71_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_81_:%.+]] = stablehlo.select [[VAR_74_]], [[VAR_78_]], [[VAR_73_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_82_:%.+]] = stablehlo.select [[VAR_74_]], [[VAR_76_]], [[VAR_72_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_83_:%.+]] = stablehlo.select [[VAR_75_]], [[VAR_79_]], [[VAR_15_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_84_:%.+]] = stablehlo.compare  GT, [[VAR_81_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_85_:%.+]] = stablehlo.select [[VAR_84_]], [[VAR_18_]], [[VAR_81_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_86_:%.+]] = stablehlo.compare  LT, [[VAR_85_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_87_:%.+]] = stablehlo.add [[VAR_85_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_88_:%.+]] = stablehlo.select [[VAR_86_]], [[VAR_87_]], [[VAR_85_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_89_:%.+]] = stablehlo.compare  LT, [[VAR_80_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_90_:%.+]] = stablehlo.add [[VAR_80_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_91_:%.+]] = stablehlo.select [[VAR_89_]], [[VAR_90_]], [[VAR_80_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_92_:%.+]] = stablehlo.concatenate [[VAR_91_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_93_:%.+]] = stablehlo.concatenate [[VAR_88_]], [[VAR_14_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_94_:%.+]] = stablehlo.concatenate [[VAR_82_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_95_:%.+]] = stablehlo.real_dynamic_slice [[VAR_83_]], [[VAR_92_]], [[VAR_93_]], [[VAR_94_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_96_:%.+]] = stablehlo.dynamic_reshape [[VAR_95_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_97_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_98_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_99_:%.+]] = stablehlo.slice [[VAR_18_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_100_:%.+]] = stablehlo.compare  LT, [[VAR_98_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_101_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_100_]], [[VAR_8_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x256xi1>
// CHECK-DAG:       [[VAR_102_:%.+]] = stablehlo.negate [[VAR_98_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_103_:%.+]] = stablehlo.add [[VAR_99_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_104_:%.+]] = stablehlo.add [[VAR_97_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_105_:%.+]] = stablehlo.reverse [[VAR_15_]], dims = [0] : tensor<2x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_106_:%.+]] = stablehlo.select [[VAR_100_]], [[VAR_103_]], [[VAR_97_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_107_:%.+]] = stablehlo.select [[VAR_100_]], [[VAR_104_]], [[VAR_99_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_108_:%.+]] = stablehlo.select [[VAR_100_]], [[VAR_102_]], [[VAR_98_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_109_:%.+]] = stablehlo.select [[VAR_101_]], [[VAR_105_]], [[VAR_15_]] : tensor<2x16x256xi1>, tensor<2x16x256xf32>
// CHECK:           [[VAR_110_:%.+]] = stablehlo.compare  GT, [[VAR_107_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_111_:%.+]] = stablehlo.select [[VAR_110_]], [[VAR_18_]], [[VAR_107_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_112_:%.+]] = stablehlo.compare  LT, [[VAR_111_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_113_:%.+]] = stablehlo.add [[VAR_111_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_114_:%.+]] = stablehlo.select [[VAR_112_]], [[VAR_113_]], [[VAR_111_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_115_:%.+]] = stablehlo.compare  LT, [[VAR_106_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_116_:%.+]] = stablehlo.add [[VAR_106_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_117_:%.+]] = stablehlo.select [[VAR_115_]], [[VAR_116_]], [[VAR_106_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_118_:%.+]] = stablehlo.concatenate [[VAR_117_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_119_:%.+]] = stablehlo.concatenate [[VAR_114_]], [[VAR_14_]], [[VAR_13_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_120_:%.+]] = stablehlo.concatenate [[VAR_108_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_121_:%.+]] = stablehlo.real_dynamic_slice [[VAR_109_]], [[VAR_118_]], [[VAR_119_]], [[VAR_120_]] : (tensor<2x16x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_122_:%.+]] = stablehlo.dynamic_reshape [[VAR_121_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_123_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-DAG:       [[VAR_124_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_125_:%.+]] = stablehlo.dynamic_reshape [[VAR_123_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_126_:%.+]] = stablehlo.dynamic_reshape [[VAR_124_]], [[VAR_7_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_127_:%.+]] = stablehlo.slice [[PARAM_3_]] [0:1, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-DAG:       [[VAR_128_:%.+]] = stablehlo.slice [[PARAM_3_]] [1:2, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_129_:%.+]] = stablehlo.dynamic_reshape [[VAR_127_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_130_:%.+]] = stablehlo.dynamic_reshape [[VAR_128_]], [[VAR_6_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_131_:%.+]] = stablehlo.transpose [[VAR_125_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_132_:%.+]] = stablehlo.transpose [[VAR_129_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_133_:%.+]] = stablehlo.transpose [[VAR_126_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-DAG:       [[VAR_134_:%.+]] = stablehlo.transpose [[VAR_130_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_135_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-DAG:       [[VAR_136_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_137_:%.+]] = stablehlo.dynamic_reshape [[VAR_135_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_138_:%.+]] = stablehlo.dynamic_reshape [[VAR_136_]], [[VAR_5_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_139_:%.+]] = stablehlo.slice [[VAR_137_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_140_:%.+]] = stablehlo.slice [[VAR_137_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_141_:%.+]] = stablehlo.slice [[VAR_137_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_142_:%.+]] = stablehlo.slice [[VAR_137_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_143_:%.+]] = stablehlo.slice [[VAR_137_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_144_:%.+]] = stablehlo.slice [[VAR_137_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_145_:%.+]] = stablehlo.slice [[VAR_137_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_146_:%.+]] = stablehlo.slice [[VAR_137_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_147_:%.+]] = stablehlo.slice [[VAR_138_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_148_:%.+]] = stablehlo.slice [[VAR_138_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_149_:%.+]] = stablehlo.slice [[VAR_138_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_150_:%.+]] = stablehlo.slice [[VAR_138_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_151_:%.+]] = stablehlo.slice [[VAR_138_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_152_:%.+]] = stablehlo.slice [[VAR_138_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_153_:%.+]] = stablehlo.slice [[VAR_138_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_154_:%.+]] = stablehlo.slice [[VAR_138_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_155_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_16_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_156_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_157_:%.+]] = stablehlo.add [[VAR_155_]], [[VAR_156_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_158_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_159_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_160_:%.+]] = stablehlo.slice [[VAR_157_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_161_:%.+]] = stablehlo.compare  LT, [[VAR_159_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_162_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_161_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x512xi1>
// CHECK-DAG:       [[VAR_163_:%.+]] = stablehlo.negate [[VAR_159_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_164_:%.+]] = stablehlo.add [[VAR_160_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_165_:%.+]] = stablehlo.add [[VAR_158_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_166_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_167_:%.+]] = stablehlo.select [[VAR_161_]], [[VAR_164_]], [[VAR_158_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_168_:%.+]] = stablehlo.select [[VAR_161_]], [[VAR_165_]], [[VAR_160_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_169_:%.+]] = stablehlo.select [[VAR_161_]], [[VAR_163_]], [[VAR_159_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_170_:%.+]] = stablehlo.select [[VAR_162_]], [[VAR_166_]], [[PARAM_0_]] : tensor<2x16x512xi1>, tensor<2x16x512xf32>
// CHECK:           [[VAR_171_:%.+]] = stablehlo.compare  GT, [[VAR_168_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_172_:%.+]] = stablehlo.select [[VAR_171_]], [[VAR_18_]], [[VAR_168_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_173_:%.+]] = stablehlo.compare  LT, [[VAR_172_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_174_:%.+]] = stablehlo.add [[VAR_172_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_175_:%.+]] = stablehlo.select [[VAR_173_]], [[VAR_174_]], [[VAR_172_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_176_:%.+]] = stablehlo.compare  LT, [[VAR_167_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_177_:%.+]] = stablehlo.add [[VAR_167_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_178_:%.+]] = stablehlo.select [[VAR_176_]], [[VAR_177_]], [[VAR_167_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_179_:%.+]] = stablehlo.concatenate [[VAR_178_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_180_:%.+]] = stablehlo.concatenate [[VAR_175_]], [[VAR_14_]], [[VAR_11_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_181_:%.+]] = stablehlo.concatenate [[VAR_169_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_182_:%.+]] = stablehlo.real_dynamic_slice [[VAR_170_]], [[VAR_179_]], [[VAR_180_]], [[VAR_181_]] : (tensor<2x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_183_:%.+]] = stablehlo.dynamic_reshape [[VAR_182_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_184_:%.+]] = stablehlo.broadcast_in_dim [[VAR_183_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_185_:%.+]] = stablehlo.broadcast_in_dim [[VAR_131_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_186_:%.+]] = stablehlo.dot [[VAR_184_]], [[VAR_185_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_187_:%.+]] = stablehlo.broadcast_in_dim [[VAR_44_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_188_:%.+]] = stablehlo.broadcast_in_dim [[VAR_132_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_189_:%.+]] = stablehlo.dot [[VAR_187_]], [[VAR_188_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_190_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_186_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_191_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_189_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_192_:%.+]] = stablehlo.add [[VAR_190_]], [[VAR_191_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_193_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_194_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_195_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_196_:%.+]] = stablehlo.compare  LT, [[VAR_194_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_197_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_196_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_198_:%.+]] = stablehlo.negate [[VAR_194_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_199_:%.+]] = stablehlo.add [[VAR_195_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_200_:%.+]] = stablehlo.add [[VAR_193_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_201_:%.+]] = stablehlo.reverse [[VAR_192_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_202_:%.+]] = stablehlo.select [[VAR_196_]], [[VAR_199_]], [[VAR_193_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_203_:%.+]] = stablehlo.select [[VAR_196_]], [[VAR_200_]], [[VAR_195_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_204_:%.+]] = stablehlo.select [[VAR_196_]], [[VAR_198_]], [[VAR_194_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_205_:%.+]] = stablehlo.select [[VAR_197_]], [[VAR_201_]], [[VAR_192_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_206_:%.+]] = stablehlo.compare  GT, [[VAR_203_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_207_:%.+]] = stablehlo.select [[VAR_206_]], [[VAR_9_]], [[VAR_203_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_208_:%.+]] = stablehlo.compare  LT, [[VAR_207_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_209_:%.+]] = stablehlo.add [[VAR_207_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_210_:%.+]] = stablehlo.select [[VAR_208_]], [[VAR_209_]], [[VAR_207_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_211_:%.+]] = stablehlo.compare  LT, [[VAR_202_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_212_:%.+]] = stablehlo.add [[VAR_202_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_213_:%.+]] = stablehlo.select [[VAR_211_]], [[VAR_212_]], [[VAR_202_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_214_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_213_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_215_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_210_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_216_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_204_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_217_:%.+]] = stablehlo.real_dynamic_slice [[VAR_205_]], [[VAR_214_]], [[VAR_215_]], [[VAR_216_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_218_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_219_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_220_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_221_:%.+]] = stablehlo.compare  LT, [[VAR_219_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_222_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_221_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_223_:%.+]] = stablehlo.negate [[VAR_219_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_224_:%.+]] = stablehlo.add [[VAR_220_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_225_:%.+]] = stablehlo.add [[VAR_218_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_226_:%.+]] = stablehlo.reverse [[VAR_192_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_227_:%.+]] = stablehlo.select [[VAR_221_]], [[VAR_224_]], [[VAR_218_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_228_:%.+]] = stablehlo.select [[VAR_221_]], [[VAR_225_]], [[VAR_220_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_229_:%.+]] = stablehlo.select [[VAR_221_]], [[VAR_223_]], [[VAR_219_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_230_:%.+]] = stablehlo.select [[VAR_222_]], [[VAR_226_]], [[VAR_192_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_231_:%.+]] = stablehlo.compare  GT, [[VAR_228_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_232_:%.+]] = stablehlo.select [[VAR_231_]], [[VAR_9_]], [[VAR_228_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_233_:%.+]] = stablehlo.compare  LT, [[VAR_232_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_234_:%.+]] = stablehlo.add [[VAR_232_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_235_:%.+]] = stablehlo.select [[VAR_233_]], [[VAR_234_]], [[VAR_232_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_236_:%.+]] = stablehlo.compare  LT, [[VAR_227_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_237_:%.+]] = stablehlo.add [[VAR_227_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_238_:%.+]] = stablehlo.select [[VAR_236_]], [[VAR_237_]], [[VAR_227_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_239_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_238_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_240_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_235_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_241_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_229_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_242_:%.+]] = stablehlo.real_dynamic_slice [[VAR_230_]], [[VAR_239_]], [[VAR_240_]], [[VAR_241_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_243_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_244_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_245_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_246_:%.+]] = stablehlo.compare  LT, [[VAR_244_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_247_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_246_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_248_:%.+]] = stablehlo.negate [[VAR_244_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_249_:%.+]] = stablehlo.add [[VAR_245_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_250_:%.+]] = stablehlo.add [[VAR_243_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_251_:%.+]] = stablehlo.reverse [[VAR_192_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_252_:%.+]] = stablehlo.select [[VAR_246_]], [[VAR_249_]], [[VAR_243_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_253_:%.+]] = stablehlo.select [[VAR_246_]], [[VAR_250_]], [[VAR_245_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_254_:%.+]] = stablehlo.select [[VAR_246_]], [[VAR_248_]], [[VAR_244_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_255_:%.+]] = stablehlo.select [[VAR_247_]], [[VAR_251_]], [[VAR_192_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_256_:%.+]] = stablehlo.compare  GT, [[VAR_253_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_257_:%.+]] = stablehlo.select [[VAR_256_]], [[VAR_9_]], [[VAR_253_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_258_:%.+]] = stablehlo.compare  LT, [[VAR_257_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_259_:%.+]] = stablehlo.add [[VAR_257_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_260_:%.+]] = stablehlo.select [[VAR_258_]], [[VAR_259_]], [[VAR_257_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_261_:%.+]] = stablehlo.compare  LT, [[VAR_252_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_262_:%.+]] = stablehlo.add [[VAR_252_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_263_:%.+]] = stablehlo.select [[VAR_261_]], [[VAR_262_]], [[VAR_252_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_264_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_263_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_265_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_260_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_266_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_254_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_267_:%.+]] = stablehlo.real_dynamic_slice [[VAR_255_]], [[VAR_264_]], [[VAR_265_]], [[VAR_266_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_268_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_269_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_270_:%.+]] = stablehlo.slice [[VAR_9_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_271_:%.+]] = stablehlo.compare  LT, [[VAR_269_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_272_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_271_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_273_:%.+]] = stablehlo.negate [[VAR_269_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_274_:%.+]] = stablehlo.add [[VAR_270_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_275_:%.+]] = stablehlo.add [[VAR_268_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_276_:%.+]] = stablehlo.reverse [[VAR_192_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_277_:%.+]] = stablehlo.select [[VAR_271_]], [[VAR_274_]], [[VAR_268_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_278_:%.+]] = stablehlo.select [[VAR_271_]], [[VAR_275_]], [[VAR_270_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_279_:%.+]] = stablehlo.select [[VAR_271_]], [[VAR_273_]], [[VAR_269_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_280_:%.+]] = stablehlo.select [[VAR_272_]], [[VAR_276_]], [[VAR_192_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_281_:%.+]] = stablehlo.compare  GT, [[VAR_278_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_282_:%.+]] = stablehlo.select [[VAR_281_]], [[VAR_9_]], [[VAR_278_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_283_:%.+]] = stablehlo.compare  LT, [[VAR_282_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_284_:%.+]] = stablehlo.add [[VAR_282_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_285_:%.+]] = stablehlo.select [[VAR_283_]], [[VAR_284_]], [[VAR_282_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_286_:%.+]] = stablehlo.compare  LT, [[VAR_277_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_287_:%.+]] = stablehlo.add [[VAR_277_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_288_:%.+]] = stablehlo.select [[VAR_286_]], [[VAR_287_]], [[VAR_277_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_289_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_288_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_290_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_285_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_291_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_279_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_292_:%.+]] = stablehlo.real_dynamic_slice [[VAR_280_]], [[VAR_289_]], [[VAR_290_]], [[VAR_291_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_293_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_217_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_294_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_139_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_295_:%.+]] = stablehlo.add [[VAR_293_]], [[VAR_294_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_296_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_295_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_297_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_143_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_298_:%.+]] = stablehlo.add [[VAR_296_]], [[VAR_297_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_299_:%.+]] = stablehlo.logistic [[VAR_298_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_300_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_267_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_301_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_141_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_302_:%.+]] = stablehlo.add [[VAR_300_]], [[VAR_301_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_303_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_302_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_304_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_145_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_305_:%.+]] = stablehlo.add [[VAR_303_]], [[VAR_304_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_306_:%.+]] = stablehlo.logistic [[VAR_305_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_307_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_292_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_308_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_142_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_309_:%.+]] = stablehlo.add [[VAR_307_]], [[VAR_308_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_310_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_309_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_311_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_146_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_312_:%.+]] = stablehlo.add [[VAR_310_]], [[VAR_311_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_313_:%.+]] = stablehlo.tanh [[VAR_312_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_314_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_306_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_315_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_70_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_316_:%.+]] = stablehlo.multiply [[VAR_314_]], [[VAR_315_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_317_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_299_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_318_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_313_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_319_:%.+]] = stablehlo.multiply [[VAR_317_]], [[VAR_318_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_320_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_316_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_321_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_319_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_322_:%.+]] = stablehlo.add [[VAR_320_]], [[VAR_321_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_323_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_242_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_324_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_140_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_325_:%.+]] = stablehlo.add [[VAR_323_]], [[VAR_324_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_326_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_325_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_327_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_144_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_328_:%.+]] = stablehlo.add [[VAR_326_]], [[VAR_327_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_329_:%.+]] = stablehlo.logistic [[VAR_328_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_330_:%.+]] = stablehlo.tanh [[VAR_322_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_331_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_329_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_332_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_330_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_333_:%.+]] = stablehlo.multiply [[VAR_331_]], [[VAR_332_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_334_:%.+]] = stablehlo.dynamic_reshape [[VAR_333_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_335_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_336_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_337_:%.+]] = stablehlo.add [[VAR_335_]], [[VAR_336_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_338_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_339_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_340_:%.+]] = stablehlo.slice [[VAR_337_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_341_:%.+]] = stablehlo.compare  LT, [[VAR_339_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_342_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_341_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x512xi1>
// CHECK-DAG:       [[VAR_343_:%.+]] = stablehlo.negate [[VAR_339_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_344_:%.+]] = stablehlo.add [[VAR_340_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_345_:%.+]] = stablehlo.add [[VAR_338_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_346_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_347_:%.+]] = stablehlo.select [[VAR_341_]], [[VAR_344_]], [[VAR_338_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_348_:%.+]] = stablehlo.select [[VAR_341_]], [[VAR_345_]], [[VAR_340_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_349_:%.+]] = stablehlo.select [[VAR_341_]], [[VAR_343_]], [[VAR_339_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_350_:%.+]] = stablehlo.select [[VAR_342_]], [[VAR_346_]], [[PARAM_0_]] : tensor<2x16x512xi1>, tensor<2x16x512xf32>
// CHECK:           [[VAR_351_:%.+]] = stablehlo.compare  GT, [[VAR_348_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_352_:%.+]] = stablehlo.select [[VAR_351_]], [[VAR_18_]], [[VAR_348_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_353_:%.+]] = stablehlo.compare  LT, [[VAR_352_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_354_:%.+]] = stablehlo.add [[VAR_352_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_355_:%.+]] = stablehlo.select [[VAR_353_]], [[VAR_354_]], [[VAR_352_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_356_:%.+]] = stablehlo.compare  LT, [[VAR_347_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_357_:%.+]] = stablehlo.add [[VAR_347_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_358_:%.+]] = stablehlo.select [[VAR_356_]], [[VAR_357_]], [[VAR_347_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_359_:%.+]] = stablehlo.concatenate [[VAR_358_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_360_:%.+]] = stablehlo.concatenate [[VAR_355_]], [[VAR_14_]], [[VAR_11_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_361_:%.+]] = stablehlo.concatenate [[VAR_349_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_362_:%.+]] = stablehlo.real_dynamic_slice [[VAR_350_]], [[VAR_359_]], [[VAR_360_]], [[VAR_361_]] : (tensor<2x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_363_:%.+]] = stablehlo.dynamic_reshape [[VAR_362_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_364_:%.+]] = stablehlo.broadcast_in_dim [[VAR_363_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_365_:%.+]] = stablehlo.broadcast_in_dim [[VAR_131_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_366_:%.+]] = stablehlo.dot [[VAR_364_]], [[VAR_365_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_367_:%.+]] = stablehlo.broadcast_in_dim [[VAR_333_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_368_:%.+]] = stablehlo.broadcast_in_dim [[VAR_132_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_369_:%.+]] = stablehlo.dot [[VAR_367_]], [[VAR_368_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_370_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_366_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_371_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_369_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_372_:%.+]] = stablehlo.add [[VAR_370_]], [[VAR_371_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_373_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_374_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_375_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_376_:%.+]] = stablehlo.compare  LT, [[VAR_374_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_377_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_376_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_378_:%.+]] = stablehlo.negate [[VAR_374_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_379_:%.+]] = stablehlo.add [[VAR_375_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_380_:%.+]] = stablehlo.add [[VAR_373_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_381_:%.+]] = stablehlo.reverse [[VAR_372_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_382_:%.+]] = stablehlo.select [[VAR_376_]], [[VAR_379_]], [[VAR_373_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_383_:%.+]] = stablehlo.select [[VAR_376_]], [[VAR_380_]], [[VAR_375_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_384_:%.+]] = stablehlo.select [[VAR_376_]], [[VAR_378_]], [[VAR_374_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_385_:%.+]] = stablehlo.select [[VAR_377_]], [[VAR_381_]], [[VAR_372_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_386_:%.+]] = stablehlo.compare  GT, [[VAR_383_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_387_:%.+]] = stablehlo.select [[VAR_386_]], [[VAR_9_]], [[VAR_383_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_388_:%.+]] = stablehlo.compare  LT, [[VAR_387_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_389_:%.+]] = stablehlo.add [[VAR_387_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_390_:%.+]] = stablehlo.select [[VAR_388_]], [[VAR_389_]], [[VAR_387_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_391_:%.+]] = stablehlo.compare  LT, [[VAR_382_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_392_:%.+]] = stablehlo.add [[VAR_382_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_393_:%.+]] = stablehlo.select [[VAR_391_]], [[VAR_392_]], [[VAR_382_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_394_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_393_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_395_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_390_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_396_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_384_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_397_:%.+]] = stablehlo.real_dynamic_slice [[VAR_385_]], [[VAR_394_]], [[VAR_395_]], [[VAR_396_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_398_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_399_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_400_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_401_:%.+]] = stablehlo.compare  LT, [[VAR_399_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_402_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_401_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_403_:%.+]] = stablehlo.negate [[VAR_399_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_404_:%.+]] = stablehlo.add [[VAR_400_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_405_:%.+]] = stablehlo.add [[VAR_398_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_406_:%.+]] = stablehlo.reverse [[VAR_372_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_407_:%.+]] = stablehlo.select [[VAR_401_]], [[VAR_404_]], [[VAR_398_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_408_:%.+]] = stablehlo.select [[VAR_401_]], [[VAR_405_]], [[VAR_400_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_409_:%.+]] = stablehlo.select [[VAR_401_]], [[VAR_403_]], [[VAR_399_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_410_:%.+]] = stablehlo.select [[VAR_402_]], [[VAR_406_]], [[VAR_372_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_411_:%.+]] = stablehlo.compare  GT, [[VAR_408_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_412_:%.+]] = stablehlo.select [[VAR_411_]], [[VAR_9_]], [[VAR_408_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_413_:%.+]] = stablehlo.compare  LT, [[VAR_412_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_414_:%.+]] = stablehlo.add [[VAR_412_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_415_:%.+]] = stablehlo.select [[VAR_413_]], [[VAR_414_]], [[VAR_412_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_416_:%.+]] = stablehlo.compare  LT, [[VAR_407_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_417_:%.+]] = stablehlo.add [[VAR_407_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_418_:%.+]] = stablehlo.select [[VAR_416_]], [[VAR_417_]], [[VAR_407_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_419_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_418_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_420_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_415_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_421_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_409_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_422_:%.+]] = stablehlo.real_dynamic_slice [[VAR_410_]], [[VAR_419_]], [[VAR_420_]], [[VAR_421_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_423_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_424_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_425_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_426_:%.+]] = stablehlo.compare  LT, [[VAR_424_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_427_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_426_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_428_:%.+]] = stablehlo.negate [[VAR_424_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_429_:%.+]] = stablehlo.add [[VAR_425_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_430_:%.+]] = stablehlo.add [[VAR_423_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_431_:%.+]] = stablehlo.reverse [[VAR_372_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_432_:%.+]] = stablehlo.select [[VAR_426_]], [[VAR_429_]], [[VAR_423_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_433_:%.+]] = stablehlo.select [[VAR_426_]], [[VAR_430_]], [[VAR_425_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_434_:%.+]] = stablehlo.select [[VAR_426_]], [[VAR_428_]], [[VAR_424_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_435_:%.+]] = stablehlo.select [[VAR_427_]], [[VAR_431_]], [[VAR_372_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_436_:%.+]] = stablehlo.compare  GT, [[VAR_433_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_437_:%.+]] = stablehlo.select [[VAR_436_]], [[VAR_9_]], [[VAR_433_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_438_:%.+]] = stablehlo.compare  LT, [[VAR_437_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_439_:%.+]] = stablehlo.add [[VAR_437_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_440_:%.+]] = stablehlo.select [[VAR_438_]], [[VAR_439_]], [[VAR_437_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_441_:%.+]] = stablehlo.compare  LT, [[VAR_432_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_442_:%.+]] = stablehlo.add [[VAR_432_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_443_:%.+]] = stablehlo.select [[VAR_441_]], [[VAR_442_]], [[VAR_432_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_444_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_443_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_445_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_440_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_446_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_434_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_447_:%.+]] = stablehlo.real_dynamic_slice [[VAR_435_]], [[VAR_444_]], [[VAR_445_]], [[VAR_446_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_448_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_449_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_450_:%.+]] = stablehlo.slice [[VAR_9_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_451_:%.+]] = stablehlo.compare  LT, [[VAR_449_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_452_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_451_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_453_:%.+]] = stablehlo.negate [[VAR_449_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_454_:%.+]] = stablehlo.add [[VAR_450_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_455_:%.+]] = stablehlo.add [[VAR_448_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_456_:%.+]] = stablehlo.reverse [[VAR_372_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_457_:%.+]] = stablehlo.select [[VAR_451_]], [[VAR_454_]], [[VAR_448_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_458_:%.+]] = stablehlo.select [[VAR_451_]], [[VAR_455_]], [[VAR_450_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_459_:%.+]] = stablehlo.select [[VAR_451_]], [[VAR_453_]], [[VAR_449_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_460_:%.+]] = stablehlo.select [[VAR_452_]], [[VAR_456_]], [[VAR_372_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_461_:%.+]] = stablehlo.compare  GT, [[VAR_458_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_462_:%.+]] = stablehlo.select [[VAR_461_]], [[VAR_9_]], [[VAR_458_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_463_:%.+]] = stablehlo.compare  LT, [[VAR_462_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_464_:%.+]] = stablehlo.add [[VAR_462_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_465_:%.+]] = stablehlo.select [[VAR_463_]], [[VAR_464_]], [[VAR_462_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_466_:%.+]] = stablehlo.compare  LT, [[VAR_457_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_467_:%.+]] = stablehlo.add [[VAR_457_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_468_:%.+]] = stablehlo.select [[VAR_466_]], [[VAR_467_]], [[VAR_457_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_469_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_468_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_470_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_465_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_471_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_459_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_472_:%.+]] = stablehlo.real_dynamic_slice [[VAR_460_]], [[VAR_469_]], [[VAR_470_]], [[VAR_471_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_473_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_397_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_474_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_139_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_475_:%.+]] = stablehlo.add [[VAR_473_]], [[VAR_474_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_476_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_475_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_477_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_143_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_478_:%.+]] = stablehlo.add [[VAR_476_]], [[VAR_477_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_479_:%.+]] = stablehlo.logistic [[VAR_478_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_480_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_447_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_481_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_141_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_482_:%.+]] = stablehlo.add [[VAR_480_]], [[VAR_481_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_483_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_482_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_484_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_145_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_485_:%.+]] = stablehlo.add [[VAR_483_]], [[VAR_484_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_486_:%.+]] = stablehlo.logistic [[VAR_485_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_487_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_472_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_488_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_142_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_489_:%.+]] = stablehlo.add [[VAR_487_]], [[VAR_488_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_490_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_489_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_491_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_146_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_492_:%.+]] = stablehlo.add [[VAR_490_]], [[VAR_491_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_493_:%.+]] = stablehlo.tanh [[VAR_492_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_494_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_486_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_495_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_322_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_496_:%.+]] = stablehlo.multiply [[VAR_494_]], [[VAR_495_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_497_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_479_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_498_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_493_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_499_:%.+]] = stablehlo.multiply [[VAR_497_]], [[VAR_498_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_500_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_496_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_501_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_499_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_502_:%.+]] = stablehlo.add [[VAR_500_]], [[VAR_501_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_503_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_422_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_504_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_140_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_505_:%.+]] = stablehlo.add [[VAR_503_]], [[VAR_504_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_506_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_505_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_507_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_144_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_508_:%.+]] = stablehlo.add [[VAR_506_]], [[VAR_507_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_509_:%.+]] = stablehlo.logistic [[VAR_508_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_510_:%.+]] = stablehlo.tanh [[VAR_502_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_511_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_509_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_512_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_510_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_513_:%.+]] = stablehlo.multiply [[VAR_511_]], [[VAR_512_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_514_:%.+]] = stablehlo.dynamic_reshape [[VAR_513_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_515_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_516_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_517_:%.+]] = stablehlo.add [[VAR_515_]], [[VAR_516_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_518_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_519_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_520_:%.+]] = stablehlo.slice [[VAR_517_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_521_:%.+]] = stablehlo.compare  LT, [[VAR_519_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_522_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_521_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x512xi1>
// CHECK-DAG:       [[VAR_523_:%.+]] = stablehlo.negate [[VAR_519_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_524_:%.+]] = stablehlo.add [[VAR_520_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_525_:%.+]] = stablehlo.add [[VAR_518_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_526_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_527_:%.+]] = stablehlo.select [[VAR_521_]], [[VAR_524_]], [[VAR_518_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_528_:%.+]] = stablehlo.select [[VAR_521_]], [[VAR_525_]], [[VAR_520_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_529_:%.+]] = stablehlo.select [[VAR_521_]], [[VAR_523_]], [[VAR_519_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_530_:%.+]] = stablehlo.select [[VAR_522_]], [[VAR_526_]], [[PARAM_0_]] : tensor<2x16x512xi1>, tensor<2x16x512xf32>
// CHECK:           [[VAR_531_:%.+]] = stablehlo.compare  GT, [[VAR_528_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_532_:%.+]] = stablehlo.select [[VAR_531_]], [[VAR_18_]], [[VAR_528_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_533_:%.+]] = stablehlo.compare  LT, [[VAR_532_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_534_:%.+]] = stablehlo.add [[VAR_532_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_535_:%.+]] = stablehlo.select [[VAR_533_]], [[VAR_534_]], [[VAR_532_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_536_:%.+]] = stablehlo.compare  LT, [[VAR_527_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_537_:%.+]] = stablehlo.add [[VAR_527_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_538_:%.+]] = stablehlo.select [[VAR_536_]], [[VAR_537_]], [[VAR_527_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_539_:%.+]] = stablehlo.concatenate [[VAR_538_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_540_:%.+]] = stablehlo.concatenate [[VAR_535_]], [[VAR_14_]], [[VAR_11_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_541_:%.+]] = stablehlo.concatenate [[VAR_529_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_542_:%.+]] = stablehlo.real_dynamic_slice [[VAR_530_]], [[VAR_539_]], [[VAR_540_]], [[VAR_541_]] : (tensor<2x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_543_:%.+]] = stablehlo.dynamic_reshape [[VAR_542_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_544_:%.+]] = stablehlo.broadcast_in_dim [[VAR_543_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_545_:%.+]] = stablehlo.broadcast_in_dim [[VAR_133_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_546_:%.+]] = stablehlo.dot [[VAR_544_]], [[VAR_545_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_547_:%.+]] = stablehlo.broadcast_in_dim [[VAR_96_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_548_:%.+]] = stablehlo.broadcast_in_dim [[VAR_134_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_549_:%.+]] = stablehlo.dot [[VAR_547_]], [[VAR_548_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_550_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_546_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_551_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_549_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_552_:%.+]] = stablehlo.add [[VAR_550_]], [[VAR_551_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_553_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_554_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_555_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_556_:%.+]] = stablehlo.compare  LT, [[VAR_554_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_557_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_556_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_558_:%.+]] = stablehlo.negate [[VAR_554_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_559_:%.+]] = stablehlo.add [[VAR_555_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_560_:%.+]] = stablehlo.add [[VAR_553_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_561_:%.+]] = stablehlo.reverse [[VAR_552_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_562_:%.+]] = stablehlo.select [[VAR_556_]], [[VAR_559_]], [[VAR_553_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_563_:%.+]] = stablehlo.select [[VAR_556_]], [[VAR_560_]], [[VAR_555_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_564_:%.+]] = stablehlo.select [[VAR_556_]], [[VAR_558_]], [[VAR_554_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_565_:%.+]] = stablehlo.select [[VAR_557_]], [[VAR_561_]], [[VAR_552_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_566_:%.+]] = stablehlo.compare  GT, [[VAR_563_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_567_:%.+]] = stablehlo.select [[VAR_566_]], [[VAR_9_]], [[VAR_563_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_568_:%.+]] = stablehlo.compare  LT, [[VAR_567_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_569_:%.+]] = stablehlo.add [[VAR_567_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_570_:%.+]] = stablehlo.select [[VAR_568_]], [[VAR_569_]], [[VAR_567_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_571_:%.+]] = stablehlo.compare  LT, [[VAR_562_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_572_:%.+]] = stablehlo.add [[VAR_562_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_573_:%.+]] = stablehlo.select [[VAR_571_]], [[VAR_572_]], [[VAR_562_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_574_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_573_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_575_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_570_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_576_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_564_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_577_:%.+]] = stablehlo.real_dynamic_slice [[VAR_565_]], [[VAR_574_]], [[VAR_575_]], [[VAR_576_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_578_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_579_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_580_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_581_:%.+]] = stablehlo.compare  LT, [[VAR_579_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_582_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_581_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_583_:%.+]] = stablehlo.negate [[VAR_579_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_584_:%.+]] = stablehlo.add [[VAR_580_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_585_:%.+]] = stablehlo.add [[VAR_578_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_586_:%.+]] = stablehlo.reverse [[VAR_552_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_587_:%.+]] = stablehlo.select [[VAR_581_]], [[VAR_584_]], [[VAR_578_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_588_:%.+]] = stablehlo.select [[VAR_581_]], [[VAR_585_]], [[VAR_580_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_589_:%.+]] = stablehlo.select [[VAR_581_]], [[VAR_583_]], [[VAR_579_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_590_:%.+]] = stablehlo.select [[VAR_582_]], [[VAR_586_]], [[VAR_552_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_591_:%.+]] = stablehlo.compare  GT, [[VAR_588_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_592_:%.+]] = stablehlo.select [[VAR_591_]], [[VAR_9_]], [[VAR_588_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_593_:%.+]] = stablehlo.compare  LT, [[VAR_592_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_594_:%.+]] = stablehlo.add [[VAR_592_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_595_:%.+]] = stablehlo.select [[VAR_593_]], [[VAR_594_]], [[VAR_592_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_596_:%.+]] = stablehlo.compare  LT, [[VAR_587_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_597_:%.+]] = stablehlo.add [[VAR_587_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_598_:%.+]] = stablehlo.select [[VAR_596_]], [[VAR_597_]], [[VAR_587_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_599_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_598_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_600_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_595_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_601_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_589_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_602_:%.+]] = stablehlo.real_dynamic_slice [[VAR_590_]], [[VAR_599_]], [[VAR_600_]], [[VAR_601_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_603_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_604_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_605_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_606_:%.+]] = stablehlo.compare  LT, [[VAR_604_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_607_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_606_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_608_:%.+]] = stablehlo.negate [[VAR_604_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_609_:%.+]] = stablehlo.add [[VAR_605_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_610_:%.+]] = stablehlo.add [[VAR_603_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_611_:%.+]] = stablehlo.reverse [[VAR_552_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_612_:%.+]] = stablehlo.select [[VAR_606_]], [[VAR_609_]], [[VAR_603_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_613_:%.+]] = stablehlo.select [[VAR_606_]], [[VAR_610_]], [[VAR_605_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_614_:%.+]] = stablehlo.select [[VAR_606_]], [[VAR_608_]], [[VAR_604_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_615_:%.+]] = stablehlo.select [[VAR_607_]], [[VAR_611_]], [[VAR_552_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_616_:%.+]] = stablehlo.compare  GT, [[VAR_613_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_617_:%.+]] = stablehlo.select [[VAR_616_]], [[VAR_9_]], [[VAR_613_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_618_:%.+]] = stablehlo.compare  LT, [[VAR_617_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_619_:%.+]] = stablehlo.add [[VAR_617_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_620_:%.+]] = stablehlo.select [[VAR_618_]], [[VAR_619_]], [[VAR_617_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_621_:%.+]] = stablehlo.compare  LT, [[VAR_612_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_622_:%.+]] = stablehlo.add [[VAR_612_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_623_:%.+]] = stablehlo.select [[VAR_621_]], [[VAR_622_]], [[VAR_612_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_624_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_623_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_625_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_620_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_626_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_614_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_627_:%.+]] = stablehlo.real_dynamic_slice [[VAR_615_]], [[VAR_624_]], [[VAR_625_]], [[VAR_626_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_628_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_629_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_630_:%.+]] = stablehlo.slice [[VAR_9_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_631_:%.+]] = stablehlo.compare  LT, [[VAR_629_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_632_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_631_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_633_:%.+]] = stablehlo.negate [[VAR_629_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_634_:%.+]] = stablehlo.add [[VAR_630_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_635_:%.+]] = stablehlo.add [[VAR_628_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_636_:%.+]] = stablehlo.reverse [[VAR_552_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_637_:%.+]] = stablehlo.select [[VAR_631_]], [[VAR_634_]], [[VAR_628_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_638_:%.+]] = stablehlo.select [[VAR_631_]], [[VAR_635_]], [[VAR_630_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_639_:%.+]] = stablehlo.select [[VAR_631_]], [[VAR_633_]], [[VAR_629_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_640_:%.+]] = stablehlo.select [[VAR_632_]], [[VAR_636_]], [[VAR_552_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_641_:%.+]] = stablehlo.compare  GT, [[VAR_638_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_642_:%.+]] = stablehlo.select [[VAR_641_]], [[VAR_9_]], [[VAR_638_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_643_:%.+]] = stablehlo.compare  LT, [[VAR_642_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_644_:%.+]] = stablehlo.add [[VAR_642_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_645_:%.+]] = stablehlo.select [[VAR_643_]], [[VAR_644_]], [[VAR_642_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_646_:%.+]] = stablehlo.compare  LT, [[VAR_637_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_647_:%.+]] = stablehlo.add [[VAR_637_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_648_:%.+]] = stablehlo.select [[VAR_646_]], [[VAR_647_]], [[VAR_637_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_649_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_648_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_650_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_645_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_651_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_639_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_652_:%.+]] = stablehlo.real_dynamic_slice [[VAR_640_]], [[VAR_649_]], [[VAR_650_]], [[VAR_651_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_653_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_577_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_654_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_147_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_655_:%.+]] = stablehlo.add [[VAR_653_]], [[VAR_654_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_656_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_655_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_657_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_151_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_658_:%.+]] = stablehlo.add [[VAR_656_]], [[VAR_657_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_659_:%.+]] = stablehlo.logistic [[VAR_658_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_660_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_627_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_661_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_149_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_662_:%.+]] = stablehlo.add [[VAR_660_]], [[VAR_661_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_663_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_662_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_664_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_153_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_665_:%.+]] = stablehlo.add [[VAR_663_]], [[VAR_664_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_666_:%.+]] = stablehlo.logistic [[VAR_665_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_667_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_652_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_668_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_150_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_669_:%.+]] = stablehlo.add [[VAR_667_]], [[VAR_668_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_670_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_669_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_671_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_154_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_672_:%.+]] = stablehlo.add [[VAR_670_]], [[VAR_671_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_673_:%.+]] = stablehlo.tanh [[VAR_672_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_674_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_666_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_675_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_122_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_676_:%.+]] = stablehlo.multiply [[VAR_674_]], [[VAR_675_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_677_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_659_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_678_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_673_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_679_:%.+]] = stablehlo.multiply [[VAR_677_]], [[VAR_678_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_680_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_676_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_681_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_679_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_682_:%.+]] = stablehlo.add [[VAR_680_]], [[VAR_681_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_683_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_602_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_684_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_148_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_685_:%.+]] = stablehlo.add [[VAR_683_]], [[VAR_684_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_686_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_685_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_687_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_152_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_688_:%.+]] = stablehlo.add [[VAR_686_]], [[VAR_687_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_689_:%.+]] = stablehlo.logistic [[VAR_688_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_690_:%.+]] = stablehlo.tanh [[VAR_682_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_691_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_689_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_692_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_690_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_693_:%.+]] = stablehlo.multiply [[VAR_691_]], [[VAR_692_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_694_:%.+]] = stablehlo.dynamic_reshape [[VAR_693_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_695_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_16_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_696_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_17_]], [[VAR_12_]], dims = [0] : (tensor<1xi64>, tensor<1xindex>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_697_:%.+]] = stablehlo.add [[VAR_695_]], [[VAR_696_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_698_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_699_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_700_:%.+]] = stablehlo.slice [[VAR_697_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_701_:%.+]] = stablehlo.compare  LT, [[VAR_699_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_702_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_701_]], [[VAR_4_]], dims = [0] : (tensor<1xi1>, tensor<3xindex>) -> tensor<2x16x512xi1>
// CHECK-DAG:       [[VAR_703_:%.+]] = stablehlo.negate [[VAR_699_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_704_:%.+]] = stablehlo.add [[VAR_700_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_705_:%.+]] = stablehlo.add [[VAR_698_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_706_:%.+]] = stablehlo.reverse [[PARAM_0_]], dims = [0] : tensor<2x16x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_707_:%.+]] = stablehlo.select [[VAR_701_]], [[VAR_704_]], [[VAR_698_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_708_:%.+]] = stablehlo.select [[VAR_701_]], [[VAR_705_]], [[VAR_700_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_709_:%.+]] = stablehlo.select [[VAR_701_]], [[VAR_703_]], [[VAR_699_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_710_:%.+]] = stablehlo.select [[VAR_702_]], [[VAR_706_]], [[PARAM_0_]] : tensor<2x16x512xi1>, tensor<2x16x512xf32>
// CHECK:           [[VAR_711_:%.+]] = stablehlo.compare  GT, [[VAR_708_]], [[VAR_18_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_712_:%.+]] = stablehlo.select [[VAR_711_]], [[VAR_18_]], [[VAR_708_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_713_:%.+]] = stablehlo.compare  LT, [[VAR_712_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_714_:%.+]] = stablehlo.add [[VAR_712_]], [[VAR_18_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_715_:%.+]] = stablehlo.select [[VAR_713_]], [[VAR_714_]], [[VAR_712_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_716_:%.+]] = stablehlo.compare  LT, [[VAR_707_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_717_:%.+]] = stablehlo.add [[VAR_707_]], [[VAR_18_]] : tensor<1xi64>
// CHECK:           [[VAR_718_:%.+]] = stablehlo.select [[VAR_716_]], [[VAR_717_]], [[VAR_707_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_719_:%.+]] = stablehlo.concatenate [[VAR_718_]], [[VAR_16_]], [[VAR_16_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_720_:%.+]] = stablehlo.concatenate [[VAR_715_]], [[VAR_14_]], [[VAR_11_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_721_:%.+]] = stablehlo.concatenate [[VAR_709_]], [[VAR_17_]], [[VAR_17_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_722_:%.+]] = stablehlo.real_dynamic_slice [[VAR_710_]], [[VAR_719_]], [[VAR_720_]], [[VAR_721_]] : (tensor<2x16x512xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_723_:%.+]] = stablehlo.dynamic_reshape [[VAR_722_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_724_:%.+]] = stablehlo.broadcast_in_dim [[VAR_723_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_725_:%.+]] = stablehlo.broadcast_in_dim [[VAR_133_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_726_:%.+]] = stablehlo.dot [[VAR_724_]], [[VAR_725_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_727_:%.+]] = stablehlo.broadcast_in_dim [[VAR_693_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_728_:%.+]] = stablehlo.broadcast_in_dim [[VAR_134_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_729_:%.+]] = stablehlo.dot [[VAR_727_]], [[VAR_728_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_730_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_726_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_731_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_729_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_732_:%.+]] = stablehlo.add [[VAR_730_]], [[VAR_731_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_733_:%.+]] = stablehlo.slice [[VAR_16_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_734_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_735_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_736_:%.+]] = stablehlo.compare  LT, [[VAR_734_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_737_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_736_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_738_:%.+]] = stablehlo.negate [[VAR_734_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_739_:%.+]] = stablehlo.add [[VAR_735_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_740_:%.+]] = stablehlo.add [[VAR_733_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_741_:%.+]] = stablehlo.reverse [[VAR_732_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_742_:%.+]] = stablehlo.select [[VAR_736_]], [[VAR_739_]], [[VAR_733_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_743_:%.+]] = stablehlo.select [[VAR_736_]], [[VAR_740_]], [[VAR_735_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_744_:%.+]] = stablehlo.select [[VAR_736_]], [[VAR_738_]], [[VAR_734_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_745_:%.+]] = stablehlo.select [[VAR_737_]], [[VAR_741_]], [[VAR_732_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_746_:%.+]] = stablehlo.compare  GT, [[VAR_743_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_747_:%.+]] = stablehlo.select [[VAR_746_]], [[VAR_9_]], [[VAR_743_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_748_:%.+]] = stablehlo.compare  LT, [[VAR_747_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_749_:%.+]] = stablehlo.add [[VAR_747_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_750_:%.+]] = stablehlo.select [[VAR_748_]], [[VAR_749_]], [[VAR_747_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_751_:%.+]] = stablehlo.compare  LT, [[VAR_742_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_752_:%.+]] = stablehlo.add [[VAR_742_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_753_:%.+]] = stablehlo.select [[VAR_751_]], [[VAR_752_]], [[VAR_742_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_754_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_753_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_755_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_750_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_756_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_744_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_757_:%.+]] = stablehlo.real_dynamic_slice [[VAR_745_]], [[VAR_754_]], [[VAR_755_]], [[VAR_756_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_758_:%.+]] = stablehlo.slice [[VAR_13_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_759_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_760_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_761_:%.+]] = stablehlo.compare  LT, [[VAR_759_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_762_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_761_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_763_:%.+]] = stablehlo.negate [[VAR_759_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_764_:%.+]] = stablehlo.add [[VAR_760_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_765_:%.+]] = stablehlo.add [[VAR_758_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_766_:%.+]] = stablehlo.reverse [[VAR_732_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_767_:%.+]] = stablehlo.select [[VAR_761_]], [[VAR_764_]], [[VAR_758_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_768_:%.+]] = stablehlo.select [[VAR_761_]], [[VAR_765_]], [[VAR_760_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_769_:%.+]] = stablehlo.select [[VAR_761_]], [[VAR_763_]], [[VAR_759_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_770_:%.+]] = stablehlo.select [[VAR_762_]], [[VAR_766_]], [[VAR_732_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_771_:%.+]] = stablehlo.compare  GT, [[VAR_768_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_772_:%.+]] = stablehlo.select [[VAR_771_]], [[VAR_9_]], [[VAR_768_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_773_:%.+]] = stablehlo.compare  LT, [[VAR_772_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_774_:%.+]] = stablehlo.add [[VAR_772_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_775_:%.+]] = stablehlo.select [[VAR_773_]], [[VAR_774_]], [[VAR_772_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_776_:%.+]] = stablehlo.compare  LT, [[VAR_767_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_777_:%.+]] = stablehlo.add [[VAR_767_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_778_:%.+]] = stablehlo.select [[VAR_776_]], [[VAR_777_]], [[VAR_767_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_779_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_778_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_780_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_775_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_781_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_769_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_782_:%.+]] = stablehlo.real_dynamic_slice [[VAR_770_]], [[VAR_779_]], [[VAR_780_]], [[VAR_781_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_783_:%.+]] = stablehlo.slice [[VAR_11_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_784_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_785_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_786_:%.+]] = stablehlo.compare  LT, [[VAR_784_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_787_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_786_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_788_:%.+]] = stablehlo.negate [[VAR_784_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_789_:%.+]] = stablehlo.add [[VAR_785_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_790_:%.+]] = stablehlo.add [[VAR_783_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_791_:%.+]] = stablehlo.reverse [[VAR_732_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_792_:%.+]] = stablehlo.select [[VAR_786_]], [[VAR_789_]], [[VAR_783_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_793_:%.+]] = stablehlo.select [[VAR_786_]], [[VAR_790_]], [[VAR_785_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_794_:%.+]] = stablehlo.select [[VAR_786_]], [[VAR_788_]], [[VAR_784_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_795_:%.+]] = stablehlo.select [[VAR_787_]], [[VAR_791_]], [[VAR_732_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_796_:%.+]] = stablehlo.compare  GT, [[VAR_793_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_797_:%.+]] = stablehlo.select [[VAR_796_]], [[VAR_9_]], [[VAR_793_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_798_:%.+]] = stablehlo.compare  LT, [[VAR_797_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_799_:%.+]] = stablehlo.add [[VAR_797_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_800_:%.+]] = stablehlo.select [[VAR_798_]], [[VAR_799_]], [[VAR_797_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_801_:%.+]] = stablehlo.compare  LT, [[VAR_792_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_802_:%.+]] = stablehlo.add [[VAR_792_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_803_:%.+]] = stablehlo.select [[VAR_801_]], [[VAR_802_]], [[VAR_792_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_804_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_803_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_805_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_800_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_806_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_794_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_807_:%.+]] = stablehlo.real_dynamic_slice [[VAR_795_]], [[VAR_804_]], [[VAR_805_]], [[VAR_806_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_808_:%.+]] = stablehlo.slice [[VAR_10_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_809_:%.+]] = stablehlo.slice [[VAR_17_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_810_:%.+]] = stablehlo.slice [[VAR_9_]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_811_:%.+]] = stablehlo.compare  LT, [[VAR_809_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_812_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_811_]], [[VAR_2_]], dims = [0] : (tensor<1xi1>, tensor<2xindex>) -> tensor<16x1024xi1>
// CHECK-DAG:       [[VAR_813_:%.+]] = stablehlo.negate [[VAR_809_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_814_:%.+]] = stablehlo.add [[VAR_810_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_815_:%.+]] = stablehlo.add [[VAR_808_]], [[VAR_17_]] : tensor<1xi64>
// CHECK-DAG:       [[VAR_816_:%.+]] = stablehlo.reverse [[VAR_732_]], dims = [1] : tensor<16x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_817_:%.+]] = stablehlo.select [[VAR_811_]], [[VAR_814_]], [[VAR_808_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_818_:%.+]] = stablehlo.select [[VAR_811_]], [[VAR_815_]], [[VAR_810_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_819_:%.+]] = stablehlo.select [[VAR_811_]], [[VAR_813_]], [[VAR_809_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_820_:%.+]] = stablehlo.select [[VAR_812_]], [[VAR_816_]], [[VAR_732_]] : tensor<16x1024xi1>, tensor<16x1024xf32>
// CHECK:           [[VAR_821_:%.+]] = stablehlo.compare  GT, [[VAR_818_]], [[VAR_9_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:           [[VAR_822_:%.+]] = stablehlo.select [[VAR_821_]], [[VAR_9_]], [[VAR_818_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_823_:%.+]] = stablehlo.compare  LT, [[VAR_822_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_824_:%.+]] = stablehlo.add [[VAR_822_]], [[VAR_9_]] : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_825_:%.+]] = stablehlo.select [[VAR_823_]], [[VAR_824_]], [[VAR_822_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_826_:%.+]] = stablehlo.compare  LT, [[VAR_817_]], [[VAR_16_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:       [[VAR_827_:%.+]] = stablehlo.add [[VAR_817_]], [[VAR_9_]] : tensor<1xi64>
// CHECK:           [[VAR_828_:%.+]] = stablehlo.select [[VAR_826_]], [[VAR_827_]], [[VAR_817_]] : tensor<1xi1>, tensor<1xi64>
// CHECK-DAG:       [[VAR_829_:%.+]] = stablehlo.concatenate [[VAR_16_]], [[VAR_828_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_830_:%.+]] = stablehlo.concatenate [[VAR_14_]], [[VAR_825_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_831_:%.+]] = stablehlo.concatenate [[VAR_17_]], [[VAR_819_]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_832_:%.+]] = stablehlo.real_dynamic_slice [[VAR_820_]], [[VAR_829_]], [[VAR_830_]], [[VAR_831_]] : (tensor<16x1024xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_833_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_757_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_834_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_147_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_835_:%.+]] = stablehlo.add [[VAR_833_]], [[VAR_834_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_836_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_835_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_837_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_151_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_838_:%.+]] = stablehlo.add [[VAR_836_]], [[VAR_837_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_839_:%.+]] = stablehlo.logistic [[VAR_838_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_840_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_807_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_841_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_149_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_842_:%.+]] = stablehlo.add [[VAR_840_]], [[VAR_841_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_843_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_842_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_844_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_153_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_845_:%.+]] = stablehlo.add [[VAR_843_]], [[VAR_844_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_846_:%.+]] = stablehlo.logistic [[VAR_845_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_847_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_832_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_848_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_150_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_849_:%.+]] = stablehlo.add [[VAR_847_]], [[VAR_848_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_850_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_849_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_851_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_154_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_852_:%.+]] = stablehlo.add [[VAR_850_]], [[VAR_851_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_853_:%.+]] = stablehlo.tanh [[VAR_852_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_854_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_846_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_855_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_682_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_856_:%.+]] = stablehlo.multiply [[VAR_854_]], [[VAR_855_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_857_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_839_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_858_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_853_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_859_:%.+]] = stablehlo.multiply [[VAR_857_]], [[VAR_858_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_860_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_856_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_861_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_859_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_862_:%.+]] = stablehlo.add [[VAR_860_]], [[VAR_861_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_863_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_782_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_864_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_148_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_865_:%.+]] = stablehlo.add [[VAR_863_]], [[VAR_864_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_866_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_865_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_867_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_152_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_868_:%.+]] = stablehlo.add [[VAR_866_]], [[VAR_867_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_869_:%.+]] = stablehlo.logistic [[VAR_868_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_870_:%.+]] = stablehlo.tanh [[VAR_862_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_871_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_869_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_872_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_870_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_873_:%.+]] = stablehlo.multiply [[VAR_871_]], [[VAR_872_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_874_:%.+]] = stablehlo.dynamic_reshape [[VAR_873_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_875_:%.+]] = stablehlo.concatenate [[VAR_334_]], [[VAR_514_]], dim = 0 : (tensor<1x1x16x256xf32>, tensor<1x1x16x256xf32>) -> tensor<2x1x16x256xf32>
// CHECK:           [[VAR_876_:%.+]] = stablehlo.concatenate [[VAR_874_]], [[VAR_694_]], dim = 0 : (tensor<1x1x16x256xf32>, tensor<1x1x16x256xf32>) -> tensor<2x1x16x256xf32>
// CHECK:           [[VAR_877_:%.+]] = stablehlo.concatenate [[VAR_875_]], [[VAR_876_]], dim = 1 : (tensor<2x1x16x256xf32>, tensor<2x1x16x256xf32>) -> tensor<2x2x16x256xf32>
// CHECK:           return [[VAR_877_]] : tensor<2x2x16x256xf32>
// CHECK:         }
}
