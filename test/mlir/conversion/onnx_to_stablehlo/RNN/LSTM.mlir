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
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.const_shape [2048] : tensor<1xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.const_shape [1024, 256] : tensor<2xindex>
// CHECK-DAG:       [[VAR_6_:%.+]] = shape.const_shape [1024, 512] : tensor<2xindex>
// CHECK-DAG:       [[VAR_7_:%.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.constant dense<768> : tensor<i64>
// CHECK-DAG:       [[VAR_9_:%.+]] = stablehlo.constant dense<512> : tensor<i64>
// CHECK-DAG:       [[VAR_10_:%.+]] = stablehlo.constant dense<256> : tensor<i64>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x16x256xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           [[VAR_15_:%.+]] = stablehlo.dynamic_slice [[VAR_11_]], [[VAR_12_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.dynamic_reshape [[VAR_15_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = stablehlo.dynamic_slice [[VAR_11_]], [[VAR_12_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = stablehlo.dynamic_reshape [[VAR_17_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = stablehlo.dynamic_slice [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = stablehlo.dynamic_reshape [[VAR_19_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = stablehlo.dynamic_slice [[VAR_11_]], [[VAR_14_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 256] : (tensor<2x16x256xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.dynamic_reshape [[VAR_21_]], [[VAR_0_]] : (tensor<1x16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = stablehlo.slice [[PARAM_2_]] [0:1, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = stablehlo.slice [[PARAM_2_]] [1:2, 0:1024, 0:512] : (tensor<2x1024x512xf32>) -> tensor<1x1024x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = stablehlo.dynamic_reshape [[VAR_23_]], [[VAR_6_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.dynamic_reshape [[VAR_24_]], [[VAR_6_]] : (tensor<1x1024x512xf32>, tensor<2xindex>) -> tensor<1024x512xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.slice [[PARAM_3_]] [0:1, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = stablehlo.slice [[PARAM_3_]] [1:2, 0:1024, 0:256] : (tensor<2x1024x256xf32>) -> tensor<1x1024x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_29_:%.+]] = stablehlo.dynamic_reshape [[VAR_27_]], [[VAR_5_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = stablehlo.dynamic_reshape [[VAR_28_]], [[VAR_5_]] : (tensor<1x1024x256xf32>, tensor<2xindex>) -> tensor<1024x256xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = stablehlo.transpose [[VAR_25_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_32_:%.+]] = stablehlo.transpose [[VAR_29_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = stablehlo.transpose [[VAR_26_]], dims = [1, 0] : (tensor<1024x512xf32>) -> tensor<512x1024xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = stablehlo.transpose [[VAR_30_]], dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = stablehlo.slice [[PARAM_1_]] [0:1, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = stablehlo.slice [[PARAM_1_]] [1:2, 0:2048] : (tensor<2x2048xf32>) -> tensor<1x2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_37_:%.+]] = stablehlo.dynamic_reshape [[VAR_35_]], [[VAR_4_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = stablehlo.dynamic_reshape [[VAR_36_]], [[VAR_4_]] : (tensor<1x2048xf32>, tensor<1xindex>) -> tensor<2048xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_39_:%.+]] = stablehlo.slice [[VAR_37_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = stablehlo.slice [[VAR_37_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = stablehlo.slice [[VAR_37_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = stablehlo.slice [[VAR_37_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = stablehlo.slice [[VAR_37_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = stablehlo.slice [[VAR_37_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = stablehlo.slice [[VAR_37_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = stablehlo.slice [[VAR_37_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = stablehlo.slice [[VAR_38_]] [0:256] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = stablehlo.slice [[VAR_38_]] [256:512] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = stablehlo.slice [[VAR_38_]] [512:768] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = stablehlo.slice [[VAR_38_]] [768:1024] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = stablehlo.slice [[VAR_38_]] [1024:1280] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = stablehlo.slice [[VAR_38_]] [1280:1536] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = stablehlo.slice [[VAR_38_]] [1536:1792] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = stablehlo.slice [[VAR_38_]] [1792:2048] : (tensor<2048xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_55_:%.+]] = stablehlo.reshape [[VAR_13_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           [[VAR_56_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_55_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 512] : (tensor<2x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_57_:%.+]] = stablehlo.dynamic_reshape [[VAR_56_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = stablehlo.broadcast_in_dim [[VAR_57_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = stablehlo.broadcast_in_dim [[VAR_31_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_60_:%.+]] = stablehlo.dot [[VAR_58_]], [[VAR_59_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_61_:%.+]] = stablehlo.broadcast_in_dim [[VAR_16_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_62_:%.+]] = stablehlo.broadcast_in_dim [[VAR_32_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_63_:%.+]] = stablehlo.dot [[VAR_61_]], [[VAR_62_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_64_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_60_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_65_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_63_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_66_:%.+]] = stablehlo.add [[VAR_64_]], [[VAR_65_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_67_:%.+]] = stablehlo.dynamic_slice [[VAR_66_]], [[VAR_12_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_68_:%.+]] = stablehlo.dynamic_slice [[VAR_66_]], [[VAR_12_]], [[VAR_10_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = stablehlo.dynamic_slice [[VAR_66_]], [[VAR_12_]], [[VAR_9_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_70_:%.+]] = stablehlo.dynamic_slice [[VAR_66_]], [[VAR_12_]], [[VAR_8_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_71_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_67_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_72_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_39_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_73_:%.+]] = stablehlo.add [[VAR_71_]], [[VAR_72_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_74_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_73_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_75_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_43_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_76_:%.+]] = stablehlo.add [[VAR_74_]], [[VAR_75_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_77_:%.+]] = stablehlo.logistic [[VAR_76_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_78_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_69_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_79_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_41_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_80_:%.+]] = stablehlo.add [[VAR_78_]], [[VAR_79_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_81_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_80_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_82_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_45_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_83_:%.+]] = stablehlo.add [[VAR_81_]], [[VAR_82_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_84_:%.+]] = stablehlo.logistic [[VAR_83_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_85_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_70_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_86_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_42_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_87_:%.+]] = stablehlo.add [[VAR_85_]], [[VAR_86_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_88_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_87_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_89_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_46_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_90_:%.+]] = stablehlo.add [[VAR_88_]], [[VAR_89_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_91_:%.+]] = stablehlo.tanh [[VAR_90_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_92_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_84_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_93_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_18_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_94_:%.+]] = stablehlo.multiply [[VAR_92_]], [[VAR_93_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_95_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_77_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_96_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_91_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = stablehlo.multiply [[VAR_95_]], [[VAR_96_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_98_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_94_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_99_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_97_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_100_:%.+]] = stablehlo.add [[VAR_98_]], [[VAR_99_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_101_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_68_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_102_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_40_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_103_:%.+]] = stablehlo.add [[VAR_101_]], [[VAR_102_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_104_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_103_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_105_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_44_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_106_:%.+]] = stablehlo.add [[VAR_104_]], [[VAR_105_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_107_:%.+]] = stablehlo.logistic [[VAR_106_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_108_:%.+]] = stablehlo.tanh [[VAR_100_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_109_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_107_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_110_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_108_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_111_:%.+]] = stablehlo.multiply [[VAR_109_]], [[VAR_110_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_112_:%.+]] = stablehlo.dynamic_reshape [[VAR_111_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_113_:%.+]] = stablehlo.reshape [[VAR_7_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           [[VAR_114_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_113_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 512] : (tensor<2x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_115_:%.+]] = stablehlo.dynamic_reshape [[VAR_114_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_116_:%.+]] = stablehlo.broadcast_in_dim [[VAR_115_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_117_:%.+]] = stablehlo.broadcast_in_dim [[VAR_31_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_118_:%.+]] = stablehlo.dot [[VAR_116_]], [[VAR_117_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_119_:%.+]] = stablehlo.broadcast_in_dim [[VAR_111_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_120_:%.+]] = stablehlo.broadcast_in_dim [[VAR_32_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_121_:%.+]] = stablehlo.dot [[VAR_119_]], [[VAR_120_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_122_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_118_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_123_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_121_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_124_:%.+]] = stablehlo.add [[VAR_122_]], [[VAR_123_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_125_:%.+]] = stablehlo.dynamic_slice [[VAR_124_]], [[VAR_12_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_126_:%.+]] = stablehlo.dynamic_slice [[VAR_124_]], [[VAR_12_]], [[VAR_10_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_127_:%.+]] = stablehlo.dynamic_slice [[VAR_124_]], [[VAR_12_]], [[VAR_9_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_128_:%.+]] = stablehlo.dynamic_slice [[VAR_124_]], [[VAR_12_]], [[VAR_8_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_129_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_125_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_130_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_39_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_131_:%.+]] = stablehlo.add [[VAR_129_]], [[VAR_130_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_132_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_131_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_133_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_43_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_134_:%.+]] = stablehlo.add [[VAR_132_]], [[VAR_133_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_135_:%.+]] = stablehlo.logistic [[VAR_134_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_136_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_127_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_137_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_41_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_138_:%.+]] = stablehlo.add [[VAR_136_]], [[VAR_137_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_139_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_138_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_140_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_45_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_141_:%.+]] = stablehlo.add [[VAR_139_]], [[VAR_140_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_142_:%.+]] = stablehlo.logistic [[VAR_141_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_143_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_128_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_144_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_42_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_145_:%.+]] = stablehlo.add [[VAR_143_]], [[VAR_144_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_146_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_145_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_147_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_46_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_148_:%.+]] = stablehlo.add [[VAR_146_]], [[VAR_147_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_149_:%.+]] = stablehlo.tanh [[VAR_148_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_150_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_142_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_151_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_100_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_152_:%.+]] = stablehlo.multiply [[VAR_150_]], [[VAR_151_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_153_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_135_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_154_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_149_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_155_:%.+]] = stablehlo.multiply [[VAR_153_]], [[VAR_154_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_156_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_152_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_157_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_155_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_158_:%.+]] = stablehlo.add [[VAR_156_]], [[VAR_157_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_159_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_126_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_160_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_40_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_161_:%.+]] = stablehlo.add [[VAR_159_]], [[VAR_160_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_162_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_161_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_163_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_44_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_164_:%.+]] = stablehlo.add [[VAR_162_]], [[VAR_163_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_165_:%.+]] = stablehlo.logistic [[VAR_164_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_166_:%.+]] = stablehlo.tanh [[VAR_158_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_167_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_165_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_168_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_166_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_169_:%.+]] = stablehlo.multiply [[VAR_167_]], [[VAR_168_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_170_:%.+]] = stablehlo.dynamic_reshape [[VAR_169_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_171_:%.+]] = stablehlo.reshape [[VAR_7_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           [[VAR_172_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_171_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 512] : (tensor<2x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_173_:%.+]] = stablehlo.dynamic_reshape [[VAR_172_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_174_:%.+]] = stablehlo.broadcast_in_dim [[VAR_173_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_175_:%.+]] = stablehlo.broadcast_in_dim [[VAR_33_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_176_:%.+]] = stablehlo.dot [[VAR_174_]], [[VAR_175_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_177_:%.+]] = stablehlo.broadcast_in_dim [[VAR_20_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_178_:%.+]] = stablehlo.broadcast_in_dim [[VAR_34_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_179_:%.+]] = stablehlo.dot [[VAR_177_]], [[VAR_178_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_180_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_176_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_181_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_179_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_182_:%.+]] = stablehlo.add [[VAR_180_]], [[VAR_181_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_183_:%.+]] = stablehlo.dynamic_slice [[VAR_182_]], [[VAR_12_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_184_:%.+]] = stablehlo.dynamic_slice [[VAR_182_]], [[VAR_12_]], [[VAR_10_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_185_:%.+]] = stablehlo.dynamic_slice [[VAR_182_]], [[VAR_12_]], [[VAR_9_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_186_:%.+]] = stablehlo.dynamic_slice [[VAR_182_]], [[VAR_12_]], [[VAR_8_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_187_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_183_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_188_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_47_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_189_:%.+]] = stablehlo.add [[VAR_187_]], [[VAR_188_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_190_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_189_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_191_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_51_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_192_:%.+]] = stablehlo.add [[VAR_190_]], [[VAR_191_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_193_:%.+]] = stablehlo.logistic [[VAR_192_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_194_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_185_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_195_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_49_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_196_:%.+]] = stablehlo.add [[VAR_194_]], [[VAR_195_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_197_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_196_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_198_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_53_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_199_:%.+]] = stablehlo.add [[VAR_197_]], [[VAR_198_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_200_:%.+]] = stablehlo.logistic [[VAR_199_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_201_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_186_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_202_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_50_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_203_:%.+]] = stablehlo.add [[VAR_201_]], [[VAR_202_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_204_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_203_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_205_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_54_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_206_:%.+]] = stablehlo.add [[VAR_204_]], [[VAR_205_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_207_:%.+]] = stablehlo.tanh [[VAR_206_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_208_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_200_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_209_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_22_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_210_:%.+]] = stablehlo.multiply [[VAR_208_]], [[VAR_209_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_211_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_193_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_212_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_207_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_213_:%.+]] = stablehlo.multiply [[VAR_211_]], [[VAR_212_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_214_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_210_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_215_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_213_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_216_:%.+]] = stablehlo.add [[VAR_214_]], [[VAR_215_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_217_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_184_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_218_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_48_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_219_:%.+]] = stablehlo.add [[VAR_217_]], [[VAR_218_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_220_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_219_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_221_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_52_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_222_:%.+]] = stablehlo.add [[VAR_220_]], [[VAR_221_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_223_:%.+]] = stablehlo.logistic [[VAR_222_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_224_:%.+]] = stablehlo.tanh [[VAR_216_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_225_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_223_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_226_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_224_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_227_:%.+]] = stablehlo.multiply [[VAR_225_]], [[VAR_226_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_228_:%.+]] = stablehlo.dynamic_reshape [[VAR_227_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_229_:%.+]] = stablehlo.reshape [[VAR_13_]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           [[VAR_230_:%.+]] = stablehlo.dynamic_slice [[PARAM_0_]], [[VAR_229_]], [[VAR_12_]], [[VAR_12_]], sizes = [1, 16, 512] : (tensor<2x16x512xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x16x512xf32>
// CHECK:           [[VAR_231_:%.+]] = stablehlo.dynamic_reshape [[VAR_230_]], [[VAR_3_]] : (tensor<1x16x512xf32>, tensor<2xindex>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_232_:%.+]] = stablehlo.broadcast_in_dim [[VAR_231_]], dims = [0, 1] : (tensor<16x512xf32>) -> tensor<16x512xf32>
// CHECK-DAG:       [[VAR_233_:%.+]] = stablehlo.broadcast_in_dim [[VAR_33_]], dims = [0, 1] : (tensor<512x1024xf32>) -> tensor<512x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_234_:%.+]] = stablehlo.dot [[VAR_232_]], [[VAR_233_]] : (tensor<16x512xf32>, tensor<512x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_235_:%.+]] = stablehlo.broadcast_in_dim [[VAR_227_]], dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_236_:%.+]] = stablehlo.broadcast_in_dim [[VAR_34_]], dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_237_:%.+]] = stablehlo.dot [[VAR_235_]], [[VAR_236_]] : (tensor<16x256xf32>, tensor<256x1024xf32>) -> tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_238_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_234_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_239_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_237_]], [[VAR_2_]], dims = [0, 1] : (tensor<16x1024xf32>, tensor<2xindex>) -> tensor<16x1024xf32>
// CHECK:           [[VAR_240_:%.+]] = stablehlo.add [[VAR_238_]], [[VAR_239_]] : tensor<16x1024xf32>
// CHECK-DAG:       [[VAR_241_:%.+]] = stablehlo.dynamic_slice [[VAR_240_]], [[VAR_12_]], [[VAR_12_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_242_:%.+]] = stablehlo.dynamic_slice [[VAR_240_]], [[VAR_12_]], [[VAR_10_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_243_:%.+]] = stablehlo.dynamic_slice [[VAR_240_]], [[VAR_12_]], [[VAR_9_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_244_:%.+]] = stablehlo.dynamic_slice [[VAR_240_]], [[VAR_12_]], [[VAR_8_]], sizes = [16, 256] : (tensor<16x1024xf32>, tensor<i64>, tensor<i64>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_245_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_241_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_246_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_47_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_247_:%.+]] = stablehlo.add [[VAR_245_]], [[VAR_246_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_248_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_247_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_249_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_51_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_250_:%.+]] = stablehlo.add [[VAR_248_]], [[VAR_249_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_251_:%.+]] = stablehlo.logistic [[VAR_250_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_252_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_243_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_253_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_49_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_254_:%.+]] = stablehlo.add [[VAR_252_]], [[VAR_253_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_255_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_254_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_256_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_53_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_257_:%.+]] = stablehlo.add [[VAR_255_]], [[VAR_256_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_258_:%.+]] = stablehlo.logistic [[VAR_257_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_259_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_244_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_260_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_50_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_261_:%.+]] = stablehlo.add [[VAR_259_]], [[VAR_260_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_262_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_261_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_263_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_54_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_264_:%.+]] = stablehlo.add [[VAR_262_]], [[VAR_263_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_265_:%.+]] = stablehlo.tanh [[VAR_264_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_266_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_258_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_267_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_216_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_268_:%.+]] = stablehlo.multiply [[VAR_266_]], [[VAR_267_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_269_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_251_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_270_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_265_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_271_:%.+]] = stablehlo.multiply [[VAR_269_]], [[VAR_270_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_272_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_268_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_273_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_271_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_274_:%.+]] = stablehlo.add [[VAR_272_]], [[VAR_273_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_275_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_242_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_276_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_48_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_277_:%.+]] = stablehlo.add [[VAR_275_]], [[VAR_276_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_278_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_277_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_279_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_52_]], [[VAR_0_]], dims = [1] : (tensor<256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_280_:%.+]] = stablehlo.add [[VAR_278_]], [[VAR_279_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_281_:%.+]] = stablehlo.logistic [[VAR_280_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_282_:%.+]] = stablehlo.tanh [[VAR_274_]] : tensor<16x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_283_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_281_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK-DAG:       [[VAR_284_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_282_]], [[VAR_0_]], dims = [0, 1] : (tensor<16x256xf32>, tensor<2xindex>) -> tensor<16x256xf32>
// CHECK:           [[VAR_285_:%.+]] = stablehlo.multiply [[VAR_283_]], [[VAR_284_]] : tensor<16x256xf32>
// CHECK-DAG:       [[VAR_286_:%.+]] = stablehlo.dynamic_reshape [[VAR_285_]], [[VAR_1_]] : (tensor<16x256xf32>, tensor<4xindex>) -> tensor<1x1x16x256xf32>
// CHECK-DAG:       [[VAR_287_:%.+]] = stablehlo.concatenate [[VAR_112_]], [[VAR_170_]], dim = 0 : (tensor<1x1x16x256xf32>, tensor<1x1x16x256xf32>) -> tensor<2x1x16x256xf32>
// CHECK:           [[VAR_288_:%.+]] = stablehlo.concatenate [[VAR_286_]], [[VAR_228_]], dim = 0 : (tensor<1x1x16x256xf32>, tensor<1x1x16x256xf32>) -> tensor<2x1x16x256xf32>
// CHECK:           [[VAR_289_:%.+]] = stablehlo.concatenate [[VAR_287_]], [[VAR_288_]], dim = 1 : (tensor<2x1x16x256xf32>, tensor<2x1x16x256xf32>) -> tensor<2x2x16x256xf32>
// CHECK:           return [[VAR_289_]] : tensor<2x2x16x256xf32>
// CHECK:         }
}
