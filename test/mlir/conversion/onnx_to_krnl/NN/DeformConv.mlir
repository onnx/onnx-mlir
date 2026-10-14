// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl=enable-parallel %s -split-input-file | FileCheck %s --check-prefix=PAR

// Smallest useful case: single channel, no bias, no mask.
func.func @test_deform_conv_basic(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x1x2x2xf32>, %arg2: tensor<1x8x2x2xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.DeformConv"(%arg0, %arg1, %arg2, %cst, %cst) {kernel_shape = [2, 2]} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>, tensor<1x8x2x2xf32>, none, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py


// CHECK-LABEL:  func.func @test_deform_conv_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x3x3xf32>, [[PARAM_1_:%.+]]: memref<1x1x2x2xf32>, [[PARAM_2_:%.+]]: memref<1x8x2x2xf32>) -> memref<1x1x2x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_9_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_10_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x2xf32>
// CHECK-DAG:       [[CST_1_11_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_12_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_13_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[CST_1_13_]], [[CST_2_]] : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_2_1_]] : index
// CHECK-DAG:       [[CST_3_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.muli [[VAR_4_]]#1, [[CST_1_1_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[VAR_4_]]#2 : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.muli [[VAR_4_]]#1, [[CST_1_12_]] : index
// CHECK-DAG:         [[CST_1_14_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.floordivsi [[VAR_4_]]#1, [[CST_1_14_]] : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_12_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[VAR_arg8_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[VAR_arg9_:%.+]] = 0 to 2) iter_args([[VAR_arg10_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_13_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:             [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_14_:%.+]] = arith.muli [[CST_0_8_]], [[CST_2_]] : index
// CHECK:                 [[VAR_15_:%.+]] = arith.addi [[VAR_14_]], [[VAR_13_]]#0 : index
// CHECK:                 [[VAR_16_:%.+]] = arith.muli [[VAR_15_]], [[CST_2_1_]] : index
// CHECK-DAG:             [[VAR_17_:%.+]] = arith.addi [[VAR_16_]], [[VAR_13_]]#1 : index
// CHECK-DAG:             [[VAR_18_:%.+]] = arith.muli [[VAR_8_]], [[VAR_2_]] : index
// CHECK:                 [[VAR_19_:%.+]] = arith.addi [[VAR_18_]], [[VAR_17_]] : index
// CHECK-DAG:             [[VAR_20_:%.+]] = arith.muli [[VAR_19_]], [[CST_2_6_]] : index
// CHECK-DAG:             [[CST_1_15_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.muli [[VAR_10_]]#0, [[CST_1_15_]] : index
// CHECK-DAG:             [[CST_1_16_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_22_:%.+]] = arith.muli [[VAR_13_]]#0, [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.addi [[VAR_21_]], [[VAR_22_]] : index
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.subi [[VAR_23_]], [[CST_0_9_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_25_:%.+]] = arith.addi [[VAR_20_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_]]#0, [[VAR_25_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<1x8x2x2xf32>
// CHECK-DAG:             [[VAR_27_:%.+]] = arith.index_cast [[VAR_24_]] : index to i64
// CHECK:                 [[VAR_28_:%.+]] = arith.sitofp [[VAR_27_]] : i64 to f32
// CHECK-DAG:             [[VAR_29_:%.+]] = arith.addf [[VAR_28_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_1_17_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.muli [[VAR_10_]]#1, [[CST_1_17_]] : index
// CHECK-DAG:             [[CST_1_18_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_31_:%.+]] = arith.muli [[VAR_13_]]#1, [[CST_1_18_]] : index
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.addi [[VAR_30_]], [[VAR_31_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.subi [[VAR_32_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[CST_1_19_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_34_:%.+]] = arith.addi [[VAR_20_]], [[CST_1_19_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_]]#0, [[VAR_34_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<1x8x2x2xf32>
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.index_cast [[VAR_33_]] : index to i64
// CHECK:                 [[VAR_37_:%.+]] = arith.sitofp [[VAR_36_]] : i64 to f32
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.addf [[VAR_37_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_20_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[CST_minus_2_dot_000000_:%.+]] = arith.constant -2.000000e+00 : f32
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.index_cast [[CST_3_2_]] : index to i64
// CHECK:                 [[VAR_40_:%.+]] = arith.sitofp [[VAR_39_]] : i64 to f32
// CHECK-DAG:             [[VAR_41_:%.+]] = arith.addf [[VAR_40_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.maxnumf [[VAR_29_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_43_:%.+]] = arith.minnumf [[VAR_42_]], [[VAR_41_]] : f32
// CHECK:                 [[VAR_44_:%.+]] = math.floor [[VAR_43_]] : f32
// CHECK:                 [[VAR_45_:%.+]] = arith.fptosi [[VAR_44_]] : f32 to i64
// CHECK-DAG:             [[VAR_46_:%.+]] = arith.index_cast [[VAR_45_]] : i64 to index
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.subf [[VAR_43_]], [[VAR_44_]] : f32
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.index_cast [[CST_3_3_]] : index to i64
// CHECK:                 [[VAR_49_:%.+]] = arith.sitofp [[VAR_48_]] : i64 to f32
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.maxnumf [[VAR_38_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_52_:%.+]] = arith.minnumf [[VAR_51_]], [[VAR_50_]] : f32
// CHECK:                 [[VAR_53_:%.+]] = math.floor [[VAR_52_]] : f32
// CHECK:                 [[VAR_54_:%.+]] = arith.fptosi [[VAR_53_]] : f32 to i64
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.index_cast [[VAR_54_]] : i64 to index
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.subf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:             [[VAR_57_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_57_]] : f32
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_61_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_59_]] : index
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.andi [[VAR_60_]], [[VAR_61_]] : i1
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.maxsi [[VAR_59_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_65_:%.+]] = arith.minsi [[VAR_64_]], [[VAR_63_]] : index
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_56_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_67_:%.+]] = arith.mulf [[VAR_58_]], [[VAR_66_]] : f32
// CHECK-DAG:             [[VAR_68_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_69_:%.+]] = arith.cmpi sge, [[VAR_55_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_70_:%.+]] = arith.cmpi sle, [[VAR_55_]], [[VAR_68_]] : index
// CHECK:                 [[VAR_71_:%.+]] = arith.andi [[VAR_69_]], [[VAR_70_]] : i1
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.andi [[VAR_62_]], [[VAR_71_]] : i1
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.maxsi [[VAR_68_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.maxsi [[VAR_55_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.minsi [[VAR_74_]], [[VAR_73_]] : index
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.cmpi sge, [[VAR_77_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.cmpi sle, [[VAR_77_]], [[VAR_78_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.andi [[VAR_79_]], [[VAR_80_]] : i1
// CHECK-DAG:             [[VAR_82_:%.+]] = arith.maxsi [[VAR_78_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.maxsi [[VAR_77_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.minsi [[VAR_83_]], [[VAR_82_]] : index
// CHECK-DAG:             [[VAR_85_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_56_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_86_:%.+]] = arith.mulf [[VAR_76_]], [[VAR_85_]] : f32
// CHECK-DAG:             [[VAR_87_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_88_:%.+]] = arith.cmpi sge, [[VAR_55_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_89_:%.+]] = arith.cmpi sle, [[VAR_55_]], [[VAR_87_]] : index
// CHECK:                 [[VAR_90_:%.+]] = arith.andi [[VAR_88_]], [[VAR_89_]] : i1
// CHECK-DAG:             [[VAR_91_:%.+]] = arith.andi [[VAR_81_]], [[VAR_90_]] : i1
// CHECK-DAG:             [[VAR_92_:%.+]] = arith.maxsi [[VAR_87_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_93_:%.+]] = arith.maxsi [[VAR_55_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_94_:%.+]] = arith.minsi [[VAR_93_]], [[VAR_92_]] : index
// CHECK-DAG:             [[VAR_95_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_95_]] : f32
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_98_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_99_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_97_]] : index
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.andi [[VAR_98_]], [[VAR_99_]] : i1
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.maxsi [[VAR_97_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_103_:%.+]] = arith.minsi [[VAR_102_]], [[VAR_101_]] : index
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.mulf [[VAR_96_]], [[VAR_56_]] : f32
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.addi [[VAR_55_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_107_:%.+]] = arith.cmpi sge, [[VAR_105_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_108_:%.+]] = arith.cmpi sle, [[VAR_105_]], [[VAR_106_]] : index
// CHECK:                 [[VAR_109_:%.+]] = arith.andi [[VAR_107_]], [[VAR_108_]] : i1
// CHECK-DAG:             [[VAR_110_:%.+]] = arith.andi [[VAR_100_]], [[VAR_109_]] : i1
// CHECK-DAG:             [[VAR_111_:%.+]] = arith.maxsi [[VAR_106_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_112_:%.+]] = arith.maxsi [[VAR_105_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_113_:%.+]] = arith.minsi [[VAR_112_]], [[VAR_111_]] : index
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.cmpi sge, [[VAR_115_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.cmpi sle, [[VAR_115_]], [[VAR_116_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.andi [[VAR_117_]], [[VAR_118_]] : i1
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.maxsi [[VAR_116_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.maxsi [[VAR_115_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.minsi [[VAR_121_]], [[VAR_120_]] : index
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.mulf [[VAR_114_]], [[VAR_56_]] : f32
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.addi [[VAR_55_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[VAR_125_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.cmpi sge, [[VAR_124_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_127_:%.+]] = arith.cmpi sle, [[VAR_124_]], [[VAR_125_]] : index
// CHECK:                 [[VAR_128_:%.+]] = arith.andi [[VAR_126_]], [[VAR_127_]] : i1
// CHECK-DAG:             [[VAR_129_:%.+]] = arith.andi [[VAR_119_]], [[VAR_128_]] : i1
// CHECK-DAG:             [[VAR_130_:%.+]] = arith.maxsi [[VAR_125_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_131_:%.+]] = arith.maxsi [[VAR_124_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_132_:%.+]] = arith.minsi [[VAR_131_]], [[VAR_130_]] : index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:             [[CST_0_13_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_134_:%.+]] = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg11_:%.+]] = 0 to 1) iter_args([[VAR_arg12_:%.+]] = [[VAR_arg10_]]) -> (f32){
// CHECK-DAG:               [[VAR_135_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_136_:%.+]] = arith.addi [[VAR_7_]], [[VAR_135_]] : index
// CHECK-DAG:               [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_136_]], [[VAR_65_]], [[VAR_75_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                   [[VAR_138_:%.+]] = arith.select [[VAR_72_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_139_:%.+]] = arith.mulf [[VAR_67_]], [[VAR_138_]] : f32
// CHECK-DAG:               [[VAR_140_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_139_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_136_]], [[VAR_84_]], [[VAR_94_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                   [[VAR_142_:%.+]] = arith.select [[VAR_91_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_143_:%.+]] = arith.mulf [[VAR_86_]], [[VAR_142_]] : f32
// CHECK-DAG:               [[VAR_144_:%.+]] = arith.addf [[VAR_140_]], [[VAR_143_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_136_]], [[VAR_103_]], [[VAR_113_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                   [[VAR_146_:%.+]] = arith.select [[VAR_110_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_147_:%.+]] = arith.mulf [[VAR_104_]], [[VAR_146_]] : f32
// CHECK-DAG:               [[VAR_148_:%.+]] = arith.addf [[VAR_144_]], [[VAR_147_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_136_]], [[VAR_122_]], [[VAR_132_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                   [[VAR_150_:%.+]] = arith.select [[VAR_129_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_151_:%.+]] = arith.mulf [[VAR_123_]], [[VAR_150_]] : f32
// CHECK-DAG:               [[VAR_152_:%.+]] = arith.addf [[VAR_148_]], [[VAR_151_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_6_]], [[VAR_135_]], [[VAR_13_]]#0, [[VAR_13_]]#1] : memref<1x1x2x2xf32>
// CHECK:                   [[VAR_154_:%.+]] = arith.mulf [[VAR_152_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:                   [[VAR_155_:%.+]] = arith.addf [[VAR_arg12_]], [[VAR_154_]] : f32
// CHECK:                   krnl.yield [[VAR_155_]] : f32
// CHECK:                 }
// CHECK:                 krnl.yield [[VAR_134_]] : f32
// CHECK:               }
// CHECK:               krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_6_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<1x1x2x2xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x2xf32>
// CHECK:         }


}

// -----

// Groups, offset groups, bias and mask all enabled.
func.func @test_deform_conv_groups_bias_mask(%arg0: tensor<1x4x8x8xf32>, %arg1: tensor<6x2x3x3xf32>, %arg2: tensor<1x36x3x3xf32>, %arg3: tensor<6xf32>, %arg4: tensor<1x18x3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.DeformConv"(%arg0, %arg1, %arg2, %arg3, %arg4) {group = 2 : si64, offset_group = 2 : si64, kernel_shape = [3, 3], strides = [2, 2]} : (tensor<1x4x8x8xf32>, tensor<6x2x3x3xf32>, tensor<1x36x3x3xf32>, tensor<6xf32>, tensor<1x18x3x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py


// CHECK-LABEL:  func.func @test_deform_conv_groups_bias_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4x8x8xf32>, [[PARAM_1_:%.+]]: memref<6x2x3x3xf32>, [[PARAM_2_:%.+]]: memref<1x36x3x3xf32>, [[PARAM_3_:%.+]]: memref<6xf32>, [[PARAM_4_:%.+]]: memref<1x18x3x3xf32>) -> memref<1x6x3x3xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_8_1_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_4_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_1_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_5_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1x6x3x3xf32>
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_6_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_7_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_8_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[CST_1_4_]], [[CST_3_]] : index
// CHECK-DAG:       [[CST_8_2_:%.+]] = arith.constant 8 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_3_1_]] : index
// CHECK-DAG:       [[CST_8_3_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3){
// CHECK:             [[VAR_3_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_3_]]#1, [[CST_3_6_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[VAR_3_]]#2 : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.muli [[VAR_3_]]#1, [[CST_2_7_]] : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.floordivsi [[VAR_3_]]#1, [[CST_1_5_]] : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_11_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[VAR_arg10_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[VAR_arg11_:%.+]] = 0 to 3) iter_args([[VAR_arg12_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_14_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:             [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_15_:%.+]] = arith.muli [[CST_0_8_]], [[CST_3_]] : index
// CHECK:                 [[VAR_16_:%.+]] = arith.addi [[VAR_15_]], [[VAR_14_]]#0 : index
// CHECK:                 [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[CST_3_1_]] : index
// CHECK-DAG:             [[VAR_18_:%.+]] = arith.addi [[VAR_17_]], [[VAR_14_]]#1 : index
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.muli [[VAR_7_]], [[VAR_1_]] : index
// CHECK:                 [[VAR_20_:%.+]] = arith.addi [[VAR_19_]], [[VAR_18_]] : index
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.muli [[VAR_20_]], [[CST_2_8_]] : index
// CHECK-DAG:             [[CST_2_9_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.muli [[VAR_9_]]#0, [[CST_2_9_]] : index
// CHECK-DAG:             [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_23_:%.+]] = arith.muli [[VAR_14_]]#0, [[CST_1_6_]] : index
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.addi [[VAR_22_]], [[VAR_23_]] : index
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.subi [[VAR_24_]], [[CST_0_9_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_26_:%.+]] = arith.addi [[VAR_21_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]]#0, [[VAR_26_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x36x3x3xf32>
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.index_cast [[VAR_25_]] : index to i64
// CHECK:                 [[VAR_29_:%.+]] = arith.sitofp [[VAR_28_]] : i64 to f32
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.addf [[VAR_29_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_2_10_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.muli [[VAR_9_]]#1, [[CST_2_10_]] : index
// CHECK-DAG:             [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_32_:%.+]] = arith.muli [[VAR_14_]]#1, [[CST_1_7_]] : index
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.addi [[VAR_31_]], [[VAR_32_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.subi [[VAR_33_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_35_:%.+]] = arith.addi [[VAR_21_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]]#0, [[VAR_35_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x36x3x3xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.index_cast [[VAR_34_]] : index to i64
// CHECK:                 [[VAR_38_:%.+]] = arith.sitofp [[VAR_37_]] : i64 to f32
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_9_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[CST_minus_2_dot_000000_:%.+]] = arith.constant -2.000000e+00 : f32
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.index_cast [[CST_8_2_]] : index to i64
// CHECK:                 [[VAR_41_:%.+]] = arith.sitofp [[VAR_40_]] : i64 to f32
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.maxnumf [[VAR_30_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_44_:%.+]] = arith.minnumf [[VAR_43_]], [[VAR_42_]] : f32
// CHECK:                 [[VAR_45_:%.+]] = math.floor [[VAR_44_]] : f32
// CHECK:                 [[VAR_46_:%.+]] = arith.fptosi [[VAR_45_]] : f32 to i64
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.index_cast [[VAR_46_]] : i64 to index
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.subf [[VAR_44_]], [[VAR_45_]] : f32
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.index_cast [[CST_8_3_]] : index to i64
// CHECK:                 [[VAR_50_:%.+]] = arith.sitofp [[VAR_49_]] : i64 to f32
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.addf [[VAR_50_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.maxnumf [[VAR_39_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_53_:%.+]] = arith.minnumf [[VAR_52_]], [[VAR_51_]] : f32
// CHECK:                 [[VAR_54_:%.+]] = math.floor [[VAR_53_]] : f32
// CHECK:                 [[VAR_55_:%.+]] = arith.fptosi [[VAR_54_]] : f32 to i64
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.index_cast [[VAR_55_]] : i64 to index
// CHECK-DAG:             [[VAR_57_:%.+]] = arith.subf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_58_]] : f32
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_61_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_62_:%.+]] = arith.cmpi sle, [[VAR_47_]], [[VAR_60_]] : index
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.andi [[VAR_61_]], [[VAR_62_]] : i1
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.maxsi [[VAR_60_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_65_:%.+]] = arith.maxsi [[VAR_47_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.minsi [[VAR_65_]], [[VAR_64_]] : index
// CHECK-DAG:             [[VAR_67_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_57_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_68_:%.+]] = arith.mulf [[VAR_59_]], [[VAR_67_]] : f32
// CHECK-DAG:             [[VAR_69_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_70_:%.+]] = arith.cmpi sge, [[VAR_56_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_71_:%.+]] = arith.cmpi sle, [[VAR_56_]], [[VAR_69_]] : index
// CHECK:                 [[VAR_72_:%.+]] = arith.andi [[VAR_70_]], [[VAR_71_]] : i1
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.andi [[VAR_63_]], [[VAR_72_]] : i1
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.maxsi [[VAR_69_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.maxsi [[VAR_56_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.minsi [[VAR_75_]], [[VAR_74_]] : index
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.addi [[VAR_47_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_9_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.cmpi sge, [[VAR_78_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.cmpi sle, [[VAR_78_]], [[VAR_79_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_82_:%.+]] = arith.andi [[VAR_80_]], [[VAR_81_]] : i1
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.maxsi [[VAR_79_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.maxsi [[VAR_78_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_85_:%.+]] = arith.minsi [[VAR_84_]], [[VAR_83_]] : index
// CHECK-DAG:             [[VAR_86_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_57_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_87_:%.+]] = arith.mulf [[VAR_77_]], [[VAR_86_]] : f32
// CHECK-DAG:             [[VAR_88_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_89_:%.+]] = arith.cmpi sge, [[VAR_56_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_90_:%.+]] = arith.cmpi sle, [[VAR_56_]], [[VAR_88_]] : index
// CHECK:                 [[VAR_91_:%.+]] = arith.andi [[VAR_89_]], [[VAR_90_]] : i1
// CHECK-DAG:             [[VAR_92_:%.+]] = arith.andi [[VAR_82_]], [[VAR_91_]] : i1
// CHECK-DAG:             [[VAR_93_:%.+]] = arith.maxsi [[VAR_88_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_94_:%.+]] = arith.maxsi [[VAR_56_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_95_:%.+]] = arith.minsi [[VAR_94_]], [[VAR_93_]] : index
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_96_]] : f32
// CHECK-DAG:             [[VAR_98_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_99_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_100_:%.+]] = arith.cmpi sle, [[VAR_47_]], [[VAR_98_]] : index
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.andi [[VAR_99_]], [[VAR_100_]] : i1
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.maxsi [[VAR_98_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_103_:%.+]] = arith.maxsi [[VAR_47_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.minsi [[VAR_103_]], [[VAR_102_]] : index
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.mulf [[VAR_97_]], [[VAR_57_]] : f32
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.addi [[VAR_56_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_107_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_9_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_108_:%.+]] = arith.cmpi sge, [[VAR_106_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_109_:%.+]] = arith.cmpi sle, [[VAR_106_]], [[VAR_107_]] : index
// CHECK:                 [[VAR_110_:%.+]] = arith.andi [[VAR_108_]], [[VAR_109_]] : i1
// CHECK-DAG:             [[VAR_111_:%.+]] = arith.andi [[VAR_101_]], [[VAR_110_]] : i1
// CHECK-DAG:             [[VAR_112_:%.+]] = arith.maxsi [[VAR_107_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_113_:%.+]] = arith.maxsi [[VAR_106_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.minsi [[VAR_113_]], [[VAR_112_]] : index
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.addi [[VAR_47_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_9_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.cmpi sge, [[VAR_116_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.cmpi sle, [[VAR_116_]], [[VAR_117_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.andi [[VAR_118_]], [[VAR_119_]] : i1
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.maxsi [[VAR_117_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.maxsi [[VAR_116_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.minsi [[VAR_122_]], [[VAR_121_]] : index
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.mulf [[VAR_115_]], [[VAR_57_]] : f32
// CHECK-DAG:             [[VAR_125_:%.+]] = arith.addi [[VAR_56_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_9_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_127_:%.+]] = arith.cmpi sge, [[VAR_125_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_128_:%.+]] = arith.cmpi sle, [[VAR_125_]], [[VAR_126_]] : index
// CHECK:                 [[VAR_129_:%.+]] = arith.andi [[VAR_127_]], [[VAR_128_]] : i1
// CHECK-DAG:             [[VAR_130_:%.+]] = arith.andi [[VAR_120_]], [[VAR_129_]] : i1
// CHECK-DAG:             [[VAR_131_:%.+]] = arith.maxsi [[VAR_126_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_132_:%.+]] = arith.maxsi [[VAR_125_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_133_:%.+]] = arith.minsi [[VAR_132_]], [[VAR_131_]] : index
// CHECK-DAG:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_3_]]#0, [[VAR_20_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x18x3x3xf32>
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:             [[CST_0_13_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_136_:%.+]] = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg13_:%.+]] = 0 to 2) iter_args([[VAR_arg14_:%.+]] = [[VAR_arg12_]]) -> (f32){
// CHECK-DAG:               [[VAR_137_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_138_:%.+]] = arith.addi [[VAR_6_]], [[VAR_137_]] : index
// CHECK-DAG:               [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_138_]], [[VAR_66_]], [[VAR_76_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                   [[VAR_140_:%.+]] = arith.select [[VAR_73_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_141_:%.+]] = arith.mulf [[VAR_68_]], [[VAR_140_]] : f32
// CHECK-DAG:               [[VAR_142_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_141_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_138_]], [[VAR_85_]], [[VAR_95_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                   [[VAR_144_:%.+]] = arith.select [[VAR_92_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_145_:%.+]] = arith.mulf [[VAR_87_]], [[VAR_144_]] : f32
// CHECK-DAG:               [[VAR_146_:%.+]] = arith.addf [[VAR_142_]], [[VAR_145_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_138_]], [[VAR_104_]], [[VAR_114_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                   [[VAR_148_:%.+]] = arith.select [[VAR_111_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_149_:%.+]] = arith.mulf [[VAR_105_]], [[VAR_148_]] : f32
// CHECK-DAG:               [[VAR_150_:%.+]] = arith.addf [[VAR_146_]], [[VAR_149_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_138_]], [[VAR_123_]], [[VAR_133_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                   [[VAR_152_:%.+]] = arith.select [[VAR_130_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_153_:%.+]] = arith.mulf [[VAR_124_]], [[VAR_152_]] : f32
// CHECK-DAG:               [[VAR_154_:%.+]] = arith.addf [[VAR_150_]], [[VAR_153_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_5_]], [[VAR_137_]], [[VAR_14_]]#0, [[VAR_14_]]#1] : memref<6x2x3x3xf32>
// CHECK:                   [[VAR_156_:%.+]] = arith.mulf [[VAR_154_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:                   [[VAR_157_:%.+]] = arith.mulf [[VAR_156_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:                   [[VAR_158_:%.+]] = arith.addf [[VAR_arg14_]], [[VAR_157_]] : f32
// CHECK:                   krnl.yield [[VAR_158_]] : f32
// CHECK:                 }
// CHECK:                 krnl.yield [[VAR_136_]] : f32
// CHECK:               }
// CHECK:               [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_5_]]{{.}} : memref<6xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[VAR_11_]], [[LOAD_PARAM_3_MEM_]] : f32
// CHECK:               krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_5_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x6x3x3xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x6x3x3xf32>
// CHECK:         }


}

// -----

// Dynamic batch and spatial dimensions.
func.func @test_deform_conv_unknown_dims(%arg0: tensor<?x1x?x?xf32>, %arg1: tensor<5x1x2x2xf32>, %arg2: tensor<?x8x?x?xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.DeformConv"(%arg0, %arg1, %arg2, %cst, %cst) : (tensor<?x1x?x?xf32>, tensor<5x1x2x2xf32>, tensor<?x8x?x?xf32>, none, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 - 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1 - 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d1 - 1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1, d2) -> (d0 - 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2) -> (d1 - 1)>
// CHECK-LABEL:  func.func @test_deform_conv_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x?x?xf32>, [[PARAM_1_:%.+]]: memref<5x1x2x2xf32>, [[PARAM_2_:%.+]]: memref<?x8x?x?xf32>) -> memref<?x5x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_4_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_2_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_dim_7_]])
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_dim_7_]])
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_12_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_dim_7_]], [[VAR_dim_12_]])
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_3_]]([[VAR_dim_7_]], [[VAR_dim_12_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_2_]], [[VAR_4_]]) alignment = 16 : memref<?x5x?x?xf32>
// CHECK-DAG:       [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_9_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[CST_1_9_]], [[CST_2_]] : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_22_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_6_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_2_1_]] : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_24_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_1_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_8_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_9_:%.+]] = arith.muli [[VAR_8_]]#1, [[CST_5_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_8_]]#2 : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_8_]]#1, [[CST_1_8_]] : index
// CHECK-DAG:         [[CST_1_10_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.floordivsi [[VAR_8_]]#1, [[CST_1_10_]] : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]])){
// CHECK-DAG:           [[VAR_14_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_16_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[VAR_arg8_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[VAR_arg9_:%.+]] = 0 to 2) iter_args([[VAR_arg10_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_18_:%.+]] = arith.muli [[CST_0_9_]], [[CST_2_]] : index
// CHECK:                 [[VAR_19_:%.+]] = arith.addi [[VAR_18_]], [[VAR_17_]]#0 : index
// CHECK:                 [[VAR_20_:%.+]] = arith.muli [[VAR_19_]], [[CST_2_1_]] : index
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.addi [[VAR_20_]], [[VAR_17_]]#1 : index
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.muli [[VAR_12_]], [[VAR_6_]] : index
// CHECK:                 [[VAR_23_:%.+]] = arith.addi [[VAR_22_]], [[VAR_21_]] : index
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.muli [[VAR_23_]], [[CST_2_5_]] : index
// CHECK-DAG:             [[CST_1_11_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.muli [[VAR_14_]]#0, [[CST_1_11_]] : index
// CHECK-DAG:             [[CST_1_12_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_26_:%.+]] = arith.muli [[VAR_17_]]#0, [[CST_1_12_]] : index
// CHECK-DAG:             [[VAR_27_:%.+]] = arith.addi [[VAR_25_]], [[VAR_26_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.subi [[VAR_27_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_29_:%.+]] = arith.addi [[VAR_24_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_]]#0, [[VAR_29_]], [[VAR_14_]]#0, [[VAR_14_]]#1] : memref<?x8x?x?xf32>
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.index_cast [[VAR_28_]] : index to i64
// CHECK:                 [[VAR_32_:%.+]] = arith.sitofp [[VAR_31_]] : i64 to f32
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_1_13_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.muli [[VAR_14_]]#1, [[CST_1_13_]] : index
// CHECK-DAG:             [[CST_1_14_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_35_:%.+]] = arith.muli [[VAR_17_]]#1, [[CST_1_14_]] : index
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addi [[VAR_34_]], [[VAR_35_]] : index
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.subi [[VAR_36_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[CST_1_15_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_38_:%.+]] = arith.addi [[VAR_24_]], [[CST_1_15_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_]]#0, [[VAR_38_]], [[VAR_14_]]#0, [[VAR_14_]]#1] : memref<?x8x?x?xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.index_cast [[VAR_37_]] : index to i64
// CHECK:                 [[VAR_41_:%.+]] = arith.sitofp [[VAR_40_]] : i64 to f32
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_13_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_16_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[CST_minus_2_dot_000000_:%.+]] = arith.constant -2.000000e+00 : f32
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.index_cast [[VAR_dim_22_]] : index to i64
// CHECK:                 [[VAR_44_:%.+]] = arith.sitofp [[VAR_43_]] : i64 to f32
// CHECK-DAG:             [[VAR_45_:%.+]] = arith.addf [[VAR_44_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_46_:%.+]] = arith.maxnumf [[VAR_33_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_47_:%.+]] = arith.minnumf [[VAR_46_]], [[VAR_45_]] : f32
// CHECK:                 [[VAR_48_:%.+]] = math.floor [[VAR_47_]] : f32
// CHECK:                 [[VAR_49_:%.+]] = arith.fptosi [[VAR_48_]] : f32 to i64
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.index_cast [[VAR_49_]] : i64 to index
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.subf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.index_cast [[VAR_dim_24_]] : index to i64
// CHECK:                 [[VAR_53_:%.+]] = arith.sitofp [[VAR_52_]] : i64 to f32
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.addf [[VAR_53_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.maxnumf [[VAR_42_]], [[CST_minus_2_dot_000000_]] : f32
// CHECK:                 [[VAR_56_:%.+]] = arith.minnumf [[VAR_55_]], [[VAR_54_]] : f32
// CHECK:                 [[VAR_57_:%.+]] = math.floor [[VAR_56_]] : f32
// CHECK:                 [[VAR_58_:%.+]] = arith.fptosi [[VAR_57_]] : f32 to i64
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.index_cast [[VAR_58_]] : i64 to index
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.subf [[VAR_56_]], [[VAR_57_]] : f32
// CHECK-DAG:             [[VAR_61_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_61_]] : f32
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.subi [[VAR_dim_22_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.cmpi sge, [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_65_:%.+]] = arith.cmpi sle, [[VAR_50_]], [[VAR_63_]] : index
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.andi [[VAR_64_]], [[VAR_65_]] : i1
// CHECK-DAG:             [[VAR_67_:%.+]] = arith.maxsi [[VAR_63_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_68_:%.+]] = arith.maxsi [[VAR_50_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_69_:%.+]] = arith.minsi [[VAR_68_]], [[VAR_67_]] : index
// CHECK-DAG:             [[VAR_70_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_71_:%.+]] = arith.mulf [[VAR_62_]], [[VAR_70_]] : f32
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.cmpi sge, [[VAR_59_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_74_:%.+]] = arith.cmpi sle, [[VAR_59_]], [[VAR_72_]] : index
// CHECK:                 [[VAR_75_:%.+]] = arith.andi [[VAR_73_]], [[VAR_74_]] : i1
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.andi [[VAR_66_]], [[VAR_75_]] : i1
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.maxsi [[VAR_72_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.maxsi [[VAR_59_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.minsi [[VAR_78_]], [[VAR_77_]] : index
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.addi [[VAR_50_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_82_:%.+]] = arith.subi [[VAR_dim_22_]], [[CST_1_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.cmpi sge, [[VAR_81_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.cmpi sle, [[VAR_81_]], [[VAR_82_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_85_:%.+]] = arith.andi [[VAR_83_]], [[VAR_84_]] : i1
// CHECK-DAG:             [[VAR_86_:%.+]] = arith.maxsi [[VAR_82_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_87_:%.+]] = arith.maxsi [[VAR_81_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_88_:%.+]] = arith.minsi [[VAR_87_]], [[VAR_86_]] : index
// CHECK-DAG:             [[VAR_89_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_90_:%.+]] = arith.mulf [[VAR_80_]], [[VAR_89_]] : f32
// CHECK-DAG:             [[VAR_91_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_92_:%.+]] = arith.cmpi sge, [[VAR_59_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_93_:%.+]] = arith.cmpi sle, [[VAR_59_]], [[VAR_91_]] : index
// CHECK:                 [[VAR_94_:%.+]] = arith.andi [[VAR_92_]], [[VAR_93_]] : i1
// CHECK-DAG:             [[VAR_95_:%.+]] = arith.andi [[VAR_85_]], [[VAR_94_]] : i1
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.maxsi [[VAR_91_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.maxsi [[VAR_59_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_98_:%.+]] = arith.minsi [[VAR_97_]], [[VAR_96_]] : index
// CHECK-DAG:             [[VAR_99_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_99_]] : f32
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.subi [[VAR_dim_22_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.cmpi sge, [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_103_:%.+]] = arith.cmpi sle, [[VAR_50_]], [[VAR_101_]] : index
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.andi [[VAR_102_]], [[VAR_103_]] : i1
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.maxsi [[VAR_101_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.maxsi [[VAR_50_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_107_:%.+]] = arith.minsi [[VAR_106_]], [[VAR_105_]] : index
// CHECK-DAG:             [[VAR_108_:%.+]] = arith.mulf [[VAR_100_]], [[VAR_60_]] : f32
// CHECK-DAG:             [[VAR_109_:%.+]] = arith.addi [[VAR_59_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_110_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_111_:%.+]] = arith.cmpi sge, [[VAR_109_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_112_:%.+]] = arith.cmpi sle, [[VAR_109_]], [[VAR_110_]] : index
// CHECK:                 [[VAR_113_:%.+]] = arith.andi [[VAR_111_]], [[VAR_112_]] : i1
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.andi [[VAR_104_]], [[VAR_113_]] : i1
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.maxsi [[VAR_110_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.maxsi [[VAR_109_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.minsi [[VAR_116_]], [[VAR_115_]] : index
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.addi [[VAR_50_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.subi [[VAR_dim_22_]], [[CST_1_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.cmpi sge, [[VAR_119_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.cmpi sle, [[VAR_119_]], [[VAR_120_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.andi [[VAR_121_]], [[VAR_122_]] : i1
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.maxsi [[VAR_120_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_125_:%.+]] = arith.maxsi [[VAR_119_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.minsi [[VAR_125_]], [[VAR_124_]] : index
// CHECK-DAG:             [[VAR_127_:%.+]] = arith.mulf [[VAR_118_]], [[VAR_60_]] : f32
// CHECK-DAG:             [[VAR_128_:%.+]] = arith.addi [[VAR_59_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[VAR_129_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_130_:%.+]] = arith.cmpi sge, [[VAR_128_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_131_:%.+]] = arith.cmpi sle, [[VAR_128_]], [[VAR_129_]] : index
// CHECK:                 [[VAR_132_:%.+]] = arith.andi [[VAR_130_]], [[VAR_131_]] : i1
// CHECK-DAG:             [[VAR_133_:%.+]] = arith.andi [[VAR_123_]], [[VAR_132_]] : i1
// CHECK-DAG:             [[VAR_134_:%.+]] = arith.maxsi [[VAR_129_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_135_:%.+]] = arith.maxsi [[VAR_128_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_136_:%.+]] = arith.minsi [[VAR_135_]], [[VAR_134_]] : index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:             [[CST_0_14_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_138_:%.+]] = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg11_:%.+]] = 0 to 1) iter_args([[VAR_arg12_:%.+]] = [[VAR_arg10_]]) -> (f32){
// CHECK-DAG:               [[VAR_139_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_140_:%.+]] = arith.addi [[VAR_11_]], [[VAR_139_]] : index
// CHECK-DAG:               [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_140_]], [[VAR_69_]], [[VAR_79_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                   [[VAR_142_:%.+]] = arith.select [[VAR_76_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_143_:%.+]] = arith.mulf [[VAR_71_]], [[VAR_142_]] : f32
// CHECK-DAG:               [[VAR_144_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_143_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_140_]], [[VAR_88_]], [[VAR_98_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                   [[VAR_146_:%.+]] = arith.select [[VAR_95_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_147_:%.+]] = arith.mulf [[VAR_90_]], [[VAR_146_]] : f32
// CHECK-DAG:               [[VAR_148_:%.+]] = arith.addf [[VAR_144_]], [[VAR_147_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_140_]], [[VAR_107_]], [[VAR_117_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                   [[VAR_150_:%.+]] = arith.select [[VAR_114_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_151_:%.+]] = arith.mulf [[VAR_108_]], [[VAR_150_]] : f32
// CHECK-DAG:               [[VAR_152_:%.+]] = arith.addf [[VAR_148_]], [[VAR_151_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_140_]], [[VAR_126_]], [[VAR_136_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                   [[VAR_154_:%.+]] = arith.select [[VAR_133_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                   [[VAR_155_:%.+]] = arith.mulf [[VAR_127_]], [[VAR_154_]] : f32
// CHECK-DAG:               [[VAR_156_:%.+]] = arith.addf [[VAR_152_]], [[VAR_155_]] : f32
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_139_]], [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<5x1x2x2xf32>
// CHECK:                   [[VAR_158_:%.+]] = arith.mulf [[VAR_156_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:                   [[VAR_159_:%.+]] = arith.addf [[VAR_arg12_]], [[VAR_158_]] : f32
// CHECK:                   krnl.yield [[VAR_159_]] : f32
// CHECK:                 }
// CHECK:                 krnl.yield [[VAR_138_]] : f32
// CHECK:               }
// CHECK:               krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_8_]]#0, [[VAR_10_]], [[VAR_14_]]#0, [[VAR_14_]]#1] : memref<?x5x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x5x?x?xf32>
// CHECK:         }


}

// -----

// With a batch of one, the parallel level must be found on an inner outer-loop
// (here the 64 output channels) rather than on the trip-count-one batch loop.
func.func @test_deform_conv_parallel(%arg0: tensor<1x4x8x8xf32>, %arg1: tensor<64x4x3x3xf32>, %arg2: tensor<1x18x6x6xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.DeformConv"(%arg0, %arg1, %arg2, %cst, %cst) {kernel_shape = [3, 3]} : (tensor<1x4x8x8xf32>, tensor<64x4x3x3xf32>, tensor<1x18x6x6xf32>, none, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// PAR-LABEL:  func.func @test_deform_conv_parallel
// PAR:          [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// PAR:          krnl.parallel([[LOOP_0_]]#2) : !krnl.loop
// PAR:          krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 64){
}
