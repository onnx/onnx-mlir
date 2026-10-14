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
// CHECK-DAG:       [[CST_1_13_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_14_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_15_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[CST_1_15_]], [[CST_2_]] : index
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
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:           [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_11_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[VAR_arg8_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[VAR_arg9_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[VAR_arg10_:%.+]] = 0 to 2) iter_args([[VAR_arg11_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_12_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:                 [[VAR_13_:%.+]] = arith.addi [[VAR_7_]], [[VAR_12_]]#0 : index
// CHECK-DAG:             [[VAR_14_:%.+]] = arith.floordivsi [[VAR_13_]], [[CST_1_13_]] : index
// CHECK-DAG:             [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_15_:%.+]] = arith.muli [[CST_0_8_]], [[CST_2_]] : index
// CHECK:                 [[VAR_16_:%.+]] = arith.addi [[VAR_15_]], [[VAR_12_]]#1 : index
// CHECK:                 [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[CST_2_1_]] : index
// CHECK-DAG:             [[VAR_18_:%.+]] = arith.addi [[VAR_17_]], [[VAR_12_]]#2 : index
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.muli [[VAR_14_]], [[VAR_2_]] : index
// CHECK:                 [[VAR_20_:%.+]] = arith.addi [[VAR_19_]], [[VAR_18_]] : index
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.muli [[VAR_20_]], [[CST_2_6_]] : index
// CHECK-DAG:             [[CST_1_16_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.muli [[VAR_9_]]#0, [[CST_1_16_]] : index
// CHECK-DAG:             [[CST_1_17_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_23_:%.+]] = arith.muli [[VAR_12_]]#1, [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.addi [[VAR_22_]], [[VAR_23_]] : index
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.subi [[VAR_24_]], [[CST_0_9_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_26_:%.+]] = arith.addi [[VAR_21_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_]]#0, [[VAR_26_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x8x2x2xf32>
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.index_cast [[VAR_25_]] : index to i64
// CHECK:                 [[VAR_29_:%.+]] = arith.sitofp [[VAR_28_]] : i64 to f32
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.addf [[VAR_29_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_1_18_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.muli [[VAR_9_]]#1, [[CST_1_18_]] : index
// CHECK-DAG:             [[CST_1_19_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_32_:%.+]] = arith.muli [[VAR_12_]]#2, [[CST_1_19_]] : index
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.addi [[VAR_31_]], [[VAR_32_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.subi [[VAR_33_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[CST_1_20_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_35_:%.+]] = arith.addi [[VAR_21_]], [[CST_1_20_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_]]#0, [[VAR_35_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x8x2x2xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.index_cast [[VAR_34_]] : index to i64
// CHECK:                 [[VAR_38_:%.+]] = arith.sitofp [[VAR_37_]] : i64 to f32
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_21_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[VAR_40_:%.+]] = math.floor [[VAR_30_]] : f32
// CHECK:                 [[VAR_41_:%.+]] = arith.fptosi [[VAR_40_]] : f32 to i64
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.index_cast [[VAR_41_]] : i64 to index
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.subf [[VAR_30_]], [[VAR_40_]] : f32
// CHECK-DAG:             [[VAR_44_:%.+]] = math.floor [[VAR_39_]] : f32
// CHECK:                 [[VAR_45_:%.+]] = arith.fptosi [[VAR_44_]] : f32 to i64
// CHECK-DAG:             [[VAR_46_:%.+]] = arith.index_cast [[VAR_45_]] : i64 to index
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.subf [[VAR_39_]], [[VAR_44_]] : f32
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_43_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.cmpi sge, [[VAR_42_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_52_:%.+]] = arith.cmpi sle, [[VAR_42_]], [[VAR_50_]] : index
// CHECK-DAG:             [[VAR_53_:%.+]] = arith.andi [[VAR_51_]], [[VAR_52_]] : i1
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.maxsi [[VAR_42_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.minsi [[VAR_54_]], [[VAR_50_]] : index
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_57_:%.+]] = arith.mulf [[VAR_49_]], [[VAR_56_]] : f32
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_60_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_58_]] : index
// CHECK:                 [[VAR_61_:%.+]] = arith.andi [[VAR_59_]], [[VAR_60_]] : i1
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.andi [[VAR_53_]], [[VAR_61_]] : i1
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_64_:%.+]] = arith.minsi [[VAR_63_]], [[VAR_58_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_13_]], [[VAR_55_]], [[VAR_64_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                 [[VAR_66_:%.+]] = arith.select [[VAR_62_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_67_:%.+]] = arith.mulf [[VAR_57_]], [[VAR_66_]] : f32
// CHECK-DAG:             [[VAR_68_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_67_]] : f32
// CHECK-DAG:             [[VAR_69_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_43_]] : f32
// CHECK-DAG:             [[VAR_70_:%.+]] = arith.addi [[VAR_42_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_71_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_21_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.cmpi sge, [[VAR_70_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.cmpi sle, [[VAR_70_]], [[VAR_71_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.andi [[VAR_72_]], [[VAR_73_]] : i1
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.maxsi [[VAR_70_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.minsi [[VAR_75_]], [[VAR_71_]] : index
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.mulf [[VAR_69_]], [[VAR_77_]] : f32
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_81_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_79_]] : index
// CHECK:                 [[VAR_82_:%.+]] = arith.andi [[VAR_80_]], [[VAR_81_]] : i1
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.andi [[VAR_74_]], [[VAR_82_]] : i1
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_85_:%.+]] = arith.minsi [[VAR_84_]], [[VAR_79_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_13_]], [[VAR_76_]], [[VAR_85_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                 [[VAR_87_:%.+]] = arith.select [[VAR_83_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_88_:%.+]] = arith.mulf [[VAR_78_]], [[VAR_87_]] : f32
// CHECK-DAG:             [[VAR_89_:%.+]] = arith.addf [[VAR_68_]], [[VAR_88_]] : f32
// CHECK-DAG:             [[VAR_90_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_43_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_91_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_90_]] : f32
// CHECK-DAG:             [[VAR_92_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_93_:%.+]] = arith.cmpi sge, [[VAR_42_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_94_:%.+]] = arith.cmpi sle, [[VAR_42_]], [[VAR_92_]] : index
// CHECK-DAG:             [[VAR_95_:%.+]] = arith.andi [[VAR_93_]], [[VAR_94_]] : i1
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.maxsi [[VAR_42_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.minsi [[VAR_96_]], [[VAR_92_]] : index
// CHECK-DAG:             [[VAR_98_:%.+]] = arith.mulf [[VAR_91_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_99_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_21_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.cmpi sge, [[VAR_99_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.cmpi sle, [[VAR_99_]], [[VAR_100_]] : index
// CHECK:                 [[VAR_103_:%.+]] = arith.andi [[VAR_101_]], [[VAR_102_]] : i1
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.andi [[VAR_95_]], [[VAR_103_]] : i1
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.maxsi [[VAR_99_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_106_:%.+]] = arith.minsi [[VAR_105_]], [[VAR_100_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_13_]], [[VAR_97_]], [[VAR_106_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                 [[VAR_108_:%.+]] = arith.select [[VAR_104_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_109_:%.+]] = arith.mulf [[VAR_98_]], [[VAR_108_]] : f32
// CHECK-DAG:             [[VAR_110_:%.+]] = arith.addf [[VAR_89_]], [[VAR_109_]] : f32
// CHECK-DAG:             [[VAR_111_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_43_]] : f32
// CHECK-DAG:             [[VAR_112_:%.+]] = arith.addi [[VAR_42_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_113_:%.+]] = arith.subi [[CST_3_2_]], [[CST_1_21_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.cmpi sge, [[VAR_112_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.cmpi sle, [[VAR_112_]], [[VAR_113_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.andi [[VAR_114_]], [[VAR_115_]] : i1
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.maxsi [[VAR_112_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.minsi [[VAR_117_]], [[VAR_113_]] : index
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.mulf [[VAR_111_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_21_]] : index
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.subi [[CST_3_3_]], [[CST_1_21_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.cmpi sge, [[VAR_120_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.cmpi sle, [[VAR_120_]], [[VAR_121_]] : index
// CHECK:                 [[VAR_124_:%.+]] = arith.andi [[VAR_122_]], [[VAR_123_]] : i1
// CHECK-DAG:             [[VAR_125_:%.+]] = arith.andi [[VAR_116_]], [[VAR_124_]] : i1
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.maxsi [[VAR_120_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_127_:%.+]] = arith.minsi [[VAR_126_]], [[VAR_121_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_13_]], [[VAR_118_]], [[VAR_127_]]{{.}} : memref<1x1x3x3xf32>
// CHECK:                 [[VAR_129_:%.+]] = arith.select [[VAR_125_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_130_:%.+]] = arith.mulf [[VAR_119_]], [[VAR_129_]] : f32
// CHECK-DAG:             [[VAR_131_:%.+]] = arith.addf [[VAR_110_]], [[VAR_130_]] : f32
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_6_]], [[VAR_12_]]#0, [[VAR_12_]]#1, [[VAR_12_]]#2] : memref<1x1x2x2xf32>
// CHECK:                 [[VAR_133_:%.+]] = arith.mulf [[VAR_131_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:                 [[VAR_134_:%.+]] = arith.addf [[VAR_arg11_]], [[VAR_133_]] : f32
// CHECK:                 krnl.yield [[VAR_134_]] : f32
// CHECK:               }
// CHECK:               krnl.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_6_]], [[VAR_9_]]#0, [[VAR_9_]]#1] : memref<1x1x2x2xf32>
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
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_8_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_9_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_10_:%.+]] = arith.constant 2 : index
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
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[VAR_arg10_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[VAR_arg11_:%.+]] = 0 to 3, [[LOOP_2_]]#2 -> [[VAR_arg12_:%.+]] = 0 to 3) iter_args([[VAR_arg13_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_13_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:                 [[VAR_14_:%.+]] = arith.addi [[VAR_6_]], [[VAR_13_]]#0 : index
// CHECK-DAG:             [[VAR_15_:%.+]] = arith.floordivsi [[VAR_14_]], [[CST_2_9_]] : index
// CHECK-DAG:             [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_16_:%.+]] = arith.muli [[CST_0_8_]], [[CST_3_]] : index
// CHECK:                 [[VAR_17_:%.+]] = arith.addi [[VAR_16_]], [[VAR_13_]]#1 : index
// CHECK:                 [[VAR_18_:%.+]] = arith.muli [[VAR_17_]], [[CST_3_1_]] : index
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.addi [[VAR_18_]], [[VAR_13_]]#2 : index
// CHECK-DAG:             [[VAR_20_:%.+]] = arith.muli [[VAR_15_]], [[VAR_1_]] : index
// CHECK:                 [[VAR_21_:%.+]] = arith.addi [[VAR_20_]], [[VAR_19_]] : index
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.muli [[VAR_21_]], [[CST_2_10_]] : index
// CHECK-DAG:             [[CST_2_11_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.muli [[VAR_8_]]#0, [[CST_2_11_]] : index
// CHECK-DAG:             [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_24_:%.+]] = arith.muli [[VAR_13_]]#1, [[CST_1_5_]] : index
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.addi [[VAR_23_]], [[VAR_24_]] : index
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_26_:%.+]] = arith.subi [[VAR_25_]], [[CST_0_9_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_27_:%.+]] = arith.addi [[VAR_22_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]]#0, [[VAR_27_]], [[VAR_8_]]#0, [[VAR_8_]]#1] : memref<1x36x3x3xf32>
// CHECK-DAG:             [[VAR_29_:%.+]] = arith.index_cast [[VAR_26_]] : index to i64
// CHECK:                 [[VAR_30_:%.+]] = arith.sitofp [[VAR_29_]] : i64 to f32
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.addf [[VAR_30_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_2_12_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.muli [[VAR_8_]]#1, [[CST_2_12_]] : index
// CHECK-DAG:             [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_33_:%.+]] = arith.muli [[VAR_13_]]#2, [[CST_1_6_]] : index
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.addi [[VAR_32_]], [[VAR_33_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_35_:%.+]] = arith.subi [[VAR_34_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_36_:%.+]] = arith.addi [[VAR_22_]], [[CST_1_7_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]]#0, [[VAR_36_]], [[VAR_8_]]#0, [[VAR_8_]]#1] : memref<1x36x3x3xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.index_cast [[VAR_35_]] : index to i64
// CHECK:                 [[VAR_39_:%.+]] = arith.sitofp [[VAR_38_]] : i64 to f32
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.addf [[VAR_39_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[VAR_41_:%.+]] = math.floor [[VAR_31_]] : f32
// CHECK:                 [[VAR_42_:%.+]] = arith.fptosi [[VAR_41_]] : f32 to i64
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.index_cast [[VAR_42_]] : i64 to index
// CHECK-DAG:             [[VAR_44_:%.+]] = arith.subf [[VAR_31_]], [[VAR_41_]] : f32
// CHECK-DAG:             [[VAR_45_:%.+]] = math.floor [[VAR_40_]] : f32
// CHECK:                 [[VAR_46_:%.+]] = arith.fptosi [[VAR_45_]] : f32 to i64
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.index_cast [[VAR_46_]] : i64 to index
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.subf [[VAR_40_]], [[VAR_45_]] : f32
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_49_]] : f32
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.cmpi sge, [[VAR_43_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_53_:%.+]] = arith.cmpi sle, [[VAR_43_]], [[VAR_51_]] : index
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.andi [[VAR_52_]], [[VAR_53_]] : i1
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.maxsi [[VAR_43_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.minsi [[VAR_55_]], [[VAR_51_]] : index
// CHECK-DAG:             [[VAR_57_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.mulf [[VAR_50_]], [[VAR_57_]] : f32
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_61_:%.+]] = arith.cmpi sle, [[VAR_47_]], [[VAR_59_]] : index
// CHECK:                 [[VAR_62_:%.+]] = arith.andi [[VAR_60_]], [[VAR_61_]] : i1
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.andi [[VAR_54_]], [[VAR_62_]] : i1
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.maxsi [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_65_:%.+]] = arith.minsi [[VAR_64_]], [[VAR_59_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_14_]], [[VAR_56_]], [[VAR_65_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                 [[VAR_67_:%.+]] = arith.select [[VAR_63_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_68_:%.+]] = arith.mulf [[VAR_58_]], [[VAR_67_]] : f32
// CHECK-DAG:             [[VAR_69_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_68_]] : f32
// CHECK-DAG:             [[VAR_70_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK-DAG:             [[VAR_71_:%.+]] = arith.addi [[VAR_43_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.cmpi sge, [[VAR_71_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.cmpi sle, [[VAR_71_]], [[VAR_72_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.andi [[VAR_73_]], [[VAR_74_]] : i1
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.maxsi [[VAR_71_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.minsi [[VAR_76_]], [[VAR_72_]] : index
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.mulf [[VAR_70_]], [[VAR_78_]] : f32
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_82_:%.+]] = arith.cmpi sle, [[VAR_47_]], [[VAR_80_]] : index
// CHECK:                 [[VAR_83_:%.+]] = arith.andi [[VAR_81_]], [[VAR_82_]] : i1
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.andi [[VAR_75_]], [[VAR_83_]] : i1
// CHECK-DAG:             [[VAR_85_:%.+]] = arith.maxsi [[VAR_47_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_86_:%.+]] = arith.minsi [[VAR_85_]], [[VAR_80_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_14_]], [[VAR_77_]], [[VAR_86_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                 [[VAR_88_:%.+]] = arith.select [[VAR_84_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_89_:%.+]] = arith.mulf [[VAR_79_]], [[VAR_88_]] : f32
// CHECK-DAG:             [[VAR_90_:%.+]] = arith.addf [[VAR_69_]], [[VAR_89_]] : f32
// CHECK-DAG:             [[VAR_91_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_92_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_91_]] : f32
// CHECK-DAG:             [[VAR_93_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_94_:%.+]] = arith.cmpi sge, [[VAR_43_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_95_:%.+]] = arith.cmpi sle, [[VAR_43_]], [[VAR_93_]] : index
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.andi [[VAR_94_]], [[VAR_95_]] : i1
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.maxsi [[VAR_43_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_98_:%.+]] = arith.minsi [[VAR_97_]], [[VAR_93_]] : index
// CHECK-DAG:             [[VAR_99_:%.+]] = arith.mulf [[VAR_92_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.addi [[VAR_47_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.cmpi sge, [[VAR_100_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_103_:%.+]] = arith.cmpi sle, [[VAR_100_]], [[VAR_101_]] : index
// CHECK:                 [[VAR_104_:%.+]] = arith.andi [[VAR_102_]], [[VAR_103_]] : i1
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.andi [[VAR_96_]], [[VAR_104_]] : i1
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.maxsi [[VAR_100_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_107_:%.+]] = arith.minsi [[VAR_106_]], [[VAR_101_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_14_]], [[VAR_98_]], [[VAR_107_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                 [[VAR_109_:%.+]] = arith.select [[VAR_105_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_110_:%.+]] = arith.mulf [[VAR_99_]], [[VAR_109_]] : f32
// CHECK-DAG:             [[VAR_111_:%.+]] = arith.addf [[VAR_90_]], [[VAR_110_]] : f32
// CHECK-DAG:             [[VAR_112_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK-DAG:             [[VAR_113_:%.+]] = arith.addi [[VAR_43_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.subi [[CST_8_2_]], [[CST_1_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.cmpi sge, [[VAR_113_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.cmpi sle, [[VAR_113_]], [[VAR_114_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.andi [[VAR_115_]], [[VAR_116_]] : i1
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.maxsi [[VAR_113_]], [[CST_0_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.minsi [[VAR_118_]], [[VAR_114_]] : index
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.mulf [[VAR_112_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.addi [[VAR_47_]], [[CST_1_8_]] : index
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.subi [[CST_8_3_]], [[CST_1_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.cmpi sge, [[VAR_121_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.cmpi sle, [[VAR_121_]], [[VAR_122_]] : index
// CHECK:                 [[VAR_125_:%.+]] = arith.andi [[VAR_123_]], [[VAR_124_]] : i1
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.andi [[VAR_117_]], [[VAR_125_]] : i1
// CHECK-DAG:             [[VAR_127_:%.+]] = arith.maxsi [[VAR_121_]], [[CST_0_12_]] : index
// CHECK:                 [[VAR_128_:%.+]] = arith.minsi [[VAR_127_]], [[VAR_122_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_14_]], [[VAR_119_]], [[VAR_128_]]{{.}} : memref<1x4x8x8xf32>
// CHECK:                 [[VAR_130_:%.+]] = arith.select [[VAR_126_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_131_:%.+]] = arith.mulf [[VAR_120_]], [[VAR_130_]] : f32
// CHECK-DAG:             [[VAR_132_:%.+]] = arith.addf [[VAR_111_]], [[VAR_131_]] : f32
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_5_]], [[VAR_13_]]#0, [[VAR_13_]]#1, [[VAR_13_]]#2] : memref<6x2x3x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_134_:%.+]] = arith.mulf [[VAR_132_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_3_]]#0, [[VAR_21_]], [[VAR_8_]]#0, [[VAR_8_]]#1] : memref<1x18x3x3xf32>
// CHECK:                 [[VAR_136_:%.+]] = arith.mulf [[VAR_134_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:                 [[VAR_137_:%.+]] = arith.addf [[VAR_arg13_]], [[VAR_136_]] : f32
// CHECK:                 krnl.yield [[VAR_137_]] : f32
// CHECK:               }
// CHECK:               [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_5_]]{{.}} : memref<6xf32>
// CHECK:               [[VAR_12_:%.+]] = arith.addf [[VAR_10_]], [[LOAD_PARAM_3_MEM_]] : f32
// CHECK:               krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_5_]], [[VAR_8_]]#0, [[VAR_8_]]#1] : memref<1x6x3x3xf32>
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
// CHECK-DAG:       [[CST_1_9_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_10_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_11_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[CST_1_11_]], [[CST_2_]] : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_24_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_6_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_2_1_]] : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_26_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_1_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_8_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_9_:%.+]] = arith.muli [[VAR_8_]]#1, [[CST_5_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_8_]]#2 : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_8_]]#1, [[CST_1_8_]] : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_7_]], [[VAR_dim_12_]], [[VAR_dim_]])){
// CHECK-DAG:           [[VAR_13_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_15_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[VAR_arg8_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[VAR_arg9_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[VAR_arg10_:%.+]] = 0 to 2) iter_args([[VAR_arg11_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:             [[VAR_16_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:                 [[VAR_17_:%.+]] = arith.addi [[VAR_11_]], [[VAR_16_]]#0 : index
// CHECK-DAG:             [[VAR_18_:%.+]] = arith.floordivsi [[VAR_17_]], [[CST_1_9_]] : index
// CHECK-DAG:             [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_19_:%.+]] = arith.muli [[CST_0_9_]], [[CST_2_]] : index
// CHECK:                 [[VAR_20_:%.+]] = arith.addi [[VAR_19_]], [[VAR_16_]]#1 : index
// CHECK:                 [[VAR_21_:%.+]] = arith.muli [[VAR_20_]], [[CST_2_1_]] : index
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[VAR_16_]]#2 : index
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.muli [[VAR_18_]], [[VAR_6_]] : index
// CHECK:                 [[VAR_24_:%.+]] = arith.addi [[VAR_23_]], [[VAR_22_]] : index
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.muli [[VAR_24_]], [[CST_2_5_]] : index
// CHECK-DAG:             [[CST_1_12_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_26_:%.+]] = arith.muli [[VAR_13_]]#0, [[CST_1_12_]] : index
// CHECK-DAG:             [[CST_1_13_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_27_:%.+]] = arith.muli [[VAR_16_]]#1, [[CST_1_13_]] : index
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.addi [[VAR_26_]], [[VAR_27_]] : index
// CHECK-DAG:             [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_29_:%.+]] = arith.subi [[VAR_28_]], [[CST_0_10_]] : index
// CHECK-DAG:             [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_30_:%.+]] = arith.addi [[VAR_25_]], [[CST_0_11_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_]]#0, [[VAR_30_]], [[VAR_13_]]#0, [[VAR_13_]]#1] : memref<?x8x?x?xf32>
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.index_cast [[VAR_29_]] : index to i64
// CHECK:                 [[VAR_33_:%.+]] = arith.sitofp [[VAR_32_]] : i64 to f32
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.addf [[VAR_33_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:             [[CST_1_14_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_35_:%.+]] = arith.muli [[VAR_13_]]#1, [[CST_1_14_]] : index
// CHECK-DAG:             [[CST_1_15_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_36_:%.+]] = arith.muli [[VAR_16_]]#2, [[CST_1_15_]] : index
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.addi [[VAR_35_]], [[VAR_36_]] : index
// CHECK-DAG:             [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.subi [[VAR_37_]], [[CST_0_12_]] : index
// CHECK-DAG:             [[CST_1_16_:%.+]] = arith.constant 1 : index
// CHECK:                 [[VAR_39_:%.+]] = arith.addi [[VAR_25_]], [[CST_1_16_]] : index
// CHECK-DAG:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_]]#0, [[VAR_39_]], [[VAR_13_]]#0, [[VAR_13_]]#1] : memref<?x8x?x?xf32>
// CHECK-DAG:             [[VAR_41_:%.+]] = arith.index_cast [[VAR_38_]] : index to i64
// CHECK:                 [[VAR_42_:%.+]] = arith.sitofp [[VAR_41_]] : i64 to f32
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK-DAG:             [[CST_0_13_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_17_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_0_dot_000000_1_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:             [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:             [[VAR_44_:%.+]] = math.floor [[VAR_34_]] : f32
// CHECK:                 [[VAR_45_:%.+]] = arith.fptosi [[VAR_44_]] : f32 to i64
// CHECK-DAG:             [[VAR_46_:%.+]] = arith.index_cast [[VAR_45_]] : i64 to index
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.subf [[VAR_34_]], [[VAR_44_]] : f32
// CHECK-DAG:             [[VAR_48_:%.+]] = math.floor [[VAR_43_]] : f32
// CHECK:                 [[VAR_49_:%.+]] = arith.fptosi [[VAR_48_]] : f32 to i64
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.index_cast [[VAR_49_]] : i64 to index
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.subf [[VAR_43_]], [[VAR_48_]] : f32
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_53_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_52_]] : f32
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_56_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_54_]] : index
// CHECK-DAG:             [[VAR_57_:%.+]] = arith.andi [[VAR_55_]], [[VAR_56_]] : i1
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.minsi [[VAR_58_]], [[VAR_54_]] : index
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_61_:%.+]] = arith.mulf [[VAR_53_]], [[VAR_60_]] : f32
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.subi [[VAR_dim_26_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.cmpi sge, [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_64_:%.+]] = arith.cmpi sle, [[VAR_50_]], [[VAR_62_]] : index
// CHECK:                 [[VAR_65_:%.+]] = arith.andi [[VAR_63_]], [[VAR_64_]] : i1
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.andi [[VAR_57_]], [[VAR_65_]] : i1
// CHECK-DAG:             [[VAR_67_:%.+]] = arith.maxsi [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_68_:%.+]] = arith.minsi [[VAR_67_]], [[VAR_62_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_17_]], [[VAR_59_]], [[VAR_68_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                 [[VAR_70_:%.+]] = arith.select [[VAR_66_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_71_:%.+]] = arith.mulf [[VAR_61_]], [[VAR_70_]] : f32
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.addf [[CST_0_dot_000000_1_]], [[VAR_71_]] : f32
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_17_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.cmpi sge, [[VAR_74_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_77_:%.+]] = arith.cmpi sle, [[VAR_74_]], [[VAR_75_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_78_:%.+]] = arith.andi [[VAR_76_]], [[VAR_77_]] : i1
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.maxsi [[VAR_74_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.minsi [[VAR_79_]], [[VAR_75_]] : index
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_51_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_82_:%.+]] = arith.mulf [[VAR_73_]], [[VAR_81_]] : f32
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.subi [[VAR_dim_26_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_84_:%.+]] = arith.cmpi sge, [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_85_:%.+]] = arith.cmpi sle, [[VAR_50_]], [[VAR_83_]] : index
// CHECK:                 [[VAR_86_:%.+]] = arith.andi [[VAR_84_]], [[VAR_85_]] : i1
// CHECK-DAG:             [[VAR_87_:%.+]] = arith.andi [[VAR_78_]], [[VAR_86_]] : i1
// CHECK-DAG:             [[VAR_88_:%.+]] = arith.maxsi [[VAR_50_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_89_:%.+]] = arith.minsi [[VAR_88_]], [[VAR_83_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_17_]], [[VAR_80_]], [[VAR_89_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                 [[VAR_91_:%.+]] = arith.select [[VAR_87_]], [[LOAD_PARAM_0_MEM_1_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_92_:%.+]] = arith.mulf [[VAR_82_]], [[VAR_91_]] : f32
// CHECK-DAG:             [[VAR_93_:%.+]] = arith.addf [[VAR_72_]], [[VAR_92_]] : f32
// CHECK-DAG:             [[VAR_94_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_95_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_94_]] : f32
// CHECK-DAG:             [[VAR_96_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_97_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_98_:%.+]] = arith.cmpi sle, [[VAR_46_]], [[VAR_96_]] : index
// CHECK-DAG:             [[VAR_99_:%.+]] = arith.andi [[VAR_97_]], [[VAR_98_]] : i1
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.maxsi [[VAR_46_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.minsi [[VAR_100_]], [[VAR_96_]] : index
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.mulf [[VAR_95_]], [[VAR_51_]] : f32
// CHECK-DAG:             [[VAR_103_:%.+]] = arith.addi [[VAR_50_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.subi [[VAR_dim_26_]], [[CST_1_17_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_105_:%.+]] = arith.cmpi sge, [[VAR_103_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.cmpi sle, [[VAR_103_]], [[VAR_104_]] : index
// CHECK:                 [[VAR_107_:%.+]] = arith.andi [[VAR_105_]], [[VAR_106_]] : i1
// CHECK-DAG:             [[VAR_108_:%.+]] = arith.andi [[VAR_99_]], [[VAR_107_]] : i1
// CHECK-DAG:             [[VAR_109_:%.+]] = arith.maxsi [[VAR_103_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_110_:%.+]] = arith.minsi [[VAR_109_]], [[VAR_104_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_17_]], [[VAR_101_]], [[VAR_110_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                 [[VAR_112_:%.+]] = arith.select [[VAR_108_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_113_:%.+]] = arith.mulf [[VAR_102_]], [[VAR_112_]] : f32
// CHECK-DAG:             [[VAR_114_:%.+]] = arith.addf [[VAR_93_]], [[VAR_113_]] : f32
// CHECK-DAG:             [[VAR_115_:%.+]] = arith.mulf [[CST_1_dot_000000_]], [[VAR_47_]] : f32
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.addi [[VAR_46_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_117_:%.+]] = arith.subi [[VAR_dim_24_]], [[CST_1_17_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.cmpi sge, [[VAR_116_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_119_:%.+]] = arith.cmpi sle, [[VAR_116_]], [[VAR_117_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_120_:%.+]] = arith.andi [[VAR_118_]], [[VAR_119_]] : i1
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.maxsi [[VAR_116_]], [[CST_0_13_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.minsi [[VAR_121_]], [[VAR_117_]] : index
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.mulf [[VAR_115_]], [[VAR_51_]] : f32
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.addi [[VAR_50_]], [[CST_1_17_]] : index
// CHECK-DAG:             [[VAR_125_:%.+]] = arith.subi [[VAR_dim_26_]], [[CST_1_17_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_126_:%.+]] = arith.cmpi sge, [[VAR_124_]], [[CST_0_13_]] : index
// CHECK-DAG:             [[VAR_127_:%.+]] = arith.cmpi sle, [[VAR_124_]], [[VAR_125_]] : index
// CHECK:                 [[VAR_128_:%.+]] = arith.andi [[VAR_126_]], [[VAR_127_]] : i1
// CHECK-DAG:             [[VAR_129_:%.+]] = arith.andi [[VAR_120_]], [[VAR_128_]] : i1
// CHECK-DAG:             [[VAR_130_:%.+]] = arith.maxsi [[VAR_124_]], [[CST_0_13_]] : index
// CHECK:                 [[VAR_131_:%.+]] = arith.minsi [[VAR_130_]], [[VAR_125_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_17_]], [[VAR_122_]], [[VAR_131_]]{{.}} : memref<?x1x?x?xf32>
// CHECK:                 [[VAR_133_:%.+]] = arith.select [[VAR_129_]], [[LOAD_PARAM_0_MEM_3_]], [[CST_0_dot_000000_1_]] : f32
// CHECK:                 [[VAR_134_:%.+]] = arith.mulf [[VAR_123_]], [[VAR_133_]] : f32
// CHECK-DAG:             [[VAR_135_:%.+]] = arith.addf [[VAR_114_]], [[VAR_134_]] : f32
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_16_]]#0, [[VAR_16_]]#1, [[VAR_16_]]#2] : memref<5x1x2x2xf32>
// CHECK:                 [[VAR_137_:%.+]] = arith.mulf [[VAR_135_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:                 [[VAR_138_:%.+]] = arith.addf [[VAR_arg11_]], [[VAR_137_]] : f32
// CHECK:                 krnl.yield [[VAR_138_]] : f32
// CHECK:               }
// CHECK:               krnl.store [[VAR_15_]], [[RES_]]{{.}}[[VAR_8_]]#0, [[VAR_10_]], [[VAR_13_]]#0, [[VAR_13_]]#1] : memref<?x5x?x?xf32>
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
