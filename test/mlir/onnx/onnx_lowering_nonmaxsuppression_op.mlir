// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func @test_nonmaxsuppression_center_point_box_format(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"]'
// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_center_point_box_format
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_21_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_27_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_21_]]#0, [[VAR_21_]]#1, [[VAR_27_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_31_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_32_:%.+]] = arith.select [[VAR_29_]], [[VAR_31_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_32_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_26_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_26_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_21_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_21_1_]]#2, [[RES_3_]]{{.}}[[VAR_21_1_]]#0, [[VAR_21_1_]]#1, [[VAR_21_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_21_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_21_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[VAR_21_2_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[VAR_21_2_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_21_2_]]#0, [[VAR_21_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_4_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:           [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_5_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_21_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:             [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_7_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_21_3_]]#0, [[VAR_21_3_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_21_3_]]#0, [[VAR_21_3_]]#1, [[LOAD_SCORES_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[LOAD_SCORES_MEM_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_1_:%.+]] = arith.cmpi eq, [[VAR_31_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_34_:%.+]] = arith.andi [[VAR_33_]], [[VAR_32_1_]] : i1
// CHECK:               scf.if [[VAR_34_]] {
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_21_3_]]#0, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_21_3_]]#1, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_29_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_40_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_41_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_43_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_43_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_45_:%.+]] = arith.cmpi eq, [[LOAD_RES_7_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_45_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[VAR_43_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[VAR_43_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[VAR_43_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_21_3_]]#0, [[VAR_43_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_50_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_54_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_60_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_62_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.maxf [[VAR_51_]], [[VAR_63_]] : f32
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.maxf [[VAR_55_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_70_:%.+]] = arith.minf [[VAR_53_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_71_:%.+]] = arith.minf [[VAR_57_]], [[VAR_61_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.subf [[VAR_70_]], [[VAR_68_]] : f32
// CHECK-DAG:                 [[VAR_73_:%.+]] = arith.subf [[VAR_71_]], [[VAR_69_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_74_:%.+]] = arith.maxf [[VAR_72_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.maxf [[VAR_73_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.mulf [[VAR_74_]], [[VAR_75_]] : f32
// CHECK-DAG:                 [[VAR_77_:%.+]] = arith.addf [[VAR_66_]], [[VAR_67_]] : f32
// CHECK:                     [[VAR_78_:%.+]] = arith.subf [[VAR_77_]], [[VAR_76_]] : f32
// CHECK:                     [[VAR_79_:%.+]] = arith.addf [[VAR_78_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_80_:%.+]] = arith.divf [[VAR_76_]], [[VAR_79_]] : f32
// CHECK:                     [[VAR_81_:%.+]] = arith.cmpf oge, [[VAR_80_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_81_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_7_]]{{.}}[[VAR_43_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc([[LOAD_RES_5_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_5_MEM_1_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_21_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_21_4_]]#0, [[VAR_21_4_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_7_:%.+]] = arith.index_cast [[RES_6_]] : index to i64
// CHECK:             krnl.store [[RES_7_]], [[RES_8_]]{{.}}[[VAR_21_4_]]#0, [[VAR_21_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_flipped_coordinates(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_flipped_coordinates
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_identical_boxes(%arg0: tensor<1x10x4xf32>, %arg1: tensor<1x1x10xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x10x4xf32>, tensor<1x1x10xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_identical_boxes
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x10x4xf32>, [[SCORES_:%.+]]: memref<1x1x10xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c10_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c10_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x10xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x10xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 10){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x10xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 9){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 10){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x10xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x10xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x10xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x10xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x10xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x10xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x10x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 10){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<10xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<10xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c10_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x10xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x10xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<10xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c10_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<10xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<10xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_limit_output_size(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_limit_output_size
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_single_box(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x1x4xf32>, tensor<1x1x1xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_single_box
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x1x4xf32>, [[SCORES_:%.+]]: memref<1x1x1xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c1_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x1xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 1){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x1xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 0){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 1){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x1xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x1xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x1xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x1xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x1xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x1xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x1x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 1){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<1xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x1xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x1xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<1xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<1xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_suppress_by_IOU(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_suppress_by_IOU
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_suppress_by_IOU_and_scores(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_nonmaxsuppression_suppress_by_IOU_and_scores
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_29_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_29_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[VAR_33_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_34_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_23_1_]]#2, [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1, [[VAR_23_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_23_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map([[VAR_23_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[VAR_23_2_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_23_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.select [[VAR_28_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_28_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_31_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.select [[VAR_31_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_29_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_23_3_]]#0, [[VAR_23_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_23_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_28_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_23_4_]]#0, [[VAR_23_4_]]#1, [[VAR_28_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_31_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_28_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_1_:%.+]] = arith.cmpi eq, [[VAR_33_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_35_]], [[VAR_34_1_]] : i1
// CHECK:               scf.if [[VAR_36_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_28_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_23_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_23_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_28_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_31_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_42_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_23_4_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf oge, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_23_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_23_5_]]#0, [[VAR_23_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_two_batches(%arg0: tensor<2x6x4xf32>, %arg1: tensor<2x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x6x4xf32>, tensor<2x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map0 = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL:  func @test_nonmaxsuppression_two_batches
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<2x6x4xf32>, [[SCORES_:%.+]]: memref<2x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_30_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1, [[VAR_30_]]{{.}} : memref<2x1x6xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_34_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.select [[VAR_32_]], [[VAR_34_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_35_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_29_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_29_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_24_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_24_1_]]#2, [[RES_3_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_24_1_]]#2] : memref<2x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_24_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map0([[VAR_24_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_24_2_]]#2] : memref<2x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOOP_1_]]{{.}} : memref<2x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<2x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<2x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_24_2_]]#2] : memref<2x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOOP_1_]]{{.}} : memref<2x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 2, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_24_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK:             [[VAR_29_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_30_1_:%.+]] = arith.select [[VAR_29_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_29_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_32_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_32_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_34_1_:%.+]] = arith.select [[VAR_32_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_30_1_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_34_1_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK:           }
// CHECK:           [[VAR_17_:%.+]] = affine.apply #map1(){{.}}[[LOAD_RES_MEM_1_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_17_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]){
// CHECK-DAG:         [[VAR_24_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_29_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_4_]]#0, [[VAR_24_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<2x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_4_]]#0, [[VAR_24_4_]]#1, [[VAR_29_1_]]{{.}} : memref<2x1x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_32_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_32_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_34_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_29_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_1_:%.+]] = arith.cmpi eq, [[VAR_34_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_37_:%.+]] = arith.andi [[VAR_36_]], [[VAR_35_1_]] : i1
// CHECK:               scf.if [[VAR_37_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_24_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_24_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_29_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[VAR_32_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_44_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_44_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_46_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_48_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_48_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.mulf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.mulf [[VAR_56_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.subf [[VAR_62_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.maxf [[VAR_64_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.mulf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.addf [[VAR_55_]], [[VAR_58_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.subf [[VAR_68_]], [[VAR_67_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.addf [[VAR_69_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.divf [[VAR_67_]], [[VAR_70_]] : f32
// CHECK:                     [[VAR_72_:%.+]] = arith.cmpf oge, [[VAR_71_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_72_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_24_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_24_5_]]#0, [[VAR_24_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_24_5_]]#0, [[VAR_24_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_two_classes(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x2x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x2x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map0 = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL:  func @test_nonmaxsuppression_two_classes
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x2x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_5_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]]){
// CHECK-DAG:         [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[VAR_30_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1, [[VAR_30_]]{{.}} : memref<1x2x6xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_34_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.select [[VAR_32_]], [[VAR_34_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_35_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_29_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_29_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_10_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x2x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_24_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_24_1_]]#2, [[RES_3_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_24_1_]]#2] : memref<1x2x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 2, [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_24_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map0([[VAR_24_2_]]#2) to 6){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_24_2_]]#2] : memref<1x2x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x2x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<1x2x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<1x2x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_24_2_]]#2] : memref<1x2x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[LOOP_1_]]{{.}} : memref<1x2x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 6){
// CHECK:             [[VAR_24_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOOP_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_29_1_:%.+]] = arith.cmpf ogt, [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_30_1_:%.+]] = arith.select [[VAR_29_1_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_1_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = arith.select [[VAR_29_1_]], [[LOOP_1_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_32_1_:%.+]] = arith.cmpf ogt, [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = arith.select [[VAR_32_1_]], [[LOAD_RES_2_MEM_1_1_]], [[LOOP_4_]] : f32
// CHECK-DAG:         [[VAR_34_1_:%.+]] = arith.select [[VAR_32_1_]], [[LOOP_4_]], [[LOAD_RES_2_MEM_1_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_30_1_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_34_1_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_3_]], [[RES_4_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[VAR_17_:%.+]] = affine.apply #map1(){{.}}[[LOAD_RES_MEM_1_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_17_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:           [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_6_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]]){
// CHECK-DAG:         [[VAR_24_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_29_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_24_4_]]#0, [[VAR_24_4_]]#1, [[LOAD_RES_1_MEM_2_1_]]{{.}} : memref<1x2x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_4_]]#0, [[VAR_24_4_]]#1, [[VAR_29_1_]]{{.}} : memref<1x2x6xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_32_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_32_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_34_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_29_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_1_:%.+]] = arith.cmpi eq, [[VAR_34_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_1_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_37_:%.+]] = arith.andi [[VAR_36_]], [[VAR_35_1_]] : i1
// CHECK:               scf.if [[VAR_37_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_29_1_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_24_4_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_24_4_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_29_1_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[VAR_32_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_43_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_44_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_44_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]){
// CHECK:                   [[VAR_46_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_48_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_48_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_24_4_]]#0, [[VAR_46_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.mulf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.mulf [[VAR_56_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.maxf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.minf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.subf [[VAR_62_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.maxf [[VAR_64_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.mulf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.addf [[VAR_55_]], [[VAR_58_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.subf [[VAR_68_]], [[VAR_67_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.addf [[VAR_69_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.divf [[VAR_67_]], [[VAR_70_]] : f32
// CHECK:                     [[VAR_72_:%.+]] = arith.cmpf oge, [[VAR_71_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_72_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_24_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_7_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_24_5_]]#0, [[VAR_24_5_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_8_:%.+]] = arith.index_cast [[RES_7_]] : index to i64
// CHECK:             krnl.store [[RES_8_]], [[RES_9_]]{{.}}[[VAR_24_5_]]#0, [[VAR_24_5_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// CHECK-DAG: #map0 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #map2 = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #map4 = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #map5 = affine_map<(d0, d1, d2) -> (d2 - 1)>
// CHECK-DAG: #map6 = affine_map<(d0, d1, d2, d3) -> (d3 + 1)>
// CHECK-DAG: #map7 = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-LABEL:  func @test_nonmaxsuppression_unknown_dims
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<?x?x?xf32>, [[SCORES_:%.+]]: memref<?x?x?xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[SCORES_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[SCORES_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[SCORES_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[VAR_7_]], [[VAR_5_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_9_:%.+]] = memref.dim [[SCORES_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = memref.dim [[SCORES_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = memref.dim [[SCORES_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_9_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_10_]]){
// CHECK-DAG:         [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_11_]]){
// CHECK:               [[VAR_38_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1, [[VAR_38_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:               [[VAR_42_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_c1_]] : index
// CHECK:               [[VAR_43_:%.+]] = arith.select [[VAR_40_]], [[VAR_42_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:               krnl.store [[VAR_43_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             [[VAR_37_:%.+]] = arith.maxsi [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:             krnl.store [[VAR_37_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:           [[VAR_16_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_16_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_18_:%.+]] = memref.dim [[SCORES_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = memref.dim [[SCORES_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = memref.dim [[SCORES_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[VAR_18_]], [[VAR_19_]], [[VAR_20_]]) {{.*}}: memref<?x?x?xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to #map0([[VAR_18_]]), [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to #map1([[VAR_18_]], [[VAR_19_]]), [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to #map2([[VAR_18_]], [[VAR_19_]], [[VAR_20_]])){
// CHECK:             [[VAR_32_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_32_1_]]#2, [[RES_3_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1, [[VAR_32_1_]]#2] : memref<?x?x?xindex>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to #map3([[VAR_18_]], [[VAR_19_]], [[VAR_20_]]), [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to #map4([[VAR_18_]], [[VAR_19_]], [[VAR_20_]]), [[LOOP_3_]]#2 -> [[I_8_:%.+]] = 0 to #map5([[VAR_18_]], [[VAR_19_]], [[VAR_20_]])){
// CHECK-DAG:         [[VAR_32_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = #map6([[VAR_18_]], [[VAR_19_]], [[VAR_20_]], [[VAR_32_2_]]#2) to #map7([[VAR_18_]], [[VAR_19_]], [[VAR_20_]], [[VAR_32_2_]]#2)){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[VAR_32_2_]]#2] : memref<?x?x?xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[LOOP_1_]]{{.}} : memref<?x?x?xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<?x?x?xf32>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<?x?x?xf32>
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORES_MEM_2_]] : f32
// CHECK:               scf.if [[LOAD_SCORES_MEM_3_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_3_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[VAR_32_2_]]#2] : memref<?x?x?xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1, [[LOOP_1_]]{{.}} : memref<?x?x?xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_24_:%.+]] = arith.muli [[VAR_3_]], [[VAR_4_]] : index
// CHECK:           [[VAR_25_:%.+]] = arith.muli [[VAR_24_]], [[LOAD_RES_MEM_1_]] : index
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[VAR_25_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_4_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:           [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_3_]], [[LOOP_5_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_4_]]){
// CHECK-DAG:         [[VAR_32_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_6_]][] : memref<index>
// CHECK:             [[RES_7_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi1>
// CHECK:             krnl.memset [[RES_7_]], [[VAR_false_]] : memref<?xi1>
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_5_]]){
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_32_3_]]#0, [[VAR_32_3_]]#1, [[LOAD_RES_1_MEM_2_]]{{.}} : memref<?x?x?xindex>
// CHECK:               [[LOAD_SCORES_MEM_4_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_32_3_]]#0, [[VAR_32_3_]]#1, [[LOAD_SCORES_MEM_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:           [[LOAD_SCORES_MEM_3_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_4_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_40_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = arith.cmpi slt, [[VAR_40_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[VAR_42_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[LOAD_SCORES_MEM_1_]]{{.}} : memref<?xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_1_:%.+]] = arith.cmpi eq, [[VAR_42_1_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_44_:%.+]] = arith.andi [[LOAD_SCORES_MEM_3_]], [[LOAD_RES_2_MEM_2_]] : i1
// CHECK:               [[VAR_45_:%.+]] = arith.andi [[VAR_44_]], [[VAR_43_1_]] : i1
// CHECK:               scf.if [[VAR_45_]] {
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c0_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c2_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[LOAD_SCORES_MEM_1_]], [[VAR_c3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_32_3_]]#0, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_32_3_]]#1, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_51_:%.+]] = arith.addi [[VAR_40_1_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_51_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[VAR_52_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_52_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_5_]]){
// CHECK:                   [[VAR_54_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_54_]]{{.}} : memref<?xi1>
// CHECK:                   [[VAR_56_:%.+]] = arith.cmpi eq, [[LOAD_RES_7_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_56_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[VAR_54_]], [[VAR_c0_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[VAR_54_]], [[VAR_c1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[VAR_54_]], [[VAR_c2_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_32_3_]]#0, [[VAR_54_]], [[VAR_c3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_61_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_63_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_67_]] : f32
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_70_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_69_]] : f32
// CHECK-DAG:                 [[VAR_71_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_71_]] : f32
// CHECK-DAG:                 [[VAR_73_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_74_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_73_]] : f32
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_75_]] : f32
// CHECK-DAG:                 [[VAR_77_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_78_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_79_:%.+]] = arith.maxf [[VAR_62_]], [[VAR_74_]] : f32
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.maxf [[VAR_66_]], [[VAR_70_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_81_:%.+]] = arith.minf [[VAR_64_]], [[VAR_76_]] : f32
// CHECK-DAG:                 [[VAR_82_:%.+]] = arith.minf [[VAR_68_]], [[VAR_72_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_83_:%.+]] = arith.subf [[VAR_81_]], [[VAR_79_]] : f32
// CHECK-DAG:                 [[VAR_84_:%.+]] = arith.subf [[VAR_82_]], [[VAR_80_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_85_:%.+]] = arith.maxf [[VAR_83_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_86_:%.+]] = arith.maxf [[VAR_84_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_87_:%.+]] = arith.mulf [[VAR_85_]], [[VAR_86_]] : f32
// CHECK-DAG:                 [[VAR_88_:%.+]] = arith.addf [[VAR_77_]], [[VAR_78_]] : f32
// CHECK:                     [[VAR_89_:%.+]] = arith.subf [[VAR_88_]], [[VAR_87_]] : f32
// CHECK:                     [[VAR_90_:%.+]] = arith.addf [[VAR_89_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_91_:%.+]] = arith.divf [[VAR_87_]], [[VAR_90_]] : f32
// CHECK:                     [[VAR_92_:%.+]] = arith.cmpf oge, [[VAR_91_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_92_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_7_]]{{.}}[[VAR_54_]]{{.}} : memref<?xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc([[LOAD_RES_5_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_5_MEM_1_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:             [[VAR_32_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_32_4_]]#0, [[VAR_32_4_]]#1] : memref<?x3xindex>
// CHECK:             [[RES_7_:%.+]] = arith.index_cast [[RES_6_]] : index to i64
// CHECK:             krnl.store [[RES_7_]], [[RES_8_]]{{.}}[[VAR_32_4_]]#0, [[VAR_32_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<?x3xi64>
// CHECK:         }
}
