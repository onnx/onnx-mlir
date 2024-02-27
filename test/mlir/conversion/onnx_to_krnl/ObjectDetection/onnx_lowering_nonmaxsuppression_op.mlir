// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_nonmaxsuppression_center_point_box_format(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"]'
// CHECK-LABEL:  func @test_nonmaxsuppression_center_point_box_format
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_14_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_19_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_14_]]#0, [[VAR_14_]]#1, [[VAR_19_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_23_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_24_:%.+]] = arith.select [[VAR_21_]], [[VAR_23_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_24_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_18_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_18_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_14_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_14_1_]]#2, [[RES_3_]]{{.}}[[VAR_14_1_]]#0, [[VAR_14_1_]]#1, [[VAR_14_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_4_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_3_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_14_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK:             [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_7_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_14_2_]]#0, [[VAR_14_2_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_14_2_]]#0, [[VAR_14_2_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_19_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_21_1_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_7_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_1_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_24_1_:%.+]] = arith.andi [[VAR_19_1_]], [[VAR_21_1_]] : i1
// CHECK:               [[VAR_25_:%.+]] = arith.andi [[VAR_24_1_]], [[VAR_23_1_]] : i1
// CHECK:               scf.if [[VAR_25_]] {
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_14_2_]]#0, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_14_2_]]#1, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_31_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_31_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_34_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_34_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_36_:%.+]] = arith.cmpi eq, [[LOAD_RES_7_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_36_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[VAR_34_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[VAR_34_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[VAR_34_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_14_2_]]#0, [[VAR_34_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_41_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_45_]] : f32
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_47_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_49_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_51_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_54_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.maxnumf [[VAR_46_]], [[VAR_50_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minnumf [[VAR_44_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.minnumf [[VAR_48_]], [[VAR_52_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.subf [[VAR_62_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.maxnumf [[VAR_63_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.maxnumf [[VAR_64_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.mulf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.addf [[VAR_57_]], [[VAR_58_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.subf [[VAR_68_]], [[VAR_67_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.addf [[VAR_69_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.divf [[VAR_67_]], [[VAR_70_]] : f32
// CHECK:                     [[VAR_72_:%.+]] = arith.cmpf oge, [[VAR_71_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_72_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_7_]]{{.}}[[VAR_34_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc([[LOAD_RES_5_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_5_MEM_1_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_14_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_14_3_]]#0, [[VAR_14_3_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_4_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_8_]]{{.}}[[VAR_14_3_]]#0, [[VAR_14_3_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_flipped_coordinates(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_flipped_coordinates
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_identical_boxes(%arg0: tensor<1x10x4xf32>, %arg1: tensor<1x1x10xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x10x4xf32>, tensor<1x1x10xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_identical_boxes
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x10x4xf32>, [[SCORES_:%.+]]: memref<1x1x10xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_10_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_10_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x10xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x10xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 10){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x10xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x10xindex>, memref<1x1x10xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x10x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 10){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x10x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x10x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<10xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<10xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_10_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x10xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x10xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<10xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_10_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<10xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<10xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_limit_output_size(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_limit_output_size
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_single_box(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x1x4xf32>, tensor<1x1x1xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_single_box
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x1x4xf32>, [[SCORES_:%.+]]: memref<1x1x1xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_1_1_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x1xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 1){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x1xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x1xindex>, memref<1x1x1xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x1x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x1x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x1x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<1xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x1xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x1xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<1xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<1xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_suppress_by_IOU(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_suppress_by_IOU
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_suppress_by_IOU_and_scores(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-LABEL:  func @test_nonmaxsuppression_suppress_by_IOU_and_scores
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_24_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_19_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_15_1_]]#2, [[RES_3_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1, [[VAR_15_1_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_15_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_20_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_20_1_]], [[VAR_19_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.select [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_19_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_24_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_22_1_]], [[RES_4_]]{{.}}[[VAR_15_2_]]#0, [[VAR_15_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_15_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_15_3_]]#0, [[VAR_15_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_20_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.andi [[VAR_20_2_]], [[VAR_22_2_]] : i1
// CHECK:               [[VAR_26_:%.+]] = arith.andi [[VAR_25_2_]], [[VAR_24_2_]] : i1
// CHECK:               scf.if [[VAR_26_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_15_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_15_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_32_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_32_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_37_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_15_3_]]#0, [[VAR_35_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_46_]] : f32
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[VAR_50_]], [[VAR_48_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.addf [[VAR_44_]], [[VAR_47_]] : f32
// CHECK:                     [[VAR_58_:%.+]] = arith.subf [[VAR_57_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.divf [[VAR_56_]], [[VAR_59_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.cmpf oge, [[VAR_60_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_61_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_35_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_15_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_15_4_]]#0, [[VAR_15_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_two_batches(%arg0: tensor<2x6x4xf32>, %arg1: tensor<2x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x6x4xf32>, tensor<2x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL:  func @test_nonmaxsuppression_two_batches
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<2x6x4xf32>, [[SCORES_:%.+]]: memref<2x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_21_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1, [[VAR_21_]]{{.}} : memref<2x1x6xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_25_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_26_:%.+]] = arith.select [[VAR_23_]], [[VAR_25_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_26_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_20_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_20_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x1x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_16_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_16_1_]]#2, [[RES_3_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_16_1_]]#2] : memref<2x1x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<2x1x6xindex>, memref<2x1x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_16_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_0_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_1_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_2_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_3_]]{{.}} : memref<2x6x4xf32>
// CHECK:             [[VAR_21_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_21_1_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.select [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_26_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_0_1_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_1_1_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_26_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_2_1_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_23_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_3_]]{{.}} : memref<2x6x4xf32>
// CHECK:           }
// CHECK:           [[VAR_12_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[LOAD_RES_MEM_1_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]]){
// CHECK:             [[VAR_16_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<2x1x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<2x1x6xf32>
// CHECK-DAG:           [[VAR_21_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_26_2_:%.+]] = arith.andi [[VAR_21_2_]], [[VAR_23_2_]] : i1
// CHECK:               [[VAR_27_:%.+]] = arith.andi [[VAR_26_2_]], [[VAR_25_2_]] : i1
// CHECK:               scf.if [[VAR_27_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_16_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_16_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_34_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_34_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_36_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_36_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_38_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_38_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_0_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_1_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_2_1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.mulf [[VAR_43_]], [[VAR_44_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.mulf [[VAR_46_]], [[VAR_47_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[VAR_52_]], [[VAR_50_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addf [[VAR_45_]], [[VAR_48_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.subf [[VAR_58_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.addf [[VAR_59_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.divf [[VAR_57_]], [[VAR_60_]] : f32
// CHECK:                     [[VAR_62_:%.+]] = arith.cmpf oge, [[VAR_61_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_62_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_36_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_16_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_16_4_]]#0, [[VAR_16_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_16_4_]]#0, [[VAR_16_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_two_classes(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x2x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x6x4xf32>, tensor<1x2x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL:  func.func @test_nonmaxsuppression_two_classes
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<1x6x4xf32>, [[SCORES_:%.+]]: memref<1x2x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[CST_6_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]]){
// CHECK:             [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[VAR_21_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1, [[VAR_21_]]{{.}} : memref<1x2x6xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_25_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_1_]] : index
// CHECK:               [[VAR_26_:%.+]] = arith.select [[VAR_23_]], [[VAR_25_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_26_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_20_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_20_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x2x6xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to 6){
// CHECK:             [[VAR_16_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_16_1_]]#2, [[RES_3_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_16_1_]]#2] : memref<1x2x6xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x2x6xindex>, memref<1x2x6xf32>, i64, i64) -> ()
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 6){
// CHECK:             [[VAR_16_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[VAR_21_1_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = arith.select [[VAR_21_1_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.select [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = arith.cmpf ogt, [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOAD_RES_2_MEM_2_]], [[LOOP_1_]] : f32
// CHECK-DAG:         [[VAR_26_1_:%.+]] = arith.select [[LOAD_RES_1_MEM_2_]], [[LOOP_1_]], [[LOAD_RES_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_26_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_23_1_]], [[RES_4_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK:           [[VAR_12_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[LOAD_RES_MEM_1_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_5_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_1_1_]], [[LOOP_4_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_2_1_]]){
// CHECK:             [[VAR_16_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_7_]][] : memref<index>
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_8_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1, [[LOAD_RES_1_MEM_1_1_]]{{.}} : memref<1x2x6xindex>
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<1x2x6xf32>
// CHECK-DAG:           [[VAR_21_2_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_2_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_1_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_26_2_:%.+]] = arith.andi [[VAR_21_2_]], [[VAR_23_2_]] : i1
// CHECK:               [[VAR_27_:%.+]] = arith.andi [[VAR_26_2_]], [[VAR_25_2_]] : i1
// CHECK:               scf.if [[VAR_27_]] {
// CHECK-DAG:             [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_16_3_]]#0, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_16_3_]]#1, [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_1_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_RES_6_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_SCORES_MEM_1_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_7_]][] : memref<index>
// CHECK:                 [[VAR_34_:%.+]] = arith.addi [[LOAD_RES_6_MEM_]], [[CST_1_1_]] : index
// CHECK:                 krnl.store [[VAR_34_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_6_]]){
// CHECK:                   [[VAR_36_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_36_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_38_:%.+]] = arith.cmpi eq, [[LOAD_RES_8_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_38_]] {
// CHECK-DAG:                 [[LOAD_RES_4_MEM_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_0_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_5_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_1_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_6_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_2_1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_4_MEM_7_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_36_]], [[CST_3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_]] : f32
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.subf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.mulf [[VAR_43_]], [[VAR_44_]] : f32
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.subf [[LOAD_RES_4_MEM_6_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.subf [[LOAD_RES_4_MEM_7_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.mulf [[VAR_46_]], [[VAR_47_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_RES_4_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[LOAD_RES_4_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[LOAD_RES_4_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_4_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[VAR_51_]], [[VAR_49_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[VAR_52_]], [[VAR_50_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addf [[VAR_45_]], [[VAR_48_]] : f32
// CHECK:                     [[VAR_59_:%.+]] = arith.subf [[VAR_58_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_60_:%.+]] = arith.addf [[VAR_59_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_61_:%.+]] = arith.divf [[VAR_57_]], [[VAR_60_]] : f32
// CHECK:                     [[VAR_62_:%.+]] = arith.cmpf oge, [[VAR_61_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_62_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_8_]]{{.}}[[VAR_36_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc([[LOAD_RES_6_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_6_MEM_1_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_16_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_16_4_]]#0, [[VAR_16_4_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_5_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_9_]]{{.}}[[VAR_16_4_]]#0, [[VAR_16_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_9_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func.func @test_nonmaxsuppression_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @test_nonmaxsuppression_unknown_dims
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<?x?x?xf32>, [[SCORES_:%.+]]: memref<?x?x?xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> {
// CHECK-DAG:       [[CST_9_dot_99999993_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[SCORES_]], [[CST_0_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[SCORES_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[SCORES_]], [[CST_2_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_3_]], [[VAR_dim_3_]] : index
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[SCORES_]], [[CST_0_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[SCORES_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[SCORES_]], [[CST_2_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_1_]] to [[VAR_dim_4_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_1_]] to [[VAR_dim_5_]]){
// CHECK:             [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_1_]][] : memref<index>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_1_]] to [[VAR_dim_6_]]){
// CHECK:               [[VAR_21_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1, [[VAR_21_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               [[VAR_25_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:               [[VAR_26_:%.+]] = arith.select [[VAR_23_]], [[VAR_25_]], [[LOAD_RES_1_MEM_]] : index
// CHECK:               krnl.store [[VAR_26_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_20_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_2_MEM_]] : index
// CHECK:             krnl.store [[VAR_20_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:           [[VAR_8_:%.+]] = arith.minsi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK:           krnl.store [[VAR_8_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_dim_9_:%.+]] = memref.dim [[SCORES_]], [[CST_0_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_10_:%.+]] = memref.dim [[SCORES_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[SCORES_]], [[CST_2_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[VAR_dim_9_]], [[VAR_dim_10_]], [[VAR_dim_11_]]) {{.*}}: memref<?x?x?xindex>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_9_]]), [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_9_]], [[VAR_dim_10_]]), [[LOOP_2_]]#2 -> [[I_5_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_9_]], [[VAR_dim_10_]], [[VAR_dim_11_]])){
// CHECK:             [[VAR_16_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_16_1_]]#2, [[RES_3_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_16_1_]]#2] : memref<?x?x?xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_3_]], [[SCORES_]], [[CST_2_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<?x?x?xindex>, memref<?x?x?xf32>, i64, i64) -> ()
// CHECK:           [[VAR_11_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_2 : index
// CHECK:           [[VAR_12_:%.+]] = arith.muli [[VAR_11_]], [[LOAD_RES_MEM_1_]] : index
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_4_]], [[CST_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[VAR_dim_]], [[LOOP_3_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[VAR_dim_]]_2){
// CHECK:             [[VAR_16_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_1_]], [[RES_6_]][] : memref<index>
// CHECK:             [[RES_7_:%.+]] = memref.alloc([[VAR_dim_3_]]) {{.*}}: memref<?xi1>
// CHECK:             krnl.memset [[RES_7_]], [[VAR_false_]] : memref<?xi1>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_8_:%.+]] = [[CST_0_1_]] to [[VAR_dim_3_]]){
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<?x?x?xindex>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:           [[VAR_21_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_1_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_6_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_1_:%.+]] = arith.cmpi slt, [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_MEM_1_]] : index
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_7_]]{{.}}[[LOAD_RES_2_MEM_2_]]{{.}} : memref<?xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_1_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_2_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_26_1_:%.+]] = arith.andi [[VAR_21_1_]], [[VAR_23_1_]] : i1
// CHECK:               [[VAR_27_:%.+]] = arith.andi [[VAR_26_1_]], [[VAR_25_1_]] : i1
// CHECK:               scf.if [[VAR_27_]] {
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_0_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_2_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_2_MEM_2_]], [[CST_3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK:                 krnl.store [[VAR_16_2_]]#0, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_0_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_16_2_]]#1, [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_4_]]{{.}}[[LOAD_RES_5_MEM_]], [[CST_2_1_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[CST_1_]] : index
// CHECK:                 krnl.store [[VAR_33_]], [[RES_6_]][] : memref<index>
// CHECK:                 [[VAR_34_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[CST_1_]] : index
// CHECK:                 krnl.store [[VAR_34_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = [[CST_0_1_]] to [[VAR_dim_3_]]){
// CHECK:                   [[VAR_36_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_36_]]{{.}} : memref<?xi1>
// CHECK:                   [[VAR_38_:%.+]] = arith.cmpi eq, [[LOAD_RES_7_MEM_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_38_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_36_]], [[CST_0_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_36_]], [[CST_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_36_]], [[CST_2_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_36_]], [[CST_3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_43_]] : f32
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_45_]] : f32
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_47_]] : f32
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_49_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_51_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[CST_2_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.maxnumf [[VAR_44_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.maxnumf [[VAR_48_]], [[VAR_52_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.minnumf [[VAR_46_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.minnumf [[VAR_50_]], [[VAR_54_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.subf [[VAR_63_]], [[VAR_61_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.subf [[VAR_64_]], [[VAR_62_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.maxnumf [[VAR_65_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.maxnumf [[VAR_66_]], [[CST_0_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.mulf [[VAR_67_]], [[VAR_68_]] : f32
// CHECK-DAG:                 [[VAR_70_:%.+]] = arith.addf [[VAR_59_]], [[VAR_60_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.subf [[VAR_70_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_72_:%.+]] = arith.addf [[VAR_71_]], [[CST_9_dot_99999993_]] : f32
// CHECK:                     [[VAR_73_:%.+]] = arith.divf [[VAR_69_]], [[VAR_72_]] : f32
// CHECK:                     [[VAR_74_:%.+]] = arith.cmpf oge, [[VAR_73_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_74_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_7_]]{{.}}[[VAR_36_]]{{.}} : memref<?xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc([[LOAD_RES_5_MEM_1_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[LOAD_RES_5_MEM_1_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_3_]]){
// CHECK:             [[VAR_16_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_4_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1] : memref<?x3xindex>
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = arith.index_cast [[LOOP_4_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_8_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<?x3xi64>
// CHECK:         }
}
