// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_16_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_16_]]#2, [[RES_1_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1, [[VAR_16_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_16_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_16_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_18_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_16_1_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_18_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[VAR_23_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_23_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_16_1_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1, [[VAR_18_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_2_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_3_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_3_]][] : memref<index>
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_3_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_16_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:             [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_5_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_23_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[VAR_23_1_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.cmpi slt, [[LOAD_RES_4_MEM_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_23_1_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi eq, [[LOAD_RES_5_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.andi [[VAR_25_]], [[VAR_27_]] : i1
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_30_]], [[VAR_29_]] : i1
// CHECK:               scf.if [[VAR_31_]] {
// CHECK:                 [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_16_2_]]#0, [[VAR_16_2_]]#1, [[VAR_23_1_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.muli [[VAR_16_2_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_38_:%.+]] = arith.addi [[VAR_16_2_]]#0, [[VAR_37_]] : index
// CHECK:                 [[VAR_39_:%.+]] = arith.addi [[VAR_38_]], [[LOAD_RES_4_MEM_]] : index
// CHECK:                 krnl.store [[VAR_16_2_]]#0, [[RES_2_]]{{.}}[[VAR_39_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_16_2_]]#1, [[RES_2_]]{{.}}[[VAR_39_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_2_]]{{.}}[[VAR_39_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_40_:%.+]] = affine.apply #map([[LOAD_RES_4_MEM_]])
// CHECK:                 krnl.store [[VAR_40_]], [[RES_4_]][] : memref<index>
// CHECK:                 [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_42_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_42_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_44_:%.+]] = arith.cmpi eq, [[LOAD_RES_5_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_44_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_42_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_42_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_42_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_16_2_]]#0, [[VAR_42_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_49_]] : f32
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_51_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_59_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_61_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_63_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = maxf [[VAR_50_]], [[VAR_62_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = maxf [[VAR_54_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_69_:%.+]] = minf [[VAR_52_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_70_:%.+]] = minf [[VAR_56_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_71_:%.+]] = arith.subf [[VAR_69_]], [[VAR_67_]] : f32
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.subf [[VAR_70_]], [[VAR_68_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_73_:%.+]] = maxf [[VAR_71_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_74_:%.+]] = maxf [[VAR_72_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.mulf [[VAR_73_]], [[VAR_74_]] : f32
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.addf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_77_:%.+]] = arith.subf [[VAR_76_]], [[VAR_75_]] : f32
// CHECK:                     [[VAR_78_:%.+]] = arith.addf [[VAR_77_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_79_:%.+]] = arith.divf [[VAR_75_]], [[VAR_78_]] : f32
// CHECK:                     [[VAR_80_:%.+]] = arith.cmpf ogt, [[VAR_79_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_80_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_5_]]{{.}}[[VAR_42_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[RES_3_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_3_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<index>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc([[LOAD_RES_3_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_3_MEM_]], [[LOOP_6_]]#1 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_16_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_18_1_:%.+]] = arith.index_cast [[RES_4_]] : index to i64
// CHECK:             krnl.store [[VAR_18_1_]], [[RES_6_]]{{.}}[[VAR_16_3_]]#0, [[VAR_16_3_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c10_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x10xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 10) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x10xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 9) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 10) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x10xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x10xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x10xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x10xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x10xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x10xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x10x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 10) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<10xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<10xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c10_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x10xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<10xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x10xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c10_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<10xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x10x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<10xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c1_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x1xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 0) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 1) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x1xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x1xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x1xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x1xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x1xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x1xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x1x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 1) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<1xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x1xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<1xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x1xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<1xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x1x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<1xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_18_]]#2, [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<1x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_18_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map([[VAR_18_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x1x6xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_25_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_18_1_]]#2] : memref<1x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1, [[VAR_20_]]{{.}} : memref<1x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_18_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_20_1_]] : f32
// CHECK-DAG:         [[VAR_25_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_20_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = select [[VAR_26_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_26_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_25_1_]], [[RES_2_]]{{.}}[[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xf32>
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_28_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_25_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.andi [[VAR_27_1_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_33_:%.+]] = arith.andi [[VAR_32_]], [[VAR_31_]] : i1
// CHECK:               scf.if [[VAR_33_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1, [[VAR_25_2_]]{{.}} : memref<1x1x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.muli [[VAR_18_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_40_:%.+]] = arith.addi [[VAR_18_3_]]#0, [[VAR_39_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[VAR_28_1_]] : index
// CHECK:                 krnl.store [[VAR_18_3_]]#0, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_18_3_]]#1, [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_41_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply #map([[VAR_28_1_]])
// CHECK:                 krnl.store [[VAR_42_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_44_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_46_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_46_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_3_]]#0, [[VAR_44_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_55_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.subf [[VAR_59_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = maxf [[VAR_61_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addf [[VAR_53_]], [[VAR_56_]] : f32
// CHECK:                     [[VAR_67_:%.+]] = arith.subf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.divf [[VAR_65_]], [[VAR_68_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.cmpf ogt, [[VAR_69_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_70_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_44_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_20_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_7_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_two_batches(%arg0: tensor<2x6x4xf32>, %arg1: tensor<2x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x6x4xf32>, tensor<2x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

// CHECK-DAG: #map0 = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_nonmaxsuppression_two_batches
// CHECK-SAME:   ([[BOXES_:%.+]]: memref<2x6x4xf32>, [[SCORES_:%.+]]: memref<2x1x6xf32>, [[MAX_OUTPUT_BOXES_PER_CLASS_:%.+]]: memref<1xi64>, [[IOU_THRESHOLD_:%.+]]: memref<1xf32>, [[SCORE_THRESHOLD_:%.+]]: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 9.99999993E-9 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x1x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_19_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_19_]]#2, [[RES_1_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1, [[VAR_19_]]#2] : memref<2x1x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_19_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map0([[VAR_19_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_21_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_19_1_]]#2] : memref<2x1x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_21_]]{{.}} : memref<2x1x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<2x1x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<2x1x6xf32>
// CHECK:               [[VAR_26_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_26_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_19_1_]]#2] : memref<2x1x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_21_]]{{.}} : memref<2x1x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_19_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[VAR_21_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_21_1_]] : f32
// CHECK-DAG:         [[VAR_26_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_27_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = select [[VAR_27_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_29_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK:             krnl.store [[VAR_26_1_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[VAR_12_:%.+]] = affine.apply #map1(){{.}}[[LOAD_RES_MEM_]]{{.}}
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]]) {
// CHECK-DAG:         [[VAR_19_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_26_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_3_]]#0, [[VAR_19_3_]]#1, [[VAR_26_2_]]{{.}} : memref<2x1x6xf32>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_26_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.andi [[VAR_28_1_]], [[VAR_30_]] : i1
// CHECK:               [[VAR_34_:%.+]] = arith.andi [[VAR_33_]], [[VAR_32_]] : i1
// CHECK:               scf.if [[VAR_34_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_3_]]#0, [[VAR_19_3_]]#1, [[VAR_26_2_]]{{.}} : memref<2x1x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = affine.apply #map2([[VAR_19_3_]]#0)
// CHECK-DAG:             [[VAR_41_:%.+]] = arith.muli [[VAR_19_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_40_]], [[VAR_41_]] : index
// CHECK:                 [[VAR_43_:%.+]] = arith.addi [[VAR_42_]], [[VAR_29_1_]] : index
// CHECK:                 krnl.store [[VAR_19_3_]]#0, [[RES_3_]]{{.}}[[VAR_43_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_19_3_]]#1, [[RES_3_]]{{.}}[[VAR_43_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_43_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_44_:%.+]] = affine.apply #map0([[VAR_29_1_]])
// CHECK:                 krnl.store [[VAR_44_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_46_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_48_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_48_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_46_]], [[VAR_c0_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_46_]], [[VAR_c1_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_46_]], [[VAR_c2_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_46_]], [[VAR_c3_]]{{.}} : memref<2x6x4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.mulf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.mulf [[VAR_56_]], [[VAR_57_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.subf [[VAR_62_]], [[VAR_60_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = maxf [[VAR_64_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.mulf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.addf [[VAR_55_]], [[VAR_58_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.subf [[VAR_68_]], [[VAR_67_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.addf [[VAR_69_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.divf [[VAR_67_]], [[VAR_70_]] : f32
// CHECK:                     [[VAR_72_:%.+]] = arith.cmpf ogt, [[VAR_71_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_72_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_46_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_19_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_19_4_]]#0, [[VAR_19_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_21_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_21_2_]], [[RES_7_]]{{.}}[[VAR_19_4_]]#0, [[VAR_19_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
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
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_3_:%.+]] = minui [[VAR_2_]], [[VAR_c6_]] : index
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x6xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_19_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_19_]]#2, [[RES_1_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1, [[VAR_19_]]#2] : memref<1x2x6xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[VAR_19_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map0([[VAR_19_1_]]#2) to 6) {
// CHECK-DAG:           [[VAR_21_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_19_1_]]#2] : memref<1x2x6xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_21_]]{{.}} : memref<1x2x6xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<1x2x6xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<1x2x6xf32>
// CHECK:               [[VAR_26_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_26_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_19_1_]]#2] : memref<1x2x6xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_21_]]{{.}} : memref<1x2x6xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x6x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_8_:%.+]] = 0 to 6) {
// CHECK:             [[VAR_19_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[VAR_21_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:             [[LOAD_SCORES_MEM_2_:%.+]] = arith.cmpf ogt, [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_SCORES_MEM_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_21_1_]] : f32
// CHECK-DAG:         [[VAR_26_1_:%.+]] = select [[LOAD_SCORES_MEM_2_]], [[VAR_21_1_]], [[LOAD_RES_1_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpf ogt, [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = select [[VAR_27_]], [[LOAD_RES_1_MEM_2_]], [[LOOP_2_]] : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = select [[VAR_27_]], [[LOOP_2_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_28_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_29_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK:             krnl.store [[VAR_26_1_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[VAR_12_:%.+]] = affine.apply #map1(){{.}}[[LOAD_RES_MEM_]]{{.}}
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:           [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c1_]], [[LOOP_4_]]#1 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]]) {
// CHECK-DAG:         [[VAR_19_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<6xi1>
// CHECK:             krnl.memset [[RES_6_]], [[VAR_false_]] : memref<6xi1>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:               [[VAR_26_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_19_3_]]#0, [[VAR_19_3_]]#1, [[VAR_26_2_]]{{.}} : memref<1x2x6xf32>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_3_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_26_2_]]{{.}} : memref<6xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.andi [[VAR_28_1_]], [[VAR_30_]] : i1
// CHECK:               [[VAR_34_:%.+]] = arith.andi [[VAR_33_]], [[VAR_32_]] : i1
// CHECK:               scf.if [[VAR_34_]] {
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_19_3_]]#0, [[VAR_19_3_]]#1, [[VAR_26_2_]]{{.}} : memref<1x2x6xindex>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[LOAD_RES_1_MEM_3_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.muli [[VAR_19_3_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_41_:%.+]] = arith.addi [[VAR_19_3_]]#0, [[VAR_40_]] : index
// CHECK:                 [[VAR_42_:%.+]] = arith.addi [[VAR_41_]], [[VAR_29_1_]] : index
// CHECK:                 krnl.store [[VAR_19_3_]]#0, [[RES_3_]]{{.}}[[VAR_42_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_19_3_]]#1, [[RES_3_]]{{.}}[[VAR_42_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_3_]], [[RES_3_]]{{.}}[[VAR_42_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_43_:%.+]] = affine.apply #map0([[VAR_29_1_]])
// CHECK:                 krnl.store [[VAR_43_]], [[RES_5_]][] : memref<index>
// CHECK:                 [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c6_]]) {
// CHECK:                   [[VAR_45_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                   [[VAR_47_:%.+]] = arith.cmpi eq, [[LOAD_RES_6_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_47_]] {
// CHECK-DAG:                 [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_45_]], [[VAR_c0_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_45_]], [[VAR_c1_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_45_]], [[VAR_c2_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[LOAD_RES_2_MEM_7_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_3_]]#0, [[VAR_45_]], [[VAR_c3_]]{{.}} : memref<1x6x4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.subf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.subf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.subf [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.subf [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_57_:%.+]] = arith.mulf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK-DAG:                 [[VAR_58_:%.+]] = maxf [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_2_MEM_5_]] : f32
// CHECK-DAG:                 [[VAR_59_:%.+]] = maxf [[LOAD_RES_2_MEM_]], [[LOAD_RES_2_MEM_4_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = minf [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_2_MEM_7_]] : f32
// CHECK-DAG:                 [[VAR_61_:%.+]] = minf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_2_MEM_6_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.subf [[VAR_60_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[VAR_61_]], [[VAR_59_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_64_:%.+]] = maxf [[VAR_62_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_65_:%.+]] = maxf [[VAR_63_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addf [[VAR_54_]], [[VAR_57_]] : f32
// CHECK:                     [[VAR_68_:%.+]] = arith.subf [[VAR_67_]], [[VAR_66_]] : f32
// CHECK:                     [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_70_:%.+]] = arith.divf [[VAR_66_]], [[VAR_69_]] : f32
// CHECK:                     [[VAR_71_:%.+]] = arith.cmpf ogt, [[VAR_70_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_71_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_6_]]{{.}}[[VAR_45_]]{{.}} : memref<6xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_2_]], [[LOAD_RES_1_MEM_1_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_1_]], [[RES_4_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc([[LOAD_RES_4_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_4_MEM_]], [[LOOP_7_]]#1 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_19_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_19_4_]]#0, [[VAR_19_4_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_21_2_:%.+]] = arith.index_cast [[RES_5_]] : index to i64
// CHECK:             krnl.store [[VAR_21_2_]], [[RES_7_]]{{.}}[[VAR_19_4_]]#0, [[VAR_19_4_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<?x3xi64>
// CHECK:         }
}

// -----

func @test_nonmaxsuppression_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
  %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"]'
// CHECK-DAG: #map0 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #map2 = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #map4 = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #map5 = affine_map<(d0, d1, d2) -> (d2 - 1)>
// CHECK-DAG: #map6 = affine_map<(d0, d1, d2, d3) -> (d3 + 1)>
// CHECK-DAG: #map7 = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-DAG: #map8 = affine_map<(d0) -> (d0 + 1)>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[SCORES_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[SCORES_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[SCORES_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_:%.+]] = krnl.load [[MAX_OUTPUT_BOXES_PER_CLASS_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_MAX_OUTPUT_BOXES_PER_CLASS_MEM_]] : i64 to index
// CHECK:           [[VAR_6_:%.+]] = minui [[VAR_5_]], [[VAR_2_]] : index
// CHECK:           krnl.store [[VAR_6_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_7_:%.+]] = memref.dim [[SCORES_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = memref.dim [[SCORES_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = memref.dim [[SCORES_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) {{.*}}: memref<?x?x?xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map0([[VAR_7_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map1([[VAR_7_]], [[VAR_8_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to #map2([[VAR_7_]], [[VAR_8_]], [[VAR_9_]])) {
// CHECK:             [[VAR_24_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[VAR_24_]]#2, [[RES_1_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1, [[VAR_24_]]#2] : memref<?x?x?xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to #map3([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to #map4([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]), [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to #map5([[VAR_7_]], [[VAR_8_]], [[VAR_9_]])) {
// CHECK-DAG:         [[VAR_24_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_6_:%.+]] = #map6([[VAR_7_]], [[VAR_8_]], [[VAR_9_]], [[VAR_24_1_]]#2) to #map7([[VAR_7_]], [[VAR_8_]], [[VAR_9_]], [[VAR_24_1_]]#2)) {
// CHECK-DAG:           [[VAR_26_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_24_1_]]#2] : memref<?x?x?xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_26_]]{{.}} : memref<?x?x?xindex>
// CHECK-DAG:           [[LOAD_SCORES_MEM_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[LOAD_RES_1_MEM_]]{{.}} : memref<?x?x?xf32>
// CHECK:               [[LOAD_SCORES_MEM_1_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[LOAD_RES_1_MEM_1_]]{{.}} : memref<?x?x?xf32>
// CHECK:               [[VAR_31_:%.+]] = arith.cmpf olt, [[LOAD_SCORES_MEM_]], [[LOAD_SCORES_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_31_]] {
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_24_1_]]#2] : memref<?x?x?xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_]], [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_26_]]{{.}} : memref<?x?x?xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[LOAD_SCORE_THRESHOLD_MEM_:%.+]] = krnl.load [[SCORE_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_IOU_THRESHOLD_MEM_:%.+]] = krnl.load [[IOU_THRESHOLD_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.muli [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[LOAD_RES_MEM_]] : index
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_17_]]) {{.*}}: memref<?x3xindex>
// CHECK:           krnl.memset [[RES_2_]], [[VAR_c_minus_1_]] : memref<?x3xindex>
// CHECK:           [[RES_3_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_3_]][] : memref<index>
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_0_]], [[LOOP_3_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_1_]]) {
// CHECK-DAG:         [[VAR_24_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK:             krnl.store [[VAR_c0_]], [[RES_4_]][] : memref<index>
// CHECK:             [[RES_5_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?xi1>
// CHECK:             krnl.memset [[RES_5_]], [[VAR_false_]] : memref<?xi1>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_2_]]) {
// CHECK:               [[VAR_31_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_SCORES_MEM_2_:%.+]] = krnl.load [[SCORES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_31_1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpf ogt, [[LOAD_SCORES_MEM_2_]], [[LOAD_SCORE_THRESHOLD_MEM_]] : f32
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.cmpi slt, [[LOAD_RES_4_MEM_]], [[LOAD_RES_MEM_]] : index
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_31_1_]]{{.}} : memref<?xi1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.cmpi eq, [[LOAD_RES_5_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK:               [[VAR_39_:%.+]] = arith.andi [[VAR_38_]], [[VAR_37_]] : i1
// CHECK:               scf.if [[VAR_39_]] {
// CHECK:                 [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1, [[VAR_31_1_]]{{.}} : memref<?x?x?xindex>
// CHECK-DAG:             [[LOAD_BOXES_MEM_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c0_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_1_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_2_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c2_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[LOAD_BOXES_MEM_3_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[LOAD_RES_1_MEM_2_]], [[VAR_c3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:             [[VAR_45_:%.+]] = arith.muli [[VAR_24_2_]]#0, [[VAR_0_]] : index
// CHECK-DAG:             [[VAR_46_:%.+]] = arith.muli [[VAR_24_2_]]#1, [[LOAD_RES_MEM_]] : index
// CHECK:                 [[VAR_47_:%.+]] = arith.addi [[VAR_45_]], [[VAR_46_]] : index
// CHECK:                 [[VAR_48_:%.+]] = arith.addi [[VAR_47_]], [[LOAD_RES_4_MEM_]] : index
// CHECK:                 krnl.store [[VAR_24_2_]]#0, [[RES_2_]]{{.}}[[VAR_48_]], [[VAR_c0_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[VAR_24_2_]]#1, [[RES_2_]]{{.}}[[VAR_48_]], [[VAR_c1_]]{{.}} : memref<?x3xindex>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_2_]]{{.}}[[VAR_48_]], [[VAR_c2_]]{{.}} : memref<?x3xindex>
// CHECK:                 [[VAR_49_:%.+]] = affine.apply #map8([[LOAD_RES_4_MEM_]])
// CHECK:                 krnl.store [[VAR_49_]], [[RES_4_]][] : memref<index>
// CHECK:                 [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_2_]]) {
// CHECK:                   [[VAR_51_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_51_]]{{.}} : memref<?xi1>
// CHECK:                   [[VAR_53_:%.+]] = arith.cmpi eq, [[LOAD_RES_5_MEM_1_]], [[VAR_false_]] : i1
// CHECK:                   scf.if [[VAR_53_]] {
// CHECK-DAG:                 [[LOAD_BOXES_MEM_4_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_51_]], [[VAR_c0_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_5_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_51_]], [[VAR_c1_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_6_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_51_]], [[VAR_c2_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[LOAD_BOXES_MEM_7_:%.+]] = krnl.load [[BOXES_]]{{.}}[[VAR_24_2_]]#0, [[VAR_51_]], [[VAR_c3_]]{{.}} : memref<?x?x?xf32>
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.subf [[LOAD_BOXES_MEM_]], [[VAR_58_]] : f32
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.divf [[LOAD_BOXES_MEM_2_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.addf [[LOAD_BOXES_MEM_]], [[VAR_60_]] : f32
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.subf [[LOAD_BOXES_MEM_1_]], [[VAR_62_]] : f32
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.divf [[LOAD_BOXES_MEM_3_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_65_:%.+]] = arith.addf [[LOAD_BOXES_MEM_1_]], [[VAR_64_]] : f32
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.subf [[LOAD_BOXES_MEM_5_]], [[VAR_66_]] : f32
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.divf [[LOAD_BOXES_MEM_7_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.addf [[LOAD_BOXES_MEM_5_]], [[VAR_68_]] : f32
// CHECK-DAG:                 [[VAR_70_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_71_:%.+]] = arith.subf [[LOAD_BOXES_MEM_4_]], [[VAR_70_]] : f32
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.divf [[LOAD_BOXES_MEM_6_]], [[VAR_cst_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_73_:%.+]] = arith.addf [[LOAD_BOXES_MEM_4_]], [[VAR_72_]] : f32
// CHECK-DAG:                 [[VAR_74_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_3_]], [[LOAD_BOXES_MEM_2_]] : f32
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.mulf [[LOAD_BOXES_MEM_7_]], [[LOAD_BOXES_MEM_6_]] : f32
// CHECK-DAG:                 [[VAR_76_:%.+]] = maxf [[VAR_59_]], [[VAR_71_]] : f32
// CHECK-DAG:                 [[VAR_77_:%.+]] = maxf [[VAR_63_]], [[VAR_67_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_78_:%.+]] = minf [[VAR_61_]], [[VAR_73_]] : f32
// CHECK-DAG:                 [[VAR_79_:%.+]] = minf [[VAR_65_]], [[VAR_69_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.subf [[VAR_78_]], [[VAR_76_]] : f32
// CHECK-DAG:                 [[VAR_81_:%.+]] = arith.subf [[VAR_79_]], [[VAR_77_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_82_:%.+]] = maxf [[VAR_80_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:                 [[VAR_83_:%.+]] = maxf [[VAR_81_]], [[VAR_cst_0_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_84_:%.+]] = arith.mulf [[VAR_82_]], [[VAR_83_]] : f32
// CHECK-DAG:                 [[VAR_85_:%.+]] = arith.addf [[VAR_74_]], [[VAR_75_]] : f32
// CHECK:                     [[VAR_86_:%.+]] = arith.subf [[VAR_85_]], [[VAR_84_]] : f32
// CHECK:                     [[VAR_87_:%.+]] = arith.addf [[VAR_86_]], [[VAR_cst_]] : f32
// CHECK:                     [[VAR_88_:%.+]] = arith.divf [[VAR_84_]], [[VAR_87_]] : f32
// CHECK:                     [[VAR_89_:%.+]] = arith.cmpf ogt, [[VAR_88_]], [[LOAD_IOU_THRESHOLD_MEM_]] : f32
// CHECK:                     scf.if [[VAR_89_]] {
// CHECK:                       krnl.store [[VAR_true_]], [[RES_5_]]{{.}}[[VAR_51_]]{{.}} : memref<?xi1>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK-DAG:         [[LOAD_SCORES_MEM_3_:%.+]] = krnl.load [[RES_3_]][] : memref<index>
// CHECK:             [[LOAD_SCORES_MEM_1_:%.+]] = arith.addi [[LOAD_SCORES_MEM_3_]], [[LOAD_RES_4_MEM_1_]] : index
// CHECK:             krnl.store [[LOAD_SCORES_MEM_1_]], [[RES_3_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<index>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc([[LOAD_RES_3_MEM_]]) {{.*}}: memref<?x3xi64>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[LOAD_RES_3_MEM_]], [[LOOP_6_]]#1 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]) {
// CHECK:             [[VAR_24_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1] : memref<?x3xindex>
// CHECK:             [[VAR_26_1_:%.+]] = arith.index_cast [[RES_4_]] : index to i64
// CHECK:             krnl.store [[VAR_26_1_]], [[RES_6_]]{{.}}[[VAR_24_3_]]#0, [[VAR_24_3_]]#1] : memref<?x3xi64>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<?x3xi64>
// CHECK:         }
}
