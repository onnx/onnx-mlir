// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_det_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<f32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[CST_3_2_]], [[CST_3_2_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_5_]]{{.}} : memref<3x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[VAR_3_]], [[VAR_5_]]{{.}} : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_3_4_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_0_:%.+]] = 0 to 3) iter_args([[I_1_:%.+]] = [[CST_1_dot_000000_]]) -> (f32){
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:             [[LOOP_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_1_]], [[VAR_3_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = math.absf [[LOOP_1_]] : f32
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[CST_3_5_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]]:2 = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg3_:%.+]] = [[MAP_0_]]([[VAR_3_1_]]) to 3) iter_args([[VAR_arg4_:%.+]] = [[VAR_5_1_]][[VAR_arg5_:%.+]] = [[VAR_3_1_]]) -> (f32index){
// CHECK-DAG:           [[VAR_18_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_]], [[VAR_3_1_]]{{.}} : memref<?x?xf32>
// CHECK:               [[VAR_20_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:               [[VAR_21_:%.+]] = arith.cmpf ogt, [[VAR_20_]], [[VAR_arg4_]] : f32
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.select [[VAR_21_]], [[VAR_20_]], [[VAR_arg4_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.select [[VAR_21_]], [[VAR_18_]], [[VAR_arg5_]] : index
// CHECK:               krnl.yield [[VAR_22_]], [[VAR_23_]] : f32, index
// CHECK:             }
// CHECK-DAG:         [[CST_3_6_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_2_:%.+]] = 0 to 3){
// CHECK:               [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_1_]], [[VAR_18_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_8_]]#1, [[VAR_18_1_]]{{.}} : memref<?x?xf32>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_3_1_]], [[VAR_18_1_]]{{.}} : memref<?x?xf32>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_8_]]#1, [[VAR_18_1_]]{{.}} : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi ne, [[VAR_8_]]#1, [[VAR_3_1_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[CST_minus_1_dot_000000_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.mulf [[I_1_]], [[VAR_11_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_1_]], [[VAR_3_1_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.mulf [[VAR_12_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_3_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[CST_3_7_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_16_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_1_]])
// CHECK-DAG:         [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_3_:%.+]] = [[MAP_0_]]([[VAR_3_1_]]) to 3){
// CHECK:               [[VAR_18_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_2_]], [[VAR_3_1_]]{{.}} : memref<?x?xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_21_1_:%.+]] = arith.select [[VAR_15_]], [[VAR_20_1_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:           [[CST_3_8_:%.+]] = arith.constant 3 : index
// CHECK-DAG:           [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_4_:%.+]] = [[MAP_1_]]([[VAR_3_1_]]) to 3){
// CHECK:                 [[VAR_23_1_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_1_]], [[VAR_23_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_18_2_]], [[VAR_23_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_26_:%.+]] = arith.mulf [[VAR_21_1_]], [[LOAD_RES_1_MEM_5_]] : f32
// CHECK:                 [[VAR_27_:%.+]] = arith.subf [[LOAD_RES_1_MEM_6_]], [[VAR_26_]] : f32
// CHECK:                 krnl.store [[VAR_27_]], [[RES_1_]]{{.}}[[VAR_18_2_]], [[VAR_23_1_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             krnl.yield [[VAR_14_]] : f32
// CHECK:           }
// CHECK:           krnl.store [[VAR_2_]], [[RES_]][] : memref<f32>
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
func.func private @test_det_2d(%arg0 : tensor<3x3xf32>) -> tensor<f32> {
  %0 = "onnx.Det"(%arg0) : (tensor<3x3xf32>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()
}

// -----

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_det_nd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x4xf32>) -> memref<2xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2xf32>
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_4_3_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc([[CST_4_2_]], [[CST_4_2_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:           [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 4){
// CHECK:                 [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_5_]], [[VAR_7_]]{{.}} : memref<2x4x4xf32>
// CHECK:                 krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[VAR_5_]], [[VAR_7_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[CST_4_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_1_:%.+]] = 0 to 4) iter_args([[I_2_:%.+]] = [[CST_1_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOOP_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[VAR_7_1_:%.+]] = math.absf [[LOOP_2_]] : f32
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK-DAG:           [[CST_4_5_:%.+]] = arith.constant 4 : index
// CHECK-DAG:           [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_0_]]([[VAR_5_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[VAR_arg4_:%.+]] = [[MAP_0_]]([[VAR_5_1_]]) to 4) iter_args([[VAR_arg5_:%.+]] = [[VAR_7_1_]][[VAR_arg6_:%.+]] = [[VAR_5_1_]]) -> (f32index){
// CHECK-DAG:             [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_22_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:                 [[VAR_23_:%.+]] = arith.cmpf ogt, [[VAR_22_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[VAR_22_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[VAR_20_]], [[VAR_arg6_]] : index
// CHECK:                 krnl.yield [[VAR_24_]], [[VAR_25_]] : f32, index
// CHECK:               }
// CHECK-DAG:           [[CST_4_6_:%.+]] = arith.constant 4 : index
// CHECK-DAG:           [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_3_:%.+]] = 0 to 4){
// CHECK:                 [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_10_]]#1, [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_10_]]#1, [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:               [[VAR_12_:%.+]] = arith.cmpi ne, [[VAR_10_]]#1, [[VAR_5_1_]] : index
// CHECK:               [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[CST_minus_1_dot_000000_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.mulf [[I_2_]], [[VAR_13_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_3_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:           [[CST_4_7_:%.+]] = arith.constant 4 : index
// CHECK-DAG:           [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:           [[VAR_18_:%.+]] = affine.apply [[MAP_0_]]([[VAR_5_1_]])
// CHECK-DAG:           [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_4_:%.+]] = [[MAP_0_]]([[VAR_5_1_]]) to 4){
// CHECK:                 [[VAR_20_2_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_22_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:             [[VAR_23_1_:%.+]] = arith.select [[VAR_17_]], [[VAR_22_1_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:             [[CST_4_8_:%.+]] = arith.constant 4 : index
// CHECK-DAG:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_5_:%.+]] = [[MAP_1_]]([[VAR_5_1_]]) to 4){
// CHECK:                   [[VAR_25_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK:                   [[VAR_28_:%.+]] = arith.mulf [[VAR_23_1_]], [[LOAD_RES_1_MEM_5_]] : f32
// CHECK:                   [[VAR_29_:%.+]] = arith.subf [[LOAD_RES_1_MEM_6_]], [[VAR_28_]] : f32
// CHECK:                   krnl.store [[VAR_29_]], [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:               krnl.yield [[VAR_16_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2xf32>
// CHECK:         }
func.func private @test_det_nd(%arg0 : tensor<2x4x4xf32>) -> tensor<2xf32> {
  %0 = "onnx.Det"(%arg0) : (tensor<2x4x4xf32>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
}

// -----

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 + 1)>
// CHECK-LABEL:  func.func private @test_det_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>) -> memref<?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc([[VAR_dim_3_]], [[VAR_dim_3_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]){{.}}[[VAR_dim_3_]]{{.}}){
// CHECK-DAG:           [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]){{.}}[[VAR_dim_3_]]{{.}}){
// CHECK:                 [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_5_]], [[VAR_7_]]{{.}} : memref<?x?x?xf32>
// CHECK:                 krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[VAR_5_]], [[VAR_7_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]){{.}}[[VAR_dim_3_]]{{.}}) iter_args([[I_2_:%.+]] = [[CST_1_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOOP_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[VAR_7_1_:%.+]] = math.absf [[LOOP_2_]] : f32
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK-DAG:           [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_5_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[VAR_arg4_:%.+]] = [[MAP_2_]]([[VAR_5_1_]]) to [[MAP_1_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}}) iter_args([[VAR_arg5_:%.+]] = [[VAR_7_1_]][[VAR_arg6_:%.+]] = [[VAR_5_1_]]) -> (f32index){
// CHECK-DAG:             [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_22_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:                 [[VAR_23_:%.+]] = arith.cmpf ogt, [[VAR_22_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[VAR_22_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[VAR_20_]], [[VAR_arg6_]] : index
// CHECK:                 krnl.yield [[VAR_24_]], [[VAR_25_]] : f32, index
// CHECK:               }
// CHECK-DAG:           [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}}){
// CHECK:                 [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_10_]]#1, [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_10_]]#1, [[VAR_20_1_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:               [[VAR_12_:%.+]] = arith.cmpi ne, [[VAR_10_]]#1, [[VAR_5_1_]] : index
// CHECK:               [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[CST_minus_1_dot_000000_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.mulf [[I_2_]], [[VAR_13_]] : f32
// CHECK-DAG:           [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_3_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:           [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:           [[VAR_18_:%.+]] = affine.apply [[MAP_3_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:           [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_6_]]) with ([[LOOP_6_]] -> [[I_4_:%.+]] = [[MAP_3_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}} to [[MAP_1_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}}){
// CHECK:                 [[VAR_20_2_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_5_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_22_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_3_]] : f32
// CHECK-DAG:             [[VAR_23_1_:%.+]] = arith.select [[VAR_17_]], [[VAR_22_1_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_5_:%.+]] = [[MAP_0_]]([[VAR_5_1_]]) to [[MAP_1_]]([[VAR_5_1_]]){{.}}[[VAR_dim_3_]]{{.}}){
// CHECK:                   [[VAR_25_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK:                   [[VAR_28_:%.+]] = arith.mulf [[VAR_23_1_]], [[LOAD_RES_1_MEM_5_]] : f32
// CHECK:                   [[VAR_29_:%.+]] = arith.subf [[LOAD_RES_1_MEM_6_]], [[VAR_28_]] : f32
// CHECK:                   krnl.store [[VAR_29_]], [[RES_1_]]{{.}}[[VAR_20_2_]], [[VAR_25_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:               krnl.yield [[VAR_16_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
func.func private @test_det_dynamic(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32> {
  %0 = "onnx.Det"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()
}
