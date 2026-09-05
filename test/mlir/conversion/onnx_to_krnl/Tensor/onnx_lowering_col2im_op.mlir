// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test the basic lowering of Col2Im with static shapes.
func.func @test_col2im(%arg0 : tensor<1x5x5xf32>) -> tensor<1x1x5x5xf32> {
  %image_shape = onnx.Constant dense<[5, 5]> : tensor<2xi64>
  %block_shape = onnx.Constant dense<[1, 5]> : tensor<2xi64>
  %0 = "onnx.Col2Im"(%arg0, %image_shape, %block_shape) : (tensor<1x5x5xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (-d0 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (-d0 + d1 - d2 + d3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 5 + d1 * 5 + d2)>
// CHECK-LABEL:  func.func @test_col2im
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x5x5xf32>) -> memref<1x1x5x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x5x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_5_:%.+]] = affine.apply [[MAP_0_]]([[VAR_4_]]#0, [[VAR_1_]]#2)
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_5_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_0_]]([[VAR_4_]]#1, [[VAR_1_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.cmpi sge, [[VAR_9_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_9_]], [[CST_1_]] : index
// CHECK:               [[VAR_12_:%.+]] = arith.andi [[VAR_10_]], [[VAR_11_]] : i1
// CHECK:               [[VAR_13_:%.+]] = arith.andi [[VAR_8_]], [[VAR_12_]] : i1
// CHECK:               scf.if [[VAR_13_]] {
// CHECK-DAG:             [[VAR_14_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_]]#1, [[VAR_1_]]#3, [[VAR_4_]]#0, [[VAR_1_]]#2)
// CHECK-DAG:             [[VAR_15_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_4_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_15_]], [[VAR_14_]]{{.}} : memref<1x5x5xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:                 [[VAR_18_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:                 krnl.store [[VAR_18_]], [[RES_1_]][] : memref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x1x5x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x5x5xf32>
// CHECK:         }
}

// -----

// Test whether the lowering is correct in the presence of dynamic dimensions.
func.func @test_col2im_dynamic_dims(%arg0 : tensor<1x?x?xf32>, %image_shape : tensor<2xi64>, %block_shape : tensor<2xi64>) -> tensor<1x?x?x?xf32> {
  %0 = "onnx.Col2Im"(%arg0, %image_shape, %block_shape) : (tensor<1x?x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x?x?x?xf32>
  "func.return"(%0) : (tensor<1x?x?x?xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (-s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (-d0 + d1)>
// CHECK-LABEL:  func.func @test_col2im_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?xf32>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<1x?x?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_1_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.floordivsi [[VAR_dim_]], [[VAR_4_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_2_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_3_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.muli [[VAR_13_]], [[VAR_11_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_11_]], [[VAR_7_]]{{.}}
// CHECK-DAG:       [[VAR_16_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_13_]], [[VAR_9_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_5_]], [[VAR_7_]], [[VAR_9_]]) {{.*}}: memref<1x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_5_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_7_]], [[VAR_9_]]){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_7_]], [[VAR_9_]]){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}){
// CHECK-DAG:         [[VAR_18_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.muli [[VAR_18_]]#1, [[VAR_14_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_3_]]([[VAR_7_]], [[VAR_9_]]){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to [[MAP_4_]]([[VAR_7_]], [[VAR_9_]]){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}){
// CHECK:               [[VAR_22_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_23_:%.+]] = affine.apply [[MAP_5_]]([[VAR_22_]]#0, [[VAR_18_]]#2)
// CHECK-DAG:           [[VAR_24_:%.+]] = arith.cmpi sge, [[VAR_23_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_23_]], [[VAR_15_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.andi [[VAR_24_]], [[VAR_25_]] : i1
// CHECK-DAG:           [[VAR_27_:%.+]] = affine.apply [[MAP_5_]]([[VAR_22_]]#1, [[VAR_18_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi sge, [[VAR_27_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[VAR_16_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.andi [[VAR_28_]], [[VAR_29_]] : i1
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_26_]], [[VAR_30_]] : i1
// CHECK:               scf.if [[VAR_31_]] {
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.muli [[VAR_22_]]#0, [[VAR_13_]] : index
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.muli [[VAR_23_]], [[VAR_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.addi [[VAR_32_]], [[VAR_22_]]#1 : index
// CHECK-DAG:             [[VAR_35_:%.+]] = arith.addi [[VAR_33_]], [[VAR_27_]] : index
// CHECK:                 [[VAR_36_:%.+]] = arith.addi [[VAR_19_]], [[VAR_34_]] : index
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_18_]]#0, [[VAR_36_]], [[VAR_35_]]{{.}} : memref<1x?x?xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:                 [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:                 krnl.store [[VAR_39_]], [[RES_1_]][] : memref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2, [[VAR_18_]]#3] : memref<1x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x?x?x?xf32>
// CHECK:         }
}

// -----

// Test the combination of a batch size > 1 with non-default strides and
// pads together (each attribute path was only exercised individually by the
// ONNX backend test suite; this exercises them jointly).
// image_shape=[6,6], block_shape=[3,3], strides=[2,2], pads=[1,1,1,1]:
// gridDim = floor((6+1+1-1*(3-1)-1)/2)+1 = 3 per axis, so L = 9 and
// C*prod(block_shape) = 1*9 = 9, giving input shape [2,9,9] and output
// shape [2,1,6,6].
func.func @test_col2im_batch_strides_pads(%arg0 : tensor<2x9x9xf32>) -> tensor<2x1x6x6xf32> {
  %image_shape = onnx.Constant dense<[6, 6]> : tensor<2xi64>
  %block_shape = onnx.Constant dense<[3, 3]> : tensor<2xi64>
  %0 = "onnx.Col2Im"(%arg0, %image_shape, %block_shape) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<2x9x9xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x1x6x6xf32>
  "func.return"(%0) : (tensor<2x1x6x6xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 - d1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-LABEL:  func.func @test_col2im_batch_strides_pads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x9x9xf32>) -> memref<2x1x6x6xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x1x6x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 6){
// CHECK-DAG:         [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_5_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#2, [[VAR_4_]]#0)
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.remsi [[VAR_5_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.cmpi eq, [[VAR_7_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.floordivsi [[VAR_5_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_9_]], [[CST_3_]] : index
// CHECK-DAG:           [[VAR_11_:%.+]] = arith.andi [[VAR_6_]], [[VAR_8_]] : i1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_12_:%.+]] = arith.andi [[VAR_11_]], [[VAR_10_]] : i1
// CHECK-DAG:           [[VAR_13_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#3, [[VAR_4_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.cmpi sge, [[VAR_13_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.remsi [[VAR_13_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.cmpi eq, [[VAR_15_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.floordivsi [[VAR_13_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.cmpi slt, [[VAR_17_]], [[CST_3_]] : index
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.andi [[VAR_14_]], [[VAR_16_]] : i1
// CHECK:               [[VAR_20_:%.+]] = arith.andi [[VAR_19_]], [[VAR_18_]] : i1
// CHECK:               [[VAR_21_:%.+]] = arith.andi [[VAR_12_]], [[VAR_20_]] : i1
// CHECK:               scf.if [[VAR_21_]] {
// CHECK-DAG:             [[VAR_22_:%.+]] = affine.apply [[MAP_1_]]([[VAR_9_]], [[VAR_17_]])
// CHECK-DAG:             [[VAR_23_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_4_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[VAR_22_]]{{.}} : memref<2x9x9xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:                 [[VAR_26_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:                 krnl.store [[VAR_26_]], [[RES_1_]][] : memref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x1x6x6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x1x6x6xf32>
// CHECK:         }
}
