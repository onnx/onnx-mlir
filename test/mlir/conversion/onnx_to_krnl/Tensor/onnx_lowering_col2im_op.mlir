// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test the basic lowering of Col2Im with static shapes.
func.func @test_col2im(%arg0 : tensor<1x5x5xf32>) -> tensor<1x1x5x5xf32> {
  %image_shape = onnx.Constant dense<[5, 5]> : tensor<2xi64>
  %block_shape = onnx.Constant dense<[1, 5]> : tensor<2xi64>
  %0 = "onnx.Col2Im"(%arg0, %image_shape, %block_shape) : (tensor<1x5x5xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (-s0 + 6)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-LABEL:  func.func @test_col2im
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x5x5xf32>) -> memref<1x1x5x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 5]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_0_]] : tensor<2xi64> to memref<2xi64>
// CHECK:           [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_3_]]{{.}}
// CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_5_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x5x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_10_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_10_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}){
// CHECK:               [[VAR_14_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_15_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]]#2, [[VAR_14_]]#0)
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.cmpi sge, [[VAR_15_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.cmpi slt, [[VAR_15_]], [[VAR_7_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.andi [[VAR_16_]], [[VAR_17_]] : i1
// CHECK-DAG:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]]#3, [[VAR_14_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.cmpi sge, [[VAR_19_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[VAR_8_]] : index
// CHECK:               [[VAR_22_:%.+]] = arith.andi [[VAR_20_]], [[VAR_21_]] : i1
// CHECK:               [[VAR_23_:%.+]] = arith.andi [[VAR_18_]], [[VAR_22_]] : i1
// CHECK:               scf.if [[VAR_23_]] {
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.muli [[VAR_14_]]#0, [[VAR_5_]] : index
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.muli [[VAR_15_]], [[VAR_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_26_:%.+]] = arith.addi [[VAR_24_]], [[VAR_14_]]#1 : index
// CHECK-DAG:             [[VAR_27_:%.+]] = arith.addi [[VAR_25_]], [[VAR_19_]] : index
// CHECK:                 [[VAR_28_:%.+]] = arith.addi [[VAR_11_]], [[VAR_26_]] : index
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_10_]]#0, [[VAR_28_]], [[VAR_27_]]{{.}} : memref<1x5x5xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:                 [[VAR_31_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:                 krnl.store [[VAR_31_]], [[RES_1_]][] : memref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_10_]]#0, [[VAR_10_]]#1, [[VAR_10_]]#2, [[VAR_10_]]#3] : memref<1x1x5x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x5x5xf32>
// CHECK:         }
}

// -----

// Test whether the lowering is correct in the presence of dynamic dimensions.
func.func @test_col2im_dynamic_dims(%arg0 : tensor<1x?x?xf32>, %image_shape : tensor<2xi64>, %block_shape : tensor<2xi64>) -> tensor<1x?x?x?xf32> {
  %0 = "onnx.Col2Im"(%arg0, %image_shape, %block_shape) : (tensor<1x?x?xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x?x?x?xf32>
  "func.return"(%0) : (tensor<1x?x?x?xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> ({{-s0 \+ s1 \+ 1|s1 - s0 \+ 1}})>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> ({{-d0 \+ d1|d0 - d1}})>
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
// CHECK-DAG:       [[VAR_15_:%.+]] = affine.apply [[MAP_0_]](){{.*}}
// CHECK-DAG:       [[VAR_16_:%.+]] = affine.apply [[MAP_0_]](){{.*}}
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
// CHECK:               [[VAR_23_:%.+]] = affine.apply [[MAP_5_]]({{.*}})
// CHECK-DAG:           [[VAR_24_:%.+]] = arith.cmpi sge, [[VAR_23_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_23_]], [[VAR_15_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.andi [[VAR_24_]], [[VAR_25_]] : i1
// CHECK-DAG:           [[VAR_27_:%.+]] = affine.apply [[MAP_5_]]({{.*}})
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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> ((-s0) floordiv 2 + 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 - d1 + 1)>
// CHECK-LABEL:  func.func @test_col2im_batch_strides_pads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x9x9xf32>) -> memref<2x1x6x6xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_0_]] : tensor<2xi64> to memref<2xi64>
// CHECK:           [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_3_]]{{.}}
// CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_5_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<2x1x6x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 6, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 6){
// CHECK-DAG:         [[VAR_10_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_10_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}){
// CHECK:               [[VAR_14_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_15_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]]#2, [[VAR_14_]]#0)
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.cmpi sge, [[VAR_15_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.remsi [[VAR_15_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.floordivsi [[VAR_15_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[VAR_7_]] : index
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.andi [[VAR_16_]], [[VAR_18_]] : i1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.andi [[VAR_21_]], [[VAR_20_]] : i1
// CHECK-DAG:           [[VAR_23_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]]#3, [[VAR_14_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_:%.+]] = arith.cmpi sge, [[VAR_23_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.remsi [[VAR_23_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_25_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.floordivsi [[VAR_23_]], [[CST_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[VAR_8_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK:               [[VAR_30_:%.+]] = arith.andi [[VAR_29_]], [[VAR_28_]] : i1
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_22_]], [[VAR_30_]] : i1
// CHECK:               scf.if [[VAR_31_]] {
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.muli [[VAR_14_]]#0, [[VAR_5_]] : index
// CHECK-DAG:             [[VAR_33_:%.+]] = arith.muli [[VAR_19_]], [[VAR_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.addi [[VAR_32_]], [[VAR_14_]]#1 : index
// CHECK-DAG:             [[VAR_35_:%.+]] = arith.addi [[VAR_33_]], [[VAR_27_]] : index
// CHECK:                 [[VAR_36_:%.+]] = arith.addi [[VAR_11_]], [[VAR_34_]] : index
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_10_]]#0, [[VAR_36_]], [[VAR_35_]]{{.}} : memref<2x9x9xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:                 [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:                 krnl.store [[VAR_39_]], [[RES_1_]][] : memref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_10_]]#0, [[VAR_10_]]#1, [[VAR_10_]]#2, [[VAR_10_]]#3] : memref<2x1x6x6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x1x6x6xf32>
// CHECK:         }
}
