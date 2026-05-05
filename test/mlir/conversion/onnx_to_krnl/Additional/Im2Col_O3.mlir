// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----



// Test basic Im2Col with static shapes, no padding, unit stride
func.func @test_im2col_basic(%arg0: tensor<1x2x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 9)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 9)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 3)>
// CHECK-LABEL:  func.func @test_im2col_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf32>) -> memref<1x18x9xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 9){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_5_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x18x9xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_5_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_5_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x18x9xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf32>
// CHECK:         }
// -----



// Test Im2Col with padding
func.func @test_im2col_with_padding(%arg0: tensor<1x3x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 16) floordiv 4 - 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 4 - 1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 16) floordiv 4 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 4 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 16) floordiv 4 - 1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 4 - 1)>
// CHECK-LABEL:  func.func @test_im2col_with_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x4x4xf32>) -> memref<1x27x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x27x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_4_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x3x4x4xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x27x16xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_4_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_4_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x3x4x4xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x27x16xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x27x16xf32>
// CHECK:         }
// -----



// Test Im2Col with strides
func.func @test_im2col_with_strides(%arg0: tensor<2x1x7x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], strides = [2, 2]} : (tensor<2x1x7x7xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 9)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 9)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (((d0 mod 9) floordiv 3) * 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 3) * 6)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (((d0 mod 9) floordiv 3) * 2 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 3) * 6 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + ((d1 mod 9) floordiv 3) * 2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 3)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2 - (d1 floordiv 3) * 6)>
// CHECK-LABEL:  func.func @test_im2col_with_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x1x7x7xf32>) -> memref<2x9x9xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x9x9xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 18){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_7_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_7_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_1_]], [[VAR_1_]])
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:               [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_2_]], [[I_1_]])
// CHECK-DAG:               [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[CST_0_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<2x1x7x7xf32>
// CHECK:                   krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<2x9x9xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]], [[VAR_1_]])
// CHECK-DAG:             [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:             [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_7_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                 [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                 scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:               [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_4_]], [[I_3_]])
// CHECK-DAG:               [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_4_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_7_]] : index
// CHECK:                   [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                   [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                   [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[CST_0_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<2x1x7x7xf32>
// CHECK:                   [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                   krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<2x9x9xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x9x9xf32>
// CHECK:         }
// -----



// Test Im2Col with dilations
func.func @test_im2col_with_dilations(%arg0: tensor<1x2x8x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], dilations = [2, 2]} : (tensor<1x2x8x8xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 16) floordiv 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 16) floordiv 4 + 4)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 4 + 4)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 16) floordiv 4)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1 mod 4)>
// CHECK-LABEL:  func.func @test_im2col_with_dilations
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x8x8xf32>) -> memref<1x18x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_8_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x2x8x8xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x18x16xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_8_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_8_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x2x8x8xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x18x16xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x16xf32>
// CHECK:         }
// -----



// Test Im2Col with multiple channels and complex parameters
func.func @test_im2col_complex(%arg0: tensor<2x4x10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2], dilations = [1, 1]} : (tensor<2x4x10x10xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 25)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 25)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (((d0 mod 25) floordiv 5) * 2 - 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 5) * 10 - 1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (((d0 mod 25) floordiv 5) * 2 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 5) * 10 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + ((d1 mod 25) floordiv 5) * 2 - 1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2 - (d1 floordiv 5) * 10 - 1)>
// CHECK-LABEL:  func.func @test_im2col_complex
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x10x10xf32>) -> memref<2x36x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x36x25xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 50){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_10_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_10_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<2x4x10x10xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<2x36x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_10_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_10_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<2x4x10x10xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<2x36x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x36x25xf32>
// CHECK:         }
// -----



// Test Im2Col with dynamic batch dimension
func.func @test_im2col_dynamic_batch(%arg0: tensor<?x3x6x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [2, 2]} : (tensor<?x3x6x6xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 25)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 25)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 mod 25)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> ((d0 mod 25) floordiv 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> ((d0 mod 25) floordiv 5 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 mod 5 + 1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 25) floordiv 5)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 2 + d2 * 4)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 5)>
// CHECK-LABEL:  func.func @test_im2col_dynamic_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x6x6xf32>) -> memref<?x12x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x6x6xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x12x25xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x6x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_6_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_6_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_7_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_9_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<?x3x6x6xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<?x12x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_7_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_6_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_9_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_6_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<?x3x6x6xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<?x12x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x12x25xf32>
// CHECK:         }
// -----



// Test Im2Col with dynamic spatial dimensions
func.func @test_im2col_dynamic_spatial(%arg0: tensor<1x2x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1]} : (tensor<1x2x?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-LABEL:  func.func @test_im2col_dynamic_spatial
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x?x?xf32>) -> memref<1x18x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x2x?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x18x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_0_]] : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_4_:%.+]] = arith.floordivsi [[VAR_3_]], [[VAR_1_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[VAR_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.subi [[VAR_3_]], [[VAR_5_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.floordivsi [[VAR_6_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.muli [[VAR_7_]], [[VAR_dim_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.subi [[VAR_6_]], [[VAR_8_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.subi [[VAR_7_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.subi [[VAR_9_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_7_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_9_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.cmpi sge, [[VAR_10_]], [[CST_0_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_11_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.andi [[VAR_14_]], [[VAR_15_]] : i1
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpi slt, [[VAR_12_]], [[VAR_dim_1_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.cmpi slt, [[VAR_13_]], [[VAR_dim_2_]] : index
// CHECK:             [[VAR_19_:%.+]] = arith.andi [[VAR_17_]], [[VAR_18_]] : i1
// CHECK:             [[VAR_20_:%.+]] = arith.andi [[VAR_16_]], [[VAR_19_]] : i1
// CHECK:             scf.if [[VAR_20_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_21_:%.+]] = affine.apply [[MAP_0_]]([[VAR_10_]], [[I_2_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_22_:%.+]] = affine.apply [[MAP_1_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_23_:%.+]] = affine.apply [[MAP_0_]]([[I_3_]], [[VAR_11_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[I_1_]], [[VAR_21_]], [[VAR_23_]]{{.}} : memref<1x2x?x?xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_22_]], [[VAR_6_]]{{.}} : memref<1x18x?xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_21_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_10_]], [[I_5_]])
// CHECK-DAG:               [[VAR_22_1_:%.+]] = arith.cmpi sge, [[VAR_21_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_23_1_:%.+]] = arith.cmpi slt, [[VAR_21_1_]], [[VAR_dim_1_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_22_1_]], [[VAR_23_1_]] : i1
// CHECK:                   [[VAR_25_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_21_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_26_:%.+]] = affine.apply [[MAP_1_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_27_:%.+]] = affine.apply [[MAP_0_]]([[I_6_]], [[VAR_11_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_28_:%.+]] = arith.cmpi sge, [[VAR_27_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[VAR_dim_2_]] : index
// CHECK:                     [[VAR_30_:%.+]] = arith.andi [[VAR_28_]], [[VAR_29_]] : i1
// CHECK:                     [[VAR_31_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_30_]] : i1
// CHECK:                     [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_27_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[I_4_]], [[VAR_25_]], [[VAR_32_]]{{.}} : memref<1x2x?x?xf32>
// CHECK:                     [[VAR_34_:%.+]] = arith.select [[VAR_31_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_34_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_26_]], [[VAR_6_]]{{.}} : memref<1x18x?xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x?xf32>
// CHECK:         }
// -----



// Test Im2Col with rectangular kernel
func.func @test_im2col_rectangular_kernel(%arg0: tensor<1x1x8x12xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [2, 4], strides = [1, 2]} : (tensor<1x1x8x12xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 35)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 35)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 35) floordiv 5)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 5) * 10)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 35) floordiv 5 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 5) * 10 + 3)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 35) floordiv 5)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 4)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2 - (d1 floordiv 5) * 10)>
// CHECK-LABEL:  func.func @test_im2col_rectangular_kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x8x12xf32>) -> memref<1x8x35xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x8x35xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 35){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_12_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_1_]], [[VAR_1_]])
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] {
// CHECK-DAG:               [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_2_]], [[I_1_]])
// CHECK-DAG:               [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[CST_0_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x1x8x12xf32>
// CHECK:                   krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x8x35xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]], [[VAR_1_]])
// CHECK-DAG:             [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:             [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_8_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                 [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                 scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_4_]] step [[CST_1_]] {
// CHECK-DAG:               [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_4_]], [[I_3_]])
// CHECK-DAG:               [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_4_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_12_]] : index
// CHECK:                   [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                   [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                   [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[CST_0_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x1x8x12xf32>
// CHECK:                   [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                   krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x8x35xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x8x35xf32>
// CHECK:         }
// -----



// Test Im2Col with asymmetric padding
func.func @test_im2col_asymmetric_padding(%arg0: tensor<1x2x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 2, 1, 0]} : (tensor<1x2x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 25)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 25)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 25) floordiv 5 - 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 5 - 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 25) floordiv 5 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 25) floordiv 5 - 1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 5 - 2)>
// CHECK-LABEL:  func.func @test_im2col_asymmetric_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf32>) -> memref<1x18x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x25xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 25){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_5_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x18x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_5_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_5_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x18x25xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x25xf32>
// CHECK:         }
// -----



// Test Im2Col with small kernel (1x1 convolution case)
func.func @test_im2col_1x1_kernel(%arg0: tensor<1x8x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [1, 1]} : (tensor<1x8x4x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 16) floordiv 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-LABEL:  func.func @test_im2col_1x1_kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x8x4x4xf32>) -> memref<1x8x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x8x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_4_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK:             scf.if [[VAR_12_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_8_]] step [[CST_1_]] {
// CHECK-DAG:             [[VAR_13_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:             [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x8x4x4xf32>
// CHECK:                 krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_3_]]{{.}} : memref<1x8x16xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_8_]] step [[CST_1_]] {
// CHECK:                 [[VAR_13_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:             [[VAR_14_1_:%.+]] = arith.cmpi sge, [[VAR_13_1_]], [[CST_0_]] : index
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_13_1_]], [[CST_4_]] : index
// CHECK:                 [[VAR_16_:%.+]] = arith.andi [[VAR_14_1_]], [[LOAD_PARAM_0_MEM_1_]] : i1
// CHECK-DAG:             [[VAR_17_:%.+]] = arith.select [[VAR_16_]], [[VAR_13_1_]], [[CST_0_]] : index
// CHECK-DAG:             [[VAR_18_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.cmpi sge, [[VAR_18_]], [[CST_0_]] : index
// CHECK-DAG:             [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_18_]], [[CST_4_]] : index
// CHECK:                 [[VAR_21_:%.+]] = arith.andi [[VAR_19_]], [[VAR_20_]] : i1
// CHECK:                 [[VAR_22_:%.+]] = arith.andi [[VAR_16_]], [[VAR_21_]] : i1
// CHECK:                 [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_18_]], [[CST_0_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_2_]], [[VAR_17_]], [[VAR_23_]]{{.}} : memref<1x8x4x4xf32>
// CHECK:                 [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:                 krnl.store [[VAR_25_]], [[RES_]]{{.}}[[VAR_2_]], [[I_2_]], [[VAR_3_]]{{.}} : memref<1x8x16xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x8x16xf32>
// CHECK:         }
// -----



// Test Im2Col with f16 type
func.func @test_im2col_f16(%arg0: tensor<1x2x5x5xf16>) -> tensor<*xf16> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf16>) -> tensor<*xf16>
  return %0 : tensor<*xf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 9)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 9)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 3)>
// CHECK-LABEL:  func.func @test_im2col_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf16>) -> memref<1x18x9xf16> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 9){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_5_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x2x5x5xf16>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x18x9xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_5_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_5_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x2x5x5xf16>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f16
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x18x9xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf16>
// CHECK:         }
// -----



// Test Im2Col with f64 type
func.func @test_im2col_f64(%arg0: tensor<1x2x5x5xf64>) -> tensor<*xf64> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf64>) -> tensor<*xf64>
  return %0 : tensor<*xf64>
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 9)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 9)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> ((d0 mod 9) floordiv 3 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 mod 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + (d1 mod 9) floordiv 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 3 + d2 * 9)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 mod 3)>
// CHECK-LABEL:  func.func @test_im2col_f64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf64>) -> memref<1x18x9xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 9){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_5_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_2_]], [[VAR_1_]])
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]], [[I_2_]], [[I_1_]])
// CHECK-DAG:                 [[VAR_17_:%.+]] = affine.apply [[MAP_8_]]([[I_3_]], [[VAR_1_]])
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]], [[VAR_15_]], [[VAR_17_]]{{.}} : memref<1x2x5x5xf64>
// CHECK:                     krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_16_]], [[VAR_3_]]{{.}} : memref<1x18x9xf64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } else {
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] {
// CHECK:                 scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK:                   [[VAR_15_1_:%.+]] = affine.apply [[MAP_6_]]([[I_5_]], [[VAR_1_]])
// CHECK-DAG:               [[VAR_16_1_:%.+]] = arith.cmpi sge, [[VAR_15_1_]], [[CST_0_]] : index
// CHECK-DAG:               [[VAR_17_1_:%.+]] = arith.cmpi slt, [[VAR_15_1_]], [[CST_5_]] : index
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.andi [[VAR_16_1_]], [[VAR_17_1_]] : i1
// CHECK:                   [[VAR_19_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_15_1_]], [[CST_0_]] : index
// CHECK:                   scf.for [[I_6_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_20_:%.+]] = affine.apply [[MAP_7_]]([[I_6_]], [[I_5_]], [[I_4_]])
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_8_]]([[I_6_]], [[VAR_1_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.cmpi sge, [[VAR_21_]], [[CST_0_]] : index
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_5_]] : index
// CHECK:                     [[VAR_24_:%.+]] = arith.andi [[VAR_22_]], [[VAR_23_]] : i1
// CHECK:                     [[VAR_25_:%.+]] = arith.andi [[LOAD_PARAM_0_MEM_1_]], [[VAR_24_]] : i1
// CHECK:                     [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_21_]], [[CST_0_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_4_]], [[VAR_19_]], [[VAR_26_]]{{.}} : memref<1x2x5x5xf64>
// CHECK:                     [[VAR_28_:%.+]] = arith.select [[VAR_25_]], [[LOAD_PARAM_0_MEM_2_]], [[CST_0_dot_000000_]] : f64
// CHECK:                     krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_20_]], [[VAR_3_]]{{.}} : memref<1x18x9xf64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf64>
// CHECK:         }
