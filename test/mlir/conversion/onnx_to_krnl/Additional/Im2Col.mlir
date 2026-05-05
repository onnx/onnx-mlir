// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


// Test basic Im2Col with static shapes, no padding, unit stride
func.func @test_im2col_basic(%arg0: tensor<1x2x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func.func @test_im2col_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf32>) -> memref<1x18x9xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_5_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_5_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x18x9xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf32>
// CHECK:         }
// -----


// Test Im2Col with padding
func.func @test_im2col_with_padding(%arg0: tensor<1x3x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 - 1)>
// CHECK-LABEL:  func.func @test_im2col_with_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x4x4xf32>) -> memref<1x27x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x27x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_4_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x3x4x4xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x27x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x27x16xf32>
// CHECK:         }
// -----


// Test Im2Col with strides
func.func @test_im2col_with_strides(%arg0: tensor<2x1x7x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], strides = [2, 2]} : (tensor<2x1x7x7xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL:  func.func @test_im2col_with_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x1x7x7xf32>) -> memref<2x9x9xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x9x9xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_7_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_7_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<2x1x7x7xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<2x9x9xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x9x9xf32>
// CHECK:         }
// -----


// Test Im2Col with dilations
func.func @test_im2col_with_dilations(%arg0: tensor<1x2x8x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], dilations = [2, 2]} : (tensor<1x2x8x8xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-LABEL:  func.func @test_im2col_with_dilations
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x8x8xf32>) -> memref<1x18x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_8_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x8x8xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x18x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x16xf32>
// CHECK:         }
// -----


// Test Im2Col with multiple channels and complex parameters
func.func @test_im2col_complex(%arg0: tensor<2x4x10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2], dilations = [1, 1]} : (tensor<2x4x10x10xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1 - 1)>
// CHECK-LABEL:  func.func @test_im2col_complex
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x10x10xf32>) -> memref<2x36x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x36x25xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 4, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_10_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_10_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<2x4x10x10xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<2x36x25xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x36x25xf32>
// CHECK:         }
// -----


// Test Im2Col with dynamic batch dimension
func.func @test_im2col_dynamic_batch(%arg0: tensor<?x3x6x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [2, 2]} : (tensor<?x3x6x6xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func.func @test_im2col_dynamic_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x6x6xf32>) -> memref<?x12x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x6x6xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x12x25xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x6x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_6_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<?x3x6x6xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<?x12x25xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x12x25xf32>
// CHECK:         }
// -----


// Test Im2Col with dynamic spatial dimensions
func.func @test_im2col_dynamic_spatial(%arg0: tensor<1x2x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 1, 1, 1]} : (tensor<1x2x?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 - 1)>
// CHECK-LABEL:  func.func @test_im2col_dynamic_spatial
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x?x?xf32>) -> memref<1x18x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x2x?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x18x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x2x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_2_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_2_]]#1, [[VAR_dim_0_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_2_]]#2, [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_2_]]#3, [[VAR_2_]]#4, [[VAR_2_]]#5)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]]#4, [[VAR_2_]]#1)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]]#5, [[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sge, [[VAR_6_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[VAR_dim_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.andi [[VAR_8_]], [[VAR_9_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sge, [[VAR_7_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[VAR_dim_2_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.andi [[VAR_11_]], [[VAR_12_]] : i1
// CHECK:             [[VAR_14_:%.+]] = arith.andi [[VAR_10_]], [[VAR_13_]] : i1
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[VAR_6_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[VAR_7_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#3, [[VAR_15_]], [[VAR_16_]]{{.}} : memref<1x2x?x?xf32>
// CHECK:             [[VAR_18_:%.+]] = arith.select [[VAR_14_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_18_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_5_]], [[VAR_4_]]{{.}} : memref<1x18x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x?xf32>
// CHECK:         }
// -----


// Test Im2Col with rectangular kernel
func.func @test_im2col_rectangular_kernel(%arg0: tensor<1x1x8x12xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [2, 4], strides = [1, 2]} : (tensor<1x1x8x12xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 4 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL:  func.func @test_im2col_rectangular_kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x8x12xf32>) -> memref<1x8x35xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x8x35xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 7, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_12_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x1x8x12xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x8x35xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x8x35xf32>
// CHECK:         }
// -----


// Test Im2Col with asymmetric padding
func.func @test_im2col_asymmetric_padding(%arg0: tensor<1x2x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3], pads = [1, 2, 1, 0]} : (tensor<1x2x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 - 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 - 2)>
// CHECK-LABEL:  func.func @test_im2col_asymmetric_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf32>) -> memref<1x18x25xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x25xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_5_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_5_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x5x5xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x18x25xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x25xf32>
// CHECK:         }
// -----


// Test Im2Col with small kernel (1x1 convolution case)
func.func @test_im2col_1x1_kernel(%arg0: tensor<1x8x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [1, 1]} : (tensor<1x8x4x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func.func @test_im2col_1x1_kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x8x4x4xf32>) -> memref<1x8x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x8x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 8, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 1){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_4_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x8x4x4xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x8x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x8x16xf32>
// CHECK:         }
// -----


// Test Im2Col with f16 type
func.func @test_im2col_f16(%arg0: tensor<1x2x5x5xf16>) -> tensor<*xf16> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf16>) -> tensor<*xf16>
  return %0 : tensor<*xf16>
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func.func @test_im2col_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf16>) -> memref<1x18x9xf16> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_5_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_5_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x5x5xf16>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f16
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x18x9xf16>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf16>
// CHECK:         }
// -----


// Test Im2Col with f64 type
func.func @test_im2col_f64(%arg0: tensor<1x2x5x5xf64>) -> tensor<*xf64> {
  %0 = "onnx.Im2Col"(%arg0) {kernel_shape = [3, 3]} : (tensor<1x2x5x5xf64>) -> tensor<*xf64>
  return %0 : tensor<*xf64>
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 9 + d1 * 3 + d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func.func @test_im2col_f64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x2x5x5xf64>) -> memref<1x18x9xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x18x9xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:6 = krnl.define_loops 6
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_0_]]#5 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]]:6 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4, [[LOOP_0_]]#5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3, [[VAR_1_]]#4, [[VAR_1_]]#5)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#4)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_1_]]#5)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_5_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.andi [[VAR_6_]], [[VAR_7_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_5_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.andi [[VAR_9_]], [[VAR_10_]] : i1
// CHECK:             [[VAR_12_:%.+]] = arith.andi [[VAR_8_]], [[VAR_11_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x5x5xf64>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_12_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_2_]]{{.}} : memref<1x18x9xf64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x18x9xf64>
// CHECK:         }
