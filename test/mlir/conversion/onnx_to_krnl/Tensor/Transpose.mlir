// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----


func.func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 5 + 1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 * 5 + 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 * 5 + 3)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 5 + 4)>
// CHECK-LABEL:  func.func private @test_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20x30x40xf32>) -> memref<40x10x30x20xf32> {
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<40x30x20x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_40_1_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_1_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 40, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 20, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_2_]]#3]
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_2_]]#2, [[VAR_2_]]#1, [[VAR_2_]]#0] : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_3_]]{{.}} : memref<40x30x20x10xf32>
// CHECK-DAG:         [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_2_]]#2, [[VAR_2_]]#1, [[VAR_2_]]#0] : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_5_]]{{.}} : memref<40x30x20x10xf32>
// CHECK-DAG:         [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_2_]]#2, [[VAR_2_]]#1, [[VAR_2_]]#0] : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_7_]]{{.}} : memref<40x30x20x10xf32>
// CHECK-DAG:         [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_9_]], [[VAR_2_]]#2, [[VAR_2_]]#1, [[VAR_2_]]#0] : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_3_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_9_]]{{.}} : memref<40x30x20x10xf32>
// CHECK-DAG:         [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_4_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_11_]], [[VAR_2_]]#2, [[VAR_2_]]#1, [[VAR_2_]]#0] : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_4_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_11_]]{{.}} : memref<40x30x20x10xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_40_2_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_10_2_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_30_2_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_2_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<40x10x30x20xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_40_3_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_10_3_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_30_3_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_3_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 40, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 10, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 30, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[CST_5_1_:%.+]] = arith.constant 5 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_2_1_]]#3]
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_0_MEM_5_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_3_1_]], [[VAR_2_1_]]#1] : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_5_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_3_1_]]{{.}} : memref<40x10x30x20xf32>
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_5_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_5_1_]], [[VAR_2_1_]]#1] : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_5_1_]]{{.}} : memref<40x10x30x20xf32>
// CHECK-DAG:         [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_7_1_]], [[VAR_2_1_]]#1] : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_7_1_]]{{.}} : memref<40x10x30x20xf32>
// CHECK-DAG:         [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_9_1_]], [[VAR_2_1_]]#1] : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_3_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_9_1_]]{{.}} : memref<40x10x30x20xf32>
// CHECK-DAG:         [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[VAR_11_1_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_4_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_11_1_]], [[VAR_2_1_]]#1] : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_4_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_11_1_]]{{.}} : memref<40x10x30x20xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<40x10x30x20xf32>
// CHECK:         }
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.

func.func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 6)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 * 6 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 * 6 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 6 + 3)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<()[s0] -> (s0 * 6 + 4)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<()[s0] -> (s0 * 6 + 5)>
// CHECK-LABEL:  func.func private @test_transpose_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<10x40x?x30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_40_1_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 40, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_1_]]#3]
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]], [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<10x40x?x30xf32>
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_4_]], [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_4_]]{{.}} : memref<10x40x?x30xf32>
// CHECK-DAG:         [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_6_]], [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_6_]]{{.}} : memref<10x40x?x30xf32>
// CHECK-DAG:         [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_8_]], [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_8_]]{{.}} : memref<10x40x?x30xf32>
// CHECK-DAG:         [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_4_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]0, [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]0] : memref<10x40x?x30xf32>
// CHECK-DAG:         [[CST_5_1_:%.+]] = arith.constant 5 : index
// CHECK-DAG:         [[VAR_12_:%.+]] = affine.apply [[MAP_6_]](){{.}}[[VAR_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_5_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]2, [[VAR_1_]]#1] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]2] : memref<10x40x?x30xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x40x?x30xf32>
// CHECK:         }
}

