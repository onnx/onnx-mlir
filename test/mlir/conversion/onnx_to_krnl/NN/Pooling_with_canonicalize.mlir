// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func private @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x?x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (s0, d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d1, d0 + d1)>
// CHECK-LABEL:  func.func private @test_pool_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x?x32xf32>) -> memref<1x3x?x31xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x3x?x31xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_0_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.max [[MAP_2_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.min [[MAP_3_]]([[VAR_2_]]#2){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.max [[MAP_2_]]([[VAR_2_]]#3)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.min [[MAP_4_]]([[VAR_2_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.subi [[VAR_4_]], [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.subi [[VAR_6_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_5_]]([[VAR_2_]]#2){{.}}[[VAR_dim_0_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_5_]]([[VAR_2_]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_16_:%.+]] = affine.max [[MAP_6_]]([[VAR_2_]]#2, [[I_4_]])
// CHECK-DAG:           [[VAR_17_:%.+]] = affine.max [[MAP_6_]]([[VAR_2_]]#3, [[I_5_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_16_]], [[VAR_17_]]{{.}} : memref<1x3x?x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_20_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<1x3x?x31xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<1x3x?x31xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.muli [[VAR_7_]], [[VAR_8_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : index to i64
// CHECK:             [[VAR_14_:%.+]] = arith.sitofp [[VAR_13_]] : i64 to f32
// CHECK:             [[VAR_15_:%.+]] = arith.divf [[LOAD_RES_MEM_]], [[VAR_14_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<1x3x?x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x?x31xf32>
// CHECK:         }
}

