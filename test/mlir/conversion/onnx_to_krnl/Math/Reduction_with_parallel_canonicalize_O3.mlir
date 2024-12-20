// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: Full reduction over all dimensions to a scalar value.
func.func @test_reduce_all_to_scalar(%arg0: tensor<?x64x?xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x64x?xf32>, none) -> tensor<*xf32>
  return %0: tensor<*xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 - 31)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s1 + ((s0 - s1) floordiv 32) * 32)>
// CHECK-LABEL:  func.func @test_reduce_all_to_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x64x?xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0xFF800000> : vector<1xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x64x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[VAR_dim_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_]]) : (memref<?x64x?xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<256xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.ceildivsi [[VAR_1_]], [[CST_8_]] : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 8){
// CHECK:             [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_8_:%.+]] = arith.muli [[VAR_7_]], [[VAR_2_]] : index
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[VAR_2_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[VAR_9_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_1_]], [[VAR_9_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]])
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             [[VAR_13_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]]{{.}}
// CHECK:             scf.for [[I_1_:%.+]] = [[VAR_8_]] to [[VAR_13_]] step [[CST_32_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[I_1_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:               [[VAR_19_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_19_]], [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[VAR_14_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_11_]], [[VAR_8_]]{{.}}
// CHECK:             scf.for [[I_2_:%.+]] = [[VAR_14_]] to [[VAR_11_]] step [[CST_1_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[I_2_]]{{.}} : memref<?xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>
// CHECK:               [[VAR_19_1_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : f32
// CHECK:               memref.store [[VAR_19_1_]], [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[VAR_12_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             [[VAR_16_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_1_MEM_2_]] : vector<32xf32> into f32
// CHECK:             memref.store [[VAR_16_]], [[RES_2_]]{{.}}[[VAR_7_]]{{.}} : memref<8xf32>
// CHECK:           }
// CHECK:           [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_8_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_7_1_]]{{.}} : memref<8xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:             [[VAR_10_1_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_3_]], [[VAR_8_1_]] : f32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_4_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           [[VAR_6_:%.+]] = vector.extract [[LOAD_RES_1_MEM_4_]][0] : f32 from vector<1xf32>
// CHECK:           krnl.store [[VAR_6_]], [[RES_3_]][] : memref<f32>
// CHECK:           return [[RES_3_]] : memref<f32>
// CHECK:         }
}

// -----

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized)
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.
// Test parallelization of ReduceMean

func.func private @test_reducemean_v13_f32_too_small(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_f32_too_small
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xf32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemean_v13_f32_big_enough(%arg0 : tensor<128x64x32xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<128x64x32xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_f32_big_enough
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128x64x32xf32>) -> memref<128x32xf32> {
// CHECK-DAG:       [[CST_6_dot_400000_:%.+]] = arith.constant 6.400000e+01 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128x32xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<128x32xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 128, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 64, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<128x64x32xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<128x32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<128x32xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 128, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<128x32xf32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[CST_6_dot_400000_]] : f32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<128x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128x32xf32>
// CHECK:         }
}

