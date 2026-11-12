// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// -----

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized)
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.
// Test parallelization of Relu

func.func @test_relu_parallel(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2 - 31)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1, s2] -> (s2 - s2 mod 32)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func @test_relu_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.maxnumf [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_7_]], [[VAR_reshape_4_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[MAP_3_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}} to [[MAP_4_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_5_1_]]{{.}} : memref<?xf32>
// CHECK:               [[VAR_7_1_:%.+]] = arith.maxnumf [[LOAD_VAR_reshape_MEM_1_]], [[CST_0_dot_000000_]] : f32
// CHECK:               krnl.store [[VAR_7_1_]], [[VAR_reshape_4_]]{{.}}[[VAR_5_1_]]{{.}} : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// Partially flattened SIMD with a single outer loop: [1, S] - [S, 1] -> [S, S]
// flattens only the innermost dim, so one outer loop over S remains. The
// region goes on that outer loop rather than on the SIMD loop inside it, which
// would fork and join once per row. As in a sliding-window attention mask.
// GROUND-THIS: --shape-info=0:1x37,1:37x1
func.func @test_sub_bcast_row_col_parallel(%arg0: tensor<1x?xi64>, %arg1: tensor<?x1xi64>) -> tensor<?x?xi64> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<1x?xi64>, tensor<?x1xi64>) -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1 - 15)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 - s1 mod 16)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @test_sub_bcast_row_col_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?xi64>, [[PARAM_1_:%.+]]: memref<?x1xi64>) -> memref<?x?xi64> {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?xi64>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_0_]], [[VAR_dim_]] : index
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_8_]] : index
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_128_]] : index
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_2_]]) alignment = 16 : memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_dim_0_]]){
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}}){
// CHECK:               [[VAR_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[CST_0_]], [[VAR_7_]]{{.}} : memref<1x?xi64>, vector<16xi64>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<?x1xi64>
// CHECK:               [[VAR_10_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : i64 to vector<16xi64>
// CHECK:               [[VAR_11_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_]], [[VAR_10_]] : vector<16xi64>
// CHECK:               vector.store [[VAR_11_]], [[VAR_view_]]{{.}}[[VAR_4_]], [[VAR_7_]]{{.}} : memref<?x?xi64>, vector<16xi64>
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = [[MAP_1_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}} to [[MAP_2_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}}){
// CHECK:               [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[VAR_7_1_]]{{.}} : memref<1x?xi64>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<?x1xi64>
// CHECK:               [[VAR_10_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : i64
// CHECK:               krnl.store [[VAR_10_1_]], [[VAR_view_]]{{.}}[[VAR_4_]], [[VAR_7_1_]]{{.}} : memref<?x?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x?xi64>
// CHECK:         }
}

// -----

// Same pattern, but the single outer loop has a literal trip count (2) below
// the parallel floor: no region on it, so the SIMD loop is parallelized
// instead, with only two regions forked.
func.func @test_sub_bcast_short_outer_parallel(%arg0: tensor<1x4096xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x4096xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<1x4096xf32>, tensor<2x1xf32>) -> tensor<2x4096xf32>
  return %0 : tensor<2x4096xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_sub_bcast_short_outer_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4096xf32>, [[PARAM_1_:%.+]]: memref<2x1xf32>) -> memref<2x4096xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<2x4096xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 4096){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[CST_0_]], [[VAR_3_]]{{.}} : memref<1x4096xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<2x1xf32>
// CHECK:               [[VAR_6_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f32 to vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]]{{.}} : memref<2x4096xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x4096xf32>
// CHECK:         }
}
