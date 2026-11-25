// RUN: onnx-mlir-opt -O3 --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// GROUND-ALL: -c="-O3 --march=arm64" -r="--disable-simd" --lower-bound=int64:-6 --upper-bound=int64:5 --rtol=1e-5 --atol=1e-5

// ScatterND row form: with k = indices.shape[-1] < rank(data), each update is
// a slice of S = prod(data.shape[k:]) contiguous elements. The k index values
// are loaded once per slice, then the slice is combined with SIMD (z16: 4
// floats per vector, a scalar tail when S is not a multiple of 4). -O3 and
// --march are both needed for SIMD. Grounded on arm64 (same vector length)
// against the element-by-element lowering, which --disable-simd selects (the
// fill of a broadcast scalar does not depend on SIMD, so it is in both). Indices
// are drawn from [-6, 5] for dims of size 6, so rows repeat and negative ones
// are normalized.

// GNN aggregation: [E, 1] index into [N, 8, 12], reduction add, dynamic N and
// E. S = 96: 24 vectors per row.
// GROUND-THIS: --shape-info=0:6x8x12,1:13x1,2:13x8x12
func.func @test_scatternd_rows_add_3d(%arg0: tensor<?x8x12xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "idx"}, %arg2: tensor<?x8x12xf32> {onnx.name = "upd"}) -> (tensor<?x8x12xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<?x8x12xf32>, tensor<?x1xi64>, tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  return %0 : tensor<?x8x12xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_scatternd_rows_add_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8x12xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x1xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x8x12xf32> {onnx.name = "upd"}) -> (memref<?x8x12xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_96_:%.+]] = arith.constant 96 : index
// CHECK-DAG:       [[CST_96_1_:%.+]] = arith.constant 96 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8x12xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_96_1_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x8x12xf32>, memref<?x8x12xf32>, i64, index, index) -> ()
// CHECK:           [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_96_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_1_]]) : (memref<?x8x12xf32>, memref<2xindex>) -> memref<?x96xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_2_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_96_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[PARAM_2_]]([[RES_2_]]) : (memref<?x8x12xf32>, memref<2xindex>) -> memref<?x96xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1xi64>
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_5_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x1xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 96){
// CHECK:               [[VAR_10_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_4_MEM_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[VAR_3_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_4_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_13_]], [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8x12xf32>
// CHECK:         }

}

// -----

// Segment max: [E, 1] into [N, 8], static. S = 8: 2 vectors per row.
func.func @test_scatternd_rows_max_2d(%arg0: tensor<6x8xf32> {onnx.name = "data"}, %arg1: tensor<7x1xi64> {onnx.name = "idx"}, %arg2: tensor<7x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "max"} : (tensor<6x8xf32>, tensor<7x1xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
  return %0 : tensor<6x8xf32>

// CHECK-LABEL:  func.func @test_scatternd_rows_max_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<7x1xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<7x8xf32> {onnx.name = "upd"}) -> (memref<6x8xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : i64
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<6x8xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_48_]], [[CST_0_]], [[CST_0_]]) : (memref<6x8xf32>, memref<6x8xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 7){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<7x1xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:               [[VAR_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]], [[VAR_8_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_8_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:               [[VAR_11_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_2_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_8_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x8xf32>
// CHECK:         }

}

// -----

// Reduction min, dynamic rows.
// GROUND-THIS: --shape-info=0:6x8,1:9x1,2:9x8
func.func @test_scatternd_rows_min_2d(%arg0: tensor<?x8xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "idx"}, %arg2: tensor<?x8xf32> {onnx.name = "upd"}) -> (tensor<?x8xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "min"} : (tensor<?x8xf32>, tensor<?x1xi64>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_scatternd_rows_min_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x1xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x8xf32> {onnx.name = "upd"}) -> (memref<?x8xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_8_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x8xf32>, memref<?x8xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1xi64>
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x1xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:               [[VAR_10_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_3_]], [[VAR_10_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_2_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8xf32>
// CHECK:         }

}

// -----

// Reduction none, with a constant permutation as the index (no duplicate rows,
// as the spec requires for none).
func.func @test_scatternd_rows_none_perm(%arg0: tensor<4x2x3xf32> {onnx.name = "data"}, %arg1: tensor<4x2x3xf32> {onnx.name = "upd"}) -> (tensor<4x2x3xf32> {onnx.name = "y"}) {
  %idx = onnx.Constant dense<[[3], [0], [2], [1]]> : tensor<4x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) {reduction = "none"} : (tensor<4x2x3xf32>, tensor<4x1xi64>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>

// CHECK-LABEL:  func.func @test_scatternd_rows_none_perm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x2x3xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x2x3xf32> {onnx.name = "upd"}) -> (memref<4x2x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 1], value = dense<{{.}}[3], [0], [2], [1]{{.}}> : tensor<4x1xi64>}> : () -> memref<4x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<4x2x3xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_24_]], [[CST_0_]], [[CST_0_]]) : (memref<4x2x3xf32>, memref<4x2x3xf32>, i64, index, index) -> ()
// CHECK:           [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[CST_4_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_6_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_1_]]) : (memref<4x2x3xf32>, memref<2xindex>) -> memref<4x6xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[CST_4_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_6_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<4x2x3xf32>, memref<2xindex>) -> memref<4x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]], [[CST_0_]]{{.}} : memref<4x1xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:               [[VAR_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_2_]], [[VAR_7_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_2_MEM_]], [[VAR_reshape_]]{{.}}[[VAR_4_]], [[VAR_7_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 4 to 6){
// CHECK:               [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_2_MEM_1_:%.+]] = krnl.load [[VAR_reshape_2_]]{{.}}[[VAR_2_]], [[VAR_7_1_]]{{.}} : memref<4x6xf32>
// CHECK:               krnl.store [[LOAD_VAR_reshape_2_MEM_1_]], [[VAR_reshape_]]{{.}}[[VAR_4_]], [[VAR_7_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x2x3xf32>
// CHECK:         }

}

// -----

// A slice that is not a multiple of the vector length: S = 6, one vector and
// a scalar tail of 2.
// GROUND-THIS: --shape-info=0:6x6,1:11x1,2:11x6
func.func @test_scatternd_rows_odd_slice(%arg0: tensor<?x6xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "idx"}, %arg2: tensor<?x6xf32> {onnx.name = "upd"}) -> (tensor<?x6xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<?x6xf32>, tensor<?x1xi64>, tensor<?x6xf32>) -> tensor<?x6xf32>
  return %0 : tensor<?x6xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_scatternd_rows_odd_slice
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x6xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x1xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x6xf32> {onnx.name = "upd"}) -> (memref<?x6xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x6xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x6xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x6xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_6_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x6xf32>, memref<?x6xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1xi64>
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x1xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:               [[VAR_11_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_3_]], [[VAR_11_]]{{.}} : memref<?x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_8_]], [[VAR_11_]]{{.}} : memref<?x6xf32>, vector<4xf32>
// CHECK:               [[VAR_14_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_2_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_8_]], [[VAR_11_]]{{.}} : memref<?x6xf32>, vector<4xf32>
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 4 to 6){
// CHECK:               [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]], [[VAR_11_1_]]{{.}} : memref<?x6xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_8_]], [[VAR_11_1_]]{{.}} : memref<?x6xf32>
// CHECK:               [[VAR_14_1_:%.+]] = arith.addf [[LOAD_RES_MEM_1_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_14_1_]], [[RES_]]{{.}}[[VAR_8_]], [[VAR_11_1_]]{{.}} : memref<?x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x6xf32>
// CHECK:         }

}

// -----

// k = 2 index into 3-D data: the slice is the last dim (S = 7), two index
// values loaded per slice.
func.func @test_scatternd_rows_k2(%arg0: tensor<6x6x7xf32> {onnx.name = "data"}, %arg1: tensor<9x2xi64> {onnx.name = "idx"}, %arg2: tensor<9x7xf32> {onnx.name = "upd"}) -> (tensor<6x6x7xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<6x6x7xf32>, tensor<9x2xi64>, tensor<9x7xf32>) -> tensor<6x6x7xf32>
  return %0 : tensor<6x6x7xf32>

// CHECK-LABEL:  func.func @test_scatternd_rows_k2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6x6x7xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<9x2xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<9x7xf32> {onnx.name = "upd"}) -> (memref<6x6x7xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_252_:%.+]] = arith.constant 252 : i64
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<6x6x7xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_252_]], [[CST_0_]], [[CST_0_]]) : (memref<6x6x7xf32>, memref<6x6x7xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 9){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<9x2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_1_]]{{.}} : memref<9x2xi64>
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_8_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_9_]], [[VAR_10_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 4){
// CHECK:               [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]], [[VAR_14_]]{{.}} : memref<9x7xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]], [[VAR_14_]]{{.}} : memref<6x6x7xf32>, vector<4xf32>
// CHECK:               [[VAR_17_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_2_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_17_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]], [[VAR_14_]]{{.}} : memref<6x6x7xf32>, vector<4xf32>
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 4 to 7){
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]], [[VAR_14_1_]]{{.}} : memref<9x7xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]], [[VAR_14_1_]]{{.}} : memref<6x6x7xf32>
// CHECK:               [[VAR_17_1_:%.+]] = arith.addf [[LOAD_RES_MEM_1_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_17_1_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]], [[VAR_14_1_]]{{.}} : memref<6x6x7xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x6x7xf32>
// CHECK:         }

}

// -----

// k = rank(data): one element per update, no slice. Unchanged path.
func.func @test_scatternd_elements_unchanged(%arg0: tensor<6x6xf32> {onnx.name = "data"}, %arg1: tensor<5x2xi64> {onnx.name = "idx"}, %arg2: tensor<5xf32> {onnx.name = "upd"}) -> (tensor<6x6xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<6x6xf32>, tensor<5x2xi64>, tensor<5xf32>) -> tensor<6x6xf32>
  return %0 : tensor<6x6xf32>

// CHECK-LABEL:  func.func @test_scatternd_elements_unchanged
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6x6xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<5x2xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<5xf32> {onnx.name = "upd"}) -> (memref<6x6xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_36_:%.+]] = arith.constant 36 : i64
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<6x6xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_36_]], [[CST_0_]], [[CST_0_]]) : (memref<6x6xf32>, memref<6x6xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<5x2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_1_]]{{.}} : memref<5x2xi64>
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_8_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_9_]], [[VAR_10_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]{{.}} : memref<5xf32>
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]]{{.}} : memref<6x6xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]]{{.}} : memref<6x6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x6xf32>
// CHECK:         }

}

// -----

// Fill instead of copy: both scatters start from the same broadcast zero
// tensor, so neither can write into it in place. Each fills its own output
// with 0 (krnl.memset) instead of copying the shared buffer (no krnl.memcpy).
func.func @test_scatternd_rows_fill(%arg0: tensor<7x1xi64> {onnx.name = "idx"}, %arg1: tensor<7x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "sum"}, tensor<6x8xf32> {onnx.name = "max"}) {
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %shape = onnx.Constant dense<[6, 8]> : tensor<2xi64>
  %data = "onnx.Expand"(%zero, %shape) : (tensor<f32>, tensor<2xi64>) -> tensor<6x8xf32>
  %0 = "onnx.ScatterND"(%data, %arg0, %arg1) {reduction = "add"} : (tensor<6x8xf32>, tensor<7x1xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
  %1 = "onnx.ScatterND"(%data, %arg0, %arg1) {reduction = "max"} : (tensor<6x8xf32>, tensor<7x1xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
  return %0, %1 : tensor<6x8xf32>, tensor<6x8xf32>
// CHECK-LABEL:  func.func @test_scatternd_rows_fill
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x1xi64> {onnx.name = "idx"}, [[PARAM_1_:%.+]]: memref<7x8xf32> {onnx.name = "upd"}) -> (memref<6x8xf32> {onnx.name = "sum"}, memref<6x8xf32> {onnx.name = "max"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [], value = dense<0.000000e+00> : tensor<f32>}> : () -> memref<f32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<6x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 6, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1] : memref<6x8xf32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<6x8xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<6x8xf32>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_1_]], [[CST_0_]]{{.}} : memref<7x1xi64>
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_7_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_8_]], [[VAR_9_]], [[VAR_7_]] : index
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 8){
// CHECK:               [[VAR_12_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_5_1_]], [[VAR_12_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[VAR_10_]], [[VAR_12_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_15_]], [[RES_1_]]{{.}}[[VAR_10_]], [[VAR_12_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<6x8xf32>
// CHECK:           krnl.memset [[RES_2_]], [[CST_0_dot_000000_]] : memref<6x8xf32>
// CHECK:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 7){
// CHECK:             [[VAR_5_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_2_]], [[CST_0_]]{{.}} : memref<7x1xi64>
// CHECK:             [[VAR_7_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_8_1_:%.+]] = arith.cmpi slt, [[VAR_7_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.addi [[VAR_7_1_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_1_:%.+]] = arith.select [[VAR_8_1_]], [[VAR_9_1_]], [[VAR_7_1_]] : index
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_5_:%.+]] = 0 to 8){
// CHECK:               [[VAR_12_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_5_2_]], [[VAR_12_1_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_10_1_]], [[VAR_12_1_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:               [[VAR_15_1_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_15_1_]], [[RES_2_]]{{.}}[[VAR_10_1_]], [[VAR_12_1_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]], [[RES_2_]] : memref<6x8xf32>, memref<6x8xf32>
// CHECK:         }

}


// -----

// A splat constant as data (static shapes: constant propagation has already
// turned Expand(0) into a constant). The output is filled with the splat
// value instead of copying the constant: krnl.memset, no krnl.memcpy.
func.func @test_scatternd_rows_fill_splat(%arg0: tensor<5x1xi64> {onnx.name = "idx"}, %arg1: tensor<5x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %data = onnx.Constant dense<0.0> : tensor<6x8xf32>
  %0 = "onnx.ScatterND"(%data, %arg0, %arg1) {reduction = "add"} : (tensor<6x8xf32>, tensor<5x1xi64>, tensor<5x8xf32>) -> tensor<6x8xf32>
  return %0 : tensor<6x8xf32>
// CHECK-LABEL:  func.func @test_scatternd_rows_fill_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x1xi64> {onnx.name = "idx"}, [[PARAM_1_:%.+]]: memref<5x8xf32> {onnx.name = "upd"}) -> (memref<6x8xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<6x8xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<6x8xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<5x1xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:               [[VAR_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[VAR_8_]]{{.}} : memref<5x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_8_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_8_]]{{.}} : memref<6x8xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x8xf32>
// CHECK:         }

}


// -----

// indices.shape[-1] dynamic: the ranks fix it (k = rank(data) + rank(indices)
// - 1 - rank(updates) = 3 + 2 - 1 - 3 = 1), so this is the same row form as the
// first case.
// GROUND-THIS: --shape-info=0:6x8x12,1:13x1,2:13x8x12
func.func @test_scatternd_rows_dynamic_k(%arg0: tensor<?x8x12xf32> {onnx.name = "data"}, %arg1: tensor<?x?xi64> {onnx.name = "idx"}, %arg2: tensor<?x8x12xf32> {onnx.name = "upd"}) -> (tensor<?x8x12xf32> {onnx.name = "y"}) {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<?x8x12xf32>, tensor<?x?xi64>, tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  return %0 : tensor<?x8x12xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_scatternd_rows_dynamic_k
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8x12xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x?xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x8x12xf32> {onnx.name = "upd"}) -> (memref<?x8x12xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_96_:%.+]] = arith.constant 96 : index
// CHECK-DAG:       [[CST_96_1_:%.+]] = arith.constant 96 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8x12xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_96_1_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x8x12xf32>, memref<?x8x12xf32>, i64, index, index) -> ()
// CHECK:           [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_96_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_1_]]) : (memref<?x8x12xf32>, memref<2xindex>) -> memref<?x96xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x8x12xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_2_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_96_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[PARAM_2_]]([[RES_2_]]) : (memref<?x8x12xf32>, memref<2xindex>) -> memref<?x96xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?xi64>
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_5_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x?xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 96){
// CHECK:               [[VAR_10_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_4_MEM_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[VAR_3_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_4_MEM_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_13_]], [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x96xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8x12xf32>
// CHECK:         }

}

