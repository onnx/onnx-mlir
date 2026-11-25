// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel placement under a loop with a dynamic trip count. A parallel region
// placed below such a loop is entered once per iteration of it, so it is only
// placed there when one entry carries enough work to repay its fork/join
// (4 x the target's forkCost). Otherwise the outer level gets the region, or
// none is placed. Generic tuning: forkCost 4000, work gate off.

// Ground against a sequential build. The copies and the in-order scatter are
// exact, hence the tight tolerance.
// GROUND-ALL: -c="-O3" -t="--parallel" --rtol=1e-6 --atol=1e-6

// Gather [N,8] by E indices: the 8 heads alone would be one region per edge.
// GROUND-THIS: --shape-info=0:944x8,1:2963 --input-value=1:min0max943
func.func @gather_dyn_outer(%data: tensor<?x8xf32> {onnx.name = "data"}, %idx: tensor<?xi64> {onnx.name = "idx"}) -> (tensor<?x8xf32> {onnx.name = "out"}) {
  %0 = "onnx.Gather"(%data, %idx) {axis = 0 : si64} : (tensor<?x8xf32>, tensor<?xi64>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @gather_dyn_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?xi64> {onnx.name = "idx"}) -> (memref<?x8xf32> {onnx.name = "out"}) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0] : memref<?xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_1_]]#1] : memref<?x8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8xf32>
// CHECK:         }

}

// -----

// All dynamic: unchanged, the outer level is parallel.
// GROUND-THIS: --shape-info=0:944x7,1:2963 --input-value=1:min0max943
func.func @gather_all_dyn(%data: tensor<?x?xf32> {onnx.name = "data"}, %idx: tensor<?xi64> {onnx.name = "idx"}) -> (tensor<?x?xf32> {onnx.name = "out"}) {
  %0 = "onnx.Gather"(%data, %idx) {axis = 0 : si64} : (tensor<?x?xf32>, tensor<?xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @gather_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?xi64> {onnx.name = "idx"}) -> (memref<?x?xf32> {onnx.name = "out"}) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) alignment = 16 : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_0_]], [[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0] : memref<?xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[VAR_dim_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_1_]]#1] : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }

}

// -----

// Expand [E,1,1] to [E,8,12].
// GROUND-THIS: --shape-info=0:2963x1x1
func.func @expand_dyn_outer(%idx: tensor<?x1x1xi64> {onnx.name = "idx"}) -> (tensor<?x8x12xi64> {onnx.name = "out"}) {
  %shape = onnx.Constant dense<[1, 8, 12]> : tensor<3xi64>
  %0 = "onnx.Expand"(%idx, %shape) : (tensor<?x1x1xi64>, tensor<3xi64>) -> tensor<?x8x12xi64>
  return %0 : tensor<?x8x12xi64>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @expand_dyn_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1xi64> {onnx.name = "idx"}) -> (memref<?x8x12xi64> {onnx.name = "out"}) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8x12xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[CST_0_]], [[CST_0_]]{{.}} : memref<?x1x1xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x8x12xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8x12xi64>
// CHECK:         }

}

// -----

// Scatter-add over E edges: the scatter axis cannot be parallel with a
// reduction, and the 8 heads alone would be one region per edge, so only the
// copy of `data` is parallel and the scatter loop is sequential.
// GROUND-THIS: --shape-info=0:944x8,1:2963x8,2:2963x8 --input-value=1:min0max943
func.func @scatter_add_dyn_outer(%data: tensor<?x8xf32> {onnx.name = "data"}, %idx: tensor<?x8xi64> {onnx.name = "idx"}, %upd: tensor<?x8xf32> {onnx.name = "upd"}) -> (tensor<?x8xf32> {onnx.name = "out"}) {
  %0 = "onnx.ScatterElements"(%data, %idx, %upd) {axis = 0 : si64, reduction = "add"} : (tensor<?x8xf32>, tensor<?x8xi64>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 * 8) ceildiv 262144)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 * -262144 + s0 * 8, 262144)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @scatter_add_dyn_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x8xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x8xf32> {onnx.name = "upd"}) -> (memref<?x8xf32> {onnx.name = "out"}) {
// CHECK-DAG:       [[CST_2097152_:%.+]] = arith.constant 2097152 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_8_]] : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[VAR_3_:%.+]] = arith.cmpi sge, [[VAR_2_]], [[CST_2097152_]] : index
// CHECK:           scf.if [[VAR_3_]] {
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_7_:%.+]] = affine.apply [[MAP_2_]]([[VAR_6_]])
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.min [[MAP_3_]]([[VAR_6_]]){{.}}[[VAR_dim_]]{{.}}
// CHECK:               [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_9_]], [[VAR_7_]], [[VAR_7_]]) : (memref<?x8xf32>, memref<?x8xf32>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           } else {
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x8xf32>, memref<?x8xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x8xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_]], [[VAR_dim_1_]]), [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[LOOP_0_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<?x8xf32>
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<?x8xi64>
// CHECK:             [[VAR_8_1_:%.+]] = arith.index_cast [[VAR_7_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.cmpi slt, [[VAR_8_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_8_1_]], [[VAR_dim_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.select [[VAR_9_1_]], [[VAR_10_]], [[VAR_8_1_]] : index
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_]], [[LOOP_0_]]#1] : memref<?x8xf32>
// CHECK:             [[VAR_13_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[VAR_6_1_]] : f32
// CHECK:             krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_11_]], [[LOOP_0_]]#1] : memref<?x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8xf32>
// CHECK:         }

}

// -----

// Unchanged: the inner level does 2048 x bodyCost 10 of work per entry, above
// 4 x forkCost, so it keeps its region under the dynamic scatter axis.
// GROUND-THIS: --shape-info=0:16x2048,1:37x2048,2:37x2048 --input-value=1:min0max15
func.func @scatter_add_wide_rows(%data: tensor<?x2048xf32> {onnx.name = "data"}, %idx: tensor<?x2048xi64> {onnx.name = "idx"}, %upd: tensor<?x2048xf32> {onnx.name = "upd"}) -> (tensor<?x2048xf32> {onnx.name = "out"}) {
  %0 = "onnx.ScatterElements"(%data, %idx, %upd) {axis = 0 : si64, reduction = "add"} : (tensor<?x2048xf32>, tensor<?x2048xi64>, tensor<?x2048xf32>) -> tensor<?x2048xf32>
  return %0 : tensor<?x2048xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2048)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 * 2048) ceildiv 262144)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 * -262144 + s0 * 2048, 262144)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @scatter_add_wide_rows
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2048xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<?x2048xi64> {onnx.name = "idx"}, [[PARAM_2_:%.+]]: memref<?x2048xf32> {onnx.name = "upd"}) -> (memref<?x2048xf32> {onnx.name = "out"}) {
// CHECK-DAG:       [[CST_2097152_:%.+]] = arith.constant 2097152 : index
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2048xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x2048xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2048xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_2048_]] : i64
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[VAR_3_:%.+]] = arith.cmpi sge, [[VAR_2_]], [[CST_2097152_]] : index
// CHECK:           scf.if [[VAR_3_]] {
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_7_:%.+]] = affine.apply [[MAP_2_]]([[VAR_6_]])
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.min [[MAP_3_]]([[VAR_6_]]){{.}}[[VAR_dim_]]{{.}}
// CHECK:               [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_9_]], [[VAR_7_]], [[VAR_7_]]) : (memref<?x2048xf32>, memref<?x2048xf32>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           } else {
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[CST_0_]], [[CST_0_]]) : (memref<?x2048xf32>, memref<?x2048xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x2048xf32>
// CHECK:           krnl.parallel([[LOOP_1_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_]], [[VAR_dim_1_]]), [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2048){
// CHECK:             [[LOOP_0_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<?x2048xf32>
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<?x2048xi64>
// CHECK:             [[VAR_8_1_:%.+]] = arith.index_cast [[VAR_7_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.cmpi slt, [[VAR_8_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_8_1_]], [[VAR_dim_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.select [[VAR_9_1_]], [[VAR_10_]], [[VAR_8_1_]] : index
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_]], [[LOOP_0_]]#1] : memref<?x2048xf32>
// CHECK:             [[VAR_13_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[VAR_6_1_]] : f32
// CHECK:             krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_11_]], [[LOOP_0_]]#1] : memref<?x2048xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2048xf32>
// CHECK:         }

}

