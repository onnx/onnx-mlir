// RUN: onnx-mlir-opt --convert-onnx-to-krnl="enable-parallel=true" --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-krnl="enable-parallel=true enable-collapse=false" --canonicalize %s -split-input-file | FileCheck %s --check-prefix=NOCOLLAPSE

// GROUND-ALL: -r=--O0 -t="-O3 --parallel" --rtol=0 --atol=0

// The search window of Expand's KrnlParallelPlan: levels [0, 2) for ranks 2
// and 3, widened to [0, 3) for rank 4 and up (never the innermost level). The
// window has no flag of its own, so grounding compares against a sequential
// -O0 build rather than against collapse off. The NOCOLLAPSE prefix pins that
// the level-2 parallel at batch 1 comes from the window, not from collapse.

// Attention mask at batch 1: 1x1x1xS -> 1x1xSxS. Levels 0 and 1 are literal 1,
// so the old two-level window found nothing; the widened window puts the
// region on level 2 (S), with or without collapse.
// GROUND-THIS: --shape-info=0:1x1x1x37
func.func @expand_bs1_mask(%arg0: tensor<1x1x1x?xf32> {onnx.name = "x"}) -> (tensor<1x1x?x?xf32> {onnx.name = "y"}) {
  %ones = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %s = "onnx.Dim"(%arg0) {axis = 3 : si64} : (tensor<1x1x1x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%ones, %s, %s) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x1x1x?xf32>, tensor<4xi64>) -> tensor<1x1x?x?xf32>
  return %0 : tensor<1x1x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @expand_bs1_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x1x?xf32> {onnx.name = "x"}) -> (memref<1x1x?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2], value = dense<1> : tensor<2xi64>}> : () -> memref<2xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x1x1x?xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<4xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_11_]]{{.}} : memref<2xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_1_]]{{.}}[[VAR_11_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1){
// CHECK:             [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_11_1_]])
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_1_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_]], [[RES_1_]]{{.}}[[LOAD_VAR_0_MEM_1_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_11_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_11_2_]])
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_2_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[LOAD_VAR_0_MEM_1_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x1x1x?xf32>
// CHECK:           [[VAR_9_:%.+]] = affine.max [[MAP_2_]](){{.}}[[VAR_8_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_9_]]) alignment = 16 : memref<1x1x?x?xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_3_]]#2) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) with ([[LOOP_3_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_5_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_1_]]){{.}}[[VAR_8_]], [[VAR_6_]]{{.}}, [[LOOP_3_]]#3 -> [[I_6_:%.+]] = 0 to [[VAR_9_]]){
// CHECK-DAG:         [[VAR_11_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = arith.select [[LOAD_VAR_0_MEM_1_1_]], [[VAR_11_3_]]#3, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]], [[CST_0_]], [[LOAD_RES_MEM_1_]]{{.}} : memref<1x1x1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_11_3_]]#0, [[VAR_11_3_]]#1, [[VAR_11_3_]]#2, [[VAR_11_3_]]#3] : memref<1x1x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<1x1x?x?xf32>
// CHECK:         }

// Collapse off: the same region on level 2, so it is the window, not collapse,
// that makes it reachable.
// NOCOLLAPSE-LABEL:  func.func @expand_bs1_mask
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#2) : !krnl.loop
}

// -----

// Attention mask at dynamic batch: Bx1x1xS -> Bx1xSxS. The widened window lets
// the collapse group grow to B*1*S (levels 0..2); before, it stopped at B*1.
// With collapse off the region stays on the dynamic batch level alone.
// GROUND-THIS: --shape-info=0:3x1x1x37
func.func @expand_dyn_batch_mask(%arg0: tensor<?x1x1x?xf32> {onnx.name = "x"}) -> (tensor<?x1x?x?xf32> {onnx.name = "y"}) {
  %one = onnx.Constant dense<[1]> : tensor<1xi64>
  %b = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x1x1x?xf32>) -> tensor<1xi64>
  %s = "onnx.Dim"(%arg0) {axis = 3 : si64} : (tensor<?x1x1x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%b, %one, %s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<?x1x1x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
  return %0 : tensor<?x1x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func @expand_dyn_batch_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x?xf32> {onnx.name = "x"}) -> (memref<?x1x?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<1> : tensor<1xi64>}> : () -> memref<1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1x?xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<1xi64>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x1x1x?xf32>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_1_]] : index to i64
// CHECK:           krnl.store [[VAR_2_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<4xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_17_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_17_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_]], [[RES_2_]]{{.}}[[VAR_17_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1){
// CHECK:             [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_17_1_]])
// CHECK-DAG:         [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_17_1_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_2_]]{{.}}[[LOAD_RES_MEM_1_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_17_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_17_2_]])
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_17_2_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_1_]], [[RES_2_]]{{.}}[[LOAD_RES_MEM_1_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:             [[VAR_17_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_MEM_1_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_17_3_]])
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_17_3_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_1_]], [[RES_2_]]{{.}}[[LOAD_RES_MEM_1_1_]]{{.}} : memref<4xi64>
// CHECK:           }
// CHECK:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1x?xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x1x1x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = affine.max [[MAP_3_]](){{.}}[[VAR_8_]], [[VAR_dim_3_]]{{.}}
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.max [[MAP_3_]](){{.}}[[VAR_12_]], [[VAR_dim_4_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[VAR_13_]], [[VAR_10_]], [[VAR_14_]]) alignment = 16 : memref<?x1x?x?xf32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_16_:%.+]] = krnl.collapse([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_16_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_16_]], [[LOOP_4_]]#3) with ([[LOOP_4_]]#0 -> [[I_4_:%.+]] = 0 to [[VAR_13_]], [[LOOP_4_]]#1 -> [[I_5_:%.+]] = 0 to 1, [[LOOP_4_]]#2 -> [[I_6_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_3_]], [[VAR_dim_4_]]){{.}}[[VAR_8_]], [[VAR_12_]], [[VAR_10_]]{{.}}, [[LOOP_4_]]#3 -> [[I_7_:%.+]] = 0 to [[VAR_14_]]){
// CHECK-DAG:         [[VAR_17_4_:%.+]]:4 = krnl.get_induction_var_value([[VAR_16_]], [[LOOP_4_]]#3) : (!krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_RES_MEM_1_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_3_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_1_:%.+]] = arith.select [[LOAD_RES_MEM_1_1_]], [[VAR_17_4_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.cmpi sgt, [[VAR_dim_4_]], [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[VAR_17_4_]]#3, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_VAR_0_MEM_1_1_]], [[CST_0_]], [[CST_0_]], [[VAR_21_]]{{.}} : memref<?x1x1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_17_4_]]#0, [[VAR_17_4_]]#1, [[VAR_17_4_]]#2, [[VAR_17_4_]]#3] : memref<?x1x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<?x1x?x?xf32>
// CHECK:         }

// Collapse off: the region stays on the dynamic batch level alone.
// NOCOLLAPSE-LABEL:  func.func @expand_dyn_batch_mask
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#0) : !krnl.loop
}

// -----

// Heads broadcast at batch 1: 1x1xSxT -> 1x12xSxT. Unchanged by the wider
// window: level 1 (12) is wide enough, so the region stays there.
// GROUND-THIS: --shape-info=0:1x1x7x37
func.func @expand_bs1_heads(%arg0: tensor<1x1x?x?xf32> {onnx.name = "x"}) -> (tensor<1x12x?x?xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 12, 1, 1]> : tensor<4xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x1x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
  return %0 : tensor<1x12x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @expand_bs1_heads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x?x?xf32> {onnx.name = "x"}) -> (memref<1x12x?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x1x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x1x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<1x12x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 12, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x1x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x12x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x12x?x?xf32>
// CHECK:         }


}

// -----

// Heads broadcast at dynamic batch: Bx1xSxT -> Bx12xSxT. Unchanged: the group
// B*12 already meets the target, so growth stops before level 2.
// GROUND-THIS: --shape-info=0:3x1x7x37
func.func @expand_dyn_batch_heads(%arg0: tensor<?x1x?x?xf32> {onnx.name = "x"}) -> (tensor<?x12x?x?xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 12, 1, 1]> : tensor<4xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<?x1x?x?xf32>, tensor<4xi64>) -> tensor<?x12x?x?xf32>
  return %0 : tensor<?x12x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @expand_dyn_batch_heads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x?x?xf32> {onnx.name = "x"}) -> (memref<?x12x?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x1x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]]) alignment = 16 : memref<?x12x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 12, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]])){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x1x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x12x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x12x?x?xf32>
// CHECK:         }


}

// -----

// Rank 3 keeps the two-level window: 1x1xS -> 1xSxS puts the region on level
// 1 (S), which the old window already reached.
// GROUND-THIS: --shape-info=0:1x1x37
func.func @expand_rank3(%arg0: tensor<1x1x?xf32> {onnx.name = "x"}) -> (tensor<1x?x?xf32> {onnx.name = "y"}) {
  %one = onnx.Constant dense<[1]> : tensor<1xi64>
  %s = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<1x1x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%one, %s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x1x?xf32>, tensor<3xi64>) -> tensor<1x?x?xf32>
  return %0 : tensor<1x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @expand_rank3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x?xf32> {onnx.name = "x"}) -> (memref<1x?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<1> : tensor<1xi64>}> : () -> memref<1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x1x?xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<3xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_11_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_1_]]{{.}}[[VAR_11_]]{{.}} : memref<3xi64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1){
// CHECK:             [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_11_1_]])
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_1_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_]], [[RES_1_]]{{.}}[[LOAD_VAR_0_MEM_1_]]{{.}} : memref<3xi64>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_11_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_11_2_]])
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_2_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[LOAD_VAR_0_MEM_1_]]{{.}} : memref<3xi64>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x1x?xf32>
// CHECK:           [[VAR_9_:%.+]] = affine.max [[MAP_2_]](){{.}}[[VAR_8_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_9_]]) alignment = 16 : memref<1x?x?xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_3_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) with ([[LOOP_3_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_1_]]){{.}}[[VAR_8_]], [[VAR_6_]]{{.}}, [[LOOP_3_]]#2 -> [[I_5_:%.+]] = 0 to [[VAR_9_]]){
// CHECK-DAG:         [[VAR_11_3_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = arith.select [[LOAD_VAR_0_MEM_1_1_]], [[VAR_11_3_]]#2, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]], [[LOAD_RES_MEM_1_]]{{.}} : memref<1x1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_11_3_]]#0, [[VAR_11_3_]]#1, [[VAR_11_3_]]#2] : memref<1x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<1x?x?xf32>
// CHECK:         }


}

// -----

// Rank 3 with both window levels literal 1: 1x1x1 -> 1x1xS. The window must
// not reach level 2, the innermost, so there is no parallel region at all.
// GROUND-THIS: --input-value=1:val37
func.func @expand_rank3_no_par(%arg0: tensor<1x1x1xf32> {onnx.name = "x"}, %arg1: tensor<1xi64> {onnx.name = "n"}) -> (tensor<1x1x?xf32> {onnx.name = "y"}) {
  %ones = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %shape = "onnx.Concat"(%ones, %arg1) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x1x1xf32>, tensor<3xi64>) -> tensor<1x1x?xf32>
  return %0 : tensor<1x1x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL:  func.func @expand_rank3_no_par
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x1xf32> {onnx.name = "x"}, [[PARAM_1_:%.+]]: memref<1xi64> {onnx.name = "n"}) -> (memref<1x1x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2], value = dense<1> : tensor<2xi64>}> : () -> memref<2xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<3xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_6_]]{{.}} : memref<2xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]]{{.}}[[VAR_6_]]{{.}} : memref<3xi64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1){
// CHECK:             [[VAR_6_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_6_1_]])
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_6_1_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[LOAD_VAR_0_MEM_1_]]{{.}} : memref<3xi64>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_RES_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_4_]]) alignment = 16 : memref<1x1x?xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_4_:%.+]] = 0 to [[VAR_4_]]){
// CHECK-DAG:         [[VAR_6_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]], [[CST_0_]]{{.}} : memref<1x1x1xf32>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_6_2_]]#0, [[VAR_6_2_]]#1, [[VAR_6_2_]]#2] : memref<1x1x?xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<1x1x?xf32>
// CHECK:         }

// The CHECK lines above (collapse on) show no krnl.parallel but would not
// notice one being added; this pins its absence, here with collapse off.
// NOCOLLAPSE-LABEL:  func.func @expand_rank3_no_par
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:3 = krnl.define_loops 3
// NOCOLLAPSE-NOT:      krnl.parallel
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2)
}

// -----

// Rank 2: 1xS -> SxS, region on level 0.
// GROUND-THIS: --shape-info=0:1x37
func.func @expand_rank2(%arg0: tensor<1x?xf32> {onnx.name = "x"}) -> (tensor<?x?xf32> {onnx.name = "y"}) {
  %s = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<1x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<1x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @expand_rank2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_9_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_9_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_]], [[RES_1_]]{{.}}[[VAR_9_]]{{.}} : memref<2xi64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1){
// CHECK:             [[VAR_9_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_9_1_]])
// CHECK-DAG:         [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_9_1_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_1_]]{{.}} : memref<2xi64>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?xf32>
// CHECK:           [[VAR_7_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_7_]]) alignment = 16 : memref<?x?xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_2_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_1_]]){{.}}[[VAR_6_]], [[VAR_4_]]{{.}}, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to [[VAR_7_]]){
// CHECK-DAG:         [[VAR_9_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.select [[LOAD_RES_MEM_1_]], [[VAR_9_2_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[LOAD_RES_MEM_2_]]{{.}} : memref<1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<?x?xf32>
// CHECK:         }


}

