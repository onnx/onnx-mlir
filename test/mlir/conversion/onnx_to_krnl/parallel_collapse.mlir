// RUN: onnx-mlir-opt -O3 --convert-onnx-to-krnl="enable-parallel enable-collapse" --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt -O3 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s --check-prefix=NOCOLLAPSE

// GROUND-ALL: -c="-O3 --parallel" -a="--enable-collapse"

// The collapse-aware parallel decision, one function per outcome it can reach.
// The NOCOLLAPSE prefix is the same input with collapse off: what it pins is that
// turning the flag on is *additive* -- the region today's code creates is still
// created, now spanning two levels instead of one -- rather than trading one
// region for another.
//
// Only Slice is covered here: it is the one site migrated so far. A group starting
// below level 0, and an excluded dimension that actually bites, need a collapse
// window wider than 2, so they arrive with Gather, Concat and LayoutTransform.

// STEP 2: a dynamic outer level absorbed into the group above the static one it
// sits on. This is the granite-4 shape, and the case the whole change exists
// for: with collapse off the region is sized on a dynamic dimension alone, so at
// batch 1 it forks a thread team around a single iteration.
//
// The trace shows why the group is {0,1} and not {1}: both candidates are wide
// enough (verdict YES), and the single level loses on cost, 4000 against 0 --
// the fork penalty for leaving the dynamic level *above* the region rather than
// absorbing it. Absorb-versus-tolerate is decided by price, not by rule.
// GROUND-THIS: --shape-info=0:3x16x7x128
func.func @test_collapse_dyn_outer(%arg0 : tensor<?x16x?x128xf32>) -> tensor<?x16x?x64xf32> {
  %axes = onnx.Constant dense<[3]> : tensor<1xi64>
  %starts = onnx.Constant dense<[64]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9223372036854775807]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x16x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x16x?x64xf32>
  "func.return"(%1) : (tensor<?x16x?x64xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 64)>
// CHECK-LABEL:  func.func @test_collapse_dyn_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x16x?x128xf32>) -> memref<?x16x?x64xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x16x?x128xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x16x?x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x16x?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 64){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_2_]]#3)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_3_]]{{.}} : memref<?x16x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x16x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x16x?x64xf32>
// CHECK:         }

// Same shape with the flag off: a region on the dynamic level alone and no
// krnl.collapse. Only the absent op and the level the region lands on are checked;
// a second transcript of the body would just give the two prefixes a place to
// drift apart.
// NOCOLLAPSE-LABEL:  func.func @test_collapse_dyn_outer
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#0) : !krnl.loop
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2, [[LOOP_OFF_]]#3)
}

// -----

// STEP 1: a single level that is literally wide enough, under a statically small
// prefix, so there is nothing to fuse and no index recovery to pay for. A group
// would be strictly worse here, and the step exists to say so.
func.func @test_par_single_static_outer(%arg0 : tensor<64x4x16xf32>) -> tensor<64x4x8xf32> {
  %axes = onnx.Constant dense<[2]> : tensor<1xi64>
  %starts = onnx.Constant dense<[8]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9223372036854775807]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<64x4x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<64x4x8xf32>
  "func.return"(%1) : (tensor<64x4x8xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-LABEL:  func.func @test_par_single_static_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x4x16xf32>) -> memref<64x4x8xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<64x4x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#2)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<64x4x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<64x4x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x4x8xf32>
// CHECK:         }

}

// -----

// STEP 2 reaching a group of one, which is the identity: KrnlBuilder::collapse
// returns the single loop and emits no op, so the result is an ordinary
// krnl.parallel. Growth stops before level 1 because what is left sequential
// inside the group would be a single element, far below minAmortWork -- there is
// no work to amortize a divide against. Worth pinning precisely because the
// answer is "no collapse" while arriving through STEP 2 rather than STEP 1.
// GROUND-THIS: --shape-info=0:5x1x1x2
func.func @test_par_one_member_group(%arg0 : tensor<?x1x1x2xf32>) -> tensor<?x1x1x1xf32> {
  %axes = onnx.Constant dense<[3]> : tensor<1xi64>
  %starts = onnx.Constant dense<[1]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9223372036854775807]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x1x1x2xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x1x1x1xf32>
  "func.return"(%1) : (tensor<?x1x1x1xf32>) -> ()
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func.func @test_par_one_member_group
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x2xf32>) -> memref<?x1x1x1xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x1x1x1xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<?x1x1x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x1x1x1xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x1x1x1xf32>
// CHECK:         }

}
