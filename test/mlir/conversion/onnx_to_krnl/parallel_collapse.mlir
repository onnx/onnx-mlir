// RUN: onnx-mlir-opt -O3 --convert-onnx-to-krnl="enable-parallel enable-collapse" --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt -O3 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s --check-prefix=NOCOLLAPSE

// GROUND-ALL: -c="-O3 --parallel" -a="--enable-collapse"

// The collapse-aware parallel decision, one function per outcome it can reach.
// The NOCOLLAPSE prefix is the same input with collapse off: what it pins is that
// turning the flag on is *additive* -- the region today's code creates is still
// created, now spanning two levels instead of one -- rather than trading one
// region for another.
//
// Covered here: Slice, Expand, Concat, Transpose's block path and Gather. Gather
// is the one with a full-rank collapse window, so it hosts the cases the others
// cannot express -- a group that starts below level 0, and a decision reached
// over more than two levels. LayoutTransform's fast path also has a full-rank
// window but needs a zTensor layout, so it lives under test/mlir/accelerators.

// STEP 2: a dynamic outer level absorbed into the group above the static one it
// sits on. This is the granite-4 shape, and the case the whole change exists
// for: with collapse off the region is sized on a dynamic dimension alone, so at
// batch 1 it forks a thread team around a single iteration.
//
// The trace shows why the group is {0,1} and not {1}: both are wide enough
// (verdict YES), but {1} leaves the dynamic level *above* the region, so the
// number of region entries is unknown and it loses on that gate. Absorbing that
// level costs {0,1} one shift and mask, 3% of the work a fused iteration does.
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

// -----

// Expand, the granite-4 shape. Same outcome and same reason as the Slice case
// above -- {0,1} has a known entry count and {1} does not -- but on a rank-5 nest, which is
// what makes the point that the group is bounded by the site's window and not by
// the nest: levels 2..4 stay sequential because Expand's window stops at 2, even
// though absorbing level 2 would make the region 16-wide rather than 4-wide.
// Widening that window is a separate decision, and on its own it would not be
// enough: growth also stops as soon as the target is met, and 4 already meets
// Expand's floor of 4.
// GROUND-THIS: --shape-info=0:3x4x1x7x128
func.func @test_collapse_expand(%arg0 : tensor<?x4x1x?x128xf32>) -> tensor<?x4x4x?x128xf32> {
  %shape = onnx.Constant dense<[1, 4, 4, 1, 128]> : tensor<5xi64>
  %1 = "onnx.Expand"(%arg0, %shape) : (tensor<?x4x1x?x128xf32>, tensor<5xi64>) -> tensor<?x4x4x?x128xf32>
  "func.return"(%1) : (tensor<?x4x4x?x128xf32>) -> ()


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_collapse_expand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x1x?x128xf32>) -> memref<?x4x4x?x128xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x1x?x128xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x4x1x?x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x4x4x?x128xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 128){
// CHECK:             [[VAR_2_:%.+]]:5 = krnl.get_induction_var_value([[VAR_1_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_0_]], [[VAR_2_]]#3, [[VAR_2_]]#4] : memref<?x4x1x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3, [[VAR_2_]]#4] : memref<?x4x4x?x128xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x4x?x128xf32>
// CHECK:         }

// Flag off: the region lands on the dynamic level alone, and the nest keeps all
// five levels.
// NOCOLLAPSE-LABEL:  func.func @test_collapse_expand
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:5 = krnl.define_loops 5
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#0) : !krnl.loop
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2, [[LOOP_OFF_]]#3, [[LOOP_OFF_]]#4)
}

// -----

// Concat with axis outside the window, so the exclusion is vacuous and the safe
// run is the whole window. Two things worth pinning: the group is {0,1} per
// input, since loopDef and the plan are both per-input, and the decision is made
// on commonUB -- the bounds the iterate actually consumes -- rather than on input
// i's own dims.
// GROUND-THIS: --shape-info=0:3x4x7x128,1:3x4x5x128
func.func @test_collapse_concat_axis_outside_window(%arg0 : tensor<?x4x?x128xf32>, %arg1 : tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1) {axis = 2 : si64} : (tensor<?x4x?x128xf32>, tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32>
  "func.return"(%1) : (tensor<?x4x?x128xf32>) -> ()


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func.func @test_collapse_concat_axis_outside_window
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x?x128xf32>, [[PARAM_1_:%.+]]: memref<?x4x?x128xf32>) -> memref<?x4x?x128xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x?x128xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x4x?x128xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<?x4x?x128xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_0_]]) {{.*}}: memref<?x4x?x128xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x4x?x128xf32>
// CHECK:           [[VAR_2_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_2_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_2_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_2_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 128){
// CHECK:             [[VAR_5_:%.+]]:4 = krnl.get_induction_var_value([[VAR_2_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2, [[VAR_5_]]#3] : memref<?x4x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2, [[VAR_5_]]#3] : memref<?x4x?x128xf32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<?x4x?x128xf32>
// CHECK:           [[VAR_4_:%.+]] = krnl.collapse([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_4_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_4_]], [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_2_]]), [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_2_]], [[VAR_dim_3_]]), [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 128){
// CHECK:             [[VAR_5_1_:%.+]]:4 = krnl.get_induction_var_value([[VAR_4_]], [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_5_1_]]#2){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#1, [[VAR_5_1_]]#2, [[VAR_5_1_]]#3] : memref<?x4x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_5_1_]]#3] : memref<?x4x?x128xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x?x128xf32>
// CHECK:         }

// Flag off: one region per input, each on the dynamic level alone.
// NOCOLLAPSE-LABEL:  func.func @test_collapse_concat_axis_outside_window
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          [[LOOP_OFF_0_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_0_]]#0) : !krnl.loop
// NOCOLLAPSE:          [[LOOP_OFF_1_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_1_]]#0) : !krnl.loop
}

// -----

// The same Concat with axis = 1, where the exclusion bites: it removes level 1
// from a window of 2, leaving {0} as the only safe level and no run of two
// adjacent ones. The frame therefore never reaches a policy -- the debug trace
// reads "Pick dim 0 because ub is dyn", STEP 0's own message -- and the IR is
// identical to flag-off rather than being subject to STEP 1's stricter rules.
// This is the exclusion path, and the only outcome it can produce for a window
// of 2.
// GROUND-THIS: --shape-info=0:3x4x7x128,1:3x4x7x128
func.func @test_par_excluded_dim_bites(%arg0 : tensor<?x4x?x128xf32>, %arg1 : tensor<?x4x?x128xf32>) -> tensor<?x8x?x128xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x4x?x128xf32>, tensor<?x4x?x128xf32>) -> tensor<?x8x?x128xf32>
  "func.return"(%1) : (tensor<?x8x?x128xf32>) -> ()


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func @test_par_excluded_dim_bites
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x?x128xf32>, [[PARAM_1_:%.+]]: memref<?x4x?x128xf32>) -> memref<?x8x?x128xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x?x128xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x4x?x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x8x?x128xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 128){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x4x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x8x?x128xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 128){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<?x4x?x128xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<?x8x?x128xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8x?x128xf32>
// CHECK:         }

// Flag off is the same IR, which is this function's whole point: with no run of
// two adjacent safe levels the frame quick-exits and the flag cannot change
// anything. Asserted rather than assumed, since it is the one property an
// exclusion has to guarantee.
// NOCOLLAPSE-LABEL:  func.func @test_par_excluded_dim_bites
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          [[LOOP_OFF_0_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_0_]]#0) : !krnl.loop
// NOCOLLAPSE:          [[LOOP_OFF_1_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_1_]]#0) : !krnl.loop
}

// -----

// Transpose's block path, perm [0,2,1,3]. The last dimension is unpermuted, so
// numLastDims is 1 and the nest is the outer [batch,16,seq] with a 128-element
// memcpy as its body -- the one covered site whose bounds are a truncation of the
// output rank and whose body is bulk work rather than one element. That body is
// declared as bodyCost, which is what puts the group's index recovery at 1 cycle
// per 100 work units: one shift and mask against a 128-element copy. Left at the
// default of 1 the same group would be priced at 200. It would still be formed
// here -- the level below it is dynamic, so the growth guard reads MAYBE rather
// than NO -- so what the declaration buys at this site is an honest price rather
// than a different answer. It is a site whose inner levels are *literal* that
// gets a different answer, LayoutTransform's retiled nest being the one in the
// tree; see the cost-model banner in KrnlParallelPlan.cpp.
// GROUND-THIS: --shape-info=0:3x16x7x128
func.func @test_collapse_block_transpose(%arg0 : tensor<?x16x?x128xf32>) -> tensor<?x?x16x128xf32> {
  %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<?x16x?x128xf32>) -> tensor<?x?x16x128xf32>
  "func.return"(%1) : (tensor<?x?x16x128xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 2048)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 2048)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-LABEL:  func.func @test_collapse_block_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x16x?x128xf32>) -> memref<?x?x16x128xf32> {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x16x?x128xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x16x?x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x?x16x128xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x16x?x128xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x16x?x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[VAR_4_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_4_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_4_]], [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_2_]], [[VAR_dim_0_]], [[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_2_]], [[VAR_dim_0_]], [[VAR_dim_1_]])){
// CHECK:             [[VAR_5_:%.+]]:3 = krnl.get_induction_var_value([[VAR_4_]], [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.muli [[VAR_5_]]#0, [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.muli [[VAR_5_]]#0, [[VAR_2_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.muli [[VAR_5_]]#1, [[VAR_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_6_]], [[VAR_8_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.apply [[MAP_4_]]([[VAR_5_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_7_]], [[VAR_10_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_5_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_9_]], [[VAR_12_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = affine.apply [[MAP_5_]]([[VAR_5_]]#1)
// CHECK:             [[VAR_15_:%.+]] = arith.addi [[VAR_11_]], [[VAR_14_]] : index
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_128_]], [[VAR_15_]], [[VAR_13_]]) : (memref<?x?x16x128xf32>, memref<?x16x?x128xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x16x128xf32>
// CHECK:         }

// Flag off: the region lands on the dynamic level alone, over the same 3-level
// truncated nest.
// NOCOLLAPSE-LABEL:  func.func @test_collapse_block_transpose
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:3 = krnl.define_loops 3
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#0) : !krnl.loop
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2)
}


// -----

// Gather, and the case no other covered site can produce: a group that starts
// *below* level 0. Level 0 has a trip count of 1, which STEP 2 refuses to lead
// with -- it buys no width and costs a recovery level -- so the run [0,4) yields
// its best candidate at d = 1, and levels 1 and 2 fuse to a guaranteed 4 while
// level 0 stays a sequential `0 to 1` wrapper above the region. Level 3 is
// rejected outright: its prefix is a static 4, above maxForkCount.
//
// Gather is also the only migrated site whose windows are the whole rank, which
// is what lets a run of three exist here at all.
// GROUND-THIS: --shape-info=0:8x2x2x7,1:1 --lower-bound=int64:0 --upper-bound=int64:7
func.func @test_collapse_gather_group_below_level_0(%arg0: tensor<8x2x2x?xf32>, %arg1: tensor<1xi64>) -> tensor<1x2x2x?xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<8x2x2x?xf32>, tensor<1xi64>) -> tensor<1x2x2x?xf32>
  return %0 : tensor<1x2x2x?xf32>


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_collapse_gather_group_below_level_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x2x2x?xf32>, [[PARAM_1_:%.+]]: memref<1xi64>) -> memref<1x2x2x?xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<8x2x2x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<1x2x2x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[VAR_1_]], [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[VAR_1_]], [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0] : memref<1xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_8_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<8x2x2x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<1x2x2x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2x2x?xf32>
// CHECK:         }


}

// -----

// Gather on an all-static shape, where the flag changes which level carries the
// region rather than widening one. Flag off, the search takes the first level
// wide enough and lands on level 2 (64), entered 2 x 2 = 4 times. Flag on, STEP 1
// refuses it because that prefix is above maxForkCount, and STEP 2 fuses {0,1}
// into a guaranteed 4 entered once. Narrower but entered once, which is the
// trade maxForkCount encodes -- worth pinning because it is the clearest case of
// the flag *moving* a region instead of growing one.
// GROUND-THIS: --shape-info=0:2x2x64,1:2 --lower-bound=int64:0 --upper-bound=int64:1
func.func @test_collapse_gather_static_moves_region(%arg0: tensor<2x2x64xf32>, %arg1: tensor<2xi64>) -> tensor<2x2x64xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2x2x64xf32>, tensor<2xi64>) -> tensor<2x2x64xf32>
  return %0 : tensor<2x2x64xf32>

// CHECK-LABEL:  func.func @test_collapse_gather_static_moves_region
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x64xf32>, [[PARAM_1_:%.+]]: memref<2xi64>) -> memref<2x2x64xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_1_]], [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 64){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[VAR_1_]], [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0] : memref<2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_2_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<2x2x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<2x2x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x64xf32>
// CHECK:         }


}

