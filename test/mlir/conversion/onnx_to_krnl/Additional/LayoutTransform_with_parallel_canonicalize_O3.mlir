// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel coverage for the generic LayoutTransform path, the perfectly-nested
// copy. NCHW4C maps its innermost result to "d1 mod 4", which is neither the
// last dimension nor a modulo of at least 16, so these cases do not take the
// retiled fast path -- see the NNPA layout-transform-parallel test for that one.
//
// This path searches only [0, 1) and demands 128 iterations, so it either
// parallelizes level 0 or nothing at all.

// 256 >= 128, so level 0 is parallelized.
func.func @test_parallel_layout_transform_wide(%arg0: tensor<256x3x32x32xf32>) -> tensor<256x3x32x32xf32> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<256x3x32x32xf32>) -> tensor<256x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<256x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<256x3x32x32xf32>
  return %1 : tensor<256x3x32x32xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 4, d2, d3, d1 mod 4)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_wide
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x3x32x32xf32>) -> memref<256x3x32x32xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x3x32x32xf32, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 256, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<256x3x32x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<256x3x32x32xf32, #map>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<256x3x32x32xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 256, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<256x3x32x32xf32, #map>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<256x3x32x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<256x3x32x32xf32>
// CHECK:         }

}

// -----

// A dynamic leading dimension is assumed wide enough, so level 0 wins even
// though nothing is known about it.
func.func @test_parallel_layout_transform_dyn(%arg0: tensor<?x3x32x32xf32>) -> tensor<?x3x32x32xf32> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<?x3x32x32xf32>) -> tensor<?x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<?x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<?x3x32x32xf32>
  return %1 : tensor<?x3x32x32xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 4, d2, d3, d1 mod 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x32x32xf32>) -> memref<?x3x32x32xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x32x32xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x32x32xf32, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x3x32x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x3x32x32xf32, #map>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x32x32xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<?x3x32x32xf32, #map>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<?x3x32x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x3x32x32xf32>
// CHECK:         }

}

// -----

// 5 < 128, and the window stops at level 0, so no parallel region is created --
// even though the levels below hold plenty of work.
func.func @test_parallel_layout_transform_narrow(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x3x32x32xf32>
  return %1 : tensor<5x3x32x32xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 4, d2, d3, d1 mod 4)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_narrow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x3x32x32xf32>) -> memref<5x3x32x32xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 5, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32, #map>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32, #map>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<5x3x32x32xf32>
// CHECK:         }

}

