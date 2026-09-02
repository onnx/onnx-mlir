// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel coverage for the retiled LayoutTransform fast path, which is the only
// LayoutTransform path with a full-rank selection window ([0, rank), floor 8).
// Reaching it needs a layout whose innermost map result is a modulo of the last
// dimension by at least 16, which the zTensor 3DS layout supplies ("d2 mod 64")
// and no plain ONNX layout does. Note the innermost level is iterated in tiles
// of 64, so its trip count here is ceilDiv(dim, 64), not dim.

// Dynamic throughout: level 0 is assumed wide enough, so the search stops there.
func.func @test_parallel_layout_transform_fast_dyn(%arg0: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<?x?x?xf16>) -> tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf16>
  return %1 : tensor<?x?x?xf16>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d0 ceildiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 * -64 + s0 - 64)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0)[s0] -> (d0 * -64 + s0)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_fast_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf16>) -> memref<?x?x?xf16> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf16>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf16>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]]) {{.*}}: memref<?x?x?xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_1_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_4_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<?x?x?xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<?x?x?xf16>
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_4_]], [[VAR_5_]]) : (memref<?x?x?xf16, #map>, memref<?x?x?xf16>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]]) {{.*}}: memref<?x?x?xf16>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_1_]], [[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_2_1_]]#2)
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<?x?x?xf16>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<?x?x?xf16, #map>
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_5_]]([[VAR_2_1_]]#2){{.}}[[VAR_dim_1_]]{{.}}
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi sge, [[VAR_6_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_7_]] {
// CHECK:               "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_64_]], [[VAR_4_1_]], [[VAR_5_1_]]) : (memref<?x?x?xf16>, memref<?x?x?xf16, #map>, i64, index, index) -> ()
// CHECK:             } else {
// CHECK:               [[VAR_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_2_1_]]#2){{.}}[[VAR_dim_1_]]{{.}}
// CHECK:               [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_1_]], [[RES_]], [[VAR_9_]], [[VAR_4_1_]], [[VAR_5_1_]]) : (memref<?x?x?xf16>, memref<?x?x?xf16, #map>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x?x?xf16>
// CHECK:         }

}

// -----

// Levels 0 and 1 have a trip count of 1, so the search reaches level 2, whose
// retiled trip count is ceilDiv(512, 64) = 8 -- exactly the floor.
func.func @test_parallel_layout_transform_fast_narrow_prefix(%arg0: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<1x1x512xf16>) -> tensor<1x1x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<1x1x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x1x512xf16>
  return %1 : tensor<1x1x512xf16>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_fast_narrow_prefix
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x512xf16>) -> memref<1x1x512xf16> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x512xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#2) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<1x1x512xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<1x1x512xf16>
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_4_]], [[VAR_5_]]) : (memref<1x1x512xf16, #map>, memref<1x1x512xf16>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x512xf16>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_1_]]#2) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_1_]]#2)
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<1x1x512xf16>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<1x1x512xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_64_]], [[VAR_4_1_]], [[VAR_5_1_]]) : (memref<1x1x512xf16>, memref<1x1x512xf16, #map>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<1x1x512xf16>
// CHECK:         }

}

// -----

// A wide static leading dimension is taken at level 0 without looking deeper.
func.func @test_parallel_layout_transform_fast_wide_outer(%arg0: tensor<32x8x512xf16>) -> tensor<32x8x512xf16> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<32x8x512xf16>) -> tensor<32x8x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "onnx.LayoutTransform"(%0) : (tensor<32x8x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<32x8x512xf16>
  return %1 : tensor<32x8x512xf16>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL:  func.func @test_parallel_layout_transform_fast_wide_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<32x8x512xf16>) -> memref<32x8x512xf16> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<32x8x512xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 32, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<32x8x512xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<32x8x512xf16>
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_4_]], [[VAR_5_]]) : (memref<32x8x512xf16, #map>, memref<32x8x512xf16>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32x8x512xf16>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 8, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_1_]]#2)
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<32x8x512xf16>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_3_1_]]{{.}} : memref<32x8x512xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_64_]], [[VAR_4_1_]], [[VAR_5_1_]]) : (memref<32x8x512xf16>, memref<32x8x512xf16, #map>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<32x8x512xf16>
// CHECK:         }

}

