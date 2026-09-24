// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl="enable-parallel enable-collapse" --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s --check-prefix=NOCOLLAPSE

// Collapse coverage for the retiled LayoutTransform fast path, the sibling of
// test/mlir/conversion/onnx_to_krnl/parallel_collapse.mlir. It lives here rather
// than there because reaching the fast path needs a layout whose innermost map
// result is a modulo of the last dimension, which the zTensor 3DS layout supplies
// ("d2 mod 64") and no plain ONNX layout does.
//
// No GROUND directives: grounding compiles *and runs*, and an NNPA model cannot
// run on a non-z host, so these are compile-time assertions only.
//
// Two site declarations matter here, and the file exists to pin both.
//
// bodyCost = modVal: one innermost iteration copies a whole 64-element tile. At
// the default of 1 the growth guard would refuse every group and drop the region
// onto the innermost level alone under an unknown entry count -- worse than
// emitting no collapse.
//
// The collapse claim stops at E1, one short of the search window. E1 is the tile
// counter this pattern just created, not a data dimension, and its body is the
// bulk memcpy. Fusing it in dissolves that: measured per 128-byte copy, a claim
// over the whole rank costs 42 instructions and 2 hardware divides, while stopping
// at E1 costs 13 and 0.25 -- the recovery arithmetic and the layout map's own
// floordiv stay amortized over a row of tiles rather than one, and the backend can
// unroll the tile loop.

// The granite-4 shape: ?x?x512 retiles to [batch, seq, 8]. The two dynamic levels
// fuse and are entered once; the 8-tile loop stays sequential inside. Verdict is
// MAYBE rather than YES -- the fused trip count is guaranteed only 1 -- and that is
// accepted deliberately: if batch and seq really are both 1 then the whole op is
// eight 128-byte copies, and no placement of a region helps. The group is wide
// exactly when there is work to be wide over.
func.func @test_collapse_layout_transform_fast(%arg0: tensor<?x?x512xf16>) -> tensor<?x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<?x?x512xf16>) -> tensor<?x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<?x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// Flag off: the region sits on the dynamic level 0 alone, and all three levels
// stay in the iterate.

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL:  func.func @test_collapse_layout_transform_fast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x512xf16>) -> memref<?x?x512xf16, #map> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x512xf16>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x512xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x?x512xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[VAR_1_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[VAR_1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[VAR_1_]], [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_0_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[VAR_1_]], [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<?x?x512xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_3_]]{{.}} : memref<?x?x512xf16>
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_4_]], [[VAR_5_]]) : (memref<?x?x512xf16, #map>, memref<?x?x512xf16>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x512xf16, #map>
// CHECK:         }

// NOCOLLAPSE-LABEL:  func.func @test_collapse_layout_transform_fast
// NOCOLLAPSE:          [[LOOP_OFF_:%.+]]:3 = krnl.define_loops 3
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_]]#0) : !krnl.loop
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2)

}

// -----

// Fully static, so nothing is unknown: level 0 is 32, wide enough on its own and
// entered once, and STEP 1 takes it. Pins that a full-rank collapse window does
// not mean the site always fuses -- when a single level is already adequate under
// a known entry count, no group is built and no recovery is paid.
func.func @test_par_layout_transform_fast_static(%arg0: tensor<32x8x512xf16>) -> tensor<32x8x512xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<32x8x512xf16>) -> tensor<32x8x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<32x8x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @test_par_layout_transform_fast_static
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT:       krnl.collapse
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 32, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
}
