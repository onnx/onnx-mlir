// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func @test_concat(%arg0: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<?x4x4x192xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #map3 = affine_map<(d0) -> (d0 + 192)>
// CHECK-LABEL:  func @test_concat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x4x192xf16, #map0>, [[PARAM_1_:%.+]]: memref<?x4x4x192xf16, #map0>) -> memref<?x4x4x384xf16, #map0> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x4x4x192xf16, #map0>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x4x4x384xf16, #map0>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x4x4x192xf16, #map0>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map1([[VAR_3_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 192){
// CHECK:             [[VAR_6_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2, [[VAR_6_]]#3] : memref<?x4x4x192xf16, #map0>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2, [[VAR_6_]]#3] : memref<?x4x4x384xf16, #map0>
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_1_]], [[VAR_c0_]] : memref<?x4x4x192xf16, #map0>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to #map2([[VAR_3_]], [[VAR_5_]]), [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 4, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 192){
// CHECK:             [[VAR_6_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply #map3([[VAR_6_1_]]#3)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1, [[VAR_6_1_]]#2, [[VAR_6_1_]]#3] : memref<?x4x4x192xf16, #map0>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1, [[VAR_6_1_]]#2, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<?x4x4x384xf16, #map0>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x4x384xf16, #map0>
// CHECK:         }
}
