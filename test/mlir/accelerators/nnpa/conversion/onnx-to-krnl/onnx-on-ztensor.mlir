// RUN: onnx-mlir-opt --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Test doing unary element-wise computation directly on zTensor.
// Taking ONNXSqrtOp as the example.
// Need to check that the buffer is correctly aligned to 4K.
func.func @test_onnx_sqrt_ztensor(%arg0: tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>) -> memref<?x3x5x7xf16, #map> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x3x5x7xf16, #map>

// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {alignment = 4096 : i64} : memref<?x3x5x7xf16, #map>

// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x3x5x7xf16, #map>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map1([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 7){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x3x5x7xf16, #map>
// CHECK:             [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f16
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x3x5x7xf16, #map>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}

// -----

// Test doing broadcasting binary element-wise computation directly on zTensor.
// Taking ONNXAddOp as the example.
// Need to check that the buffer is correctly aligned to 4K.
func.func @test_onnx_add_ztensor(%arg0: tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>, tensor<?x3x5x1xf32, #zhigh.encoding<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf32, #zhigh.encoding<{dataLayout = "4D"}>>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_onnx_add_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x7xf16, #map> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[VAR_c0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max #map1(){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]

// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {alignment = 4096 : i64} : memref<?x3x5x7xf16, #map>

// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map2([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[VAR_c1_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]]#0, [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[VAR_c1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_2_]]#0, [[VAR_c0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_]], [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_c0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f16
// CHECK:             krnl.store [[VAR_9_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x3x5x7xf16, #map>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}
