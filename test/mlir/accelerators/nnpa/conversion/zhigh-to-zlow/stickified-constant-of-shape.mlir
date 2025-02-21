// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_stickified_constant_of_shape(%arg0: tensor<?x10xf16>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>> {
  %0 = onnx.Constant dense<8.000000e+00> : tensor<f32>
  %2 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x10xf16>) -> tensor<1xi64>
  %3 = onnx.Constant dense<10> : tensor<1xi64>
  %4 = "onnx.Concat"(%2, %3) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %5 = "zhigh.StickifiedConstantOfShape"(%4) {layout = "2D", value = 8.000000e+00 : f32} : (tensor<2xi64>) -> tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %5 : tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func.func @test_stickified_constant_of_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf16>) -> memref<?x10xf16, #map> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_17408_:%.+]] = arith.constant 17408 : i16
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index

// CHECK-DAG:       {{.*}} = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-DAG:       {{.*}} = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc({{.*}}) {{.*}}: memref<?x10xf16, #map>

// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<i16>
// CHECK:           krnl.store [[CST_17408_]], [[RES_2_]][] : memref<i16>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() : memref<f16>
// CHECK:           "krnl.memcpy"([[RES_3_]], [[RES_2_]], [[CST_1_]], [[CST_0_]], [[CST_0_]]) : (memref<f16>, memref<i16>, i64, index, index) -> ()

// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to {{.*}}, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 10){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_3_]][] : memref<f16>
// CHECK:             krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf16, #map>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x10xf16, #map>
// CHECK:         }
}
