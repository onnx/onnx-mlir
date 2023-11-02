// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_stickified_constant_of_shape(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32, #zhigh.layout<{dataLayout = "2D"}>> {
  %0 = onnx.Constant dense<8.000000e+00> : tensor<f32>
  %2 = "onnx.Dim"(%arg0) {axis = 0 : si64} : (tensor<?x10xf32>) -> tensor<1xi64>
  %3 = onnx.Constant dense<10> : tensor<1xi64>
  %4 = "onnx.Concat"(%2, %3) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %5 = "zhigh.StickifiedConstantOfShape"(%4) {layout = "2D", value = 8.000000e+00 : f32} : (tensor<2xi64>) -> tensor<?x10xf32, #zhigh.layout<{dataLayout = "2D"}>>
  return %5 : tensor<?x10xf32, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func.func @test_stickified_constant_of_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf16, #map> {
// CHECK-DAG:       [[CST_1_dot_740800_:%.+]] = arith.constant 1.740800e+04 : f16
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_RES_1_MEM_]] : i64 to index
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?x10xf16, #map>
// CHECK:           krnl.memset [[RES_2_]], [[CST_1_dot_740800_]] {delayed = true} : memref<?x10xf16, #map>
// CHECK:           return [[RES_2_]] : memref<?x10xf16, #map>
// CHECK:         }
}
