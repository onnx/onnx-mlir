// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @reduce_max_axes_defined_noop_0(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>> { 
  %0 = "zhigh.ReduceMax"(%arg0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @reduce_max_axes_defined_noop_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x1xf16, #map> {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[CST_3_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_4_]], [[RES_1_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_5_]], [[RES_1_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<8192xi8>
// CHECK:           "zlow.reducemax"([[PARAM_0_]], [[RES_2_]], [[RES_1_]], [[RES_]]) {layout = "3DS"} : (memref<3x4x5xf16, #map>, memref<8192xi8>, memref<3xi64>, memref<3x4x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<3x4x1xf16, #map>
// CHECK:         }
}
