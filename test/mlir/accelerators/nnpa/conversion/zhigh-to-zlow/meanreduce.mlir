// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow(%arg0: tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x5x7x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x5x7x3xf16, #map>) -> memref<1x1x1x3xf16, #map> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_]]{{.}}[[VAR_c0_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_]]{{.}}[[VAR_c1_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_]]{{.}}[[VAR_c2_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_]]{{.}}[[VAR_c3_]]{{.}} : memref<4xi64>
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x3xf16, #map>
// CHECK:           "zlow.meanreduce2d"([[PARAM_0_]], [[RES_]], [[RES_1_]]) : (memref<1x5x7x3xf16, #map>, memref<4xi64>, memref<1x1x1x3xf16, #map>) -> ()
// CHECK:           return [[RES_1_]] : memref<1x1x1x3xf16, #map>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MeanReduce2d"(%arg0) : (tensor<1x?x?x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x3xf16, #map>) -> memref<1x1x1x3xf16, #map> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x3xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x3xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_]]{{.}}[[VAR_c0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_c1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_c2_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_]]{{.}}[[VAR_c3_]]{{.}} : memref<4xi64>
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x3xf16, #map>
// CHECK:           "zlow.meanreduce2d"([[PARAM_0_]], [[RES_]], [[RES_1_]]) : (memref<1x?x?x3xf16, #map>, memref<4xi64>, memref<1x1x1x3xf16, #map>) -> ()
// CHECK:           return [[RES_1_]] : memref<1x1x1x3xf16, #map>
// CHECK:         }
}
