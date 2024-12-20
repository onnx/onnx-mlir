// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> { 
  %0 = "zhigh.Tanh"(%arg0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.tanh"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {layout = "3D"} : (memref<3x4x5xf16, #map>, memref<3xi64>, memref<3x4x5xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<3x4x5xf16, #map>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<3x?x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> { 
  %0 = "zhigh.Tanh"(%arg0) : (tensor<3x?x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?x5xf16, #map>) -> memref<3x?x5xf16, #map> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<3x?x5xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<3x?x5xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_3_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.tanh"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {layout = "3D"} : (memref<3x?x5xf16, #map>, memref<3xi64>, memref<3x?x5xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<3x?x5xf16, #map>
// CHECK:         }
}
