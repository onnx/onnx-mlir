// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @matmul(%arg0: tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg1: tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg2: tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
 %0 ="zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> 
 return %0 : tensor<*xf32> 

// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8xf16, #map0>, [[PARAM_1_:%.+]]: memref<8x16xf16, #map0>, [[PARAM_2_:%.+]]: memref<16xf16, #map1>) -> memref<4x16xf16, #map0> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c16_i64_:%.+]] = arith.constant 16 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map0>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c8_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c16_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.matmul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<4x8xf16, #map0>, memref<8x16xf16, #map0>, memref<16xf16, #map1>, memref<3xi64>, memref<4x16xf16, #map0>) -> ()
// CHECK:           return [[RES_]] : memref<4x16xf16, #map0>
// CHECK:         }
}

// -----

func.func @matmul_stack(%arg0: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1: tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg2: tensor<2x16xf32, #zhigh.encoding<{dataLayout = "2DS"}>>) -> tensor<*xf32> {
 %0 ="zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x8x16xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<2x16xf32, #zhigh.encoding<{dataLayout = "2DS"}>>) -> tensor<*xf32> 
 return %0 : tensor<*xf32> 

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d0, d1 floordiv 64, 0, 0, 31, d1 mod 64)>
// CHECK-LABEL:  func @matmul_stack
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8xf16, #map0>, [[PARAM_1_:%.+]]: memref<2x8x16xf16, #map0>, [[PARAM_2_:%.+]]: memref<2x16xf16, #map1>) -> memref<2x4x16xf16, #map0> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c16_i64_:%.+]] = arith.constant 16 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c2_i64_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x16xf16, #map0>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK:           krnl.store [[VAR_c2_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c8_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c16_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<4xi64>
// CHECK:           "zlow.matmul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {is_bcast = 0 : si64, is_stacked = -1 : si64} : (memref<2x4x8xf16, #map0>, memref<2x8x16xf16, #map0>, memref<2x16xf16, #map1>, memref<4xi64>, memref<2x4x16xf16, #map0>) -> ()
// CHECK:           return [[RES_]] : memref<2x4x16xf16, #map0>
// CHECK:         }
}

// -----

func.func @matmul_broadcast(%arg0: tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %arg1: tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg2: tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
 %0 ="zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<2x4x8xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<8x16xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<16xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> 
 return %0 : tensor<*xf32> 

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @matmul_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8xf16, #map0>, [[PARAM_1_:%.+]]: memref<8x16xf16, #map1>, [[PARAM_2_:%.+]]: memref<16xf16, #map2>) -> memref<2x4x16xf16, #map0> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c16_i64_:%.+]] = arith.constant 16 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c2_i64_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x16xf16, #map0>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK:           krnl.store [[VAR_c2_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c8_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[VAR_c16_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<4xi64>
// CHECK:           "zlow.matmul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {is_bcast = -1 : si64, is_stacked = 0 : si64} : (memref<2x4x8xf16, #map0>, memref<8x16xf16, #map1>, memref<16xf16, #map2>, memref<4xi64>, memref<2x4x16xf16, #map0>) -> ()
// CHECK:           return [[RES_]] : memref<2x4x16xf16, #map0>
// CHECK:         }
}

// -----

func.func @matmul_unknown_dims(%arg0: tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg1: tensor<8x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, %arg2: tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
 %0 ="zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<4x8xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<8x?xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> 
 return %0 : tensor<*xf32> 

// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @matmul_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8xf16, #map0>, [[PARAM_1_:%.+]]: memref<8x?xf16, #map0>, [[PARAM_2_:%.+]]: memref<?xf16, #map1>) -> memref<4x?xf16, #map0> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_1_]], [[VAR_c1_]] : memref<8x?xf16, #map0>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_1_]], [[VAR_c1_]] : memref<8x?xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<4x?xf16, #map0>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c8_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_4_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.matmul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<4x8xf16, #map0>, memref<8x?xf16, #map0>, memref<?xf16, #map1>, memref<3xi64>, memref<4x?xf16, #map0>) -> ()
// CHECK:           return [[RES_]] : memref<4x?xf16, #map0>
// CHECK:         }
}
