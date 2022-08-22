// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test whether the lowering is correct in the presence of dynamic dimensions.
func.func private @test_space_to_depth_dynamic_dims(%arg0 : tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32> {
  %0 = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64} : (tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32>
  "func.return"(%0) : (tensor<1x?x32x?xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-LABEL:  func private @test_space_to_depth_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x8x?xf32>) -> memref<1x?x2x?xf32> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x8x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x8x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply #map0(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map1(){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xindex>
// CHECK:           krnl.store [[VAR_c1_]], [[RES_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_c2_]], [[RES_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_c4_]], [[RES_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_c4_]], [[RES_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : memref<1x?x8x?xf32> to tensor<1x?x8x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = builtin.unrealized_conversion_cast [[RES_]] : memref<6xindex> to tensor<6xi64>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_6_]]) {allowzero = 0 : si64} : (tensor<1x?x8x?xf32>, tensor<6xi64>) -> tensor<?x?x?x?x?x?xf32>
// CHECK:           [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[VAR_7_]] : tensor<?x?x?x?x?x?xf32> to memref<?x?x?x?x?x?xf32>
// CHECK:           [[VAR_9_:%.+]] = memref.cast [[VAR_8_]] : memref<?x?x?x?x?x?xf32> to memref<1x?x2x4x?x4xf32>
// CHECK:           [[VAR_10_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]] : memref<1x?x2x4x?x4xf32> to tensor<1x?x2x4x?x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_10_]]) {perm = [0, 1, 3, 5, 2, 4]} : (tensor<1x?x2x4x?x4xf32>) -> tensor<1x?x4x4x2x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xindex>
// CHECK:           krnl.store [[VAR_c1_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[VAR_2_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[VAR_c2_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[VAR_3_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<4xindex>
// CHECK:           [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<4xindex> to tensor<4xi64>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Reshape"([[VAR_11_]], [[VAR_13_]]) {allowzero = 0 : si64} : (tensor<1x?x4x4x2x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK:           [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]] : tensor<?x?x?x?xf32> to memref<?x?x?x?xf32>
// CHECK:           [[VAR_16_:%.+]] = memref.cast [[VAR_15_]] : memref<?x?x?x?xf32> to memref<1x?x2x?xf32>
// CHECK:           return [[VAR_16_]] : memref<1x?x2x?xf32>
// CHECK:         }

}
