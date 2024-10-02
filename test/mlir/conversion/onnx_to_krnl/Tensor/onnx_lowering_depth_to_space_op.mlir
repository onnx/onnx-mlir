// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test whether the lowering is correct in the presence of dynamic dimensions.
func.func private @test_depth_to_space_dynamic_dims(%arg0 : tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64} : (tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32>
  "func.return"(%0) : (tensor<1x?x32x?xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 floordiv 16)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-LABEL:  func.func private @test_depth_to_space_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x8x?xf32>) -> memref<1x?x32x?xf32> {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x8x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x?x8x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xindex>
// CHECK:           krnl.store [[CST_1_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[CST_4_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[CST_4_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_3_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[CST_8_]], [[RES_]]{{.}}[[CST_4_]]{{.}} : memref<6xindex>
// CHECK:           krnl.store [[VAR_dim_0_]], [[RES_]]{{.}}[[CST_5_]]{{.}} : memref<6xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[RES_]] : memref<6xindex> to tensor<6xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : memref<1x?x8x?xf32> to tensor<1x?x8x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x?x8x?xf32>, tensor<6xi64>) -> tensor<?x?x?x?x?x?xf32>
// CHECK:           [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[VAR_4_]] : tensor<?x?x?x?x?x?xf32> to memref<?x?x?x?x?x?xf32>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[VAR_5_]] : memref<?x?x?x?x?x?xf32> to memref<1x4x4x?x8x?xf32>
// CHECK:           [[VAR_6_:%.+]] = builtin.unrealized_conversion_cast [[VAR_cast_]] : memref<1x4x4x?x8x?xf32> to tensor<1x4x4x?x8x?xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x4x4x?x8x?xf32>) -> tensor<1x?x8x4x?x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xindex>
// CHECK:           krnl.store [[CST_1_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[VAR_0_]], [[RES_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[CST_32_]], [[RES_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xindex>
// CHECK:           krnl.store [[VAR_1_]], [[RES_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xindex>
// CHECK:           [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<4xindex> to tensor<4xi64>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_7_]], [[VAR_8_]]) {allowzero = 0 : si64} : (tensor<1x?x8x4x?x4xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK:           [[VAR_10_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]] : tensor<?x?x?x?xf32> to memref<?x?x?x?xf32>
// CHECK:           [[VAR_cast_2_:%.+]] = memref.cast [[VAR_10_]] : memref<?x?x?x?xf32> to memref<1x?x32x?xf32>
// CHECK:           return [[VAR_cast_2_]] : memref<1x?x32x?xf32>
}
