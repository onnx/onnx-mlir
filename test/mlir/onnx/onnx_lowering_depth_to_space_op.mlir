// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test whether the lowering is correct in the presence of dynamic dimensions.
func private @test_depth_to_space_dynamic_dims(%arg0 : tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64} : (tensor<1x?x8x?xf32>) -> tensor<1x?x32x?xf32>
  "func.return"(%0) : (tensor<1x?x32x?xf32>) -> ()

  // CHECK: [[MAP0:#.+]] = affine_map<()[s0] -> (s0 floordiv 16)>
  // CHECK: [[MAP1:#.+]] = affine_map<()[s0] -> (s0 * 4)>
  // CHECK-LABEL: test_depth_to_space_dynamic_dims
  // CHECK:  [[C3:%.+]] = arith.constant 3 : index
  // CHECK:  [[C2:%.+]] = arith.constant 2 : index
  // CHECK:  [[C1:%.+]] = arith.constant 1 : index
  // CHECK:  [[C0:%.+]] = arith.constant 0 : index
  // CHECK:  [[C32:%.+]] = arith.constant 32 : index
  // CHECK:  [[C5:%.+]] = arith.constant 5 : index
  // CHECK:  [[C4:%.+]] = arith.constant 4 : index
  // CHECK:  [[C8:%.+]] = arith.constant 8 : index
  // CHECK:  [[DIM1:%.+]] = memref.dim %arg0, [[C1]] : memref<1x?x8x?xf32>
  // CHECK:  [[DIM3:%.+]] = memref.dim %arg0, [[C3]] : memref<1x?x8x?xf32>
  // CHECK:  [[MAP0_1:%.+]] = affine.apply [[MAP0]](){{.*}}[[DIM1]]]{{.*}}
  // CHECK:  [[MAP1_3:%.+]] = affine.apply [[MAP1]](){{.*}}[[DIM3]]{{.*}}
  // CHECK:  [[ALLOCA1:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<6xindex>
  // CHECK:  krnl.store [[C1]], [[ALLOCA1]]{{.*}}[[C0]]{{.*}} : memref<6xindex>
  // CHECK:  krnl.store [[C4]], [[ALLOCA1]]{{.*}}[[C1]]{{.*}} : memref<6xindex>
  // CHECK:  krnl.store [[C4]], [[ALLOCA1]]{{.*}}[[C2]]{{.*}} : memref<6xindex>
  // CHECK:  krnl.store [[MAP0_1]], [[ALLOCA1]]{{.*}}[[C3]]{{.*}} : memref<6xindex>
  // CHECK:  krnl.store [[C8]], [[ALLOCA1]]{{.*}}[[C4]]{{.*}} : memref<6xindex>
  // CHECK:  krnl.store [[DIM3]], [[ALLOCA1]]{{.*}}[[C5]]{{.*}} : memref<6xindex>
  // CHECK:  [[RESHAPE1:%.+]] = "onnx.Reshape"(%arg0, [[ALLOCA1]]) : (memref<1x?x8x?xf32>, memref<6xindex>) -> memref<?x?x?x?x?x?xf32>
  // CHECK:  [[CAST1:%.+]] = memref.cast [[RESHAPE1]] : memref<?x?x?x?x?x?xf32> to memref<1x4x4x?x8x?xf32>
  // CHECK:  [[TRANSPOSE:%.+]] = "onnx.Transpose"([[CAST1]]) {perm = [0, 3, 4, 1, 5, 2]} : (memref<1x4x4x?x8x?xf32>) -> memref<1x?x8x4x?x4xf32>
  // CHECK:  [[ALLOCA2:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xindex>
  // CHECK:  krnl.store [[C1]], [[ALLOCA2]]{{.*}}[[C0]]{{.*}} : memref<4xindex>
  // CHECK:  krnl.store [[MAP0_1]], [[ALLOCA2]]{{.*}}[[C1]]{{.*}} : memref<4xindex>
  // CHECK:  krnl.store [[C32]], [[ALLOCA2]]{{.*}}[[C2]]{{.*}} : memref<4xindex>
  // CHECK:  krnl.store [[MAP1_3]], [[ALLOCA2]]{{.*}}[[C3]]{{.*}} : memref<4xindex>
  // CHECK:  [[RESHAPE2:%.+]] = "onnx.Reshape"([[TRANSPOSE]], [[ALLOCA2]]) : (memref<1x?x8x4x?x4xf32>, memref<4xindex>) -> memref<?x?x?x?xf32>
  // CHECK:  [[CAST2:%.+]] = memref.cast [[RESHAPE2]] : memref<?x?x?x?xf32> to memref<1x?x32x?xf32>
  // CHECK:  return [[CAST2]] : memref<1x?x32x?xf32>
}
