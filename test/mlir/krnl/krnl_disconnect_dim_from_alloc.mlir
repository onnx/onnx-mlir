// RUN: onnx-mlir-opt --disconnect-dims %s -split-input-file | FileCheck %s

func @test_krnl_dim_lowering(%arg0: memref<?x?xf32>) -> index {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = alloc(%0) : memref<?x10xf32>
  %d0 = "krnl.dim"(%1, %c0) : (memref<?x10xf32>, index) -> index
  %d1 = "krnl.dim"(%1, %c1) : (memref<?x10xf32>, index) -> index
  %e = addi %d0, %d1 : index
  return %e : index

  // CHECK-LABEL: test_krnl_dim_lowering
  // CHECK: [[CONST0:%.+]] = constant 0 : index
  // CHECK: [[CONST10:%.+]] = constant 10 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: [[SUM:%.+]] = addi [[DIM]], [[CONST10]] : index
  // CHECK: return [[SUM]] : index
}

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
func @test_krnl_dim_lowering_with_map(%arg0: memref<?x?xf32>) -> index {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = alloc(%0) : memref<?x10xf32, #map>
  %d0 = "krnl.dim"(%1, %c0) : (memref<?x10xf32, #map>, index) -> index
  %d1 = "krnl.dim"(%1, %c1) : (memref<?x10xf32, #map>, index) -> index
  %e = addi %d0, %d1 : index
  return %e : index

  // CHECK: [[MAP:#.+]] = affine_map<(d0, d1) -> (d1, d0)>
  // CHECK-LABEL: test_krnl_dim_lowering_with_map
  // CHECK: [[CONST0:%.+]] = constant 0 : index
  // CHECK: [[CONST10:%.+]] = constant 10 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = alloc([[DIM]]) : memref<?x10xf32, [[MAP]]>
  // CHECK: [[SUM:%.+]] = addi [[DIM]], [[CONST10]] : index
  // CHECK: return [[SUM]] : index
}
