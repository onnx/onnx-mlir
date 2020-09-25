// RUN: onnx-mlir-opt --lower-krnl-shape-to-std %s -split-input-file | FileCheck %s

/// Lower krnl.dim when input MemRef does not have an affine map.
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

/// Lower krnl.dim when input MemRef has an affine map.
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

// -----

/// Lower krnl.dim to constant when first argument of krnl.dim is an input arg
/// and the dimensions is static.
func @test_krnl_dim_lowering_with_const_arg(%arg0: memref<10x20xf32>) -> index {
  %c0 = constant 0 : index
  %0 = "krnl.dim"(%arg0, %c0) : (memref<10x20xf32>, index) -> index
  return %0 : index

  // CHECK-LABEL: test_krnl_dim_lowering_with_const_arg
  // CHECK: [[CONST10:%.+]] = constant 10 : index
  // CHECK: return [[CONST10]] : index
}

// -----

/// Lower krnl.dim to a standard dim operation when first argument of krnl.dim
/// is an input arg and the dimensions is dynamic.
func @test_krnl_dim_lowering_with_dynamic_arg(%arg0: memref<10x?xf32>) -> index {
  %c0 = constant 1 : index
  %0 = "krnl.dim"(%arg0, %c0) : (memref<10x?xf32>, index) -> index
  return %0 : index

  // CHECK-LABEL: test_krnl_dim_lowering_with_dynamic_arg
  // CHECK: [[CONST1:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[CONST1]] : memref<10x?xf32>
  // CHECK: return [[DIM]] : index
}
