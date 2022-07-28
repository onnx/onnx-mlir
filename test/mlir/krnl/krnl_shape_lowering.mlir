// RUN: onnx-mlir-opt -O3 --lower-krnl-shape %s -split-input-file | FileCheck %s

func.func @test_krnl_shape_lowering(%arg0: memref<?x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.alloc(%0) : memref<?x10xf32>
  %shape = "krnl.shape"(%1) : (memref<?x10xf32>) -> memref<2xindex>
  %e = memref.load %shape[%c0] : memref<2xindex>
  return %e : index

  // CHECK-LABEL: test_krnl_shape_lowering
  // CHECK: %[[CONST0:.+]] = arith.constant 0 : index
  // CHECK: %[[CONST1:.+]] = arith.constant 1 : index
  // CHECK: [[DIM:%.+]] = memref.dim %arg0, %[[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = memref.alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: [[SHAPE:%.+]] = memref.alloc() : memref<2xindex>
  // CHECK: [[DIM0:%.+]] = "krnl.dim"([[ALLOC]], %[[CONST0]]) : (memref<?x10xf32>, index) -> index
  // CHECK: store [[DIM0]], [[SHAPE]][%[[CONST0]]] : memref<2xindex>
  // CHECK: [[DIM1:%.+]] = "krnl.dim"([[ALLOC]], %[[CONST1]]) : (memref<?x10xf32>, index) -> index
  // CHECK: store [[DIM1]], [[SHAPE]][%[[CONST1]]] : memref<2xindex>
  // CHECK: [[RES:%.+]] = memref.load [[SHAPE]][%[[CONST0]]] : memref<2xindex>
  // CHECK: return [[RES]] : index
}

// -----

// COM: check krnl.shape lowering when its input is a MemRef with affine_map.

#map0 = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 4, d0 mod 2, d1 mod 4)>

func.func @test_krnl_shape_lowering_with_affine_map(%arg0: memref<?x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.alloc(%0) : memref<?x10xf32, #map0>
  %shape = "krnl.shape"(%1) : (memref<?x10xf32, #map0>) -> memref<2xindex>
  %e = memref.load %shape[%c0] : memref<2xindex>
  return %e : index

  // CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 4, d0 mod 2, d1 mod 4)>
  // CHECK-LABEL: test_krnl_shape_lowering_with_affine_map
  // CHECK: %[[CONST0:.+]] = arith.constant 0 : index
  // CHECK: %[[CONST1:.+]] = arith.constant 1 : index
  // CHECK: [[DIM:%.+]] = memref.dim %arg0, %[[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = memref.alloc([[DIM]]) : memref<?x10xf32, #[[MAP0]]>
  // CHECK: [[SHAPE:%.+]] = memref.alloc() : memref<2xindex>
  // CHECK: [[DIM0:%.+]] = "krnl.dim"([[ALLOC]], %[[CONST0]]) : (memref<?x10xf32, #[[MAP0]]>, index) -> index
  // CHECK: store [[DIM0]], [[SHAPE]][%[[CONST0]]] : memref<2xindex>
  // CHECK: [[DIM1:%.+]] = "krnl.dim"([[ALLOC]], %[[CONST1]]) : (memref<?x10xf32, #[[MAP0]]>, index) -> index
  // CHECK: store [[DIM1]], [[SHAPE]][%[[CONST1]]] : memref<2xindex>
  // CHECK: [[RES:%.+]] = memref.load [[SHAPE]][%[[CONST0]]] : memref<2xindex>
  // CHECK: return [[RES]] : index
}
