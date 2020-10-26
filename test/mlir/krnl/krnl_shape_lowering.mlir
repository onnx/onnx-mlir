// RUN: onnx-mlir-opt --lower-krnl-shape %s -split-input-file | FileCheck %s

func @test_krnl_shape_lowering(%arg0: memref<?x?xf32>) -> index {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = alloc(%0) : memref<?x10xf32>
  %shape = "krnl.shape"(%1) : (memref<?x10xf32>) -> !shape.shape
  %d1 = "shape.get_extent"(%shape, %c1) : (!shape.shape, index) -> !shape.size
  %ind = "shape.size_to_index"(%d1) : (!shape.size) -> index
  %e = addi %ind, %ind : index
  return %e : index

  // CHECK-LABEL: test_krnl_shape_lowering
  // CHECK: [[CONST0:%.+]] = constant 0 : index
  // CHECK: [[CONST1:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: [[DIM0:%.+]] = "krnl.dim"([[ALLOC]], [[CONST0]]) : (memref<?x10xf32>, index) -> index
  // CHECK: [[DIM1:%.+]] = "krnl.dim"([[ALLOC]], [[CONST1]]) : (memref<?x10xf32>, index) -> index
  // CHECK: [[SHAPE:%.+]] = shape.from_extents [[DIM0]], [[DIM1]]
  // CHECK: [[EXTENT:%.+]] = shape.get_extent [[SHAPE]], [[CONST1]] : !shape.shape, index -> !shape.size
  // CHECK: [[EXTENT_AS_INDEX:%.+]] = shape.size_to_index [[EXTENT]] : !shape.size
  // CHECK: [[RES:%.+]] = addi [[EXTENT_AS_INDEX]], [[EXTENT_AS_INDEX]] : index
  // CHECK: return [[RES]] : index
}

// -----

// COM: check krnl.shape lowering when its input is a MemRef with affine_map.

#map0 = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 4, d0 mod 2, d1 mod 4)>

func @test_krnl_shape_lowering_with_affine_map(%arg0: memref<?x?xf32>) -> index {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = alloc(%0) : memref<?x10xf32, #map0>
  %shape = "krnl.shape"(%1) : (memref<?x10xf32, #map0>) -> !shape.shape
  %d1 = "shape.get_extent"(%shape, %c1) : (!shape.shape, index) -> !shape.size
  %ind = "shape.size_to_index"(%d1) : (!shape.size) -> index
  %e = addi %ind, %ind : index
  return %e : index

  // CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 4, d0 mod 2, d1 mod 4)>
  // CHECK-LABEL: test_krnl_shape_lowering_with_affine_map
  // CHECK: [[CONST0:%.+]] = constant 0 : index
  // CHECK: [[CONST1:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[CONST0]] : memref<?x?xf32>
  // CHECK: [[ALLOC:%.+]] = alloc([[DIM]]) : memref<?x10xf32, #[[MAP0]]>
  // CHECK: [[DIM0:%.+]] = "krnl.dim"([[ALLOC]], [[CONST0]]) : (memref<?x10xf32, #[[MAP0]]>, index) -> index
  // CHECK: [[DIM1:%.+]] = "krnl.dim"([[ALLOC]], [[CONST1]]) : (memref<?x10xf32, #[[MAP0]]>, index) -> index
  // CHECK: [[SHAPE:%.+]] = shape.from_extents [[DIM0]], [[DIM1]]
  // CHECK: [[EXTENT:%.+]] = shape.get_extent [[SHAPE]], [[CONST1]] : !shape.shape, index -> !shape.size
  // CHECK: [[EXTENT_AS_INDEX:%.+]] = shape.size_to_index [[EXTENT]] : !shape.size
  // CHECK: [[RES:%.+]] = addi [[EXTENT_AS_INDEX]], [[EXTENT_AS_INDEX]] : index
  // CHECK: return [[RES]] : index
}
