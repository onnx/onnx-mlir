// RUN: onnx-mlir-opt --simplify-krnl %s | FileCheck %s

func @test_simplify_1(%arg0: memref<?x?xf32>) -> memref<?x10xf32> {
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %c1 = constant 1 : index
  %1 = dim %arg0, %c1 : memref<?x?xf32>
  %3 = cmpi "sgt", %0, %0 : index
  %4 = select %3, %0, %1 : index
  %5 = alloc(%4) : memref<?x10xf32>
  return %5 : memref<?x10xf32>

  // CHECK-LABEL: test_simplify_1
  // CHECK: [[DIM2:%.+]] = dim %arg0, %c1 : memref<?x?xf32>
  // CHECK-NOT: cmpi "sgt", %0, %0 : index
  // CHECK: [[RES:%.+]] = alloc([[DIM2]]) : memref<?x10xf32>
}

func @test_simplify_2(%arg0: memref<?x?xf32>) -> memref<?x10xf32> {
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %c1 = constant 1 : index
  %1 = dim %arg0, %c1 : memref<?x?xf32>
  %3 = cmpi "eq", %0, %0 : index
  %4 = select %3, %0, %1 : index
  %5 = alloc(%4) : memref<?x10xf32>
  return %5 : memref<?x10xf32>

  // CHECK-LABEL: test_simplify_2
  // CHECK: [[DIM1:%.+]] = dim %arg0, %c0 : memref<?x?xf32>
  // CHECK-NOT: cmpi "sgt", %0, %0 : index
  // CHECK: [[RES:%.+]] = alloc([[DIM1]]) : memref<?x10xf32>
}