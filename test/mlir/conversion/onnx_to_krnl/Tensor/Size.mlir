// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_size_known(%arg0: tensor<2x2xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0) : (tensor<2x2xf32>) -> tensor<i64>
  "func.return"(%1) : (tensor<i64>) -> ()

  // CHECK-LABEL: test_size_known
  // CHECK:      [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK-NEXT  [[SIZE:%.+]] = arith.constant 4 : i64
  // CHECK-NEXT  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT  return [[RES]] : memref<i64>

}

// -----

func.func private @test_size_unknown(%arg0 : tensor<?x2x?xf32>) -> tensor<i64> {

  // CHECK-LABEL: test_size_unknown
  // CHECK:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK-NEXT:  [[INIT:%.+]] = arith.constant 2 : i64
  // CHECK-NEXT:  [[IND1:%.+]] = arith.constant 0 : index
  // CHECK-NEXT:  [[DIM1:%.+]] = memref.dim %arg0, [[IND1]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[CAST1:%.+]] = arith.index_cast [[DIM1]] : index to i64
  // CHECK-NEXT:  [[TMP1:%.+]] = arith.muli [[INIT]], [[CAST1]] : i64
  // CHECK-NEXT:  [[IND2:%.+]] = arith.constant 2 : index
  // CHECK-NEXT:  [[DIM2:%.+]] = memref.dim %arg0, [[IND2]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[IND3:%.+]] = arith.index_cast [[DIM2]] : index to i64
  // CHECK-NEXT:  [[SIZE:%.+]] = arith.muli [[TMP1]], [[IND3]] : i64
  // CHECK-NEXT:  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT:  return [[RES]] : memref<i64>

  %1 = "onnx.Size"(%arg0)  : (tensor<?x2x?xf32>) -> tensor<i64>
  "func.return"(%1) : (tensor<i64>) -> ()
}

