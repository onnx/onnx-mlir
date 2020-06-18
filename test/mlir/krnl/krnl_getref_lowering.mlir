// RUN: onnx-mlir-opt --lower-krnl --lower-all-llvm %s -split-input-file | FileCheck %s

func @test_getref_lowering(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
  %c13_i64 = constant 13 : i64
  %1 = alloc() : memref<10x10xf32>
  %2 = krnl.getref(%1, %c13_i64) : (memref<10x10xf32>, i64) -> memref<2x2xf32>
  return %2 : memref<2x2xf32>

  // CHECK-LABEL: test_getref_lowering
  // CHECK: %[[OFFSET:.+]] = llvm.mlir.constant(13 : i64) : !llvm.i64
  // CHECK: [[CONST_10_0:%.+]] = llvm.mlir.constant(10 : index) : !llvm.i64
  // CHECK: [[CONST_10_1:%.+]] = llvm.mlir.constant(10 : index) : !llvm.i64
  // CHECK: [[MUL1:%.+]] = llvm.mul [[CONST_10_0]], [[CONST_10_1]] : !llvm.i64
  // CHECK: [[FLOAT_STAR:%.+]] = llvm.mlir.null : !llvm<"float*">
  // CHECK: %[[CONST_1:.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: [[ELEM1:%.+]] = llvm.getelementptr [[FLOAT_STAR]][%[[CONST_1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  // CHECK: [[ELEM_SIZE:%.+]] = llvm.ptrtoint [[ELEM1]] : !llvm<"float*"> to !llvm.i64
  // CHECK: [[MUL2:%.+]] = llvm.mul [[MUL1]], [[ELEM_SIZE]] : !llvm.i64
  // CHECK: [[MEMPOOL:%.+]] = llvm.call @malloc([[MUL2]]) : (!llvm.i64) -> !llvm<"i8*">
  // CHECK: [[TYPED_MEMPOOL:%.+]] = llvm.bitcast [[MEMPOOL]] : !llvm<"i8*"> to !llvm<"float*">
  // CHECK: [[MEMPOOL_MEMREF:%.+]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL]], [[MEMPOOL_MEMREF]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL]], [[MEMREF1]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.mlir.constant
  // CHECK: llvm.insertvalue
  // CHECK: llvm.mlir.constant
  // CHECK: llvm.mlir.constant
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: [[MEMPOOL1:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[MEMPOOL_ALLOC:%.+]] = llvm.getelementptr [[MEMPOOL1]][%[[OFFSET]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  // CHECK: [[TYPED_MEMPOOL_ALLOC:%.+]] = llvm.bitcast [[MEMPOOL_ALLOC]] : !llvm<"float*"> to !llvm<"float*">
  // CHECK: [[MEMPOOLED_MEMREF:%.+]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL_ALLOC]], [[MEMPOOLED_MEMREF]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL_ALLOC]], [[MEMREF3]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
}
