// RUN: onnx-mlir-opt -O3 --convert-seq-to-memref %s -split-input-file | FileCheck %s

// -----

func.func @test_seqstore(%arg0: memref<?x3xf32>, %arg1: memref<?xmemref<?x?xf32>>, %arg2: index) -> memref<?x3xf32>  {
    %0 = "onnx.Constant"(){value_int = 0 : si64 }: () -> tensor<i64>
    "krnl.seqstore"(%arg0, %arg1, %arg2) : (memref<?x3xf32>, memref<?xmemref<?x?xf32>>, index) -> ()
    return %arg0 : memref<?x3xf32>
// CHECK-LABEL:  func @test_seqstore
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3xf32>, [[PARAM_1_:%.+]]: memref<?xmemref<?x?xf32>>, [[PARAM_2_:%.+]]: index) -> memref<?x3xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x3xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3xf32>
// CHECK:           memref.copy [[PARAM_0_]], [[RES_]] : memref<?x3xf32> to memref<?x3xf32>
// CHECK:           [[VAR_2_:%.+]] = memref.cast [[RES_]] : memref<?x3xf32> to memref<?x?xf32>
// CHECK:           memref.store [[VAR_2_]], [[PARAM_1_]]{{.}}[[PARAM_2_]]{{.}} : memref<?xmemref<?x?xf32>>
// CHECK:           return [[PARAM_0_]] : memref<?x3xf32>
}

// -----

func.func @test_seqextract(%arg1: memref<?xmemref<?x3xf32>>, %arg2: index) -> memref<?x3xf32>  {
    %0 = "krnl.seqextract"(%arg1, %arg2) : (memref<?xmemref<?x3xf32>>, index) -> (memref<?x3xf32>)
    return %0 : memref<?x3xf32>
// CHECK-LABEL:  func @test_seqextract
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xmemref<?x3xf32>>, [[PARAM_1_:%.+]]: index) -> memref<?x3xf32> {
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[PARAM_1_]]{{.}} : memref<?xmemref<?x3xf32>>
// CHECK:           return [[LOAD_PARAM_0_MEM_]] : memref<?x3xf32>
}
