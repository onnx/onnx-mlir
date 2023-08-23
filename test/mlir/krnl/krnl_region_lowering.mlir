// RUN: onnx-mlir-opt -O3 --lower-krnl-region  %s -split-input-file | FileCheck %s

func.func @test_krnlregion(%arg2: memref<1xi64>) -> memref<1xi64> {
  %0 = memref.alloc() : memref<1xi64> 
  "krnl.region"() ({
     %c0 = arith.constant 0 : index
     %18 = memref.load %arg2[%c0] : memref<1xi64>
     %19 = arith.addi %18, %18 : i64 
     memref.store %19, %0[%c0] : memref<1xi64>
  }) : () -> ()
  return %0 : memref<1xi64>
// CHECK-LABEL:  func @test_krnlregion
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1xi64>) -> memref<1xi64> {
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1xi64>
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_]] : i64
// CHECK:           memref.store [[VAR_2_]], [[RES_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           return [[RES_]] : memref<1xi64>
// CHECK:         }
}
