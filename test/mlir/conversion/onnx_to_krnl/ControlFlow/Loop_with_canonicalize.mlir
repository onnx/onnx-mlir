// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_loop_tiny_yolo() -> tensor<?xi32> {
    %0 = onnx.Constant dense<7> : tensor<i64>
    %1 = onnx.Constant dense<true> : tensor<i1>
    %2 = onnx.Constant dense<0> : tensor<i32>
    %3:2 = "onnx.Loop"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<i32>):  // no predecessors
      %4 = onnx.Constant dense<1> : tensor<i32>
      %5 = "onnx.Add"(%arg2, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      onnx.Yield %arg1, %5, %arg2 : tensor<i1>, tensor<i32>, tensor<i32>
    }) : (tensor<i64>, tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<?xi32>)
    return %3#1 : tensor<?xi32>

// CHECK-LABEL:  func @test_loop_tiny_yolo
// CHECK-SAME:   () -> memref<?xi32> {
// CHECK-DAG:       [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[ONE_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<7> : tensor<i64>} : () -> memref<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<true> : tensor<i1>} : () -> memref<i1>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<0> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i32>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi32>
// CHECK-DAG:       [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]][] : memref<i32>
// CHECK-DAG:       krnl.store [[LOAD_VAR_2_MEM_]], [[RES_]][] : memref<i32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK-DAG:       krnl.store [[LOAD_VAR_1_MEM_]], [[RES_2_]][] : memref<i1>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> %arg0 = [[ZERO]] to [[VAR_12_]]){
// CHECK-DAG:         [[I_0_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<i1>
// CHECK-DAG:         scf.if [[LOAD_RES_2_MEM_]] {
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.index_cast [[I_0_]] : index to i64
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloc() : memref<i64>
// CHECK-DAG:           krnl.store [[VAR_14_]], [[RES_3_]][] : memref<i64>
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloc() : memref<i32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               [[LOAD_ONE_MEM_:%.+]] = krnl.load [[ONE_]][] : memref<i32>
// CHECK:               [[VAR_20_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_ONE_MEM_]] : i32
// CHECK:               krnl.store [[VAR_20_]], [[RES_4_]][] : memref<i32>
// CHECK:               [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK:               krnl.store [[LOAD_VAR_1_MEM_1_]], [[RES_2_]][] : memref<i1>
// CHECK:               [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[I_0_]]{{.}} : memref<?xi32>
// CHECK:               [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_4_MEM_]], [[RES_]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xi32>
// CHECK:         }
}
