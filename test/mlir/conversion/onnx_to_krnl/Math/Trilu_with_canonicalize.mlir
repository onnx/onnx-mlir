// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_trilu_lower(%arg0: tensor<4x5xi64>, %arg1: tensor<i64>) -> tensor<4x5xi64> {
  %0 = "onnx.Trilu"(%arg0, %arg1) {upper = 0 : si64} : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi64>
  return %0 : tensor<4x5xi64>

// CHECK-LABEL:  func.func @test_trilu_lower
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<4x5xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_1_]], [[VAR_3_]]#0 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[VAR_3_]]#1 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[LOAD_PARAM_0_MEM_]] : i64
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xi64>
// CHECK:         }
}

// -----

func.func @test_trilu_upper(%arg0: tensor<4x5xi64>, %arg1: tensor<i64>) -> tensor<4x5xi64> {
  %0 = "onnx.Trilu"(%arg0, %arg1) {upper = 1 : si64} : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi64>
  return %0 : tensor<4x5xi64>

// CHECK-LABEL:  func.func @test_trilu_upper
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<4x5xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_1_]], [[VAR_3_]]#0 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_4_]], [[VAR_3_]]#1 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[LOAD_PARAM_0_MEM_]] : i64
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xi64>
// CHECK:         }
}
