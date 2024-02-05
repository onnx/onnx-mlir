// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = onnx.Constant dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_4_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = onnx.Constant dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_:%.+]]#0, [[LOOP_0_:%.+]]#1, [[LOOP_0_:%.+]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_7_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 1, second example in ONNX for Gather.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = onnx.Constant dense<[[0, 2]]> : tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [1, 2], value = dense<{{.}}[0, 2]{{.}}> : tensor<1x2xi64>} : () -> memref<1x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#1, [[IV]]#2] : memref<1x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_4_]]] : memref<3x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x1x2xf32>
// CHECK:         }
}
