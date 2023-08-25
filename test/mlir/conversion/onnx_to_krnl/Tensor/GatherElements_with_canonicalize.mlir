// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// COM: Test GatherElements along axis 0. Positive indices, so no select.
func.func @test_gather_elements_axis0(%arg0 : tensor<3x3xf32>) -> tensor<2x3xf32> {
  %indices = onnx.Constant dense<[[1, 2, 0], [2, 0, 0]]> : tensor<2x3xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<2x3xi64>) -> tensor<2x3xf32>
  "func.return"(%0) : (tensor<2x3xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x3xf32>) -> memref<2x3xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 3], value = dense<{{.}}[1, 2, 0], [2, 0, 0]{{.}}> : tensor<2x3xi64>} : () -> memref<2x3xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x3xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[INDEX_CAST]], [[IV]]#1] : memref<3x3xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x3xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x3xf32>
// CHECK:         }
}

// -----

// Test GatherElements along axis 0. Negative indices.
func.func @test_gather_elements_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = onnx.Constant dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0:%.+]]#0, [[LOOP_0:%.+]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[INDEX_CAST]], [[CST_0]] : index
// CHECK-DAG:         [[VAR_1:%.+]] = arith.addi [[INDEX_CAST]], [[CST_3]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[VAR_1]], [[INDEX_CAST]] : index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0_]][[[SEL]], [[IV]]#1] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherElements along axis 1. Positive indices, so no select.
func.func @test_gather_elements_axis1(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = onnx.Constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 0], [1, 0]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[IV]]#0, [[INDEX_CAST]]] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}
