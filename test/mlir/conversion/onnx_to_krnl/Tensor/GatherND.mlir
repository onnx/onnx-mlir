// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// COM: Test GatherND with indices_shape[-1] == rank(data) - batch_dims
func.func @test_gather_nd_1(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x2xi64>) -> tensor<2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
// CHECK-LABEL:  @test_gather_nd_1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<2x2xf32>, [[PARAM_1:%.+]]: memref<2x2xi64>) -> memref<2xf32> {
// CHECK:           [[RESHAPED_INDICES:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xi64> to memref<1x2x2xi64>
// CHECK:           [[RESHAPED_DATA:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xf32> to memref<1x2x2xf32>
// CHECK-DAG:       [[RES_BUFFER:%.+]] = memref.alloc() : memref<2xf32>
// CHECK-DAG:       [[RES_BUFFER_INDEX:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[CST_0_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_0:%.+]] = arith.constant 1 : index
// CHECK:           krnl.store [[CST_0_0]], [[RES_BUFFER_INDEX]][] : memref<index>
// CHECK:           [[LOOP:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP]]#0, [[LOOP]]#1) with ([[LOOP]]#0 -> [[I_0:%.+]] = 0 to 1, [[LOOP]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK-DAG:         [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[CST_0_1:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_INDEX_1:%.+]] = krnl.load [[RESHAPED_INDICES]][[[IV]]#0, [[IV]]#1, [[CST_0_1]]] : memref<1x2x2xi64>
// CHECK-DAG:         [[INDEX_1:%.+]] = arith.index_cast [[LOAD_INDEX_1]] : i64 to index
// CHECK-DAG:         [[CST_1_1:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_INDEX_2:%.+]] = krnl.load [[RESHAPED_INDICES]][[[IV]]#0, [[IV]]#1, [[CST_1_1]]] : memref<1x2x2xi64>
// CHECK:             [[INDEX_2:%.+]] = arith.index_cast [[LOAD_INDEX_2]] : i64 to index
// CHECK-DAG:         [[DATA_VAL:%.+]] = krnl.load [[RESHAPED_DATA]][[[IV]]#0, [[INDEX_1]], [[INDEX_2]]] : memref<1x2x2xf32>
// CHECK-DAG:         [[RES_BUFFER_INDEX_VAL:%.+]] = krnl.load [[RES_BUFFER_INDEX]][] : memref<index>
// CHECK:             krnl.store [[DATA_VAL]], [[RES_BUFFER]][[[RES_BUFFER_INDEX_VAL]]] : memref<2xf32>
// CHECK:             [[PLUS_ONE:%.+]] = arith.addi [[RES_BUFFER_INDEX_VAL]], [[CST_1_0]] : index
// CHECK:             krnl.store [[PLUS_ONE]], [[RES_BUFFER_INDEX]][] : memref<index>
// CHECK:           }
// CHECK:          [[RES:%.+]] = memref.reinterpret_cast [[RES_BUFFER]] to offset: [0], sizes: [2], strides: [1] : memref<2xf32> to memref<2xf32>
// CHECK:           return [[RES]] : memref<2xf32>
}

// -----

// COM: Test GatherND with indices_shape[-1] < rank(data) - batch_dims
func.func @test_gather_nd_2(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x1x2xi64>) -> tensor<2x1x2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1x2xf32>
  "func.return"(%0) : (tensor<2x1x2xf32>) -> ()
// CHECK-LABEL:  func @test_gather_nd_2
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<2x2x2xf32>, [[PARAM_1:%.+]]: memref<2x1x2xi64>) -> memref<2x1x2xf32> {
// CHECK-DAG:       [[RESHAPED_INDICES:%.+]] = memref.reinterpret_cast [[PARAM_1]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x1x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[RESHAPED_DATA:%.+]] = memref.reinterpret_cast [[PARAM_0]] to offset: [0], sizes: [1, 2, 2, 2], strides: [8, 4, 2, 1] : memref<2x2x2xf32> to memref<1x2x2x2xf32>
// CHECK-DAG:       [[RES_BUFFER:%.+]] = memref.alloc() : memref<4xf32>
// CHECK:           [[CST_0_0:%.+]] = arith.constant 0 : index
// CHECK:           [[CST_1_0:%.+]] = arith.constant 1 : index
// CHECK:           [[RES_INDEX_BUFFER:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_0]], [[RES_INDEX_BUFFER]][] : memref<index>
// CHECK:           [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[CST_0_1:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_INDEX_1:%.+]] = krnl.load [[RESHAPED_INDICES]]{{.}}[[IV]]#0, [[IV]]#1, [[CST_0_1]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[INDEX_1:%.+]] = arith.index_cast [[LOAD_INDEX_1]] : i64 to index
// CHECK-DAG:         [[CST_1_1:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_INDEX_2:%.+]] = krnl.load [[RESHAPED_INDICES]]{{.}}[[IV]]#0, [[IV]]#1, [[CST_1_1]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[INDEX_2:%.+]] = arith.index_cast [[LOAD_INDEX_2]] : i64 to index
// CHECK-DAG:         [[CST_0_2:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[DATA_1:%.+]] = krnl.load [[RESHAPED_DATA]]{{.}}[[IV]]#0, [[INDEX_1]], [[INDEX_2]], [[CST_0_2]]{{.}} : memref<1x2x2x2xf32>
// CHECK-DAG:         [[RES_INDEX_1:%.+]] = krnl.load [[RES_INDEX_BUFFER]][] : memref<index>
// CHECK:             krnl.store [[DATA_1]], [[RES_BUFFER]]{{.}}[[RES_INDEX_1]]{{.}} : memref<4xf32>
// CHECK:             [[PLUS_ONE:%.+]] = arith.addi [[RES_INDEX_1]], [[CST_1_0]] : index
// CHECK:             krnl.store [[PLUS_ONE]], [[RES_INDEX_BUFFER]][] : memref<index>
// CHECK:             [[CST_1_2:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[DATA_2:%.+]] = krnl.load [[RESHAPED_DATA]]{{.}}[[IV]]#0, [[INDEX_1]], [[INDEX_2]], [[CST_1_2]]{{.}} : memref<1x2x2x2xf32>
// CHECK-DAG:         [[RES_INDEX_2:%.+]] = krnl.load [[RES_INDEX_BUFFER]][] : memref<index>
// CHECK:             krnl.store [[DATA_2]], [[RES_BUFFER]]{{.}}[[RES_INDEX_2]]{{.}} : memref<4xf32>
// CHECK:             [[PLUS_ONE_1:%.+]] = arith.addi [[RES_INDEX_2]], [[CST_1_0]] : index
// CHECK:             krnl.store [[PLUS_ONE_1]], [[RES_INDEX_BUFFER]][] : memref<index>
// CHECK:           }
// CHECK:           [[RES:%.+]] = memref.reinterpret_cast [[RES_BUFFER]] to offset: [0], sizes: [2, 1, 2], strides: [2, 2, 1] : memref<4xf32> to memref<2x1x2xf32>
// CHECK:           return [[RES]] : memref<2x1x2xf32>
}

// -----

// COM: Test GatherND with dynamic shape
func.func @test_gather_nd_with_dynamic_shape_int(%arg0 : tensor<2x2xi32>, %arg1 : tensor<?x2xi64>) -> tensor<2xi32> {
// COM:  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xi32>, tensor<?x2xi64>) -> tensor<?xi32>
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xi32>, tensor<?x2xi64>) -> tensor<2xi32>
  "func.return"(%0) : (tensor<2xi32>) -> ()
}

// COM:XXX // -----
// COM:XXX 
// COM:XXX // COM: Test GatherND with dynamic shape
// COM:XXX func.func @test_gather_nd_with_dynamic_shape(%arg0 : tensor<2x2xf32>, %arg1 : tensor<?x2xi64>) -> tensor<?xf32> {
// COM:XXX   %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<?x2xi64>) -> tensor<?xf32>
// COM:XXX   "func.return"(%0) : (tensor<?xf32>) -> ()
// COM:XXX }
