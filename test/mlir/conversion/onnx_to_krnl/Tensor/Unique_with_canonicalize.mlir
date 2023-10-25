// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @unique_without_axis(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) : (tensor<2x2xi64>) -> (tensor<*xi64>, none, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_without_axis
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_minus_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_minus_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<?xi64>, memref<0xi64>, memref<0xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<?xi64>
// CHECK:         }

// -----

func.func @unique_with_axis(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 0 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, none, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_axis
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<?x2xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x2xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<?x2xi64>, memref<0xi64>, memref<0xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<?x2xi64>
// CHECK:         }

// -----

func.func @unique_with_axis_3d(%arg0: tensor<2x2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 0 : si64} : (tensor<2x2x2xi64>) -> (tensor<*xi64>, none, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_axis_3d
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2x2xi64>) -> memref<?x2x2xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?x2x2xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<?x2x2xi64>, memref<0xi64>, memref<0xi64>, memref<0xi64>, memref<2x2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<?x2x2xi64>
// CHECK:         }

// -----

func.func @unique_with_negative_axis(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = -1 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, none, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_negative_axis
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<2x?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<2x?xi64>, memref<0xi64>, memref<0xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<2x?xi64>
// CHECK:         }

// -----

func.func @unique_with_sort(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {sorted = 1 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, none, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_sort
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_minus_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_minus_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<?xi64>, memref<0xi64>, memref<0xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<?xi64>
// CHECK:         }

// -----

func.func @unique_with_indices(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 1 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, tensor<*xi64>, none, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_indices
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<2x?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_1, [[RES_1_]]_0, [[RES_1_]]_0, [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<2x?xi64>, memref<?xi64>, memref<0xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<2x?xi64>
// CHECK:         }

// -----

func.func @unique_with_inverse_indices(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 1 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, none, tensor<*xi64>, none)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_inverse_indices
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<2x?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[RES_3_]] : memref<2xi64> to memref<?xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[VAR_cast_]], [[RES_1_]]_0, [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<2x?xi64>, memref<0xi64>, memref<?xi64>, memref<0xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<2x?xi64>
// CHECK:         }

// -----

func.func @unique_with_counts(%arg0: tensor<2x2xi64>) -> tensor<*xi64> {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 1 : si64} : (tensor<2x2xi64>) -> (tensor<*xi64>, none, none, tensor<*xi64>)
  return %Y : tensor<*xi64>
}

// mlir2FileCheck.py -a '["X"]'
// CHECK-LABEL:  func.func @unique_with_counts
// CHECK-SAME:   ([[X_:%.+]]: memref<2x2xi64>) -> memref<2x?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_0, [[RES_1_]]_1, [[X_]], [[CST_1_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<2x?xi64>, memref<0xi64>, memref<0xi64>, memref<?xi64>, memref<2x2xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]] : memref<2x?xi64>
// CHECK:         }

// -----

func.func @unique_with_dynamic_inputs(%arg0: tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %Y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = 0 : si64, sorted = 1 : si64} : (tensor<?xi64>) -> (tensor<?xi64>, none, tensor<?xi64>, tensor<?xi64>)
  return %Y, %inverse_indices, %counts : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @unique_with_dynamic_inputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xi64>) -> (memref<?xi64>, memref<?xi64>, memref<?xi64>) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUniqueCount", numOfOutput = 1 : si64} : (memref<index>, memref<?xi64>, i64, i64) -> ()
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<0xi64>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xi64>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc([[LOAD_RES_MEM_]]) {{.*}}: memref<?xi64>
// CHECK:           krnl.store [[CST_0_1_]], [[RES_]][] : memref<index>
// CHECK:           "krnl.call"([[RES_]], [[RES_1_]], [[RES_1_]]_0, [[RES_1_]]_1, [[RES_1_]]_2, [[PARAM_0_]], [[CST_0_]], [[CST_1_]]) {funcName = "omTensorUnique", numOfOutput = 5 : si64} : (memref<index>, memref<?xi64>, memref<0xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>, i64, i64) -> ()
// CHECK:           return [[RES_1_]], [[RES_1_]]_1, [[RES_1_]]_2 : memref<?xi64>, memref<?xi64>, memref<?xi64>
}

