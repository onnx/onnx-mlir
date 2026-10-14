// RUN: onnx-mlir-opt --buffer-deallocation-pipeline --eliminate-entry-arg-clone %s -split-input-file | FileCheck %s

// Returned input of the entry function: no clone.
module {
  func.func @main_graph(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?xi64>) -> memref<?x?x?x?xf32> attributes {llvm.emit_c_interface} {
    return %arg0 : memref<?x?x?x?xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ([[PARAM_0_:%[a-z0-9]+]]: memref<?x?x?x?xf32>, {{.*}}) -> memref<?x?x?x?xf32>
// CHECK-NOT:      bufferization.clone
// CHECK:          return [[PARAM_0_]] : memref<?x?x?x?xf32>
}

// -----

// Input returned with a type change and a view of an input, next to a local
// alloc that is returned.
module {
  func.func @main_graph(%arg0: memref<10xf32>, %arg1: memref<10xf32>) -> (memref<?xf32>, memref<5xf32, strided<[1]>>, memref<10xf32>) {
    %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
    %1 = memref.subview %arg1[0] [5] [1] : memref<10xf32> to memref<5xf32, strided<[1]>>
    %2 = memref.alloc() : memref<10xf32>
    return %0, %1, %2 : memref<?xf32>, memref<5xf32, strided<[1]>>, memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 3 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ([[PARAM_0_:%[a-z0-9]+]]: memref<10xf32>, [[PARAM_1_:%[a-z0-9]+]]: memref<10xf32>)
// CHECK-NOT:      bufferization.clone
// CHECK-DAG:      [[VAR_cast_:%.+]] = memref.cast [[PARAM_0_]] : memref<10xf32> to memref<?xf32>
// CHECK-DAG:      [[VAR_subview_:%.+]] = memref.subview [[PARAM_1_]][0] [5] [1]
// CHECK-DAG:      [[RES_:%.+]] = memref.alloc() : memref<10xf32>
// CHECK:          return [[VAR_cast_]], [[VAR_subview_]], [[RES_]]
}

// -----

// Clones in functions that are not entry points are kept.
module {
  func.func private @helper(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  func.func @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    %0 = call @helper(%arg0) : (memref<10xf32>) -> memref<10xf32>
    return %0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-LABEL:  func.func private @helper
// CHECK:          bufferization.clone
// CHECK-LABEL:  func.func @main_graph
// CHECK-NOT:      bufferization.clone
// CHECK:          return
}

// -----

// Returned constant: the clone is kept.
module {
  func.func @main_graph() -> memref<3xi64> {
    %0 = "krnl.global"() {name = "constant_0", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
    return %0 : memref<3xi64>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 0 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK:          [[VAR_0_:%.+]] = "krnl.global"
// CHECK:          [[VAR_1_:%.+]] = bufferization.clone [[VAR_0_]]
// CHECK:          return [[VAR_1_]]
}
