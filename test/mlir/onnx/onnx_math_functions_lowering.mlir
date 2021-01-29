// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// ----

/// onnx.Erf lowering to krnl.erf.
func @erf_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Erf"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL erf_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ERF:%.+]]  = "krnl.erf"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ERF]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
/// onnx.Acos lowering to krnl.acos.
func @acos_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Acos"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL acos_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.acos"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
