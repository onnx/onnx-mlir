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
 
 
/// onnx.Acosh lowering to krnl.acosh.
func @acosh_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Acosh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL acosh_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.acosh"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
 
/// onnx.Asin lowering to krnl.asin.
func @asin_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Asin"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL asin_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.asin"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
 
/// onnx.Asinh lowering to krnl.asinh.
func @asinh_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Asinh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL asinh_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.asinh"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
/// onnx.Atan lowering to krnl.atan.
func @atan_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Atan"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL atan_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.atan"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
  

/// onnx.Atanh lowering to krnl.atanh.
func @atanh_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Atanh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL atanh_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.atanh"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
 
/// onnx.Tan lowering to krnl.tan.
func @tan_function(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Tan"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL tan_function
// CHECK: [[ALLOC:%.+]] = alloc() : memref<10x10xf32>
// CHECK: krnl.define_loops 2
// CHECK: krnl.iterate
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[%arg1, %arg2] : memref<10x10xf32>
// CHECK: [[ACOS:%.+]]  = "krnl.tan"([[LOAD]]) : (f32) -> f32
// CHECK: {{.*}}store [[ACOS]], [[ALLOC]][%arg1, %arg2] : memref<10x10xf32>
// CHECK: return [[ALLOC]] : memref<10x10xf32>
 
