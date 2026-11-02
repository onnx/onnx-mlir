// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s --check-prefixes=CHECK,SCALAR
// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s --check-prefixes=CHECK,SIMD
// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s --check-prefix=PAR

// The MatMul and rounded results are the only output-sized i32 buffers.
// Adjustment, signed saturation, and truncation must share one loop body.
func.func @final_i8_tail(%a: tensor<9x7xi8>, %b: tensor<7x65xi8>, %scale: tensor<f32>, %zero: tensor<i8>) -> tensor<9x65xi8> {
  %y = "onnx.QLinearMatMul"(%a, %scale, %zero, %b, %scale, %zero, %scale, %zero) : (tensor<9x7xi8>, tensor<f32>, tensor<i8>, tensor<7x65xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<9x65xi8>
  return %y : tensor<9x65xi8>
}
// CHECK-LABEL: func.func @final_i8_tail
// SCALAR-COUNT-2: memref.alloc{{.*}}memref<9x65xi32>
// SIMD: memref.alloc{{.*}}memref<9x65xi32>
// SIMD: memref.view {{.*}} to memref<9x65xi32>
// CHECK-NOT: memref.alloc{{.*}}memref<9x65xi32>
// SCALAR: [[ADD:%.+]] = arith.addi {{.*}} : i32
// SCALAR-NOT: krnl.iterate
// SCALAR: [[LOW:%.+]] = arith.maxsi [[ADD]], {{.*}} : i32
// SCALAR: [[HIGH:%.+]] = arith.minsi [[LOW]], {{.*}} : i32
// SCALAR: [[BYTE:%.+]] = arith.trunci [[HIGH]] : i32 to i8
// SCALAR-NOT: krnl.iterate
// SCALAR: krnl.store [[BYTE]]
// SIMD: [[VADD:%.+]] = arith.addi {{.*}} : vector<32xi32>
// SIMD-NOT: krnl.iterate
// SIMD: [[VLOW:%.+]] = arith.maxsi [[VADD]], {{.*}} : vector<32xi32>
// SIMD: [[VHIGH:%.+]] = arith.minsi [[VLOW]], {{.*}} : vector<32xi32>
// SIMD: [[VBYTE:%.+]] = arith.trunci [[VHIGH]] : vector<32xi32> to vector<32xi8>
// SIMD: vector.store [[VBYTE]]
// SIMD: [[TAILADD:%.+]] = arith.addi {{.*}} : i32
// SIMD: [[TAILLOW:%.+]] = arith.maxsi [[TAILADD]], {{.*}} : i32
// SIMD: [[TAILHIGH:%.+]] = arith.minsi [[TAILLOW]], {{.*}} : i32
// SIMD: arith.trunci [[TAILHIGH]] : i32 to i8
// CHECK-NOT: memref.alloc{{.*}}memref<9x65xi32>
// CHECK: return

// -----

// Constant unsigned column zero points exercise restoration from centered i8.
func.func @final_ui8_columns(%a: tensor<9x7xui8>, %b: tensor<7x65xui8>) -> tensor<9x65xui8> {
  %scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
  %zero = "onnx.Constant"() {value = dense<128> : tensor<ui8>} : () -> tensor<ui8>
  %columns = "onnx.Constant"() {value = dense<131> : tensor<65xui8>} : () -> tensor<65xui8>
  %yscale = "onnx.Constant"() {value = dense<1.0> : tensor<65xf32>} : () -> tensor<65xf32>
  %y = "onnx.QLinearMatMul"(%a, %scale, %zero, %b, %scale, %zero, %yscale, %columns) : (tensor<9x7xui8>, tensor<f32>, tensor<ui8>, tensor<7x65xui8>, tensor<f32>, tensor<ui8>, tensor<65xf32>, tensor<65xui8>) -> tensor<9x65xui8>
  return %y : tensor<9x65xui8>
}
// CHECK-LABEL: func.func @final_ui8_columns
// SCALAR-COUNT-2: memref.alloc{{.*}}memref<9x65xi32>
// SIMD: memref.alloc{{.*}}memref<9x65xi32>
// SIMD: memref.view {{.*}} to memref<9x65xi32>
// CHECK-NOT: memref.alloc{{.*}}memref<9x65xi32>
// SCALAR: arith.maxsi {{.*}} : i32
// SCALAR: arith.minsi {{.*}} : i32
// SCALAR: arith.trunci {{.*}} : i32 to i8
// SIMD: arith.maxsi {{.*}} : vector<32xi32>
// SIMD: arith.minsi {{.*}} : vector<32xi32>
// SIMD: arith.trunci {{.*}} : vector<32xi32> to vector<32xi8>
// CHECK-NOT: arith.maxui
// CHECK-NOT: arith.minui
// CHECK-NOT: memref.alloc{{.*}}memref<9x65xi32>
// CHECK: return

// -----

// Dynamic innermost broadcasting must retain scalar access selections.
func.func @final_dynamic(%a: tensor<?x?xi8>, %b: tensor<?x?xi8>, %scale: tensor<f32>, %zero: tensor<i8>, %yscale: tensor<?xf32>, %yzero: tensor<?xi8>) -> tensor<?x?xi8> {
  %y = "onnx.QLinearMatMul"(%a, %scale, %zero, %b, %scale, %zero, %yscale, %yzero) : (tensor<?x?xi8>, tensor<f32>, tensor<i8>, tensor<?x?xi8>, tensor<f32>, tensor<i8>, tensor<?xf32>, tensor<?xi8>) -> tensor<?x?xi8>
  return %y : tensor<?x?xi8>
}
// CHECK-LABEL: func.func @final_dynamic
// CHECK: arith.fptosi
// CHECK: arith.select
// CHECK: arith.maxsi {{.*}} : i32
// CHECK: arith.minsi {{.*}} : i32
// CHECK: arith.trunci {{.*}} : i32 to i8
// CHECK: return

// -----

// Vector dot products produce rank-zero outputs.
func.func @final_scalar(%a: tensor<7xi8>, %b: tensor<7xi8>, %scale: tensor<f32>, %zero: tensor<i8>) -> tensor<i8> {
  %y = "onnx.QLinearMatMul"(%a, %scale, %zero, %b, %scale, %zero, %scale, %zero) : (tensor<7xi8>, tensor<f32>, tensor<i8>, tensor<7xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<i8>
  return %y : tensor<i8>
}
// CHECK-LABEL: func.func @final_scalar
// CHECK: arith.fptosi
// CHECK: [[SCALARADD:%.+]] = arith.addi {{.*}} : i32
// CHECK: [[SCALARLOW:%.+]] = arith.maxsi [[SCALARADD]], {{.*}} : i32
// CHECK: [[SCALARHIGH:%.+]] = arith.minsi [[SCALARLOW]], {{.*}} : i32
// CHECK: [[SCALARBYTE:%.+]] = arith.trunci [[SCALARHIGH]] : i32 to i8
// CHECK: krnl.store [[SCALARBYTE]], {{.*}}[] : memref<i8>
// CHECK: return

// -----

// The final stage can parallelize eligible outer rows, with SIMD within each row.
func.func @final_rows_parallel(%a: tensor<128x7xi8>, %b: tensor<7x64xi8>, %scale: tensor<f32>, %zero: tensor<i8>) -> tensor<128x64xi8> {
  %y = "onnx.QLinearMatMul"(%a, %scale, %zero, %b, %scale, %zero, %scale, %zero) : (tensor<128x7xi8>, tensor<f32>, tensor<i8>, tensor<7x64xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<128x64xi8>
  return %y : tensor<128x64xi8>
}
// SIMD-LABEL: func.func @final_rows_parallel
// SIMD-NOT: krnl.parallel
// SIMD: arith.maxsi {{.*}} : vector<32xi32>
// SIMD: return
// PAR-LABEL: func.func @final_rows_parallel
// PAR: arith.fptosi
// PAR: krnl.parallel
// PAR: arith.maxsi {{.*}} : vector<32xi32>
// PAR: arith.minsi {{.*}} : vector<32xi32>
// PAR: arith.trunci {{.*}} : vector<32xi32> to vector<32xi8>
// PAR: vector.store
// PAR: return
