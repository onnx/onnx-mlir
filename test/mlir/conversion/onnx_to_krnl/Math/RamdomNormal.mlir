// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_random_normal1() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal1
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

// -----

func.func @test_random_normal2() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 11 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal2
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

// -----

func.func @test_random_normal3() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 11 : si64, mean = 0.0 :f32, scale = 1.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal3
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}
