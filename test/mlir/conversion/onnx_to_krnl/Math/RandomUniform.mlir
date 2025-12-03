// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_random_uniform_f32type() -> tensor<*xf32> {
  %0 = "onnx.RandomUniform"() {shape = [3, 4, 5], dtype = 1 : si64, low = 0.0 :f32, high = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_random_uniform_f32type
// CHECK-SAME:   () -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK:           "krnl.call"([[RES_]], [[CST_0_dot_000000_]], [[CST_1_dot_000000_]], [[CST_2_dot_000000_]]) {funcName = "run_uniform_random", numOfOutput = 1 : si64} : (memref<3x4x5xf32>, f32, f32, f32) -> ()
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

func.func @test_random_uniform_f64type() -> tensor<*xf64> {
  %0 = "onnx.RandomUniform"() {shape = [3, 4, 5], dtype = 11 : si64, low = 0.0 :f32, high = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()

// CHECK-LABEL:  func.func @test_random_uniform_f64type
// CHECK-SAME:   () -> memref<3x4x5xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf64>
// CHECK:           "krnl.call"([[RES_]], [[CST_0_dot_000000_]], [[CST_1_dot_000000_]], [[CST_2_dot_000000_]]) {funcName = "run_uniform_random", numOfOutput = 1 : si64} : (memref<3x4x5xf64>, f32, f32, f32) -> ()
// CHECK:           return [[RES_]] : memref<3x4x5xf64>
// CHECK:         }
}


func.func @test_random_uniform_without_seed() -> tensor<*xf32> {
  %0 = "onnx.RandomUniform"() {shape = [3, 4, 5], dtype = 1 : si64, low = 0.0 :f32, high = 1.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_random_uniform_without_seed
// CHECK-SAME:   () -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED_:%.+]] = arith.constant 
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK:           "krnl.call"([[RES_]], [[CST_0_dot_000000_]], [[CST_1_dot_000000_]], [[SEED_]]) {funcName = "run_uniform_random", numOfOutput = 1 : si64} : (memref<3x4x5xf32>, f32, f32, f32) -> ()
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}


