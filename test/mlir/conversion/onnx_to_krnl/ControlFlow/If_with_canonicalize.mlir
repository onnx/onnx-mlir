// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Check nested if lowering (function computes scalar Sign).
func.func @test_if_sign(%arg0: tensor<f32>) -> tensor<i32> {
  %zero = onnx.Constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %plus = onnx.Constant {value = dense<1> : tensor<i32>} : tensor<i32>
  %minus = onnx.Constant {value = dense<-1> : tensor<i32>} : tensor<i32>
  %0 = onnx.Constant {value = dense<0.0> : tensor<f32>} : tensor<f32>
  %1 = "onnx.Less"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2 = "onnx.If"(%1) ({
    onnx.Yield %minus : tensor<i32>
  }, {
    %3 = "onnx.Greater"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = "onnx.If"(%3) ({
      onnx.Yield %plus : tensor<i32>
    }, {
      onnx.Yield %zero : tensor<i32>
    }) : (tensor<i1>) -> tensor<i32>
    onnx.Yield %4 : tensor<i32>
  }) : (tensor<i1>) -> tensor<i32>
  return %2 : tensor<i32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_if_sign
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<f32>) -> memref<i32> {
// CHECK-DAG:       [[CONSTANT_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<0> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[CONSTANT_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[CONSTANT_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<-1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<0.000000e+00> : tensor<f32>} : () -> memref<f32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK-DAG:       krnl.store [[VAR_3_]], [[RES_]][] : memref<i1>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<i1>
// CHECK-DAG:       [[VAR_5_:%.+]] = scf.if [[LOAD_RES_MEM_]] -> (memref<i32>) {
// CHECK-DAG:         scf.yield [[CONSTANT_3_]] : memref<i32>
// CHECK-DAG:       } else {
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK-DAG:         krnl.store [[VAR_8_]], [[RES_1_]][] : memref<i1>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             [[VAR_10_:%.+]] = arith.select [[LOAD_RES_1_MEM_]], [[CONSTANT_2_]], [[CONSTANT_1_]] : memref<i32>
// CHECK:             scf.yield [[VAR_10_]] : memref<i32>
// CHECK:           }
// CHECK:           return [[VAR_5_]] : memref<i32>
// CHECK:         }
}

