// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file --mlir-print-debuginfo | FileCheck %s


//===----------------------------------------------------------------------===//
/// Commutative tests

// CHECK-LABEL: @test_add_constant_1_loc
func.func @test_add_constant_1_loc(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = onnx.Constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32> loc("Constant")
  %1 = "onnx.Add"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>  loc("Add")
  "onnx.Return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32> loc([[LOC_CONST:#.+]])
  // CHECK-NEXT: [[ADD:%.+]] = "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32> loc([[LOC_ADD:#.+]])
  // CHECK-DAG:      [[LOC_CONST]] = loc("Constant")
  // CHECK-DAG:      [[LOC_ADD]] = loc("Add")
}

// -----

// CHECK-LABEL: @test_mul_constant_1_loc
func.func @test_mul_constant_1_loc(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = onnx.Constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32> loc("Constant")
  %1 = "onnx.Mul"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>  loc("Mul")
  "onnx.Return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32> loc([[LOC_CONST:#.+]])
  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32> loc([[LOC_MUL:#.+]])
  // CHECK-DAG:      [[LOC_CONST]] = loc("Constant")
  // CHECK-DAG:      [[LOC_MUL]] = loc("Mul")
}

