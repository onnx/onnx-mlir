// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_constant_dense_2d_value(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3, 2], value = dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK: return [[GLOBAL]] : memref<3x2xf32>
}

// -----

func.func @test_constant_string() -> tensor<!onnx.String> {
  %0 = onnx.Constant dense<"1"> : tensor<!onnx.String>
  "func.return"(%0) : (tensor<!onnx.String>) -> ()
  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_constant_string
  // CHECK-SAME:   () -> memref<!krnl.string> {
  // CHECK:           [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<"1"> : tensor<!krnl.string>} : () -> memref<!krnl.string>
  // CHECK:           return [[VAR_0_]] : memref<!krnl.string>
}

// -----

func.func @test_constant_string_3elem() -> tensor<3x!onnx.String> {
  %0 = onnx.Constant dense<["1", "2", "3"]> : tensor<3x!onnx.String>
  "func.return"(%0) : (tensor<3x!onnx.String>) -> ()
  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_constant_string_3elem
  // CHECK-SAME:   () -> memref<3x!krnl.string> {
  // CHECK:           [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [3], value = dense<["1", "2", "3"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK:           return [[VAR_0_]] : memref<3x!krnl.string>
}

// -----

func.func @test_constant_string_3elem2() -> tensor<3x!onnx.String> {
  %0 = onnx.Constant dense<"1"> : tensor<3x!onnx.String>
  "func.return"(%0) : (tensor<3x!onnx.String>) -> ()
  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_constant_string_3elem2
  // CHECK-SAME:   () -> memref<3x!krnl.string> {
  // CHECK:           [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [3], value = dense<"1"> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK:           return [[VAR_0_]] : memref<3x!krnl.string>
  // CHECK:         }
}
