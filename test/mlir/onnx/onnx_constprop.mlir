// RUN: onnx-mlir-opt --constprop-onnx %s -split-input-file | FileCheck %s

  /// Test ConstantOp assoc for add

// CHECK-LABEL: @test_add_constant_1(%arg0: tensor<3xf32>) -> tensor<3xf32>
func @test_add_constant_1(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "std.return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// Test ConstantOp assoc for add
// CHECK-LABEL: @test_add_constant_2(%arg0: tensor<3xf32>) -> tensor<3xf32>
func @test_add_constant_2(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "std.return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// change (x+c1)+c2 to x+(c1+c2)
// CHECK-LABEL: @test_add_constant_3(%arg0: tensor<3xf32>) -> tensor<3xf32> 
func @test_add_constant_3(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<[10.0, 11.0, 12.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "std.return"(%3) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[CONST2:%.+]] = "onnx.Constant"() {value = dense<[1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"([[CONST1]], [[CONST2]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"(%arg0, [[ADD1]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}
