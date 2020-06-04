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

/// Change (x+c1)+c2 to x+(c1+c2)
// CHECK-LABEL: @test_add_constant_3(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func @test_add_constant_3(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "std.return"(%3) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Same test as above, but with a use of an intermediary result
/// change (x+c1)+c2 + (x+c1) to x+(c1+c2) + (x+c1)
// CHECK-LABEL: @test_add_constant_4(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func @test_add_constant_4(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Add"(%2, %3) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "std.return"(%4) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[CONST2:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"(%arg0, [[CONST2]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[ADD3:%.+]] = "onnx.Add"([[ADD1]], [[ADD2]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Test broadcast 1 -> 2d

// CHECK-LABEL: @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "std.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// Test broadcast 2d (size one) -> 2d

// CHECK-LABEL: @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1]]> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "std.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// check 1d -> 2d

  // CHECK-LABEL: @test_broadcast_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_3(%arg0 : tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1], [2], [3]]> : tensor<3x1xi32>} : () -> tensor<3x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 11], [21, 22], [31, 32]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "std.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[11, 12], [23, 24], [34, 35]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}
