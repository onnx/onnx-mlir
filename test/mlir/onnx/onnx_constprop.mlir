// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Common tests. Use ONNXAddOp as example.

// -----

// CHECK-LABEL: @test_scalar_attr() -> tensor<f32>
func @test_scalar_attr() -> tensor<f32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32> , tensor<f32>) -> tensor<f32>
  "func.return"(%2) : (tensor<f32>) -> ()
  // CHECK: [[CONST:%.+]] = "onnx.Constant"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
}

// -----

// CHECK-LABEL: @test_single_value_attr() -> tensor<1xf32>
func @test_single_value_attr() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%2) : (tensor<1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = "onnx.Constant"() {value = dense<3.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
}

// -----

// CHECK-LABEL: @test_splat_attr() -> tensor<3xf32>
func @test_splat_attr() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%2) : (tensor<3xf32>) -> ()
  // CHECK: [[CONST:%.+]] = "onnx.Constant"() {value = dense<3.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
}

// -----

// CHECK-LABEL: @test_splat_nonsplat_attrs() -> tensor<3xf32>
func @test_splat_nonsplat_attrs() -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%2) : (tensor<3xf32>) -> ()
  // CHECK: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// ADD tests 

/// Test ConstantOp assoc for add
// -----
// CHECK-LABEL: @test_add_constant_1(%arg0: tensor<3xf32>) -> tensor<3xf32>
func @test_add_constant_1(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// Test ConstantOp assoc for add
// -----
// CHECK-LABEL: @test_add_constant_2(%arg0: tensor<3xf32>) -> tensor<3xf32>
func @test_add_constant_2(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "func.return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// Change (x+c1)+c2 to x+(c1+c2)
// -----
// CHECK-LABEL: @test_add_constant_3(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func @test_add_constant_3(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "func.return"(%3) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Same test as above, but with a use of an intermediary result
/// change (x+c1)+c2 + (x+c1) to x+x + (c1+c2+c3)
// -----
// CHECK-LABEL: @test_add_constant_4(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func @test_add_constant_4(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Add"(%2, %3) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "func.return"(%4) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg0) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 13, 16]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Change (x+c0)+y  + (z+c1) to  (x+y)+z + (c1+c2)
// -----
// CHECK-LABEL: @test_add_constant_5(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> 
func @test_add_constant_5(%arg0 : tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%2, %arg1) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Add"(%1, %arg2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %5 = "onnx.Add"(%3, %4) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "func.return"(%5) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], %arg2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[ADD3:%.+]] = "onnx.Add"([[ADD2]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Test broadcast 1 -> 2d
// -----
// CHECK-LABEL: @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "func.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// Test broadcast 2d (size one) -> 2d
// -----
// CHECK-LABEL: @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1]]> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "func.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// check 1d -> 2d
// -----
// CHECK-LABEL: @test_broadcast_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_broadcast_3(%arg0 : tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1], [2], [3]]> : tensor<3x1xi32>} : () -> tensor<3x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 11], [21, 22], [31, 32]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "func.return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[11, 12], [23, 24], [34, 35]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}


//===----------------------------------------------------------------------===//  
/// MUL tests (same as add, so have only two).
  
/// Change (x*c1)*c2 to x*(c1*c2)
// -----
// CHECK-LABEL: @test_mul_constant_3(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func @test_mul_constant_3(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Mul"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "func.return"(%3) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0, 11, 24]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Change (x*c0)*y  * (z*c1) to  (x*y)*z * (c1*c2)
// -----
// CHECK-LABEL: @test_mul_constant_5(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> 
func @test_mul_constant_5(%arg0 : tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Mul"(%2, %arg1) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Mul"(%1, %arg2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %5 = "onnx.Mul"(%3, %4) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "func.return"(%5) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[MUL2:%.+]] = "onnx.Mul"([[MUL1]], %arg2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0, 11, 24]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: [[MUL3:%.+]] = "onnx.Mul"([[MUL2]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

//===----------------------------------------------------------------------===//
/// SUB and NEG tests.

// check of sub two constants
// -----  
// CHECK-LABEL: @test_sub_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_sub_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[2]]> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %2 = "onnx.Sub"(%0, %1) : (tensor<3x2xi32>, tensor<1x1xi32>) -> tensor<3x2xi32>
  "func.return"(%2) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[0, 1], [2, 3], [4, 5]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
}

/// check sub to add of negative
// -----
// CHECK-LABEL: @test_neg_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_neg_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "onnx.Sub"(%arg0, %0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "func.return"(%1) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[-2, -3], [-4, -5], [-6, -7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

// -----
// CHECK-LABEL: @test_neg_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_neg_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10]]> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %2 = "onnx.Sub"(%arg0, %0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %5 = "onnx.Add"(%2, %1) : (tensor<3x2xi32> , tensor<1x1xi32>) -> tensor<3x2xi32>
  "func.return"(%5) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

// -----
// CHECK-LABEL: @test_neg_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func @test_neg_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10]]> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %2 = "onnx.Neg"(%0) : (tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%arg0, %2) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %4 = "onnx.Add"(%3, %1) : (tensor<3x2xi32> , tensor<1x1xi32>) -> tensor<3x2xi32>
  "func.return"(%4) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

//===----------------------------------------------------------------------===//
/// Transpose tests.

// -----  
// CHECK-LABEL: test_default_transpose_const_1
  func @test_default_transpose_const_1() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 211], [121, 221], [131, 231]{{.}}, [{{.}}112, 212], [122, 222], [132, 232]{{.}}, [{{.}}113, 213], [123, 223], [133, 233]{{.}}, [{{.}}114, 214], [124, 224], [134, 234]{{.}}]> : tensor<4x3x2xi32>} : () -> tensor<4x3x2xi32>
  // CHECK: return [[RES]] : tensor<4x3x2xi32>
}

// -----  
// CHECK-LABEL: test_default_transpose_const_2
func @test_default_transpose_const_2() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 2, 1]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 121, 131], [112, 122, 132], [113, 123, 133], [114, 124, 134]{{.}}, [{{.}}211, 221, 231], [212, 222, 232], [213, 223, 233], [214, 224, 234]{{.}}]> : tensor<2x4x3xi32>} : () -> tensor<2x4x3xi32>
  // CHECK: return [[RES]] : tensor<2x4x3xi32>
}

// -----  
// CHECK-LABEL: test_default_transpose_const_3
func @test_default_transpose_const_3() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [1, 0, 2]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "func.return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] =  "onnx.Constant"() {value = dense<[{{.}}[111, 112, 113, 114], [211, 212, 213, 214]{{.}}, [{{.}}121, 122, 123, 124], [221, 222, 223, 224]{{.}}, [{{.}}131, 132, 133, 134], [231, 232, 233, 234]{{.}}]> : tensor<3x2x4xi32>} : () -> tensor<3x2x4xi32>
  // CHECK: return [[RES]] : tensor<3x2x4xi32>
}

//===----------------------------------------------------------------------===//
/// Div tests

// -----

// CHECK-LABEL: @test_div(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32>
func @test_div(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "onnx.Constant"() {value = dense<[[2.0]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %2 = "onnx.Div"(%0, %1) : (tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "func.return"(%2) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{\]}}> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Div"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Sqrt tests

// -----

// CHECK-LABEL: @test_sqrt() -> tensor<1x2xf32>
func @test_sqrt() -> tensor<1x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[4.0, 16.0]]> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "func.return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[2.000000e+00, 4.000000e+00]{{\]}}> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Sqrt"{{.*}}
}

/// Relu tests

// -----

// CHECK-LABEL: @test_relu() -> tensor<1x2xf32>
func @test_relu() -> tensor<1x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[-4.0, 16.0]]> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "func.return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.600000e+01]{{\]}}> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Relu"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Unsqueeze tests

// -----

// CHECK-LABEL: @test_unsqueeze() -> tensor<2x1x1xf32>
func @test_unsqueeze() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[4.0, 16.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%0, %1) : (tensor<2xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
}

// -----

// CHECK-LABEL: @test_unsqueezev11() -> tensor<2x1x1xf32>
func @test_unsqueezev11() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[4.0, 16.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes = [1, 2]} : (tensor<2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Squeeze tests

// -----

// CHECK-LABEL: @test_squeeze() -> tensor<2xf32>
func @test_squeeze() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
  %1 = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %2 = "onnx.Squeeze"(%0, %1) : (tensor<2x1x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK: return [[RES]] : tensor<2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
}

// -----

// CHECK-LABEL: @test_squeezev11() -> tensor<2xf32>
func @test_squeezev11() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes = [1, 2]} : (tensor<2x1x1xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK: return [[RES]] : tensor<2xf32>
  // CHECK-NOT: {{.*}} = "onnx.SqueezeV11"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Split tests

// -----

// CHECK-LABEL: @test_split_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func @test_split_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %split = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) {axis = 0 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "func.return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// CHECK-LABEL: @test_split_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func @test_split_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %split = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "func.return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK: {{.*}}  = "onnx.Constant"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// COM: There is no constant propagation if the split's input is not a constant.

// CHECK-LABEL: @test_split_axis_2(%arg0: tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func @test_split_axis_2(%arg0 : tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %0 = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1, %2 = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "func.return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
}

// -----

// CHECK-LABEL: @test_splitv11_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func @test_splitv11_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 0 : si64, split = [1, 1]} : (tensor<2x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "func.return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}

// -----

// CHECK-LABEL: @test_splitv11_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func @test_splitv11_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 1 : si64, split = [5, 5]} : (tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "func.return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK: {{.*}}  = "onnx.Constant"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}

// -----

// COM: There is no constant propagation if the split's input is not a constant.

// CHECK-LABEL: @test_splitv11_axis_2(%arg0: tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func @test_splitv11_axis_2(%arg0 : tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %1, %2 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [5, 5]} : (tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "func.return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [5, 5]} : (tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
}

// -----

func @test_mul_folding(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[[[0.0234164055, 0.0228030644], [2.442580e-02, 0.0237577036]]], [[[-0.0410864502, 0.0488203131], [0.164448678, -0.0200194642]]], [[[-4.34581793E-9, 0.025325032], [0.0373019315, 0.165243402]]], [[[-0.0198689923, 0.131284416], [0.0572107285, 2.33985098E-8]]], [[[0.0187684372, -0.148515195], [0.0154875498, 0.019133633]]], [[[0.0176953916, -0.0154658081], [0.0233727545, -0.274110436]]], [[[-0.021181887, 0.0936150252], [0.135688141, -0.0202601217]]], [[[-0.0201558527, 0.0192655921], [0.227748245, -0.196346223]]]]> : tensor<8x1x2x2xf32>} : () -> tensor<8x1x2x2xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.Constant"() {value = dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
  %3 = "onnx.UnsqueezeV11"(%2) {axes = [3]} : (tensor<8x1x1xf32>) -> tensor<8x1x1x1xf32>
  %4 = "onnx.Mul"(%0, %3) : (tensor<8x1x2x2xf32>, tensor<8x1x1x1xf32>) -> tensor<8x1x2x2xf32>
  %5 = "onnx.Conv"(%arg0, %4, %1) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<*xf32>
  return %5 : tensor<*xf32>

  // CHECK-LABEL:  func @test_mul_folding
  // CHECK-SAME:   ([[X:%.+]]: tensor<1x1x28x28xf32>) -> tensor<1x8x27x27xf32> {
  // CHECK: [[NOBIAS:%.+]] = "onnx.NoValue"() {value} : () -> none    
  // CHECK: [[W:%.+]] = "onnx.Constant"() {value = dense<{{.*}}[-0.00378267956, -0.00368360057], [-0.00394573715, -0.00383781269]{{.*}}, {{.*}}[0.0178247672, -0.0211799927], [-7.134370e-02, 0.00868515763]{{.*}}, {{.*}}[-3.9825665E-10, 0.00232082023], [0.00341839972, 0.01514313]{{.*}}, {{.*}}[3.34836572E-4, -0.00221243338], [-9.64127597E-4, -3.94316746E-10]{{.*}}, {{.*}}[-0.00122044468, 0.00965741463], [-0.00100710022, -0.00124419201]{{.*}}, {{.*}}[-0.00233115326, 0.00203743274], [-0.003079077, 0.0361107253]{{.*}}, {{.*}}[-4.32482251E-4, 0.00191138953], [0.00277041947, -4.13662056E-4]{{.*}}, {{.*}}[2.441080e-03, -0.00233326037], [-0.0275826417, 0.0237795357]{{.*}}> : tensor<8x1x2x2xf32>} : () -> tensor<8x1x2x2xf32>
  // CHECK: [[RES:%.+]] = "onnx.Conv"([[X]], [[W]], [[NOBIAS]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<1x8x27x27xf32>
  // CHECK: return [[RES]] : tensor<1x8x27x27xf32>
}
