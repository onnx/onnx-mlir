// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Common tests. Use ONNXAddOp as example.

// -----

// CHECK-LABEL: @test_scalar_attr() -> tensor<f32>
func.func @test_scalar_attr() -> tensor<f32> {
  %0 = onnx.Constant dense<1.0> : tensor<f32>
  %1 = onnx.Constant dense<2.0> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) : (tensor<f32> , tensor<f32>) -> tensor<f32>
  "onnx.Return"(%2) : (tensor<f32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<f32>
}

// -----

// CHECK-LABEL: @test_single_value_attr() -> tensor<1xf32>
func.func @test_single_value_attr() -> tensor<1xf32> {
  %0 = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %1 = onnx.Constant dense<[2.0]> : tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "onnx.Return"(%2) : (tensor<1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1xf32>
}

// -----

// CHECK-LABEL: @test_splat_attr() -> tensor<3xf32>
func.func @test_splat_attr() -> tensor<3xf32> {
  %0 = onnx.Constant dense<1.0> : tensor<3xf32>
  %1 = onnx.Constant dense<2.0> : tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "onnx.Return"(%2) : (tensor<3xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<3xf32>
}

// -----

// CHECK-LABEL: @test_splat_nonsplat_attrs() -> tensor<3xf32>
func.func @test_splat_nonsplat_attrs() -> tensor<3xf32> {
  %0 = onnx.Constant dense<1.0> : tensor<3xf32>
  %1 = onnx.Constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "onnx.Return"(%2) : (tensor<3xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// ADD tests 

/// Test ConstantOp assoc for add

// -----

// CHECK-LABEL: @test_add_constant_1(%arg0: tensor<3xf32>) -> tensor<3xf32>
func.func @test_add_constant_1(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = onnx.Constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "onnx.Return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// Test ConstantOp assoc for add

// -----

// CHECK-LABEL: @test_add_constant_2(%arg0: tensor<3xf32>) -> tensor<3xf32>
func.func @test_add_constant_2(%arg0 : tensor<3xf32>) -> tensor<3xf32> {
  %0 = onnx.Constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<3xf32> , tensor<3xf32>) -> tensor<3xf32>
  "onnx.Return"(%1) : (tensor<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
}

/// Change (x+c1)+c2 to x+(c1+c2)

// -----

// CHECK-LABEL: @test_add_constant_3(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func.func @test_add_constant_3(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = onnx.Constant dense<[10, 11, 12]> : tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "onnx.Return"(%3) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<[10, 12, 14]> : tensor<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Same test as above, but with a use of an intermediary result
/// change (x+c1)+c2 + (x+c1) to x+x + (c1+c2+c3)

// -----

func.func @test_add_constant_4(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = onnx.Constant dense<[10, 11, 12]> : tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %5 = "onnx.Add"(%3, %4) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "onnx.Return"(%5) : (tensor<3xi32>) -> ()
// CHECK-LABEL: @test_add_constant_4(%arg0: tensor<3xi32>) -> tensor<3xi32> 
  // CHECK-DAG: [[CONST1:%.+]] = onnx.Constant dense<[10, 13, 16]> : tensor<3xi32>
  // CHECK-DAG: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg0) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Change (x+c0)+y  + (z+c1) to  (x+y)+z + (c1+c2)

// -----

// CHECK-LABEL: @test_add_constant_5(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> 
func.func @test_add_constant_5(%arg0 : tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = onnx.Constant dense<[10, 11, 12]> : tensor<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Add"(%2, %arg1) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Add"(%1, %arg2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %5 = "onnx.Add"(%3, %4) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "onnx.Return"(%5) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<[10, 12, 14]> : tensor<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], %arg2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[ADD3:%.+]] = "onnx.Add"([[ADD2]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

// -----

func.func @test_add_const_associative2_2uses(%x: tensor<i32>, %y: tensor<i32>, %z: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %c = onnx.Constant dense<1> : tensor<i32>
  %1 = "onnx.Add"(%x, %c) : (tensor<i32> , tensor<i32>) -> tensor<i32>
  %2 = "onnx.Add"(%1, %y) : (tensor<i32> , tensor<i32>) -> tensor<i32>
  %3 = "onnx.Add"(%1, %z) : (tensor<i32> , tensor<i32>) -> tensor<i32>
  onnx.Return %2, %3 : tensor<i32>, tensor<i32>
// CHECK-LABEL:  func.func @test_add_const_associative2_2uses
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i32>, [[PARAM_1_:%.+]]: tensor<i32>, [[PARAM_2_:%.+]]: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_1_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_2_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           onnx.Return [[VAR_2_]], [[VAR_3_]] : tensor<i32>, tensor<i32>
// CHECK:         }
}

// -----

// CHECK-LABEL: @test_add_zeros(%arg0: tensor<3xi32>) -> tensor<3xi32>
func.func @test_add_zeros(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  onnx.Return %1 : tensor<3xi32>
  // CHECK: onnx.Return %arg0 : tensor<3xi32>
}

/// Test broadcast 1 -> 2d

// -----

// CHECK-LABEL: @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_broadcast_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi32>
  %1 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// Test broadcast 2d (size one) -> 2d

// -----

// CHECK-LABEL: @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_broadcast_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[1]]> : tensor<1x1xi32>
  %1 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

/// check 1d -> 2d

// -----

// CHECK-LABEL: @test_broadcast_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_broadcast_3(%arg0 : tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[1], [2], [3]]> : tensor<3x1xi32>
  %1 = onnx.Constant dense<[[10, 11], [21, 22], [31, 32]]> : tensor<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (tensor<3x1xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%3) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[11, 12], [23, 24], [34, 35]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}


//===----------------------------------------------------------------------===//  
/// MUL tests (same as add, so have only two).
  
/// Change (x*c1)*c2 to x*(c1*c2)

// -----

// CHECK-LABEL: @test_mul_constant_3(%arg0: tensor<3xi32>) -> tensor<3xi32> 
func.func @test_mul_constant_3(%arg0 : tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = onnx.Constant dense<[10, 11, 12]> : tensor<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Mul"(%1, %2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "onnx.Return"(%3) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<[0, 11, 24]> : tensor<3xi32>
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

/// Change (x*c0)*y  * (z*c1) to  (x*y)*z * (c1*c2)

// -----

// CHECK-LABEL: @test_mul_constant_5(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> 
func.func @test_mul_constant_5(%arg0 : tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = onnx.Constant dense<[10, 11, 12]> : tensor<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %3 = "onnx.Mul"(%2, %arg1) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %4 = "onnx.Mul"(%1, %arg2) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  %5 = "onnx.Mul"(%3, %4) : (tensor<3xi32> , tensor<3xi32>) -> tensor<3xi32>
  "onnx.Return"(%5) : (tensor<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<[0, 11, 24]> : tensor<3xi32>
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[MUL2:%.+]] = "onnx.Mul"([[MUL1]], %arg2) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  // CHECK-NEXT: [[MUL3:%.+]] = "onnx.Mul"([[MUL2]], [[CONST1]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
}

// -----

// CHECK-LABEL: @test_mul_ones(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
func.func @test_mul_ones(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %1 = "onnx.Mul"(%arg0, %0) : (tensor<2x2xf32> , tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %1 : tensor<2x2xf32>
  // CHECK: onnx.Return %arg0 : tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
/// SUB and NEG tests.

// check of sub two constants

// -----  

// CHECK-LABEL: @test_sub_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_sub_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = onnx.Constant dense<[[2]]> : tensor<1x1xi32>
  %2 = "onnx.Sub"(%0, %1) : (tensor<3x2xi32>, tensor<1x1xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%2) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[0, 1], [2, 3], [4, 5]]> : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_sub_zeros(%arg0: tensor<f32>) -> tensor<f32>
func.func @test_sub_zeros(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = onnx.Constant dense<0.0> : tensor<f32>
  %1 = "onnx.Sub"(%arg0, %0) : (tensor<f32> , tensor<f32>) -> tensor<f32>
  onnx.Return %1 : tensor<f32>
  // CHECK: onnx.Return %arg0 : tensor<f32>
}

/// check sub to add of negative

// -----

// CHECK-LABEL: @test_neg_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_neg_1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = "onnx.Sub"(%arg0, %0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%1) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[-2, -3], [-4, -5], [-6, -7]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_neg_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_neg_2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = onnx.Constant dense<[[10]]> : tensor<1x1xi32>
  %2 = "onnx.Sub"(%arg0, %0) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %5 = "onnx.Add"(%2, %1) : (tensor<3x2xi32> , tensor<1x1xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%5) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_neg_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32>
func.func @test_neg_3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = onnx.Constant dense<[[10]]> : tensor<1x1xi32>
  %2 = "onnx.Neg"(%0) : (tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = "onnx.Add"(%arg0, %2) : (tensor<3x2xi32> , tensor<3x2xi32>) -> tensor<3x2xi32>
  %4 = "onnx.Add"(%3, %1) : (tensor<3x2xi32> , tensor<1x1xi32>) -> tensor<3x2xi32>
  "onnx.Return"(%4) : (tensor<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = onnx.Constant dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
}

//===----------------------------------------------------------------------===//
/// Transpose tests.

// -----  

// CHECK-LABEL: test_default_transpose_const_1
  func.func @test_default_transpose_const_1() -> tensor<*xi32> {
  %0 = onnx.Constant dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "onnx.Return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = onnx.Constant dense<[{{.}}[111, 211], [121, 221], [131, 231]{{.}}, [{{.}}112, 212], [122, 222], [132, 232]{{.}}, [{{.}}113, 213], [123, 223], [133, 233]{{.}}, [{{.}}114, 214], [124, 224], [134, 234]{{.}}]> : tensor<4x3x2xi32>
  // CHECK: onnx.Return [[RES]] : tensor<4x3x2xi32>
}

// -----  

// CHECK-LABEL: test_default_transpose_const_2
func.func @test_default_transpose_const_2() -> tensor<*xi32> {
  %0 = onnx.Constant dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 2, 1]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "onnx.Return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] = onnx.Constant dense<[{{.}}[111, 121, 131], [112, 122, 132], [113, 123, 133], [114, 124, 134]{{.}}, [{{.}}211, 221, 231], [212, 222, 232], [213, 223, 233], [214, 224, 234]{{.}}]> : tensor<2x4x3xi32>
  // CHECK: onnx.Return [[RES]] : tensor<2x4x3xi32>
}

// -----  

// CHECK-LABEL: test_default_transpose_const_3
func.func @test_default_transpose_const_3() -> tensor<*xi32> {
  %0 = onnx.Constant dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [1, 0, 2]} : (tensor<2x3x4xi32>) -> tensor<*xi32>
  "onnx.Return"(%1) : (tensor<*xi32>) -> ()
  // CHECK: [[RES:%.+]] =  onnx.Constant dense<[{{.}}[111, 112, 113, 114], [211, 212, 213, 214]{{.}}, [{{.}}121, 122, 123, 124], [221, 222, 223, 224]{{.}}, [{{.}}131, 132, 133, 134], [231, 232, 233, 234]{{.}}]> : tensor<3x2x4xi32>
  // CHECK: onnx.Return [[RES]] : tensor<3x2x4xi32>
}

//===----------------------------------------------------------------------===//
/// Div tests

// -----

// CHECK-LABEL: @test_div() -> tensor<3x2xf32>
func.func @test_div() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[2.0]]> : tensor<1x1xf32>
  %2 = "onnx.Div"(%0, %1) : (tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "onnx.Return"(%2) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{\]}}> : tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Div"{{.*}}
}

// -----

// CHECK-LABEL: @test_div_ones(%arg0: tensor<1x2xui8>) -> tensor<1x2xui8>
func.func @test_div_ones(%arg0 : tensor<1x2xui8>) -> tensor<1x2xui8> {
  %0 = onnx.Constant dense<[[1, 1]]> : tensor<1x2xui8>
  %1 = "onnx.Div"(%arg0, %0) : (tensor<1x2xui8> , tensor<1x2xui8>) -> tensor<1x2xui8>
  onnx.Return %1 : tensor<1x2xui8>
  // CHECK: onnx.Return %arg0 : tensor<1x2xui8>
}

//===----------------------------------------------------------------------===//
/// Equal test

// -----

// CHECK-LABEL: @test_equal() -> tensor<3xi1>
func.func @test_equal() -> tensor<3xi1> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %1 = onnx.Constant dense<[-1, 0, 2]> : tensor<3xi64>
  %2 = "onnx.Equal"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  "onnx.Return"(%2) : (tensor<3xi1>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<[false, true, false]> : tensor<3xi1>
  // CHECK-NOT: {{.*}} = "onnx.Equal"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Less test

// -----

// CHECK-LABEL: @test_less() -> tensor<3xi1>
func.func @test_less() -> tensor<3xi1> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %1 = onnx.Constant dense<[-1, 0, 2]> : tensor<3xi64>
  %2 = "onnx.Less"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  "onnx.Return"(%2) : (tensor<3xi1>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<[false, false, true]> : tensor<3xi1>
  // CHECK-NOT: {{.*}} = "onnx.Less"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Greater test

// -----

// CHECK-LABEL: @test_greater() -> tensor<3xi1>
func.func @test_greater() -> tensor<3xi1> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %1 = onnx.Constant dense<[-1, 0, 2]> : tensor<3xi64>
  %2 = "onnx.Greater"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  "onnx.Return"(%2) : (tensor<3xi1>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<[true, false, false]> : tensor<3xi1>
  // CHECK-NOT: {{.*}} = "onnx.Greater"{{.*}}
}

//===----------------------------------------------------------------------===//
/// LessOrEqual test

// -----

// CHECK-LABEL: @test_lessorequal() -> tensor<3xi1>
func.func @test_lessorequal() -> tensor<3xi1> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %1 = onnx.Constant dense<[-1, 0, 2]> : tensor<3xi64>
  %2 = "onnx.LessOrEqual"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  "onnx.Return"(%2) : (tensor<3xi1>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<[false, true, true]> : tensor<3xi1>
  // CHECK-NOT: {{.*}} = "onnx.LessOrEqual"{{.*}}
}

//===----------------------------------------------------------------------===//
/// GreaterOrEqual test

// -----

// CHECK-LABEL: @test_greaterorequal() -> tensor<3xi1>
func.func @test_greaterorequal() -> tensor<3xi1> {
  %0 = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %1 = onnx.Constant dense<[-1, 0, 2]> : tensor<3xi64>
  %2 = "onnx.GreaterOrEqual"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  "onnx.Return"(%2) : (tensor<3xi1>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<[true, true, false]> : tensor<3xi1>
  // CHECK-NOT: {{.*}} = "onnx.GreaterOrEqual"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Modulo tests

// -----

// CHECK-LABEL: @test_modulo_int_both_neg() -> tensor<i64>
func.func @test_modulo_int_both_neg() -> tensor<i64> {
  %0 = onnx.Constant dense<-7> : tensor<i64>
  %1 = onnx.Constant dense<-5> : tensor<i64>
  %2 = "onnx.Mod"(%0, %1) : (tensor<i64> , tensor<i64>) -> tensor<i64>
  "onnx.Return"(%2) : (tensor<i64>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<-2> : tensor<i64>
}

// -----

// CHECK-LABEL: @test_modulo_int_neg() -> tensor<i64>
func.func @test_modulo_int_neg() -> tensor<i64> {
  %0 = onnx.Constant dense<-4> : tensor<i64>
  %1 = onnx.Constant dense<2> : tensor<i64>
  %2 = "onnx.Mod"(%0, %1) : (tensor<i64> , tensor<i64>) -> tensor<i64>
  "onnx.Return"(%2) : (tensor<i64>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<0> : tensor<i64>
}

// -----

// CHECK-LABEL: @test_modulo_int_pos() -> tensor<i64>
func.func @test_modulo_int_pos() -> tensor<i64> {
  %0 = onnx.Constant dense<5> : tensor<i64>
  %1 = onnx.Constant dense<8> : tensor<i64>
  %2 = "onnx.Mod"(%0, %1) : (tensor<i64> , tensor<i64>) -> tensor<i64>
  "onnx.Return"(%2) : (tensor<i64>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<5> : tensor<i64>
}

// -----

// CHECK-LABEL: @test_modulo_float() -> tensor<1xf32>
func.func @test_modulo_float() -> tensor<1xf32> {
  %0 = onnx.Constant dense<[2.0]> : tensor<1xf32>
  %1 = onnx.Constant dense<[7.0]> : tensor<1xf32>
  %2 = "onnx.Mod"(%0, %1) {fmod = 1 : si64} : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "onnx.Return"(%2) : (tensor<1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1xf32>
  // CHECK-NOT: {{.*}} = "onnx.Mod"{{.*}}
}

// -----

// CHECK-LABEL: @test_modulo_float_mixed() -> tensor<1xf32>
func.func @test_modulo_float_mixed() -> tensor<1xf32> {
  %0 = onnx.Constant dense<[-4.3]> : tensor<1xf32>
  %1 = onnx.Constant dense<[2.1]> : tensor<1xf32>
  %2 = "onnx.Mod"(%0, %1) {fmod = 1 : si64} : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "onnx.Return"(%2) : (tensor<1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<-0.100000381> : tensor<1xf32>
  // CHECK-NOT: {{.*}} = "onnx.Mod"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Sqrt test

// -----

// CHECK-LABEL: @test_sqrt() -> tensor<1x2xf32>
func.func @test_sqrt() -> tensor<1x2xf32> {
  %0 = onnx.Constant dense<[[4.0, 16.0]]> : tensor<1x2xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "onnx.Return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[2.000000e+00, 4.000000e+00]{{\]}}> : tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Sqrt"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Relu test

// -----

// CHECK-LABEL: @test_relu() -> tensor<1x2xf32>
func.func @test_relu() -> tensor<1x2xf32> {
  %0 = onnx.Constant dense<[[-4.0, 16.0]]> : tensor<1x2xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "onnx.Return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.600000e+01]{{\]}}> : tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Relu"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Where tests

// -----

// CHECK-LABEL: @test_where() -> tensor<3x2xf32>
func.func @test_where() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<[true, false]> : tensor<2xi1>
  %1 = onnx.Constant dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>
  %2 = onnx.Constant dense<[[2.0]]> : tensor<1x1xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "onnx.Return"(%3) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[2.000000e+00, 2.000000e+00], [6.000000e+00, 2.000000e+00], [1.000000e+01, 2.000000e+00]{{\]}}> : tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Where"{{.*}}
}

// -----

// CHECK-LABEL: @test_where_true() -> tensor<3x2xf32>
func.func @test_where_true() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<true> : tensor<2xi1>
  %1 = onnx.Constant dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>
  %2 = onnx.Constant dense<[[2.0]]> : tensor<1x1xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "onnx.Return"(%3) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[2.000000e+00, 4.000000e+00], [6.000000e+00, 8.000000e+00], [1.000000e+01, 1.200000e+01]{{\]}}> : tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Where"{{.*}}
}

// -----

// CHECK-LABEL: @test_where_false() -> tensor<3x2xf32>
func.func @test_where_false() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<false> : tensor<2xi1>
  %1 = onnx.Constant dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>
  %2 = onnx.Constant dense<[[2.0]]> : tensor<1x1xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "onnx.Return"(%3) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<2.000000e+00> : tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Where"{{.*}}
}

// -----

// CHECK-LABEL: @test_where_splat_branches() -> tensor<3x2xf32>
func.func @test_where_splat_branches() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<[true, false]> : tensor<2xi1>
  %1 = onnx.Constant dense<1.0> : tensor<3x2xf32>
  %2 = onnx.Constant dense<2.0> : tensor<1x1xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<3x2xf32>
  "onnx.Return"(%3) : (tensor<3x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00]{{\]}}> : tensor<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Where"{{.*}}
}

//===----------------------------------------------------------------------===//
/// MatMul tests

// -----

// CHECK-LABEL: @test_matmul_lhs_zero(%arg0: tensor<3x2xi32>) -> tensor<4x2xi32>
func.func @test_matmul_lhs_zero(%arg0: tensor<3x2xi32>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<0> : tensor<4x3xi32>
  %2 = "onnx.MatMul"(%0, %arg0) : (tensor<4x3xi32>, tensor<3x2xi32>) -> tensor<4x2xi32>
  "onnx.Return"(%2) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmul_rhs_zero(%arg0: tensor<4x3xi64>) -> tensor<4x2xi64>
func.func @test_matmul_rhs_zero(%arg0: tensor<4x3xi64>) -> tensor<4x2xi64> {
  %0 = onnx.Constant dense<0> : tensor<3x2xi64>
  %2 = "onnx.MatMul"(%arg0, %0) : (tensor<4x3xi64>, tensor<3x2xi64>) -> tensor<4x2xi64>
  "onnx.Return"(%2) : (tensor<4x2xi64>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi64>
  // CHECK-NOT: {{.*}} = "onnx.MatMul"{{.*}}
}

// -----

// The MatMulZerosOnRhs constant propagation pattern doesn't match fp types.
func.func @test_matmul_rhs_zero_f16(%arg0: tensor<4x3xf16>) -> tensor<4x2xf16> {
  %0 = onnx.Constant dense<0.0> : tensor<3x2xf16>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<4x3xf16>, tensor<3x2xf16>) -> tensor<4x2xf16>
  "onnx.Return"(%1) : (tensor<4x2xf16>) -> ()
  // CHECK-LABEL:  func.func @test_matmul_rhs_zero_f16
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x3xf16>) -> tensor<4x2xf16> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3x2xf16>
  // CHECK:           [[VAR_1_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<4x3xf16>, tensor<3x2xf16>) -> tensor<4x2xf16>
  // CHECK:           onnx.Return [[VAR_1_]] : tensor<4x2xf16>
  // CHECK:         }
}

// -----

func.func @test_matmul_2d() -> (tensor<2x1xf32>) {
  %0 = "onnx.Constant"() {value = dense<1.> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "onnx.Constant"() {value = dense<1.> : tensor<3x1xf32>} : () -> tensor<3x1xf32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<2x3xf32>, tensor<3x1xf32>) -> tensor<2x1xf32>
  onnx.Return %3 : tensor<2x1xf32>
  // CHECK-LABEL: test_matmul_2d
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<2x1xf32>
}

// -----

func.func @test_matmul_2d_batch() -> (tensor<4x2x1xf64>) {
  %0 = "onnx.Constant"() {value = dense<100.> : tensor<4x2x3xf64>} : () -> tensor<4x2x3xf64>
  %1 = "onnx.Constant"() {value = dense<100.> : tensor<4x3x1xf64>} : () -> tensor<4x3x1xf64>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<4x2x3xf64>, tensor<4x3x1xf64>) -> tensor<4x2x1xf64>
  onnx.Return %3 : tensor<4x2x1xf64>
  // CHECK-LABEL: test_matmul_2d_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3.000000e+04> : tensor<4x2x1xf64>
}

// -----

func.func @test_matmul_2d_batch_lhs() -> (tensor<4x2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<1> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %1 = "onnx.Constant"() {value = dense<3> : tensor<3x1xi32>} : () -> tensor<3x1xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<4x2x3xi32>, tensor<3x1xi32>) -> tensor<4x2x1xi32>
  onnx.Return %3 : tensor<4x2x1xi32>
  // CHECK-LABEL: test_matmul_2d_batch_lhs
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<9> : tensor<4x2x1xi32>
}

// -----

func.func @test_matmul_2d_batch_broadcast() -> (tensor<5x4x2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<1> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %1 = "onnx.Constant"() {value = dense<-1> : tensor<5x1x3x1xi32>} : () -> tensor<5x1x3x1xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<4x2x3xi32>, tensor<5x1x3x1xi32>) -> tensor<5x4x2x1xi32>
  onnx.Return %3 : tensor<5x4x2x1xi32>
  // CHECK-LABEL: test_matmul_2d_batch_broadcast
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<-3> : tensor<5x4x2x1xi32>
}

// -----

func.func @test_matmul_lhs_vector() -> (tensor<3xi32>) {
  %0 = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "onnx.Constant"() {value = dense<10> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<2xi32>, tensor<2x3xi32>) -> tensor<3xi32>
  onnx.Return %3 : tensor<3xi32>
  // CHECK-LABEL: test_matmul_lhs_vector
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3000> : tensor<3xi32>
}

// -----

func.func @test_matmul_lhs_vector_batch() -> (tensor<4x3xi32>) {
  %0 = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "onnx.Constant"() {value = dense<10> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<2xi32>, tensor<4x2x3xi32>) -> tensor<4x3xi32>
  onnx.Return %3 : tensor<4x3xi32>
  // CHECK-LABEL: test_matmul_lhs_vector_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3000> : tensor<4x3xi32>
}

// -----

func.func @test_matmul_rhs_vector() -> (tensor<2xi32>) {
  %0 = "onnx.Constant"() {value = dense<100> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 20, 30]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2xi32>
  onnx.Return %3 : tensor<2xi32>
  // CHECK-LABEL: test_matmul_rhs_vector
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<6000> : tensor<2xi32>
}

// -----

func.func @test_matmul_rhs_vector_batch() -> (tensor<4x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<100> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 20, 30]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<4x2x3xi32>, tensor<3xi32>) -> tensor<4x2xi32>
  onnx.Return %3 : tensor<4x2xi32>
  // CHECK-LABEL: test_matmul_rhs_vector_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<6000> : tensor<4x2xi32>
}

// -----

// 1486 == 200 * 7 + 40 * 2 + 2 * 3
func.func @test_matmul_two_vectors() -> (tensor<ui32>) {
  %0 = "onnx.Constant"() {value = dense<[200, 40, 2]> : tensor<3xui32>} : () -> tensor<3xui32>
  %1 = "onnx.Constant"() {value = dense<[7, 2, 3]> : tensor<3xui32>} : () -> tensor<3xui32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<3xui32>, tensor<3xui32>) -> tensor<ui32>
  onnx.Return %3 : tensor<ui32>
  // CHECK-LABEL: test_matmul_two_vectors
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<1486> : tensor<ui32>
}

//===----------------------------------------------------------------------===//
/// MatMulInteger tests

// -----

// CHECK-LABEL: @test_matmulinteger_lhs_zero(%arg0: tensor<3x2xui8>) -> tensor<4x2xi32>
func.func @test_matmulinteger_lhs_zero(%arg0: tensor<3x2xui8>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<0> : tensor<4x3xi8>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.MatMulInteger"(%0, %arg0, %1, %1) : (tensor<4x3xi8>, tensor<3x2xui8>, none, none) -> tensor<4x2xi32>
  "onnx.Return"(%2) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_lhs_scalar(%arg0: tensor<3x2xui8>) -> tensor<4x2xi32>
func.func @test_matmulinteger_lhs_scalar(%arg0: tensor<3x2xui8>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<7> : tensor<4x3xui8>
  %1 = onnx.Constant dense<7> : tensor<ui8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %arg0, %1, %2) : (tensor<4x3xui8>, tensor<3x2xui8>, tensor<ui8>, none) -> tensor<4x2xi32>
  "onnx.Return"(%3) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_lhs_vector(%arg0: tensor<3x4xui8>) -> tensor<2x4xi32>
func.func @test_matmulinteger_lhs_vector(%arg0: tensor<3x4xui8>) -> tensor<2x4xi32> {
  %0 = onnx.Constant dense<[[7, 7, 7], [9, 9, 9]]> : tensor<2x3xui8>
  %1 = onnx.Constant dense<[7, 9]> : tensor<2xui8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %arg0, %1, %2) : (tensor<2x3xui8>, tensor<3x4xui8>, tensor<2xui8>, none) -> tensor<2x4xi32>
  "onnx.Return"(%3) : (tensor<2x4xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<2x4xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_lhs_matrix(%arg0: tensor<3x4xui8>) -> tensor<2x4xi32>
func.func @test_matmulinteger_lhs_matrix(%arg0: tensor<3x4xui8>) -> tensor<2x4xi32> {
  %0 = onnx.Constant dense<[[7, 7, 7], [9, 9, 9]]> : tensor<2x3xui8>
  %1 = onnx.Constant dense<[[7], [9]]> : tensor<2x1xui8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %arg0, %1, %2) : (tensor<2x3xui8>, tensor<3x4xui8>, tensor<2x1xui8>, none) -> tensor<2x4xi32>
  "onnx.Return"(%3) : (tensor<2x4xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<2x4xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_rhs_zero_none(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32>
func.func @test_matmulinteger_rhs_zero_none(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<0> : tensor<3x2xui8>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.MatMulInteger"(%arg0, %0, %1, %1) : (tensor<4x3xi8>, tensor<3x2xui8>, none, none) -> tensor<4x2xi32>
  "onnx.Return"(%2) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_rhs_zero_scalar(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32>
func.func @test_matmulinteger_rhs_zero_scalar(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<42> : tensor<3x2xui8>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<42> : tensor<ui8>
  %3 = "onnx.MatMulInteger"(%arg0, %0, %1, %2) : (tensor<4x3xi8>, tensor<3x2xui8>, none, tensor<ui8>) -> tensor<4x2xi32>
  "onnx.Return"(%3) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_rhs_zero_vector(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32>
func.func @test_matmulinteger_rhs_zero_vector(%arg0: tensor<4x3xi8>) -> tensor<4x2xi32> {
  %0 = onnx.Constant dense<[[1, 2], [1, 2], [1, 2]]> : tensor<3x2xui8>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[1, 2]> : tensor<2xui8>
  %3 = "onnx.MatMulInteger"(%arg0, %0, %1, %2) : (tensor<4x3xi8>, tensor<3x2xui8>, none, tensor<2xui8>) -> tensor<4x2xi32>
  "onnx.Return"(%3) : (tensor<4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

// CHECK-LABEL: @test_matmulinteger_rhs_zero_tensor(%arg0: tensor<1x4x3xi8>) -> tensor<1x4x2xi32>
func.func @test_matmulinteger_rhs_zero_tensor(%arg0: tensor<1x4x3xi8>) -> tensor<1x4x2xi32> {
  %0 = onnx.Constant dense<[[[1, 2], [1, 2], [1, 2]]]> : tensor<1x3x2xui8>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[[[1, 2]]]> : tensor<1x1x2xui8>
  %3 = "onnx.MatMulInteger"(%arg0, %0, %1, %2) : (tensor<1x4x3xi8>, tensor<1x3x2xui8>, none, tensor<1x1x2xui8>) -> tensor<1x4x2xi32>
  "onnx.Return"(%3) : (tensor<1x4x2xi32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<0> : tensor<1x4x2xi32>
  // CHECK-NOT: {{.*}} = "onnx.MatMulInteger"{{.*}}
}

// -----

func.func @test_matmulinteger_2d() -> (tensor<2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<1> : tensor<2x3xi8>} : () -> tensor<2x3xi8>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<3x1xi8>} : () -> tensor<3x1xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<2x3xi8>, tensor<3x1xi8>, none, none) -> tensor<2x1xi32>
  onnx.Return %3 : tensor<2x1xi32>
  // CHECK-LABEL: test_matmulinteger_2d
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3> : tensor<2x1xi32>
}

// -----

func.func @test_matmulinteger_2d_batch() -> (tensor<4x2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<100> : tensor<4x2x3xi8>} : () -> tensor<4x2x3xi8>
  %1 = "onnx.Constant"() {value = dense<100> : tensor<4x3x1xi8>} : () -> tensor<4x3x1xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<4x2x3xi8>, tensor<4x3x1xi8>, none, none) -> tensor<4x2x1xi32>
  onnx.Return %3 : tensor<4x2x1xi32>
  // CHECK-LABEL: test_matmulinteger_2d_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<30000> : tensor<4x2x1xi32>
}

// -----

func.func @test_matmulinteger_2d_batch_lhs() -> (tensor<4x2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<1> : tensor<4x2x3xi8>} : () -> tensor<4x2x3xi8>
  %1 = "onnx.Constant"() {value = dense<3> : tensor<3x1xi8>} : () -> tensor<3x1xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<4x2x3xi8>, tensor<3x1xi8>, none, none) -> tensor<4x2x1xi32>
  onnx.Return %3 : tensor<4x2x1xi32>
  // CHECK-LABEL: test_matmulinteger_2d_batch_lhs
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<9> : tensor<4x2x1xi32>
}

// -----

func.func @test_matmulinteger_2d_batch_broadcast() -> (tensor<5x4x2x1xi32>) {
  %0 = "onnx.Constant"() {value = dense<1> : tensor<4x2x3xi8>} : () -> tensor<4x2x3xi8>
  %1 = "onnx.Constant"() {value = dense<-1> : tensor<5x1x3x1xi8>} : () -> tensor<5x1x3x1xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<4x2x3xi8>, tensor<5x1x3x1xi8>, none, none) -> tensor<5x4x2x1xi32>
  onnx.Return %3 : tensor<5x4x2x1xi32>
  // CHECK-LABEL: test_matmulinteger_2d_batch_broadcast
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<-3> : tensor<5x4x2x1xi32>
}

// -----

func.func @test_matmulinteger_lhs_vector() -> (tensor<3xi32>) {
  %0 = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xui8>} : () -> tensor<2xui8>
  %1 = "onnx.Constant"() {value = dense<10> : tensor<2x3xi8>} : () -> tensor<2x3xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<2xui8>, tensor<2x3xi8>, none, none) -> tensor<3xi32>
  onnx.Return %3 : tensor<3xi32>
  // CHECK-LABEL: test_matmulinteger_lhs_vector
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3000> : tensor<3xi32>
}

// -----

func.func @test_matmulinteger_lhs_vector_batch() -> (tensor<4x3xi32>) {
  %0 = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xui8>} : () -> tensor<2xui8>
  %1 = "onnx.Constant"() {value = dense<10> : tensor<4x2x3xi8>} : () -> tensor<4x2x3xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<2xui8>, tensor<4x2x3xi8>, none, none) -> tensor<4x3xi32>
  onnx.Return %3 : tensor<4x3xi32>
  // CHECK-LABEL: test_matmulinteger_lhs_vector_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<3000> : tensor<4x3xi32>
}

// -----

func.func @test_matmulinteger_rhs_vector() -> (tensor<2xi32>) {
  %0 = "onnx.Constant"() {value = dense<100> : tensor<2x3xui8>} : () -> tensor<2x3xui8>
  %1 = "onnx.Constant"() {value = dense<[10, 20, 30]> : tensor<3xi8>} : () -> tensor<3xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<2x3xui8>, tensor<3xi8>, none, none) -> tensor<2xi32>
  onnx.Return %3 : tensor<2xi32>
  // CHECK-LABEL: test_matmulinteger_rhs_vector
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<6000> : tensor<2xi32>
}

// -----

func.func @test_matmulinteger_rhs_vector_batch() -> (tensor<4x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<100> : tensor<4x2x3xui8>} : () -> tensor<4x2x3xui8>
  %1 = "onnx.Constant"() {value = dense<[10, 20, 30]> : tensor<3xi8>} : () -> tensor<3xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<4x2x3xui8>, tensor<3xi8>, none, none) -> tensor<4x2xi32>
  onnx.Return %3 : tensor<4x2xi32>
  // CHECK-LABEL: test_matmulinteger_rhs_vector_batch
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<6000> : tensor<4x2xi32>
}

// -----

// 1486 == 200 * 7 + 40 * 2 + 2 * 3
func.func @test_matmulinteger_two_vectors() -> (tensor<i32>) {
  %0 = "onnx.Constant"() {value = dense<[200, 40, 2]> : tensor<3xui8>} : () -> tensor<3xui8>
  %1 = "onnx.Constant"() {value = dense<[7, 2, 3]> : tensor<3xi8>} : () -> tensor<3xi8>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.MatMulInteger"(%0, %1, %2, %2) : (tensor<3xui8>, tensor<3xi8>, none, none) -> tensor<i32>
  onnx.Return %3 : tensor<i32>
  // CHECK-LABEL: test_matmulinteger_two_vectors
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<1486> : tensor<i32>
}

// -----

// This example is taken from the onnx MatMulInteger operation specification.
func.func @test_matmulinteger_with_1Dzeros() -> (tensor<4x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]> : tensor<4x3xui8>} : () -> tensor<4x3xui8>
  %1 = "onnx.Constant"() {value = dense<[[1, 4], [2, 5], [3, 6]]> : tensor<3x2xui8>} : () -> tensor<3x2xui8>
  %2 = "onnx.Constant"() {value = dense<[12]> : tensor<1xui8>} : () -> tensor<1xui8>
  %3 = "onnx.Constant"() {value = dense<[0]> : tensor<1xui8>} : () -> tensor<1xui8>
  %4 = "onnx.MatMulInteger"(%0, %1, %2, %3) : (tensor<4x3xui8>, tensor<3x2xui8>, tensor<1xui8>, tensor<1xui8>) -> tensor<4x2xi32>
  onnx.Return %4 : tensor<4x2xi32>
  // CHECK-LABEL: test_matmulinteger_with_1Dzeros
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[-38, -83], [-44, -98], [-50, -113], [-56, -128]{{\]}}> : tensor<4x2xi32>
}

// -----

func.func @test_matmulinteger_with_0dzeros() -> (tensor<4x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]> : tensor<4x3xui8>} : () -> tensor<4x3xui8>
  %1 = "onnx.Constant"() {value = dense<[[1, 4], [2, 5], [3, 6]]> : tensor<3x2xui8>} : () -> tensor<3x2xui8>
  %2 = "onnx.Constant"() {value = dense<12> : tensor<ui8>} : () -> tensor<ui8>
  %3 = "onnx.Constant"() {value = dense<0> : tensor<ui8>} : () -> tensor<ui8>
  %4 = "onnx.MatMulInteger"(%0, %1, %2, %3) : (tensor<4x3xui8>, tensor<3x2xui8>, tensor<ui8>, tensor<ui8>) -> tensor<4x2xi32>
  onnx.Return %4 : tensor<4x2xi32>
  // CHECK-LABEL: test_matmulinteger_with_0dzeros
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[-38, -83], [-44, -98], [-50, -113], [-56, -128]{{\]}}> : tensor<4x2xi32>
}

//===----------------------------------------------------------------------===//
/// Gemm tests

// -----

func.func @test_gemm_no_bias() -> (tensor<2x2xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.25], [0.5, 0.75]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Gemm"(%0, %1, %2) : (tensor<2x2xf32>, tensor<2x2xf32>, none) -> tensor<2x2xf32>
  onnx.Return %3 : tensor<2x2xf32>
  // CHECK-LABEL: test_gemm_no_bias
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[7.500000e-01, 1.000000e+00], [2.750000e+00, 4.000000e+00]{{\]}}> : tensor<2x2xf32>
}

// -----

func.func @test_gemm_no_bias_transposed() -> (tensor<2x2xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Constant"() {value = dense<[[0.0, 0.25], [0.5, 0.75]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Gemm"(%0, %1, %2) {transA = 1 : si64, transB = 1 : si64} : (tensor<2x2xf32>, tensor<2x2xf32>, none) -> tensor<2x2xf32>
  onnx.Return %3 : tensor<2x2xf32>
  // CHECK-LABEL: test_gemm_no_bias_transposed
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[7.500000e-01, 2.750000e+00], [1.000000e+00, 4.000000e+00]{{\]}}> : tensor<2x2xf32>
}

// -----

func.func @test_gemm() -> (tensor<2x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %2 = "onnx.Constant"() {value = dense<[[1000, 2000], [3000, 4000]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %3 = "onnx.Gemm"(%0, %1, %2) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  onnx.Return %3 : tensor<2x2xi32>
  // CHECK-LABEL: test_gemm
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[1070, 2100], [3150, 4220]{{\]}}> : tensor<2x2xi32>
}

// -----

func.func @test_gemm_beta1000() -> (tensor<2x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %2 = "onnx.Gemm"(%0, %1, %0) {beta = 1000.0 : f32} : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  onnx.Return %2 : tensor<2x2xi32>
  // CHECK-LABEL: test_gemm_beta1000
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[1070, 2100], [3150, 4220]{{\]}}> : tensor<2x2xi32>
}

// -----

func.func @test_gemm_alpha0() -> (tensor<2x2xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Gemm"(%0, %0, %0) {alpha = 0.0 : f32} : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  onnx.Return %1 : tensor<2x2xi32>
  // CHECK-LABEL: test_gemm_alpha0
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{\[}}[1, 2], [3, 4]{{\]}}> : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
/// Reduce tests

// -----

// CHECK-LABEL: @test_reduce_sum_positive_axis() -> tensor<2x1xi32>
func.func @test_reduce_sum_positive_axis() -> tensor<2x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<2x2xi32>, tensor<i64>) -> tensor<2x1xi32>
  "onnx.Return"(%2) : (tensor<2x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[3], [7]{{.}}> : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_negative_axis() -> tensor<1x2xi32>
func.func @test_reduce_sum_negative_axis() -> tensor<1x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<2x2xi32>, tensor<1xi64>) -> tensor<1x2xi32>
  "onnx.Return"(%2) : (tensor<1x2xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[4, 6]{{.}}> : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_all_axes() -> tensor<1x1xi32>
func.func @test_reduce_sum_all_axes() -> tensor<1x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<2x2xi32>, tensor<2xi64>) -> tensor<1x1xi32>
  "onnx.Return"(%2) : (tensor<1x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<10> : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_keedims_false() -> tensor<2xi32>
func.func @test_reduce_sum_keedims_false() -> tensor<2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceSum"(%0, %1) {keepdims = 0 : si64} : (tensor<2x2xi32>, tensor<i64>) -> tensor<2xi32>
  "onnx.Return"(%2) : (tensor<2xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<[3, 7]> : tensor<2xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_noop_with_empty_axes_unset() -> tensor<1x1xi32>
func.func @test_reduce_sum_noop_with_empty_axes_unset() -> tensor<1x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<2x2xi32>, none) -> tensor<1x1xi32>
  "onnx.Return"(%2) : (tensor<1x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<10> : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_noop_with_empty_axes_true() -> tensor<2x2xi32>
func.func @test_reduce_sum_noop_with_empty_axes_true() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.ReduceSum"(%0, %1) {noop_with_empty_axes = 1 : si64} : (tensor<2x2xi32>, none) -> tensor<2x2xi32>
  "onnx.Return"(%2) : (tensor<2x2xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[1, 2], [3, 4]{{.}}> : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_empty() -> tensor<1x1xi32>
func.func @test_reduce_sum_empty() -> tensor<1x1xi32> {
  %0 = "onnx.Constant"() {value = dense<> : tensor<0x2xi32>} : () -> tensor<0x2xi32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<0x2xi32>, none) -> tensor<1x1xi32>
  "onnx.Return"(%2) : (tensor<1x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<0> : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_scalar() -> tensor<i32>
func.func @test_reduce_sum_scalar() -> tensor<i32> {
  %0 = "onnx.Constant"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.ReduceSum"(%0, %1) : (tensor<i32>, none) -> tensor<i32>
  "onnx.Return"(%2) : (tensor<i32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<42> : tensor<i32>
}

// -----

// CHECK-LABEL: @test_reduce_prod_positive_axis() -> tensor<2x1xi32>
func.func @test_reduce_prod_positive_axis() -> tensor<2x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceProd"(%0, %1) : (tensor<2x2xi32>, tensor<i64>) -> tensor<2x1xi32>
  "onnx.Return"(%2) : (tensor<2x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[2], [12]{{.}}> : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_prod_empty() -> tensor<1x1xf32>
func.func @test_reduce_prod_empty() -> tensor<1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<> : tensor<0x2xf32>} : () -> tensor<0x2xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.ReduceProd"(%0, %1) : (tensor<0x2xf32>, none) -> tensor<1x1xf32>
  "onnx.Return"(%2) : (tensor<1x1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_min_positive_axis() -> tensor<2x1xi32>
func.func @test_reduce_min_positive_axis() -> tensor<2x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceMin"(%0, %1) : (tensor<2x2xi32>, tensor<i64>) -> tensor<2x1xi32>
  "onnx.Return"(%2) : (tensor<2x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[1], [3]{{.}}> : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_max_positive_axis() -> tensor<2x1xi32>
func.func @test_reduce_max_positive_axis() -> tensor<2x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceMax"(%0, %1) : (tensor<2x2xi32>, tensor<i64>) -> tensor<2x1xi32>
  "onnx.Return"(%2) : (tensor<2x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[2], [4]{{.}}> : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_mean_i32() -> tensor<2x1xi32>
func.func @test_reduce_mean_i32() -> tensor<2x1xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [4, 6]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceMean"(%0, %1) : (tensor<2x2xi32>, tensor<i64>) -> tensor<2x1xi32>
  "onnx.Return"(%2) : (tensor<2x1xi32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[1], [5]{{.}}> : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: @test_reduce_mean_f32() -> tensor<2x1xf32>
func.func @test_reduce_mean_f32() -> tensor<2x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [4.0, 6.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "onnx.ReduceMean"(%0, %1) : (tensor<2x2xf32>, tensor<i64>) -> tensor<2x1xf32>
  "onnx.Return"(%2) : (tensor<2x1xf32>) -> ()
  // CHECK: [[CONST:%.+]] = onnx.Constant dense<{{.}}[1.500000e+00], [5.000000e+00]{{.}}> : tensor<2x1xf32>
}

//===----------------------------------------------------------------------===//
/// Unsqueeze tests

// -----

// CHECK-LABEL: @test_unsqueeze() -> tensor<2x1x1xf32>
func.func @test_unsqueeze() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[4.0, 16.0]> : tensor<2xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%0, %1) : (tensor<2xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
}

// -----

// CHECK-LABEL: @test_unsqueezev11() -> tensor<2x1x1xf32>
func.func @test_unsqueezev11() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[4.0, 16.0]> : tensor<2xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes = [1, 2]} : (tensor<2xf32>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Squeeze tests

// -----

// CHECK-LABEL: @test_squeeze() -> tensor<2xf32>
func.func @test_squeeze() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Squeeze"(%0, %1) : (tensor<2x1x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
  // CHECK: [[RES:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
}

// -----

// CHECK-LABEL: @test_squeezev11() -> tensor<2xf32>
func.func @test_squeezev11() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes = [1, 2]} : (tensor<2x1x1xf32>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: [[RES:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2xf32>
  // CHECK-NOT: {{.*}} = "onnx.SqueezeV11"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Split tests

// -----

// CHECK-LABEL: @test_split_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func.func @test_split_axis_0() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %split = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) {axis = 0 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "onnx.Return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// CHECK-LABEL: @test_split_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func.func @test_split_axis_1() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %split = onnx.Constant dense<[5, 5]> : tensor<2xi64>
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "onnx.Return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>
  // CHECK: {{.*}}  = onnx.Constant dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// COM: There is no constant propagation if the split's input is not a constant.

// CHECK-LABEL: @test_split_axis_2(%arg0: tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func.func @test_split_axis_2(%arg0 : tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %0 = onnx.Constant dense<[5, 5]> : tensor<2xi64>
  %1, %2 = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "onnx.Return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<2xi64>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
}

// -----

func.func @test_mul_folding(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[[0.0234164055, 0.0228030644], [2.442580e-02, 0.0237577036]]], [[[-0.0410864502, 0.0488203131], [0.164448678, -0.0200194642]]], [[[-4.34581793E-9, 0.025325032], [0.0373019315, 0.165243402]]], [[[-0.0198689923, 0.131284416], [0.0572107285, 2.33985098E-8]]], [[[0.0187684372, -0.148515195], [0.0154875498, 0.019133633]]], [[[0.0176953916, -0.0154658081], [0.0233727545, -0.274110436]]], [[[-0.021181887, 0.0936150252], [0.135688141, -0.0202601217]]], [[[-0.0201558527, 0.0192655921], [0.227748245, -0.196346223]]]]> : tensor<8x1x2x2xf32>
  %1 = "onnx.NoValue"() {value} : () -> none
  %2 = onnx.Constant dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>
  %3 = "onnx.UnsqueezeV11"(%2) {axes = [3]} : (tensor<8x1x1xf32>) -> tensor<8x1x1x1xf32>
  %4 = "onnx.Mul"(%0, %3) : (tensor<8x1x2x2xf32>, tensor<8x1x1x1xf32>) -> tensor<8x1x2x2xf32>
  %5 = "onnx.Conv"(%arg0, %4, %1) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<*xf32>
  onnx.Return %5 : tensor<*xf32>

  // CHECK-LABEL:  func @test_mul_folding
  // CHECK-SAME:   ([[X:%.+]]: tensor<1x1x28x28xf32>) -> tensor<1x8x27x27xf32> {
  // CHECK-DAG: [[NOBIAS:%.+]] = "onnx.NoValue"() {value} : () -> none    
  // CHECK-DAG: [[W:%.+]] = onnx.Constant dense<{{.*}}[-0.00378267956, -0.00368360057], [-0.00394573715, -0.00383781269]{{.*}}, {{.*}}[0.0178247672, -0.0211799927], [-7.134370e-02, 0.00868515763]{{.*}}, {{.*}}[-3.9825665E-10, 0.00232082023], [0.00341839972, 0.01514313]{{.*}}, {{.*}}[3.34836572E-4, -0.00221243338], [-9.64127597E-4, -3.94316746E-10]{{.*}}, {{.*}}[-0.00122044468, 0.00965741463], [-0.00100710022, -0.00124419201]{{.*}}, {{.*}}[-0.00233115326, 0.00203743274], [-0.003079077, 0.0361107253]{{.*}}, {{.*}}[-4.32482251E-4, 0.00191138953], [0.00277041947, -4.13662056E-4]{{.*}}, {{.*}}[2.441080e-03, -0.00233326037], [-0.0275826417, 0.0237795357]{{.*}}> : tensor<8x1x2x2xf32>
  // CHECK: [[RES:%.+]] = "onnx.Conv"([[X]], [[W]], [[NOBIAS]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<1x8x27x27xf32>
  // CHECK: onnx.Return [[RES]] : tensor<1x8x27x27xf32>
}

// -----

func.func @test_cast_i32_i1_i32() -> tensor<4xi32> {
  %0 = onnx.Constant dense<[-1, 0, 1, 2]> : tensor<4xi32>
  %1 = "onnx.Cast"(%0) {to = i1} : (tensor<4xi32>) -> tensor<4xi1>
  %2 = "onnx.Cast"(%1) {to = i32} : (tensor<4xi1>) -> tensor<4xi32>
  "onnx.Return"(%2) : (tensor<4xi32>) -> ()

  // CHECK-LABEL:  func @test_cast_i32_i1_i32
  // CHECK-SAME:   () -> tensor<4xi32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 0, 1, 1]> : tensor<4xi32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<4xi32>
  // CHECK:         }
}

// -----

func.func @test_cast_i32_i64() -> tensor<3x2xi64> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = "onnx.Cast"(%0) {to = i64} : (tensor<3x2xi32>) -> tensor<3x2xi64>
  "onnx.Return"(%1) : (tensor<3x2xi64>) -> ()

  // CHECK-LABEL:  func @test_cast_i32_i64
  // CHECK-SAME:   () -> tensor<3x2xi64> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2, 3], [4, 5], [6, 7]{{.}}> : tensor<3x2xi64>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x2xi64>
  // CHECK:         }
}

// -----

func.func @test_cast_i64_i32() -> tensor<3x2xi32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi64>
  %1 = "onnx.Cast"(%0) {to = i32} : (tensor<3x2xi64>) -> tensor<3x2xi32>
  "onnx.Return"(%1) : (tensor<3x2xi32>) -> ()

  // CHECK-LABEL:  func @test_cast_i64_i32
  // CHECK-SAME:   () -> tensor<3x2xi32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2, 3], [4, 5], [6, 7]{{.}}> : tensor<3x2xi32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x2xi32>
  // CHECK:         }
}

// -----

func.func @test_cast_i32_f32() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<3x2xi32>) -> tensor<3x2xf32>
  "onnx.Return"(%1) : (tensor<3x2xf32>) -> ()

  // CHECK-LABEL:  func @test_cast_i32_f32
  // CHECK-SAME:   () -> tensor<3x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00]{{.}}> : tensor<3x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x2xf32>
  // CHECK:         }
}

// -----

func.func @test_cast_f32_i32() -> tensor<8xi32> {
  // COM: 0x7F800000/0xFF800000 are +/-INF
  // COM: 0x7F800001/0xFFFFFFFF are smallest positive NaN/largest negative NaN
  %0 = onnx.Constant dense<[2.3, 3.6, -1.0e10, 1.0e10, 0x7F800000, 0xFF800000, 0x7F800001, 0xFFFFFFFF]> : tensor<8xf32>
  %1 = "onnx.Cast"(%0) {to = i32} : (tensor<8xf32>) -> tensor<8xi32>
  "onnx.Return"(%1) : (tensor<8xi32>) -> ()

  // CHECK-LABEL:  func @test_cast_f32_i32
  // CHECK-SAME:   () -> tensor<8xi32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3, -2147483648, 2147483647, 2147483647, -2147483648, 0, 0]> : tensor<8xi32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<8xi32>
  // CHECK:         }
}

// -----

func.func @test_cast_f32_i64() -> tensor<3x2xi64> {
  %0 = onnx.Constant dense<[[2.3, 3.6], [4.5, 5.5], [6.0, 7.0]]> : tensor<3x2xf32>
  %1 = "onnx.Cast"(%0) {to = i64} : (tensor<3x2xf32>) -> tensor<3x2xi64>
  "onnx.Return"(%1) : (tensor<3x2xi64>) -> ()

  // CHECK-LABEL:  func @test_cast_f32_i64
  // CHECK-SAME:   () -> tensor<3x2xi64> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2, 3], [4, 5], [6, 7]{{.}}> : tensor<3x2xi64>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x2xi64>
  // CHECK:         }
}

// -----

func.func @test_cast_i64_f32() -> tensor<3x2xf32> {
  %0 = onnx.Constant dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi64>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<3x2xi64>) -> tensor<3x2xf32>
  "onnx.Return"(%1) : (tensor<3x2xf32>) -> ()

  // CHECK-LABEL:  func @test_cast_i64_f32
  // CHECK-SAME:   () -> tensor<3x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00]{{.}}> : tensor<3x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x2xf32>
  // CHECK:         }
}

// -----

// Among the float literals below
// 0x7F800000, 0xFF800000 mean +/-infinity,
// 0x7F800001, 0xFF800001 mean NaN,
// see https://en.wikipedia.org/wiki/Single-precision_floating-point_format

func.func @test_cast_f32_f8E4M3FN() -> (tensor<12xf8E4M3FN>, tensor<12xf8E4M3FN>) {
  %0 = onnx.Constant dense<[0.0, -0.0, 400.0, -400.0, 448.0, -448.0, 600.0, -600.0, 0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001]> : tensor<12xf32>
  %1 = "onnx.Cast"(%0) {to = f8E4M3FN, saturate = 0 : si64} : (tensor<12xf32>) -> tensor<12xf8E4M3FN>
  %2 = "onnx.Cast"(%0) {to = f8E4M3FN, saturate = 1 : si64} : (tensor<12xf32>) -> tensor<12xf8E4M3FN>
  onnx.Return %1, %2 : tensor<12xf8E4M3FN>, tensor<12xf8E4M3FN>

// f8E4M3FN literals 0x7F, 0xFF mean NaN
// CHECK-LABEL:  func.func @test_cast_f32_f8E4M3FN
// CHECK-SAME:   () -> (tensor<12xf8E4M3FN>, tensor<12xf8E4M3FN>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0.000000e+00, -0.000000e+00, 3.840000e+02, -3.840000e+02, 4.480000e+02, -4.480000e+02, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF]> : tensor<12xf8E4M3FN>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0.000000e+00, -0.000000e+00, 3.840000e+02, -3.840000e+02, 4.480000e+02, -4.480000e+02, 4.480000e+02, -4.480000e+02, 4.480000e+02, -4.480000e+02, 0x7F, 0xFF]> : tensor<12xf8E4M3FN>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]] : tensor<12xf8E4M3FN>, tensor<12xf8E4M3FN>
// CHECK:         }
}

// -----

func.func @test_cast_f32_f8E4M3FNUZ() -> (tensor<12xf8E4M3FNUZ>, tensor<12xf8E4M3FNUZ>) {
  %0 = onnx.Constant dense<[0.0, -0.0, 200.0, -200.0, 240.0, -240.0, 400.0, -400.0, 0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001]> : tensor<12xf32>
  %1 = "onnx.Cast"(%0) {to = f8E4M3FNUZ, saturate = 0 : si64} : (tensor<12xf32>) -> tensor<12xf8E4M3FNUZ>
  %2 = "onnx.Cast"(%0) {to = f8E4M3FNUZ, saturate = 1 : si64} : (tensor<12xf32>) -> tensor<12xf8E4M3FNUZ>
  onnx.Return %1, %2 : tensor<12xf8E4M3FNUZ>, tensor<12xf8E4M3FNUZ>

// f8E4M3FNUZ literal 0x80 means NaN
// CHECK-LABEL:  func.func @test_cast_f32_f8E4M3FNUZ
// CHECK-SAME:   () -> (tensor<12xf8E4M3FNUZ>, tensor<12xf8E4M3FNUZ>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.920000e+02, -1.920000e+02, 2.400000e+02, -2.400000e+02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]> : tensor<12xf8E4M3FNUZ>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.920000e+02, -1.920000e+02, 2.400000e+02, -2.400000e+02, 2.400000e+02, -2.400000e+02, 2.400000e+02, -2.400000e+02, 0x80, 0x80]> : tensor<12xf8E4M3FNUZ>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]] : tensor<12xf8E4M3FNUZ>, tensor<12xf8E4M3FNUZ>
// CHECK:         }
}

// -----

func.func @test_cast_f32_f8E5M2() -> (tensor<12xf8E5M2>, tensor<12xf8E5M2>) {
  %0 = onnx.Constant dense<[0.0, -0.0, 40000.0, -40000.0, 57344.0, -57344.0, 70000.0, -70000.0, 0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001]> : tensor<12xf32>
  %1 = "onnx.Cast"(%0) {to = f8E5M2, saturate = 0 : si64} : (tensor<12xf32>) -> tensor<12xf8E5M2>
  %2 = "onnx.Cast"(%0) {to = f8E5M2, saturate = 1 : si64} : (tensor<12xf32>) -> tensor<12xf8E5M2>
  onnx.Return %1, %2 : tensor<12xf8E5M2>, tensor<12xf8E5M2>

// f8E5M2 literals 0x7C, 0xFC mean +/-INF and 0x7E, 0xFE mean NaN
// CHECK-LABEL:  func.func @test_cast_f32_f8E5M2
// CHECK-SAME:   () -> (tensor<12xf8E5M2>, tensor<12xf8E5M2>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0.000000e+00, -0.000000e+00, 4.096000e+04, -4.096000e+04, 5.734400e+04, -5.734400e+04, 0x7C, 0xFC, 0x7C, 0xFC, 0x7E, 0xFE]> : tensor<12xf8E5M2>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0.000000e+00, -0.000000e+00, 4.096000e+04, -4.096000e+04, 5.734400e+04, -5.734400e+04, 5.734400e+04, -5.734400e+04, 5.734400e+04, -5.734400e+04, 0x7E, 0xFE]> : tensor<12xf8E5M2>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]] : tensor<12xf8E5M2>, tensor<12xf8E5M2>
// CHECK:         }
}

// -----

func.func @test_cast_f32_f8E5M2FNUZ() -> (tensor<12xf8E5M2FNUZ>, tensor<12xf8E5M2FNUZ>) {
  %0 = onnx.Constant dense<[0.0, -0.0, 40000.0, -40000.0, 57344.0, -57344.0, 70000.0, -70000.0, 0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001]> : tensor<12xf32>
  %1 = "onnx.Cast"(%0) {to = f8E5M2FNUZ, saturate = 0 : si64} : (tensor<12xf32>) -> tensor<12xf8E5M2FNUZ>
  %2 = "onnx.Cast"(%0) {to = f8E5M2FNUZ, saturate = 1 : si64} : (tensor<12xf32>) -> tensor<12xf8E5M2FNUZ>
  onnx.Return %1, %2 : tensor<12xf8E5M2FNUZ>, tensor<12xf8E5M2FNUZ>

// f8E5M2FNUZ literal 0x80 means NaN
// CHECK-LABEL:  func.func @test_cast_f32_f8E5M2FNUZ
// CHECK-SAME:   () -> (tensor<12xf8E5M2FNUZ>, tensor<12xf8E5M2FNUZ>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 4.096000e+04, -4.096000e+04, 5.734400e+04, -5.734400e+04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]> : tensor<12xf8E5M2FNUZ>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 4.096000e+04, -4.096000e+04, 5.734400e+04, -5.734400e+04, 5.734400e+04, -5.734400e+04, 5.734400e+04, -5.734400e+04, 0x80, 0x80]> : tensor<12xf8E5M2FNUZ>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]] : tensor<12xf8E5M2FNUZ>, tensor<12xf8E5M2FNUZ>
// CHECK:         }
}

// -----

func.func @test_slice() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]> : tensor<3x2xf32>
  %starts = onnx.Constant dense<[0, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<3x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_slice
  // CHECK-SAME:   () -> tensor<1x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[2.000000e+00, 3.000000e+00]{{.}}> : tensor<1x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<1x2xf32>
  // CHECK:         }
}

// -----

func.func @test_slice_reversed() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]> : tensor<3x2xf32>
  %starts = onnx.Constant dense<[-1, 2]> : tensor<2xi64>
  %ends = onnx.Constant dense<[0, 0]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[-1, -1]> : tensor<2xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<3x2xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_slice_reversed
  // CHECK-SAME:   () -> tensor<2x1xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[7.000000e+00], [5.000000e+00]{{.}}> : tensor<2x1xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<2x1xf32>
  // CHECK:         }
}

// -----

func.func @test_slice_empty() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[2.0, 3.0, 4.0, 5.0]> : tensor<4xf32>
  %starts = onnx.Constant dense<0> : tensor<1xi64>
  %ends = onnx.Constant dense<0> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_slice_empty
  // CHECK-SAME:   () -> tensor<0xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<> : tensor<0xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<0xf32>
  // CHECK:         }
}

// -----

func.func @test_pad() -> tensor<*xf32> {
  %data = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %pads = onnx.Constant dense<[0, 2, 0, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<3x2xf32>, tensor<4xi64>, none, none) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad
// CHECK-SAME:   () -> tensor<3x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.200000e+00], [0.000000e+00, 0.000000e+00, 2.300000e+00, 3.400000e+00], [0.000000e+00, 0.000000e+00, 4.500000e+00, 5.700000e+00]{{.}}> : tensor<3x4xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3x4xf32>
// CHECK:         }
}

func.func @test_pad_rank0() -> tensor<*xf32> {
  %data = onnx.Constant dense<3.14> : tensor<f32>
  %pads = onnx.Constant dense<[]> : tensor<0xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<f32>, tensor<0xi64>, none, none) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_pad_rank0
// CHECK-SAME:   () -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.140000e+00> : tensor<f32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<f32>
// CHECK:         }
}

func.func @test_pad_empty() -> tensor<*xf16> {
  %data = onnx.Constant dense<[[], []]> : tensor<2x0xf16>
  %pads = onnx.Constant dense<[0, 1, 1, 0]> : tensor<4xi64>
  %val = onnx.Constant dense<3.14> : tensor<f16>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %val, %non) { mode = "constant" } : (tensor<2x0xf16>, tensor<4xi64>, tensor<f16>, none) -> tensor<*xf16>
  onnx.Return %1 : tensor<*xf16>

// CHECK-LABEL:  func.func @test_pad_empty
// CHECK-SAME:   () -> tensor<3x1xf16> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.140630e+00> : tensor<3x1xf16>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3x1xf16>
// CHECK:         }
}

// pad const prop doesn't support edge mode
func.func @test_pad_edge() -> tensor<*xf16> {
  %data = onnx.Constant dense<3.14> : tensor<3x2xf16>
  %pads = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "edge" } : (tensor<3x2xf16>, tensor<4xi64>, none, none) -> tensor<*xf16>
  onnx.Return %1 : tensor<*xf16>

// CHECK-LABEL:  func.func @test_pad_edge
// CHECK-SAME:   () -> tensor<3x2xf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<3.140630e+00> : tensor<3x2xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_3_:%.+]] = "onnx.Pad"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_2_]]) {mode = "edge"} : (tensor<3x2xf16>, tensor<4xi64>, none, none) -> tensor<3x2xf16>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<3x2xf16>
// CHECK:         }
}

// -----

func.func @test_concat() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]> : tensor<3x2xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_concat
  // CHECK-SAME:   () -> tensor<6x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01]{{.}}> : tensor<6x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<6x2xf32>
  // CHECK:         }
}

// -----

func.func @test_concat_integer() -> tensor<*xi32> {
  %0 = onnx.Constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %1 = onnx.Constant dense<[[11, 12], [13, 14], [15, 16]]> : tensor<3x2xi32>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<*xi32>
  "onnx.Return"(%2) : (tensor<*xi32>) -> ()

  // CHECK-LABEL:  func @test_concat_integer
  // CHECK-SAME:   () -> tensor<6x2xi32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2], [3, 4], [5, 6], [11, 12], [13, 14], [15, 16]{{.}}> : tensor<6x2xi32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<6x2xi32>
  // CHECK:         }
}

// -----

func.func @test_concat_3_operands() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]> : tensor<3x2xf32>
  %2 = onnx.Constant dense<[[21.0, 22.0], [23.0, 24.0], [25.0, 26.0]]> : tensor<3x2xf32>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%3) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_concat_3_operands
  // CHECK-SAME:   () -> tensor<9x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01], [2.100000e+01, 2.200000e+01], [2.300000e+01, 2.400000e+01], [2.500000e+01, 2.600000e+01]{{.}}> : tensor<9x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<9x2xf32>
  // CHECK:         }
}

// -----

func.func @test_concat_negative_axis() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]> : tensor<3x2xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = -1 : si64} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_concat_negative_axis
  // CHECK-SAME:   () -> tensor<3x4xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 2.000000e+00, 1.100000e+01, 1.200000e+01], [3.000000e+00, 4.000000e+00, 1.300000e+01, 1.400000e+01], [5.000000e+00, 6.000000e+00, 1.500000e+01, 1.600000e+01]{{.}}> : tensor<3x4xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x4xf32>
  // CHECK:         }
}

// -----

func.func @test_expand() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[2, 3, 2]> : tensor<3xi64>
  %2 = "onnx.Expand"(%0, %1) : (tensor<3x2xf32>, tensor<3xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_expand
  // CHECK-SAME:   () -> tensor<2x3x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{.}}, {{.}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{.}}{{.}}> : tensor<2x3x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<2x3x2xf32>
  // CHECK:         }
}

// -----

func.func @test_expand_broadcast() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[1.0], [3.0], [5.0]]> : tensor<3x1xf32>
  %1 = onnx.Constant dense<[2, 3, 2]> : tensor<3xi64>
  %2 = "onnx.Expand"(%0, %1) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_expand_broadcast
  // CHECK-SAME:   () -> tensor<2x3x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00], [5.000000e+00, 5.000000e+00]{{.}}, {{.}}[1.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00], [5.000000e+00, 5.000000e+00]{{.}}{{.}}> : tensor<2x3x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<2x3x2xf32>
  // CHECK:         }
}

// -----

// Expand's shape can be shorter than the data input shape.
func.func @test_expand_2_broadcast() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[1.0], [3.0], [5.0]]]> : tensor<1x3x1xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Expand"(%0, %1) : (tensor<1x3x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func.func @test_expand_2_broadcast
  // CHECK-SAME:   () -> tensor<1x3x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00], [5.000000e+00, 5.000000e+00]{{.}}{{.}}> : tensor<1x3x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<1x3x2xf32>
  // CHECK:         }
}

// -----

func.func @test_gather_axis_0() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
  %2 = "onnx.Gather"(%0, %1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_gather_axis_0
  // CHECK-SAME:   () -> tensor<2x2x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 1.200000e+00], [2.300000e+00, 3.400000e+00]{{.}}, {{.}}[2.300000e+00, 3.400000e+00], [4.500000e+00, 5.700000e+00]{{.}}{{.}}> : tensor<2x2x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2x2xf32>
  // CHECK:         }
}

// -----

func.func @test_gather_axis_1() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]]> : tensor<3x3xf32>
  %1 = onnx.Constant dense<[[0, 2]]> : tensor<1x2xi64>
  %2 = "onnx.Gather"(%0, %1) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_gather_axis_1
  // CHECK-SAME:   () -> tensor<3x1x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 1.900000e+00]{{.}}, {{.}}[2.300000e+00, 3.900000e+00]{{.}}, {{.}}[4.500000e+00, 5.900000e+00]{{.}}{{.}}> : tensor<3x1x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x1x2xf32>
  // CHECK:         }
}

// -----

func.func @test_gather_negative_index() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]]> : tensor<3x3xf32>
  %1 = onnx.Constant dense<[[0, -1]]> : tensor<1x2xi64>
  %2 = "onnx.Gather"(%0, %1) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_gather_negative_index
  // CHECK-SAME:   () -> tensor<3x1x2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 1.900000e+00]{{.}}, {{.}}[2.300000e+00, 3.900000e+00]{{.}}, {{.}}[4.500000e+00, 5.900000e+00]{{.}}{{.}}> : tensor<3x1x2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<3x1x2xf32>
  // CHECK:         }
}

// -----

func.func @test_gather_rank0_int32_indices() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<1> : tensor<i32>
  %2 = "onnx.Gather"(%0, %1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<i32>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_gather_rank0_int32_indices
  // CHECK-SAME:   () -> tensor<2xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2.300000e+00, 3.400000e+00]> : tensor<2xf32>
  // CHECK:           onnx.Return [[VAR_0_]] : tensor<2xf32>
  // CHECK:         }
}

// -----

func.func @test_reshape() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]]> : tensor<3x3xf32>
  %1 = onnx.Constant dense<[1, -1]> : tensor<2xi64>
  %2 = "onnx.Reshape"(%0, %1) : (tensor<3x3xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reshape
// CHECK-SAME:   () -> tensor<1x9xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 1.200000e+00, 1.900000e+00, 2.300000e+00, 3.400000e+00, 3.900000e+00, 4.500000e+00, 5.700000e+00, 5.900000e+00]{{.}}> : tensor<1x9xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<1x9xf32>
// CHECK:         }
}

// -----

func.func @test_constant_of_shape() -> tensor<3xi64> {
  %0 = onnx.Constant dense<3> : tensor<1xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<2> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<3xi64>
  "onnx.Return"(%1) : (tensor<3xi64>) -> ()

// CHECK-LABEL:  func.func @test_constant_of_shape
// CHECK-SAME:   () -> tensor<3xi64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<3xi64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xi64>
// CHECK:         }
}

// -----

func.func @test_constant_of_shape_empty_tensor() -> tensor<f32> {
  %0 = onnx.Constant dense<> : tensor<0xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<2.0> : tensor<1xf32>} : (tensor<0xi64>) -> tensor<f32>
  "onnx.Return"(%1) : (tensor<f32>) -> ()

// CHECK-LABEL:  func.func @test_constant_of_shape_empty_tensor
// CHECK-SAME:   () -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<f32>
// CHECK:         }
}

// -----

func.func @test_range_int() -> tensor<8xi16> {
  %start = onnx.Constant dense<2> : tensor<i16>
  %limit = onnx.Constant dense<10> : tensor<i16>
  %delta = onnx.Constant dense<1> : tensor<i16>
  %1 = "onnx.Range"(%start, %limit, %delta) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<8xi16>
  onnx.Return %1 : tensor<8xi16>

// CHECK-LABEL:  func.func @test_range_int
// CHECK-SAME:   () -> tensor<8xi16> {
// CHECK:           [[VAR:%.+]] = onnx.Constant dense<[2, 3, 4, 5, 6, 7, 8, 9]> : tensor<8xi16>
// CHECK:           onnx.Return [[VAR]] : tensor<8xi16>
// CHECK:         }
}

// -----

func.func @test_range_fp() -> tensor<3xf32> {
  %start = onnx.Constant dense<0.2> : tensor<f32>
  %limit = onnx.Constant dense<0.5> : tensor<f32>
  %delta = onnx.Constant dense<0.1> : tensor<f32>
  %1 = "onnx.Range"(%start, %limit, %delta) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  onnx.Return %1 : tensor<3xf32>

// CHECK-LABEL:  func.func @test_range_fp
// CHECK-SAME:   () -> tensor<3xf32> {
// CHECK:           [[VAR:%.+]] = onnx.Constant dense<[2.000000e-01, 3.000000e-01, 4.000000e-01]> : tensor<3xf32>
// CHECK:           onnx.Return [[VAR]] : tensor<3xf32>
// CHECK:         }
}

// -----

func.func @test_nonzero() -> tensor<2x?xi64> {
  %0 = "onnx.Constant"() {value = dense<[[2, 1], [0, 2], [0, 1]]> : tensor<3x2xi8>} : () -> tensor<3x2xi8>
  %1 = "onnx.NonZero"(%0) : (tensor<3x2xi8>) -> tensor<2x?xi64>
  onnx.Return %1 : tensor<2x?xi64>

// CHECK-LABEL:  func.func @test_nonzero
// CHECK-SAME:   () -> tensor<2x?xi64> {
// CHECK:           [[VAR:%.+]] = onnx.Constant dense<{{\[}}[0, 0, 1, 2], [0, 1, 1, 1]{{\]}}> : tensor<2x4xi64>
// CHECK:           onnx.Return [[VAR]] : tensor<2x4xi64>
// CHECK:         }
}

// -----

func.func @test_max_1_input() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %2 = "onnx.Max"(%0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  "onnx.Return"(%2) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_max_1_input
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2], [3, 4]{{.}}> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }
}

// -----

func.func @test_max_2_inputs() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "onnx.Max"(%0, %1) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  "onnx.Return"(%2) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_max_2_inputs
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2], [3, 4]{{.}}> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }
}

// -----

func.func @test_max_3_inputs() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "onnx.Constant"() {value = dense<[[5], [6]]> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %3 = "onnx.Max"(%0, %1, %2) : (tensor<2x2xi32>, tensor<i32>, tensor<2x1xi32>) -> tensor<2x2xi32>
  "onnx.Return"(%3) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_max_3_inputs
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[5, 5], [6, 6]{{.}}> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }

}

// -----

func.func @test_min_1_input() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %2 = "onnx.Min"(%0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  "onnx.Return"(%2) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_min_1_input
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2], [3, 4]{{.}}> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }
}

// -----

func.func @test_min_2_inputs() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %2 = "onnx.Min"(%0, %1) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  "onnx.Return"(%2) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_min_2_inputs
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2], [2, 2]{{.}}> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }
}

// -----

func.func @test_min_3_inputs() -> tensor<2x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "onnx.Constant"() {value = dense<[[1], [6]]> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %3 = "onnx.Min"(%0, %1, %2) : (tensor<2x2xi32>, tensor<i32>, tensor<2x1xi32>) -> tensor<2x2xi32>
  "onnx.Return"(%3) : (tensor<2x2xi32>) -> ()

// CHECK-LABEL:  func.func @test_min_3_inputs
// CHECK-SAME:   () -> tensor<2x2xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<2x2xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xi32>
// CHECK:         }
}

// -----

func.func @test_sum_1_input() -> tensor<2x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %2 = "onnx.Sum"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  "onnx.Return"(%2) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func.func @test_sum_1_input
// CHECK-SAME:   () -> tensor<2x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]{{.}}> : tensor<2x2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xf32>
// CHECK:         }
}

// -----

func.func @test_sum_2_inputs() -> tensor<2x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "onnx.Sum"(%0, %1) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  "onnx.Return"(%2) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func.func @test_sum_2_inputs
// CHECK-SAME:   () -> tensor<2x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{.}}> : tensor<2x2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xf32>
// CHECK:         }
}

// -----

func.func @test_sum_3_inputs() -> tensor<2x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "onnx.Constant"() {value = dense<[[1.0], [6.0]]> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %3 = "onnx.Sum"(%0, %1, %2) : (tensor<2x2xf32>, tensor<f32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  "onnx.Return"(%3) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func.func @test_sum_3_inputs
// CHECK-SAME:   () -> tensor<2x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01]{{.}}> : tensor<2x2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xf32>
// CHECK:         }
}

// -----

func.func @test_if_true(%arg0 : tensor<*xf16>, %arg1 : tensor<1xi64>, %arg2 : tensor<*xf16>) -> tensor<?xi64> {
    %487 = onnx.Constant dense<true> : tensor<1xi1>
    %488 = "onnx.If"(%487) ({
      %6277 = onnx.Constant dense<1> : tensor<1xi64>
      %6278 = "onnx.Squeeze"(%arg0, %arg1) : (tensor<*xf16>, tensor<1xi64>) -> tensor<?x?x?xf16>
      onnx.Yield %6278 : tensor<?x?x?xf16>
    }, {
      %6277 = "onnx.Identity"(%arg2) : (tensor<*xf16>) -> tensor<?x?x?x?xf16>
      onnx.Yield %6277 : tensor<?x?x?x?xf16>
    }) : (tensor<1xi1>) -> tensor<*xf16>
    %490 = "onnx.Shape"(%488) { start = 0 : si64} : (tensor<*xf16>) -> tensor<?xi64>
   onnx.Return %490 : tensor<?xi64>
}
// CHECK-LABEL:  func.func @test_if_true
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf16>, [[PARAM_1_:%.+]]: tensor<1xi64>, [[PARAM_2_:%.+]]: tensor<*xf16>) -> tensor<?xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<*xf16>, tensor<1xi64>) -> tensor<?x?x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x?x?xf16>) -> tensor<?xi64>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?xi64>
// CHECK:         }

// -----

func.func @test_if_false(%arg0 : tensor<*xf16>, %arg1 : tensor<1xi64>, %arg2 : tensor<*xf16>) -> tensor<?xi64> {
    %487 = onnx.Constant dense<false> : tensor<1xi1>
    %488 = "onnx.If"(%487) ({
      %6277 = onnx.Constant dense<1> : tensor<1xi64>
      %6278 = "onnx.Squeeze"(%arg0, %arg1) : (tensor<*xf16>, tensor<1xi64>) -> tensor<?x?x?xf16>
      onnx.Yield %6278 : tensor<?x?x?xf16>
    }, {
      %6277 = "onnx.Identity"(%arg2) : (tensor<*xf16>) -> tensor<?x?x?x?xf16>
      onnx.Yield %6277 : tensor<?x?x?x?xf16>
    }) : (tensor<1xi1>) -> tensor<*xf16>
    %490 = "onnx.Shape"(%488) { start = 0 : si64} : (tensor<*xf16>) -> tensor<?xi64>
   onnx.Return %490 : tensor<?xi64>
}
// CHECK-LABEL:  func.func @test_if_false
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf16>, [[PARAM_1_:%.+]]: tensor<1xi64>, [[PARAM_2_:%.+]]: tensor<*xf16>) -> tensor<?xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Identity"([[PARAM_2_]]) : (tensor<*xf16>) -> tensor<?x?x?x?xf16>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x?x?x?xf16>) -> tensor<?xi64>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<?xi64>
// CHECK:         }

// -----

func.func @test_pow() -> tensor<2x2xf32> {
  %0 = onnx.Constant dense<64.0> : tensor<2x2xf32>
  %1 = onnx.Constant dense<0.5> : tensor<f32>
  %2 = "onnx.Pow"(%0, %1) : (tensor<2x2xf32> , tensor<f32>) -> tensor<2x2xf32>
  onnx.Return %2 : tensor<2x2xf32>

// CHECK-LABEL:  func.func @test_pow
// CHECK-SAME:   () -> tensor<2x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<8.000000e+00> : tensor<2x2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x2xf32>
// CHECK:         }
}

//===----------------------------------------------------------------------===//
/// Reciprocal test

// -----

// CHECK-LABEL: @test_reciprocal() -> tensor<1x2xf32>
func.func @test_reciprocal() -> tensor<1x2xf32> {
  %0 = onnx.Constant dense<[[-4.0, 16.0]]> : tensor<1x2xf32>
  %1 = "onnx.Reciprocal"(%0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  "onnx.Return"(%1) : (tensor<1x2xf32>) -> ()
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[-2.500000e-01, 6.250000e-02]{{\]}}> : tensor<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Reciprocal"{{.*}}
}
