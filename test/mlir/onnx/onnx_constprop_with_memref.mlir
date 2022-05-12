// RUN: onnx-mlir-opt --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// ADD tests 

/// Test ConstantOp assoc for add
// -----
// CHECK-LABEL: @test_add_constant_1(%arg0: memref<3xf32>) -> memref<3xf32>
func @test_add_constant_1(%arg0 : memref<3xf32>) -> memref<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> memref<3xf32>
  %1 = "onnx.Add"(%0, %arg0) : (memref<3xf32> , memref<3xf32>) -> memref<3xf32>
  "func.return"(%1) : (memref<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> memref<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (memref<3xf32>, memref<3xf32>) -> memref<3xf32>
}

/// Test ConstantOp assoc for add
// -----
// CHECK-LABEL: @test_add_constant_2(%arg0: memref<3xf32>) -> memref<3xf32>
func @test_add_constant_2(%arg0 : memref<3xf32>) -> memref<3xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> memref<3xf32>
  %1 = "onnx.Add"(%arg0, %0) : (memref<3xf32> , memref<3xf32>) -> memref<3xf32>
  "func.return"(%1) : (memref<3xf32>) -> ()
  // CHECK-NEXT: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> memref<3xf32>
  // CHECK-NEXT: [[ADD:%.+]] =  "onnx.Add"(%arg0, [[CONST]]) : (memref<3xf32>, memref<3xf32>) -> memref<3xf32>
}

/// Change (x+c1)+c2 to x+(c1+c2)
// -----
// CHECK-LABEL: @test_add_constant_3(%arg0: memref<3xi32>) -> memref<3xi32> 
func @test_add_constant_3(%arg0 : memref<3xi32>) -> memref<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> memref<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %3 = "onnx.Add"(%1, %2) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  "func.return"(%3) : (memref<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
}

/// Same test as above, but with a use of an intermediary result
/// change (x+c1)+c2 + (x+c1) to x+x + (c1+c2+c3)
// -----
// CHECK-LABEL: @test_add_constant_4(%arg0: memref<3xi32>) -> memref<3xi32> 
func @test_add_constant_4(%arg0 : memref<3xi32>) -> memref<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> memref<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %3 = "onnx.Add"(%1, %2) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %4 = "onnx.Add"(%2, %3) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  "func.return"(%4) : (memref<3xi32>) -> ()
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg0) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 13, 16]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], [[CONST1]]) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
}

/// Change (x+c0)+y  + (z+c1) to  (x+y)+z + (c1+c2)
// -----
// CHECK-LABEL: @test_add_constant_5(%arg0: memref<3xi32>, %arg1: memref<3xi32>, %arg2: memref<3xi32>) -> memref<3xi32> 
func @test_add_constant_5(%arg0 : memref<3xi32>, %arg1: memref<3xi32>, %arg2: memref<3xi32>) -> memref<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> memref<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "onnx.Add"(%0, %arg0) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %3 = "onnx.Add"(%2, %arg1) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %4 = "onnx.Add"(%1, %arg2) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %5 = "onnx.Add"(%3, %4) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  "func.return"(%5) : (memref<3xi32>) -> ()
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, %arg1) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
  // CHECK-NEXT: [[ADD2:%.+]] = "onnx.Add"([[ADD1]], %arg2) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[10, 12, 14]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-NEXT: [[ADD3:%.+]] = "onnx.Add"([[ADD2]], [[CONST1]]) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
}

/// Test broadcast 1 -> 2d
// -----
// CHECK-LABEL: @test_broadcast_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_broadcast_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi32>} : () -> memref<1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (memref<1xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  "func.return"(%3) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}

/// Test broadcast 2d (size one) -> 2d
// -----
// CHECK-LABEL: @test_broadcast_2(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_broadcast_2(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1]]> : tensor<1x1xi32>} : () -> memref<1x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (memref<1x1xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  "func.return"(%3) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[3, 4], [5, 6], [7, 8]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}

/// check 1d -> 2d
// -----
// CHECK-LABEL: @test_broadcast_3(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_broadcast_3(%arg0 : memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1], [2], [3]]> : tensor<3x1xi32>} : () -> memref<3x1xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 11], [21, 22], [31, 32]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %2 = "onnx.Add"(%0, %1) : (memref<3x1xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  %3 = "onnx.Add"(%2, %arg0) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  "func.return"(%3) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[11, 12], [23, 24], [34, 35]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}


//===----------------------------------------------------------------------===//  
/// MUL tests (same as add, so have only two).
  
/// Change (x*c1)*c2 to x*(c1*c2)
// -----
// CHECK-LABEL: @test_mul_constant_3(%arg0: memref<3xi32>) -> memref<3xi32> 
func @test_mul_constant_3(%arg0 : memref<3xi32>) -> memref<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> memref<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %3 = "onnx.Mul"(%1, %2) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  "func.return"(%3) : (memref<3xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0, 11, 24]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, [[CONST1]]) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
}

/// Change (x*c0)*y  * (z*c1) to  (x*y)*z * (c1*c2)
// -----
// CHECK-LABEL: @test_mul_constant_5(%arg0: memref<3xi32>, %arg1: memref<3xi32>, %arg2: memref<3xi32>) -> memref<3xi32> 
func @test_mul_constant_5(%arg0 : memref<3xi32>, %arg1: memref<3xi32>, %arg2: memref<3xi32>) -> memref<3xi32> {
  %0 = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> memref<3xi32>
  %1 = "onnx.Constant"() {value = dense<[10, 11, 12]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "onnx.Mul"(%0, %arg0) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %3 = "onnx.Mul"(%2, %arg1) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %4 = "onnx.Mul"(%1, %arg2) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  %5 = "onnx.Mul"(%3, %4) : (memref<3xi32> , memref<3xi32>) -> memref<3xi32>
  "func.return"(%5) : (memref<3xi32>) -> ()
  // CHECK-NEXT: [[MUL1:%.+]] = "onnx.Mul"(%arg0, %arg1) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
  // CHECK-NEXT: [[MUL2:%.+]] = "onnx.Mul"([[MUL1]], %arg2) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<[0, 11, 24]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-NEXT: [[MUL3:%.+]] = "onnx.Mul"([[MUL2]], [[CONST1]]) : (memref<3xi32>, memref<3xi32>) -> memref<3xi32>
}

//===----------------------------------------------------------------------===//
/// SUB and NEG tests.

// check of sub two constants
// -----  
// CHECK-LABEL: @test_sub_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_sub_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[2]]> : tensor<1x1xi32>} : () -> memref<1x1xi32>
  %2 = "onnx.Sub"(%0, %1) : (memref<3x2xi32>, memref<1x1xi32>) -> memref<3x2xi32>
  "func.return"(%2) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[0, 1], [2, 3], [4, 5]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
}

/// check sub to add of negative
// -----
// CHECK-LABEL: @test_neg_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_neg_1(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %1 = "onnx.Sub"(%arg0, %0) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  "func.return"(%1) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[-2, -3], [-4, -5], [-6, -7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}

// -----
// CHECK-LABEL: @test_neg_2(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_neg_2(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10]]> : tensor<1x1xi32>} : () -> memref<1x1xi32>
  %2 = "onnx.Sub"(%arg0, %0) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  %5 = "onnx.Add"(%2, %1) : (memref<3x2xi32> , memref<1x1xi32>) -> memref<3x2xi32>
  "func.return"(%5) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}

// -----
// CHECK-LABEL: @test_neg_3(%arg0: memref<3x2xi32>) -> memref<3x2xi32>
func @test_neg_3(%arg0: memref<3x2xi32>) -> memref<3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[2, 3], [4, 5], [6, 7]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10]]> : tensor<1x1xi32>} : () -> memref<1x1xi32>
  %2 = "onnx.Neg"(%0) : (memref<3x2xi32>) -> memref<3x2xi32>
  %3 = "onnx.Add"(%arg0, %2) : (memref<3x2xi32> , memref<3x2xi32>) -> memref<3x2xi32>
  %4 = "onnx.Add"(%3, %1) : (memref<3x2xi32> , memref<1x1xi32>) -> memref<3x2xi32>
  "func.return"(%4) : (memref<3x2xi32>) -> ()
  // CHECK-NEXT: [[CONST1:%.+]] = "onnx.Constant"() {value = dense<{{.}}[8, 7], [6, 5], [4, 3]]> : tensor<3x2xi32>} : () -> memref<3x2xi32>
  // CHECK-NEXT: [[ADD1:%.+]] = "onnx.Add"(%arg0, [[CONST1]]) : (memref<3x2xi32>, memref<3x2xi32>) -> memref<3x2xi32>
}

//===----------------------------------------------------------------------===//
/// Transpose tests.

// -----  
// CHECK-LABEL: test_default_transpose_const_1
  func @test_default_transpose_const_1() -> memref<4x3x2xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> memref<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [2, 1, 0]} : (memref<2x3x4xi32>) -> memref<4x3x2xi32>
  "func.return"(%1) : (memref<4x3x2xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 211], [121, 221], [131, 231]{{.}}, [{{.}}112, 212], [122, 222], [132, 232]{{.}}, [{{.}}113, 213], [123, 223], [133, 233]{{.}}, [{{.}}114, 214], [124, 224], [134, 234]{{.}}]> : tensor<4x3x2xi32>} : () -> memref<4x3x2xi32>
  // CHECK: return [[RES]] : memref<4x3x2xi32>
}

// -----  
// CHECK-LABEL: test_default_transpose_const_2
func @test_default_transpose_const_2() -> memref<2x4x3xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> memref<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 2, 1]} : (memref<2x3x4xi32>) -> memref<2x4x3xi32>
  "func.return"(%1) : (memref<2x4x3xi32>) -> ()
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[{{.}}[111, 121, 131], [112, 122, 132], [113, 123, 133], [114, 124, 134]{{.}}, [{{.}}211, 221, 231], [212, 222, 232], [213, 223, 233], [214, 224, 234]{{.}}]> : tensor<2x4x3xi32>} : () -> memref<2x4x3xi32>
  // CHECK: return [[RES]] : memref<2x4x3xi32>
}

// -----  
// CHECK-LABEL: test_default_transpose_const_3
func @test_default_transpose_const_3() -> memref<3x2x4xi32> {
  %0 = "onnx.Constant"() {value = dense<[[[111, 112, 113, 114], [121, 122, 123, 124], [131, 132, 133, 134]], [[211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234]]]> : tensor<2x3x4xi32>} : () -> memref<2x3x4xi32>
  %1 = "onnx.Transpose"(%0) {perm = [1, 0, 2]} : (memref<2x3x4xi32>) -> memref<3x2x4xi32>
  "func.return"(%1) : (memref<3x2x4xi32>) -> ()
  // CHECK: [[RES:%.+]] =  "onnx.Constant"() {value = dense<[{{.}}[111, 112, 113, 114], [211, 212, 213, 214]{{.}}, [{{.}}121, 122, 123, 124], [221, 222, 223, 224]{{.}}, [{{.}}131, 132, 133, 134], [231, 232, 233, 234]{{.}}]> : tensor<3x2x4xi32>} : () -> memref<3x2x4xi32>
  // CHECK: return [[RES]] : memref<3x2x4xi32>
}

//===----------------------------------------------------------------------===//
/// Div tests

// -----

// CHECK-LABEL: @test_div(%arg0: memref<3x2xf32>) -> memref<3x2xf32>
func @test_div(%arg0: memref<3x2xf32>) -> memref<3x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  %1 = "onnx.Constant"() {value = dense<[[2.0]]> : tensor<1x1xf32>} : () -> memref<1x1xf32>
  %2 = "onnx.Div"(%0, %1) : (memref<3x2xf32>, memref<1x1xf32>) -> memref<3x2xf32>
  "func.return"(%2) : (memref<3x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]{{\]}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Div"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Sqrt tests

// -----

// CHECK-LABEL: @test_sqrt() -> memref<1x2xf32>
func @test_sqrt() -> memref<1x2xf32> {
  %0 = "onnx.Constant"() {value = dense<[[4.0, 16.0]]> : tensor<1x2xf32>} : () -> memref<1x2xf32>
  %1 = "onnx.Sqrt"(%0) : (memref<1x2xf32>) -> memref<1x2xf32>
  "func.return"(%1) : (memref<1x2xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[2.000000e+00, 4.000000e+00]{{\]}}> : tensor<1x2xf32>} : () -> memref<1x2xf32>
  // CHECK-NOT: {{.*}} = "onnx.Sqrt"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Unsqueeze tests

// -----

// CHECK-LABEL: @test_unsqueeze() -> memref<2x1x1xf32>
func @test_unsqueeze() -> memref<2x1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[4.0, 16.0]> : tensor<2xf32>} : () -> memref<2xf32>
  %1 = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
  %2 = "onnx.Unsqueeze"(%0, %1) : (memref<2xf32>, memref<2xi64>) -> memref<2x1x1xf32>
  "func.return"(%2) : (memref<2x1x1xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>} : () -> memref<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
}

// -----

// CHECK-LABEL: @test_unsqueezev11() -> memref<2x1x1xf32>
func @test_unsqueezev11() -> memref<2x1x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[4.0, 16.0]> : tensor<2xf32>} : () -> memref<2xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes = [1, 2]} : (memref<2xf32>) -> memref<2x1x1xf32>
  "func.return"(%1) : (memref<2x1x1xf32>) -> ()
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}{{\[}}[4.000000e+00]{{\]}}, {{\[}}[1.600000e+01]{{\]}}{{\]}}> : tensor<2x1x1xf32>} : () -> memref<2x1x1xf32>
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
}

//===----------------------------------------------------------------------===//
/// Split tests

// -----

// CHECK-LABEL: @test_split_axis_0() -> (memref<1x10xf32>, memref<1x10xf32>) {
func @test_split_axis_0() -> (memref<1x10xf32>, memref<1x10xf32>) {
  %split = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64>} : () -> memref<2xi64>
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> memref<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) { axis = 0 : si64} : (memref<2x10xf32>, memref<2xi64>) -> (memref<1x10xf32>, memref<1x10xf32>)
  "func.return"(%1, %2) : (memref<1x10xf32>, memref<1x10xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>} : () -> memref<1x10xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>} : () -> memref<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// CHECK-LABEL: @test_split_axis_1() -> (memref<2x5xf32>, memref<2x5xf32>) {
func @test_split_axis_1() -> (memref<2x5xf32>, memref<2x5xf32>) {
  %split = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64>} : () -> memref<2xi64>
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> memref<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %split) { axis = 1 : si64} : (memref<2x10xf32>, memref<2xi64>) -> (memref<2x5xf32>, memref<2x5xf32>)
  "func.return"(%1, %2) : (memref<2x5xf32>, memref<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>} : () -> memref<2x5xf32>
  // CHECK: {{.*}}  = "onnx.Constant"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>} : () -> memref<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// CHECK-LABEL: @test_splitv11_axis_0() -> (memref<1x10xf32>, memref<1x10xf32>) {
func @test_splitv11_axis_0() -> (memref<1x10xf32>, memref<1x10xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> memref<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 0 : si64, split = [1, 1]} : (memref<2x10xf32>) -> (memref<1x10xf32>, memref<1x10xf32>)
  "func.return"(%1, %2) : (memref<1x10xf32>, memref<1x10xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>} : () -> memref<1x10xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>} : () -> memref<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}

// -----

// CHECK-LABEL: @test_splitv11_axis_1() -> (memref<2x5xf32>, memref<2x5xf32>) {
func @test_splitv11_axis_1() -> (memref<2x5xf32>, memref<2x5xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> memref<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 1 : si64, split = [5, 5]} : (memref<2x10xf32>) -> (memref<2x5xf32>, memref<2x5xf32>)
  "func.return"(%1, %2) : (memref<2x5xf32>, memref<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>} : () -> memref<2x5xf32>
  // CHECK: {{.*}}  = "onnx.Constant"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>} : () -> memref<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}
