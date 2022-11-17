// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.clamp"([[PARAM_0_]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----
func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] =  "tosa.clamp"([[PARAM_0_]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:    }
}

// -----
func.func @test_neg(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.negate"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----
func.func @test_floor(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.floor"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}


// -----
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----
func.func @test_add_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_add_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
}


// -----
func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----
func.func @test_sub_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_sub_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
}