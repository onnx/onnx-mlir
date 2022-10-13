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

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] =  "tosa.clamp"([[PARAM_0_]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:    }
}

func.func @test_neg(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.negate"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

func.func @test_floor(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.floor"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

func.func @test_leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.5 : f32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
}


