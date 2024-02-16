// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_reducemax_v13_negative_inf_f32(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_f32
  // CHECK: arith.constant 0xFF800000 : f32
}

// -----

func.func private @test_reducemax_v13_negative_inf_f64(%arg0 : tensor<2x3xf64>) -> tensor<*xf64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf64>)-> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_f64
  // CHECK: arith.constant 0xFFF0000000000000 : f64
}

// -----

func.func private @test_reducemax_v13_negative_inf_i8(%arg0 : tensor<2x3xi8>) -> tensor<*xi8> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi8>)-> tensor<*xi8>
  "func.return"(%0) : (tensor<*xi8>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i8
  // CHECK: arith.constant -128 : i8
}

// -----

func.func private @test_reducemax_v13_negative_inf_i32(%arg0 : tensor<2x3xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i32
  // CHECK: arith.constant -2147483648 : i32
}

// -----

func.func private @test_reducemax_v13_negative_inf_i64(%arg0 : tensor<2x3xi64>) -> tensor<*xi64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi64>)-> tensor<*xi64>
  "func.return"(%0) : (tensor<*xi64>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i64
  // CHECK: arith.constant -9223372036854775808 : i64
}

// -----

func.func private @test_reducemin_v13_positive_inf_f32(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_f32
  // CHECK: arith.constant 0x7F800000 : f32
}

// -----

func.func private @test_reducemin_v13_positive_inf_f64(%arg0 : tensor<2x3xf64>) -> tensor<*xf64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf64>)-> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_f64
  // CHECK: arith.constant 0x7FF0000000000000 : f64
}

// -----

func.func private @test_reducemin_v13_positive_inf_i8(%arg0 : tensor<2x3xi8>) -> tensor<*xi8> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi8>)-> tensor<*xi8>
  "func.return"(%0) : (tensor<*xi8>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i8
  // CHECK: arith.constant 127 : i8
}

// -----

func.func private @test_reducemin_v13_positive_inf_i32(%arg0 : tensor<2x3xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i32
  // CHECK: arith.constant 2147483647 : i32
}

// -----

func.func private @test_reducemin_v13_positive_inf_i64(%arg0 : tensor<2x3xi64>) -> tensor<*xi64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi64>)-> tensor<*xi64>
  "func.return"(%0) : (tensor<*xi64>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i64
  // CHECK: arith.constant 9223372036854775807 : i64
}
