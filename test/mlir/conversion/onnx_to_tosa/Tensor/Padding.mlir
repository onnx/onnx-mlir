// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_pad(%arg0: tensor<20x16x44x32xf32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<24x22x52x42xf32> 
    return %2 :   tensor<24x22x52x42xf32> 
// CHECK-LABEL: test_pad
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() {value = dense<4.500000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[VAR2:.*]] = "tosa.pad"(%arg0, %[[VAR0]], %[[VAR1]])
}

// -----
func.func @test_no_pad(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x44x32xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_no_pad
// CHECK: return %arg0
}

// -----
func.func @test_novalue_pad(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x45x33xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, none, none) -> tensor<20x16x45x33xf32> 
    return %2 :   tensor<20x16x45x33xf32> 
// CHECK-LABEL: test_novalue_pad
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[{{\[}}0, 0], [0, 0], [1, 0], [1, 0]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: "tosa.pad"(%arg0, %[[VAR0]], %[[VAR1]])
}

// -----
func.func @test_novalue_no_pad(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x44x32xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, none, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_novalue_no_pad
// CHECK: return %arg0
}

// -----
func.func @test_no_const_pad(%arg0: tensor<20x16x44x32xf32>, %arg1: tensor<8xi64>, %arg2: tensor<1xf32>) ->  tensor<20x16x44x32xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %arg1, %arg2, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_no_const_pad
// CHECK: "onnx.Pad"
}
