// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_pad_f32(%arg0: tensor<20x16x44x32xf32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<24x22x52x42xf32> 
    return %2 :   tensor<24x22x52x42xf32> 
// CHECK-LABEL: test_pad_f32
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<4.500000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_no_pad_f32(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x44x32xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_no_pad_f32
// CHECK: return %arg0
}

// -----
func.func @test_novalue_pad_f32(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x45x33xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, none, none) -> tensor<20x16x45x33xf32> 
    return %2 :   tensor<20x16x45x33xf32> 
// CHECK-LABEL: test_novalue_pad_f32
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 0], [0, 0], [1, 0], [1, 0]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_novalue_no_pad_f32(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x44x32xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, none, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_novalue_no_pad_f32
// CHECK: return %arg0
}

// -----
func.func @test_no_const_pad_f32(%arg0: tensor<20x16x44x32xf32>, %arg1: tensor<8xi64>, %arg2: tensor<1xf32>) ->  tensor<20x16x44x32xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %arg1, %arg2, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<20x16x44x32xf32> 
    return %2 :   tensor<20x16x44x32xf32> 
// CHECK-LABEL: test_no_const_pad_f32
// CHECK: "onnx.Pad"
}

// -----
func.func @test_pad_i64(%arg0: tensor<20x16x44x32xi64>) ->  tensor<24x22x52x42xi64>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4]> : tensor<1xi64>} : () -> tensor<1xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, tensor<1xi64>, none) -> tensor<24x22x52x42xi64> 
    return %2 :   tensor<24x22x52x42xi64> 
// CHECK-LABEL: test_pad_i64
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_no_pad_i64(%arg0: tensor<20x16x44x32xi64>) ->  tensor<20x16x44x32xi64>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4]> : tensor<1xi64>} : () -> tensor<1xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, tensor<1xi64>, none) -> tensor<20x16x44x32xi64> 
    return %2 :   tensor<20x16x44x32xi64> 
// CHECK-LABEL: test_no_pad_i64
// CHECK: return %arg0
}

// -----
func.func @test_novalue_pad_i64(%arg0: tensor<20x16x44x32xi64>) ->  tensor<20x16x45x33xi64>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, none, none) -> tensor<20x16x45x33xi64> 
    return %2 :   tensor<20x16x45x33xi64> 
// CHECK-LABEL: test_novalue_pad_i64
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 0], [0, 0], [1, 0], [1, 0]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK: tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_novalue_no_pad_i64(%arg0: tensor<20x16x44x32xi64>) ->  tensor<20x16x44x32xi64>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1, %1) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, none, none) -> tensor<20x16x44x32xi64> 
    return %2 :   tensor<20x16x44x32xi64> 
// CHECK-LABEL: test_novalue_no_pad_i64
// CHECK: return %arg0
}

// -----
func.func @test_no_const_pad_i64(%arg0: tensor<20x16x44x32xi64>, %arg1: tensor<8xi64>, %arg2: tensor<1xi64>) ->  tensor<20x16x44x32xi64>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %arg1, %arg2, %noval) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, tensor<1xi64>, none) -> tensor<20x16x44x32xi64> 
    return %2 :   tensor<20x16x44x32xi64> 
// CHECK-LABEL: test_no_const_pad_i64
// CHECK: "onnx.Pad"
}

// -----
func.func @test_pad_ui32(%arg0: tensor<20x16x44x32xui32>) ->  tensor<24x22x52x42xui32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4]> : tensor<1xui32>} : () -> tensor<1xui32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xui32>, tensor<8xi64>, tensor<1xui32>, none) -> tensor<24x22x52x42xui32> 
    return %2 :   tensor<24x22x52x42xui32> 
// CHECK-LABEL: test_pad_ui32
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<4> : tensor<ui32>}> : () -> tensor<ui32>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_pad_bf16(%arg0: tensor<20x16x44x32xbf16>) ->  tensor<24x22x52x42xbf16>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.500000e+00]> : tensor<1xbf16>} : () -> tensor<1xbf16> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xbf16>, tensor<8xi64>, tensor<1xbf16>, none) -> tensor<24x22x52x42xbf16> 
    return %2 :   tensor<24x22x52x42xbf16> 
// CHECK-LABEL: test_pad_bf16
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<4.500000e+00> : tensor<bf16>}> : () -> tensor<bf16>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]]
}

// -----
func.func @test_pad_f16_constant_none(%arg0: tensor<256x1x1x5x1xf16>) -> tensor<256x1x1x5x2xf16> {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<10xi64>} : () -> tensor<10xi64>
    %1 = "onnx.Pad"(%arg0, %0, %noval, %noval) {mode = "constant"} : (tensor<256x1x1x5x1xf16>, tensor<10xi64>, none, none) -> tensor<256x1x1x5x2xf16>
    return %1 :   tensor<256x1x1x5x2xf16>
// CHECK-LABEL: test_pad_f16_constant_none
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{value = dense<[{{\[}}0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]> : tensor<5x2xi64>}> : () -> tensor<5x2xi64>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]] : (tensor<256x1x1x5x1xf16>, tensor<5x2xi64>, tensor<f16>) -> tensor<256x1x1x5x2xf16>
// CHECK: return %[[VAR2]] : tensor<256x1x1x5x2xf16>
}
