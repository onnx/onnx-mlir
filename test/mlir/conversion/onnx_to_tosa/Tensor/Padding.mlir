// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa --canonicalize --cse %s -split-input-file | FileCheck %s

func.func @test_pad_f32(%arg0: tensor<20x16x44x32xf32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<24x22x52x42xf32> 
    return %2 :   tensor<24x22x52x42xf32> 
// CHECK-LABEL:  func.func @test_pad_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xf32>) -> tensor<24x22x52x42xf32> {
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<4.500000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           [[VAR_5_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_3_]], [[VAR_4_]] : (tensor<20x16x44x32xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<24x22x52x42xf32>
// CHECK:           return [[VAR_5_]] : tensor<24x22x52x42xf32>
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
// CHECK-LABEL:  func.func @test_novalue_pad_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xf32>) -> tensor<20x16x45x33xf32> {
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape {value = dense<[0, 0, 0, 0, 1, 0, 1, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           [[VAR_4_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_2_]], [[VAR_3_]] : (tensor<20x16x44x32xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<20x16x45x33xf32>
// CHECK:           return [[VAR_4_]] : tensor<20x16x45x33xf32>
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
// CHECK-LABEL:  func.func @test_pad_i64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xi64>) -> tensor<24x22x52x42xi64> {
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           [[VAR_6_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_3_]], [[VAR_5_]] : (tensor<20x16x44x32xi64>, !tosa.shape<8>, tensor<i64>) -> tensor<24x22x52x42xi64>
// CHECK:           return [[VAR_6_]] : tensor<24x22x52x42xi64>

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
// CHECK-LABEL:  func.func @test_novalue_pad_i64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xi64>) -> tensor<20x16x45x33xi64> {
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape {value = dense<[0, 0, 0, 0, 1, 0, 1, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
// CHECK:           [[VAR_4_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_2_]], [[VAR_3_]] : (tensor<20x16x44x32xi64>, !tosa.shape<8>, tensor<i64>) -> tensor<20x16x45x33xi64>
// CHECK:           return [[VAR_4_]] : tensor<20x16x45x33xi64>
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
// CHECK-LABEL:  func.func @test_pad_ui32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xui32>) -> tensor<24x22x52x42xui32> {
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<4> : tensor<ui32>}> : () -> tensor<ui32>
// CHECK:           [[VAR_6_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_3_]], [[VAR_5_]] : (tensor<20x16x44x32xui32>, !tosa.shape<8>, tensor<ui32>) -> tensor<24x22x52x42xui32>
// CHECK:           return [[VAR_6_]] : tensor<24x22x52x42xui32>
}

// -----
func.func @test_pad_bf16(%arg0: tensor<20x16x44x32xbf16>) ->  tensor<24x22x52x42xbf16>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.500000e+00]> : tensor<1xbf16>} : () -> tensor<1xbf16> 
    %2 = "onnx.Pad"(%arg0, %0, %1, %noval) {mode = "constant"} : (tensor<20x16x44x32xbf16>, tensor<8xi64>, tensor<1xbf16>, none) -> tensor<24x22x52x42xbf16> 
    return %2 :   tensor<24x22x52x42xbf16> 
// CHECK-LABEL:  func.func @test_pad_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xbf16>) -> tensor<24x22x52x42xbf16> {
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<4.500000e+00> : tensor<bf16>}> : () -> tensor<bf16>
// CHECK:           [[VAR_6_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_3_]], [[VAR_5_]] : (tensor<20x16x44x32xbf16>, !tosa.shape<8>, tensor<bf16>) -> tensor<24x22x52x42xbf16>
// CHECK:           return [[VAR_6_]] : tensor<24x22x52x42xbf16>
}

// -----
func.func @test_pad_f16_constant_none(%arg0: tensor<256x1x1x5x1xf16>) -> tensor<256x1x1x5x2xf16> {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<10xi64>} : () -> tensor<10xi64>
    %1 = "onnx.Pad"(%arg0, %0, %noval, %noval) {mode = "constant"} : (tensor<256x1x1x5x1xf16>, tensor<10xi64>, none, none) -> tensor<256x1x1x5x2xf16>
    return %1 :   tensor<256x1x1x5x2xf16>
// CHECK-LABEL: test_pad_f16_constant_none
// CHECK: %[[VAR0:.*]] = tosa.const_shape {value = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<10xindex>} : () -> !tosa.shape<10>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[VAR1]] : (tensor<256x1x1x5x1xf16>, !tosa.shape<10>, tensor<f16>) -> tensor<256x1x1x5x2xf16>
// CHECK: return %[[VAR2]] : tensor<256x1x1x5x2xf16>
}

// -----

func.func @test_pad_f32_non_constant_padval(%arg0: tensor<20x16x44x32xf32>, %arg1: tensor<f32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %arg1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<24x22x52x42xf32> 
    return %2 :   tensor<24x22x52x42xf32> 
// CHECK-LABEL:  func.func @test_pad_f32_non_constant_padval
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xf32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<24x22x52x42xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK:           [[VAR_1_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]] : (tensor<20x16x44x32xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<24x22x52x42xf32>
// CHECK:           return [[VAR_1_]] : tensor<24x22x52x42xf32>
}

// -----

func.func @test_pad_f32_non_constant_1Dpadval(%arg0: tensor<20x16x44x32xf32>, %arg1: tensor<1xf32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %arg1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<24x22x52x42xf32> 
    return %2 :   tensor<24x22x52x42xf32> 
// CHECK-LABEL:  func.func @test_pad_f32_non_constant_1Dpadval
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<24x22x52x42xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAL_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64>} : (tensor<1xf32>) -> tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAL_1_]] : (tensor<20x16x44x32xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<24x22x52x42xf32>
// CHECK:           return [[VAR_2_]] : tensor<24x22x52x42xf32>
}

// -----

func.func @test_pad_i64_non_constant_padval(%arg0: tensor<20x16x44x32xi64>, %arg1: tensor<i64>) ->  tensor<24x22x52x42xi64>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %arg1, %noval) {mode = "constant"} : (tensor<20x16x44x32xi64>, tensor<8xi64>, tensor<i64>, none) -> tensor<24x22x52x42xi64> 
    return %2 :   tensor<24x22x52x42xi64> 
// CHECK-LABEL:  func.func @test_pad_i64_non_constant_padval
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xi64>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<24x22x52x42xi64> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK:           [[VAR_1_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]] : (tensor<20x16x44x32xi64>, !tosa.shape<8>, tensor<i64>) -> tensor<24x22x52x42xi64>
// CHECK:           return [[VAR_1_]] : tensor<24x22x52x42xi64>
}

// -----
func.func @test_pad_f16_non_constant_padval(%arg0: tensor<20x16x44x32xf16>, %arg1: tensor<f16>) ->  tensor<24x22x52x42xf16>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %2 = "onnx.Pad"(%arg0, %0, %arg1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf16>, tensor<8xi64>, tensor<f16>, none) -> tensor<24x22x52x42xf16> 
    return %2 :   tensor<24x22x52x42xf16> 
// CHECK-LABEL:  func.func @test_pad_f16_non_constant_padval
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x16x44x32xf16>, [[PARAM_1_:%.+]]: tensor<f16>) -> tensor<24x22x52x42xf16> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape {value = dense<[0, 4, 1, 5, 2, 6, 3, 7]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK:           [[VAR_1_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]] : (tensor<20x16x44x32xf16>, !tosa.shape<8>, tensor<f16>) -> tensor<24x22x52x42xf16>
// CHECK:           return [[VAR_1_]] : tensor<24x22x52x42xf16>
}
