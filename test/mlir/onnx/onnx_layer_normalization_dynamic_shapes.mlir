// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

func @test_f32(%arg : tensor<*xf32>) -> (tensor<*xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<*xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    return %41 : tensor<*xf32>

    // CHECK-LABEL: test_f32
    // CHECK: %[[SCALE:.*]] = "onnx.Constant"() {value = dense<5.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[BIAS:.*]]  = "onnx.Constant"() {value = dense<6.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[MEAN:.*]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} {{.*}} tensor<*xf32>
    // CHECK: %[[DIFF:.*]] = "onnx.Sub"(%arg0, %[[MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SQUARED:.*]] = "onnx.Mul"(%[[DIFF]], %[[DIFF]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[VARIANCE:.*]] = "onnx.ReduceMean"(%[[SQUARED]]) {axes = [-1], keepdims = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[EPSILON:.*]] = "onnx.Constant"() {value = dense<5.000000e-06> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[DENOM1:.*]] = "onnx.Add"(%[[VARIANCE]], %[[EPSILON]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[DENOM2:.*]] = "onnx.Sqrt"(%[[DENOM1]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED:.*]] = "onnx.Div"(%[[DIFF]], %[[DENOM2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED1:.*]] = "onnx.Mul"(%[[SCALED]], %[[SCALE]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[NORM:.*]] = "onnx.Add"(%[[SCALED1]], %[[BIAS]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[ONE:.*]] = "onnx.Constant"() {value = 1 : i64} : () -> tensor<*xf32>
    // CHECK: %[[INV_VAR:.*]] = "onnx.Div"(%[[ONE]], %[[VARIANCE]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: return %[[NORM]] : tensor<*xf32>
}

// -----

func @test_bf16(%arg : tensor<*xbf16>) -> (tensor<*xbf16>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xbf16>} : () -> tensor<768xbf16>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xbf16>} : () -> tensor<768xbf16>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<*xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> (tensor<*xbf16>, tensor<*xbf16>, tensor<*xbf16>)
    return %41: tensor<*xbf16>

    // CHECK-LABEL: test_bf16
    // CHECK: %[[SCALE:.*]] = "onnx.Constant"() {value = dense<4.997250e-04> : tensor<768xbf16>} : () -> tensor<768xbf16>
    // CHECK: %[[BIAS:.*]]  = "onnx.Constant"() {value = dense<5.989070e-04> : tensor<768xbf16>} : () -> tensor<768xbf16>
    // CHECK: %[[MEAN:.*]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} {{.*}} tensor<*xbf16>
    // CHECK: %[[DIFF:.*]] = "onnx.Sub"(%arg0, %[[MEAN]]) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: %[[SQUARED:.*]] = "onnx.Mul"(%[[DIFF]], %[[DIFF]]) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: %[[VARIANCE:.*]] = "onnx.ReduceMean"(%[[SQUARED]]) {axes = [-1], keepdims = 1 : si64} : (tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: %[[EPSILON:.*]] = "onnx.Constant"() {value = dense<5.006790e-06> : tensor<1xbf16>} : () -> tensor<1xbf16>
    // CHECK: %[[DENOM1:.*]] = "onnx.Add"(%[[VARIANCE]], %[[EPSILON]]) : (tensor<*xbf16>, tensor<1xbf16>) -> tensor<*xbf16>
    // CHECK: %[[DENOM2:.*]] = "onnx.Sqrt"(%[[DENOM1]]) : (tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: %[[SCALED:.*]] = "onnx.Div"(%[[DIFF]], %[[DENOM2]]) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: %[[SCALED1:.*]] = "onnx.Mul"(%[[SCALED]], %[[SCALE]]) : (tensor<*xbf16>, tensor<768xbf16>) -> tensor<*xbf16>
    // CHECK: %[[NORM:.*]] = "onnx.Add"(%[[SCALED1]], %[[BIAS]]) : (tensor<*xbf16>, tensor<768xbf16>) -> tensor<*xbf16>
    // CHECK: %[[ONE:.*]] = "onnx.Constant"() {value = 1 : i64} : () -> tensor<*xbf16>
    // CHECK: %[[INV_VAR:.*]] = "onnx.Div"(%[[ONE]], %[[VARIANCE]]) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
    // CHECK: return %[[NORM]] : tensor<*xbf16>
}

// -----

func @test_return_optional_inv_var(%arg : tensor<*xf32>) -> (tensor<*xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %41, %saved_mean_1, %x = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<*xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<*xf32>, tensor<*xf32>, none)
    return %41 : tensor<*xf32>

    // CHECK-LABEL: test_return_optional_inv_var
    // CHECK: %[[SCALE:.*]] = "onnx.Constant"() {value = dense<5.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[BIAS:.*]]  = "onnx.Constant"() {value = dense<6.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[MEAN:.*]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} {{.*}} tensor<*xf32>
    // CHECK: %[[DIFF:.*]] = "onnx.Sub"(%arg0, %[[MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SQUARED:.*]] = "onnx.Mul"(%[[DIFF]], %[[DIFF]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[VARIANCE:.*]] = "onnx.ReduceMean"(%[[SQUARED]]) {axes = [-1], keepdims = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[EPSILON:.*]] = "onnx.Constant"() {value = dense<5.000000e-06> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[DENOM1:.*]] = "onnx.Add"(%[[VARIANCE]], %[[EPSILON]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[DENOM2:.*]] = "onnx.Sqrt"(%[[DENOM1]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED:.*]] = "onnx.Div"(%[[DIFF]], %[[DENOM2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED1:.*]] = "onnx.Mul"(%[[SCALED]], %[[SCALE]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[NORM:.*]] = "onnx.Add"(%[[SCALED1]], %[[BIAS]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>

}

// -----

    func @test_return_optional_inv_var_and_mean(%arg : tensor<*xf32>) -> (tensor<*xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %41, %noop, %noop1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<*xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<*xf32>, none, none)
    return %41 : tensor<*xf32>

    // CHECK-LABEL: test_return_optional_inv_var_and_mean
    // CHECK: %[[SCALE:.*]] = "onnx.Constant"() {value = dense<5.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[BIAS:.*]]  = "onnx.Constant"() {value = dense<6.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[MEAN:.*]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} {{.*}} tensor<*xf32>
    // CHECK: %[[DIFF:.*]] = "onnx.Sub"(%arg0, %[[MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SQUARED:.*]] = "onnx.Mul"(%[[DIFF]], %[[DIFF]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[VARIANCE:.*]] = "onnx.ReduceMean"(%[[SQUARED]]) {axes = [-1], keepdims = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[EPSILON:.*]] = "onnx.Constant"() {value = dense<5.000000e-06> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[DENOM1:.*]] = "onnx.Add"(%[[VARIANCE]], %[[EPSILON]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[DENOM2:.*]] = "onnx.Sqrt"(%[[DENOM1]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED:.*]] = "onnx.Div"(%[[DIFF]], %[[DENOM2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED1:.*]] = "onnx.Mul"(%[[SCALED]], %[[SCALE]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[NORM:.*]] = "onnx.Add"(%[[SCALED1]], %[[BIAS]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>

}

// -----

func @test_optional_bias_parameter(%arg : tensor<*xf32>, %bias : none) -> (tensor<*xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<1xf32>} : () -> tensor<1xf32>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %bias) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<*xf32>, tensor<1xf32>, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    return %41 : tensor<*xf32>

    // CHECK-LABEL: test_optional_bias_parameter
    // CHECK: %[[SCALE:.*]] = "onnx.Constant"() {value = dense<5.000000e-04> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[MEAN:.*]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} {{.*}} tensor<*xf32>
    // CHECK: %[[DIFF:.*]] = "onnx.Sub"(%arg0, %[[MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SQUARED:.*]] = "onnx.Mul"(%[[DIFF]], %[[DIFF]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[VARIANCE:.*]] = "onnx.ReduceMean"(%[[SQUARED]]) {axes = [-1], keepdims = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[EPSILON:.*]] = "onnx.Constant"() {value = dense<5.000000e-06> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: %[[DENOM1:.*]] = "onnx.Add"(%[[VARIANCE]], %[[EPSILON]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[DENOM2:.*]] = "onnx.Sqrt"(%[[DENOM1]]) : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED:.*]] = "onnx.Div"(%[[DIFF]], %[[DENOM2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: %[[SCALED1:.*]] = "onnx.Mul"(%[[SCALED]], %[[SCALE]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: %[[ONE:.*]] = "onnx.Constant"() {value = 1 : i64} : () -> tensor<*xf32>
    // CHECK: %[[INV_VAR:.*]] = "onnx.Div"(%[[ONE]], %[[VARIANCE]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: return %[[SCALED1]] : tensor<*xf32>
}

