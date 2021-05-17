// RUN: onnx-mlir-opt --convert-onnx-to-krnl %s -npu -split-input-file | FileCheck %s


func @test_f32(%arg : tensor<1x32x768xf32>) -> (tensor<1x32x768xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x32x768xf32>, tensor<1x32x1xf32>, tensor<1x32x1xf32>)
    return %41 : tensor<1x32x768xf32>

    // CHECK: func private @tvp_LayerNormalization[[UNIQUE_ID:.*]](memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, memref<1x32x1xf32>) attributes {axis = -1 : si64, epsilon = 5.000000e-06 : f32, tvp.layerNormalization = true}
    // CHECK-LABEL: test_f32
    // CHECK: %[[SCALE:.*]] = "krnl.global"() {name = "constant_0", shape = [768], value = dense<5.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[BIAS:.*]] = "krnl.global"() {name = "constant_1", shape = [768], value = dense<6.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[RESULT:.*]]:3 = call @tvp_LayerNormalization[[UNIQUE_ID]](%[[INPUT:.*]], %[[SCALE]], %[[BIAS]]) : (memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, memref<1x32x1xf32>)
    // CHECK: return %[[RESULT]]#0 : memref<1x32x768xf32>
}

// -----

func @test_bf16(%arg : tensor<1x32x768xbf16>) -> (tensor<1x32x768xbf16>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xbf16>} : () -> tensor<768xbf16>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xbf16>} : () -> tensor<768xbf16>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> (tensor<1x32x768xbf16>, tensor<1x32x1xbf16>, tensor<1x32x1xbf16>)
    return %41: tensor<1x32x768xbf16>

    // CHECK: func private @tvp_LayerNormalization[[UNIQUE_ID:.*]](memref<1x32x768xbf16>, memref<768xbf16>, memref<768xbf16>) -> (memref<1x32x768xbf16>, memref<1x32x1xbf16>, memref<1x32x1xbf16>) attributes {axis = -1 : si64, epsilon = 5.006790e-06 : bf16, tvp.layerNormalization = true}
    // CHECK-LABEL: test_bf16
    // CHECK: %[[SCALE:.*]] = "krnl.global"() {name = "constant_0", shape = [768], value = dense<4.997250e-04> : tensor<768xbf16>} : () -> memref<768xbf16>
    // CHECK: %[[BIAS:.*]] = "krnl.global"() {name = "constant_1", shape = [768], value = dense<5.989070e-04> : tensor<768xbf16>} : () -> memref<768xbf16>
    // CHECK: %[[RESULT:.*]]:3 = call @tvp_LayerNormalization[[UNIQUE_ID]](%[[INPUT:.*]], %[[SCALE]], %[[BIAS]]) : (memref<1x32x768xbf16>, memref<768xbf16>, memref<768xbf16>) -> (memref<1x32x768xbf16>, memref<1x32x1xbf16>, memref<1x32x1xbf16>)
    // CHECK: return %[[RESULT]]#0 : memref<1x32x768xbf16>
}

// -----

// expected-error{{Only normalization on the last axis supported}}
func @test_reduction_axis_must_be_last(%arg : tensor<1x32x768xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = 1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x32x768xf32>, tensor<1x32x1xf32>, tensor<1x32x1xf32>)
    return
}

// -----

func @test_return_optional_inv_var(%arg : tensor<1x32x768xf32>) -> (tensor<1x32x768xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %41, %saved_mean_1, %x = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x32x768xf32>, tensor<1x32x1xf32>, none)
    return %41 : tensor<1x32x768xf32>

    // CHECK: func private @tvp_LayerNormalization[[UNIQUE_ID:.*]](memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, none) attributes {axis = -1 : si64, epsilon = 5.000000e-06 : f32, tvp.layerNormalization = true}
    // CHECK-LABEL: test_return_optional_inv_var
    // CHECK: %[[SCALE:.*]] = "krnl.global"() {name = "constant_0", shape = [768], value = dense<5.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[BIAS:.*]] = "krnl.global"() {name = "constant_1", shape = [768], value = dense<6.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[RESULT:.*]]:3 = call @tvp_LayerNormalization[[UNIQUE_ID]](%[[INPUT:.*]], %[[SCALE]], %[[BIAS]]) : (memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, none)
    // CHECK: return %[[RESULT]]#0 : memref<1x32x768xf32>
}

// -----

    func @test_return_optional_inv_var_and_mean(%arg : tensor<1x32x768xf32>) -> (tensor<1x32x768xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %40 = "onnx.Constant"() {value = dense<6.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %41, %noop, %noop1 = "onnx.LayerNormalization"(%arg, %39, %40) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x32x768xf32>, none, none)
    return %41 : tensor<1x32x768xf32>

    // CHECK: func private @tvp_LayerNormalization[[UNIQUE_ID:.*]](memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, none, none) attributes {axis = -1 : si64, epsilon = 5.000000e-06 : f32, tvp.layerNormalization = true}
    // CHECK-LABEL: test_return_optional_inv_var_and_mean
    // CHECK: %[[SCALE:.*]] = "krnl.global"() {name = "constant_0", shape = [768], value = dense<5.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[BIAS:.*]] = "krnl.global"() {name = "constant_1", shape = [768], value = dense<6.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[RESULT:.*]]:3 = call @tvp_LayerNormalization[[UNIQUE_ID]](%[[INPUT:.*]], %[[SCALE]], %[[BIAS]]) : (memref<1x32x768xf32>, memref<768xf32>, memref<768xf32>) -> (memref<1x32x768xf32>, none, none)
    // CHECK: return %[[RESULT]]#0 : memref<1x32x768xf32>

}

// -----

func @test_optional_bias_parameter(%arg : tensor<1x32x768xf32>, %bias : none) -> (tensor<1x32x768xf32>) {
    %39 = "onnx.Constant"() {value = dense<5.000000E-004> : tensor<768xf32>} : () -> tensor<768xf32>
    %41, %saved_mean_1, %saved_inv_std_var_1 = "onnx.LayerNormalization"(%arg, %39, %bias) {axis = -1 : si64, epsilon = 5.0E-06 : f32} : (tensor<1x32x768xf32>, tensor<768xf32>, none) -> (tensor<1x32x768xf32>, tensor<1x32x1xf32>, tensor<1x32x1xf32>)
    return %41 : tensor<1x32x768xf32>

    // CHECK: func private @tvp_LayerNormalization[[UNIQUE_ID:.*]](memref<1x32x768xf32>, memref<768xf32>, none) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, memref<1x32x1xf32>) attributes {axis = -1 : si64, epsilon = 5.000000e-06 : f32, tvp.layerNormalization = true}
    // CHECK-LABEL: test_optional_bias_parameter
    // CHECK: %[[SCALE:.*]] = "krnl.global"() {name = "constant_0", shape = [768], value = dense<5.000000e-04> : tensor<768xf32>} : () -> memref<768xf32>
    // CHECK: %[[RESULT:.*]]:3 = call @tvp_LayerNormalization[[UNIQUE_ID]](%[[INPUT:.*]], %[[SCALE]], %[[INPUT2:.*]]) : (memref<1x32x768xf32>, memref<768xf32>, none) -> (memref<1x32x768xf32>, memref<1x32x1xf32>, memref<1x32x1xf32>)
    // CHECK: return %[[RESULT]]#0 : memref<1x32x768xf32>
}

