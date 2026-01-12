// RUN: onnx-mlir-opt --qdq-canonicalize="remove-qdq-around-ops=true" %s -split-input-file | FileCheck %s

// -----

// Test that Reshape optimization DOES happen when shape is constant
func.func @reshape_with_constant_shape(%arg0: tensor<1x4xui8>) -> tensor<2x2xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<2> : tensor<2xi64>
    %3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %4 = "onnx.Reshape"(%3, %2) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
    %5 = "onnx.QuantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<2x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x2xui8>
    return %5 : tensor<2x2xui8>
}

// CHECK-LABEL:  func.func @reshape_with_constant_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8>) -> tensor<2x2xui8> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x4xui8>, tensor<2xi64>) -> tensor<2x2xui8>
// CHECK:           return [[VAR_1_]] : tensor<2x2xui8>
// CHECK:         }

// -----

// Test that Reshape optimization DOES NOT happen when shape is NOT constant (runtime input)
func.func @reshape_with_dynamic_shape(%arg0: tensor<1x4xui8>, %arg1: tensor<2xi64>) -> tensor<?x?xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %3 = "onnx.Reshape"(%2, %arg1) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<?x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<?x?xui8>
    return %4 : tensor<?x?xui8>
}

// CHECK-LABEL:  func.func @reshape_with_dynamic_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8>, [[PARAM_1_:%.+]]: tensor<2xi64>) -> tensor<?x?xui8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK:           [[VAR_2_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[PARAM_1_]]) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.QuantizeLinear"([[VAR_3_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<?x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<?x?xui8>
// CHECK:           return [[VAR_4_]] : tensor<?x?xui8>
// CHECK:         }

// -----

// Test that Unsqueeze optimization DOES happen when axes is constant
func.func @unsqueeze_with_constant_axes(%arg0: tensor<1x3xi8>) -> tensor<1x1x3xi8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<-128> : tensor<1xi8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x3xf32>
    %4 = "onnx.Unsqueeze"(%3, %2) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
    %5 = "onnx.QuantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x1x3xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x1x3xi8>
    return %5 : tensor<1x1x3xi8>
}

// CHECK-LABEL:  func.func @unsqueeze_with_constant_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3xi8>) -> tensor<1x1x3xi8> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x3xi8>, tensor<1xi64>) -> tensor<1x1x3xi8>
// CHECK:           return [[VAR_1_]] : tensor<1x1x3xi8>
// CHECK:         }

// -----

// Test that Unsqueeze optimization DOES NOT happen when axes is NOT constant
func.func @unsqueeze_with_dynamic_axes(%arg0: tensor<1x3xi8>, %arg1: tensor<1xi64>) -> tensor<?x?x?xi8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<-128> : tensor<1xi8>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x3xf32>
    %3 = "onnx.Unsqueeze"(%2, %arg1) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<?x?x?xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<?x?x?xi8>
    return %4 : tensor<?x?x?xi8>
}

// CHECK-LABEL:  func.func @unsqueeze_with_dynamic_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3xi8>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?x?xi8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-128> : tensor<1xi8>
// CHECK:           [[VAR_2_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Unsqueeze"([[VAR_2_]], [[PARAM_1_]]) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.QuantizeLinear"([[VAR_3_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<?x?x?xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<?x?x?xi8>
// CHECK:           return [[VAR_4_]] : tensor<?x?x?xi8>
// CHECK:         }

// -----

// Test that Squeeze optimization DOES happen when axes is constant
func.func @squeeze_with_constant_axes(%arg0: tensor<1x1x3xi8>) -> tensor<1x3xi8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<-128> : tensor<1xi8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x1x3xf32>
    %4 = "onnx.Squeeze"(%3, %2) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
    %5 = "onnx.QuantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x3xi8>
    return %5 : tensor<1x3xi8>
}

// CHECK-LABEL:  func.func @squeeze_with_constant_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xi8>) -> tensor<1x3xi8> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x1x3xi8>, tensor<1xi64>) -> tensor<1x3xi8>
// CHECK:           return [[VAR_1_]] : tensor<1x3xi8>
// CHECK:         }

// -----

// Test that Squeeze optimization DOES NOT happen when axes is NOT constant
func.func @squeeze_with_dynamic_axes(%arg0: tensor<1x1x3xi8>, %arg1: tensor<1xi64>) -> tensor<?x?xi8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<-128> : tensor<1xi8>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x1x3xf32>
    %3 = "onnx.Squeeze"(%2, %arg1) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<?x?xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<?x?xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<?x?xi8>
    return %4 : tensor<?x?xi8>
}

// CHECK-LABEL:  func.func @squeeze_with_dynamic_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3xi8>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xi8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-128> : tensor<1xi8>
// CHECK:           [[VAR_2_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Squeeze"([[VAR_2_]], [[PARAM_1_]]) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.QuantizeLinear"([[VAR_3_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<?x?xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<?x?xi8>
// CHECK:           return [[VAR_4_]] : tensor<?x?xi8>
// CHECK:         }

// -----

// Test that Gather optimization DOES happen when indices is constant
func.func @gather_with_constant_indices(%arg0: tensor<4xui8>) -> tensor<2xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<0> : tensor<1xui8>
    %2 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
    %3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<4xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<4xf32>
    %4 = "onnx.Gather"(%3, %2) {axis = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<2xf32>
    %5 = "onnx.QuantizeLinear"(%4, %0, %1) {axis = 1 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<2xui8>
    return %5 : tensor<2xui8>
}

// CHECK-LABEL:  func.func @gather_with_constant_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xui8>) -> tensor<2xui8> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Gather"([[PARAM_0_]], [[VAR_0_]]) {axis = 0 : si64} : (tensor<4xui8>, tensor<2xi64>) -> tensor<2xui8>
// CHECK:           return [[VAR_1_]] : tensor<2xui8>
// CHECK:         }

// -----

// Test that Gather optimization DOES NOT happen when indices is NOT constant
func.func @gather_with_dynamic_indices(%arg0: tensor<4xui8>, %arg1: tensor<?xi64>) -> tensor<?xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<0> : tensor<1xui8>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<4xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<4xf32>
    %3 = "onnx.Gather"(%2, %arg1) {axis = 0 : si64} : (tensor<4xf32>, tensor<?xi64>) -> tensor<?xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, saturate = 1 : si64} : (tensor<?xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<?xui8>
    return %4 : tensor<?xui8>
}

// CHECK-LABEL:  func.func @gather_with_dynamic_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xui8>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<?xui8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0> : tensor<1xui8>
// CHECK:           [[VAR_2_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Gather"([[VAR_2_]], [[PARAM_1_]]) {axis = 0 : si64} : (tensor<4xf32>, tensor<?xi64>) -> tensor<?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.QuantizeLinear"([[VAR_3_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<?xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<?xui8>
// CHECK:           return [[VAR_4_]] : tensor<?xui8>
// CHECK:         }

// -----

// Test that Slice optimization DOES happen when all control parameters are constant
func.func @slice_with_constant_params(%arg0: tensor<1x4xui8>) -> tensor<1x2xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = onnx.Constant dense<3> : tensor<1xi64>
    %4 = onnx.Constant dense<1> : tensor<1xi64>
    %5 = onnx.Constant dense<1> : tensor<1xi64>
    %6 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %7 = "onnx.Slice"(%6, %2, %3, %4, %5) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xf32>
    %8 = "onnx.QuantizeLinear"(%7, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
    return %8 : tensor<1x2xui8>
}

// CHECK-LABEL:  func.func @slice_with_constant_params
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8>) -> tensor<1x2xui8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_0_]]) : (tensor<1x4xui8>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xui8>
// CHECK:           return [[VAR_2_]] : tensor<1x2xui8>
// CHECK:         }

// -----

// Test that Slice optimization DOES NOT happen when starts is NOT constant
func.func @slice_with_dynamic_starts(%arg0: tensor<1x4xui8>, %arg1: tensor<1xi64>) -> tensor<1x?xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<3> : tensor<1xi64>
    %3 = onnx.Constant dense<1> : tensor<1xi64>
    %4 = onnx.Constant dense<1> : tensor<1xi64>
    %5 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %6 = "onnx.Slice"(%5, %arg1, %2, %3, %4) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?xf32>
    %7 = "onnx.QuantizeLinear"(%6, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x?xui8>
    return %7 : tensor<1x?xui8>
}

// CHECK-LABEL:  func.func @slice_with_dynamic_starts
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<1x?xui8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Slice"([[VAR_4_]], [[PARAM_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_3_]]) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.QuantizeLinear"([[VAR_5_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x?xui8>
// CHECK:           return [[VAR_6_]] : tensor<1x?xui8>
// CHECK:         }

// -----

// Test that Slice optimization DOES NOT happen when ends is NOT constant
func.func @slice_with_dynamic_ends(%arg0: tensor<1x4xui8>, %arg1: tensor<1xi64>) -> tensor<1x?xui8> {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = onnx.Constant dense<1> : tensor<1xi64>
    %4 = onnx.Constant dense<1> : tensor<1xi64>
    %5 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %6 = "onnx.Slice"(%5, %2, %arg1, %3, %4) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?xf32>
    %7 = "onnx.QuantizeLinear"(%6, %0, %1) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x?xui8>
    return %7 : tensor<1x?xui8>
}

// CHECK-LABEL:  func.func @slice_with_dynamic_ends
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<1x?xui8> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<128> : tensor<ui8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Slice"([[VAR_3_]], [[VAR_2_]], [[PARAM_1_]], [[VAR_2_]], [[VAR_2_]]) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.QuantizeLinear"([[VAR_4_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x?xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x?xui8>
// CHECK:           return [[VAR_5_]] : tensor<1x?xui8>
// CHECK:         }

// -----

// Test that Resize optimization DOES happen when sizes is constant, scales is NoValue, and mode is "nearest"
func.func @resize_with_constant_sizes_nearest(%arg0: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %2 = onnx.Constant dense<0> : tensor<1xui8>
    %3 = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
    %4 = "onnx.DequantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %5 = "onnx.Resize"(%4, %0, %0, %3) {mode = "nearest"} : (tensor<1x3x64x64xf32>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xf32>
    %6 = "onnx.QuantizeLinear"(%5, %1, %2) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
    return %6 : tensor<1x3x128x128xui8>
}


// CHECK-LABEL: func.func @resize_with_constant_sizes_nearest
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8>
// CHECK-DAG:     %[[NONE:.*]] = "onnx.NoValue"
// CHECK-DAG:     %[[SIZES:.*]] = onnx.Constant dense<[1, 3, 128, 128]> : tensor<4xi64>
// CHECK:         %[[RESIZE:.*]] = "onnx.Resize"(%[[ARG0]], %[[NONE]], %[[NONE]], %[[SIZES]])
// CHECK-SAME:      mode = "nearest"
// CHECK-SAME:      (tensor<1x3x64x64xui8>, none, none, tensor<4xi64>) -> tensor<1x3x128x128xui8>
// CHECK:         return %[[RESIZE]] : tensor<1x3x128x128xui8>

// -----

// Test that Resize optimization DOES happen when scales is constant, sizes is NoValue, and mode is "nearest"
func.func @resize_with_constant_scales_nearest(%arg0: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
    %2 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %3 = onnx.Constant dense<0> : tensor<1xui8>
    %4 = "onnx.DequantizeLinear"(%arg0, %2, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %5 = "onnx.Resize"(%4, %0, %1, %0) {mode = "nearest"} : (tensor<1x3x64x64xf32>, none, tensor<4xf32>, none) -> tensor<1x3x128x128xf32>
    %6 = "onnx.QuantizeLinear"(%5, %2, %3) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
    return %6 : tensor<1x3x128x128xui8>
}


// CHECK-LABEL: func.func @resize_with_constant_scales_nearest
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8>
// CHECK-DAG:     %[[NONE:.*]] = "onnx.NoValue"
// CHECK-DAG:     %[[RESIZE_SCALES:.*]] = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
// CHECK:         %[[RESIZE:.*]] = "onnx.Resize"(%[[ARG0]], %[[NONE]], %[[RESIZE_SCALES]], %[[NONE]])
// CHECK-SAME:      mode = "nearest"
// CHECK-SAME:      (tensor<1x3x64x64xui8>, none, tensor<4xf32>, none) -> tensor<1x3x128x128xui8>
// CHECK:         return %[[RESIZE]] : tensor<1x3x128x128xui8>

// -----

// Test that Resize optimization DOES NOT happen when sizes is NOT constant
func.func @resize_with_dynamic_sizes(%arg0: tensor<1x3x64x64xui8>, %arg1: tensor<4xi64>) -> tensor<?x?x?x?xui8> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %2 = onnx.Constant dense<0> : tensor<1xui8>
    %3 = "onnx.DequantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %4 = "onnx.Resize"(%3, %0, %0, %arg1) {mode = "nearest"} : (tensor<1x3x64x64xf32>, none, none, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    %5 = "onnx.QuantizeLinear"(%4, %1, %2) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<?x?x?x?xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<?x?x?x?xui8>
    return %5 : tensor<?x?x?x?xui8>
}

// CHECK-LABEL: func.func @resize_with_dynamic_sizes
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x3x64x64xui8>, %[[ARG1:.*]]: tensor<4xi64>) -> tensor<?x?x?x?xui8>
// CHECK-DAG:     %[[NONE:.*]] = "onnx.NoValue"
// CHECK-DAG:     %[[SCALE:.*]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:     %[[ZP:.*]] = onnx.Constant dense<0> : tensor<1xui8>
// CHECK:         %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[ARG0]], %[[SCALE]], %[[ZP]])
// CHECK:         %[[RESIZE:.*]] = "onnx.Resize"(%[[DEQUANT]], %[[NONE]], %[[NONE]], %[[ARG1]])
// CHECK:         mode = "nearest"
// CHECK:         %[[QUANT:.*]] = "onnx.QuantizeLinear"(%[[RESIZE]], %[[SCALE]], %[[ZP]])
// CHECK:         return %[[QUANT]]

// -----

// Test that Resize optimization DOES NOT happen when mode is NOT "nearest" (e.g., "linear")
func.func @resize_with_linear_mode(%arg0: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
    %2 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %3 = onnx.Constant dense<0> : tensor<1xui8>
    %4 = "onnx.DequantizeLinear"(%arg0, %2, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x64x64xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x64x64xf32>
    %5 = "onnx.Resize"(%4, %0, %1, %0) {mode = "linear"} : (tensor<1x3x64x64xf32>, none, tensor<4xf32>, none) -> tensor<1x3x128x128xf32>
    %6 = "onnx.QuantizeLinear"(%5, %2, %3) {axis = 1 : si64, block_size = 0 : si64, saturate = 1 : si64} : (tensor<1x3x128x128xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<1x3x128x128xui8>
    return %6 : tensor<1x3x128x128xui8>
}

// CHECK-LABEL: func.func @resize_with_linear_mode
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x3x64x64xui8>) -> tensor<1x3x128x128xui8>
// CHECK-DAG:     %[[NONE:.*]] = "onnx.NoValue"
// CHECK-DAG:     %[[RESIZE_SCALES:.*]] = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
// CHECK-DAG:     %[[QDQ_SCALE:.*]] = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
// CHECK-DAG:     %[[ZP:.*]] = onnx.Constant dense<0> : tensor<1xui8>
// CHECK:         %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[ARG0]], %[[QDQ_SCALE]], %[[ZP]])
// CHECK:         %[[RESIZE:.*]] = "onnx.Resize"(%[[DEQUANT]], %[[NONE]], %[[RESIZE_SCALES]], %[[NONE]])
// CHECK:         mode = "linear"
// CHECK:         %[[QUANT:.*]] = "onnx.QuantizeLinear"(%[[RESIZE]], %[[QDQ_SCALE]], %[[ZP]])
// CHECK:         return %[[QUANT]]
