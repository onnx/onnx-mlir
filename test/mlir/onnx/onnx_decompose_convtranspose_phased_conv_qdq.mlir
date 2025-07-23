// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-phased %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-phased --constprop-onnx %s -split-input-file | FileCheck %s --check-prefix=CONSTPROP
func.func @test_convtrans_stride11(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<1xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %13 : tensor<1x1x13x57xf32>

// CHECK-LABEL:   func.func @test_convtrans_stride11(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.QuantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_22]] : tensor<1x1x13x57xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride11(
// CONSTPROP-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CONSTPROP-SAME:                                       %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.QuantizeLinear"(%[[VAL_13]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.DequantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_15]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

func.func @test_convtrans_stride11_with_relu(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<1xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %12 = "onnx.Relu"(%11) {} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
  %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %14 : tensor<1x1x13x57xf32>

// CHECK-LABEL:   func.func @test_convtrans_stride11_with_relu(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Relu"(%[[VAL_20]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_23]] : tensor<1x1x13x57xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride11_with_relu(
// CONSTPROP-SAME:                                                 %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CONSTPROP-SAME:                                                 %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.Relu"(%[[VAL_13]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_16]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

func.func @test_convtrans_stride11_with_leakyrelu(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<1xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %12 = "onnx.LeakyRelu"(%11) {alpha = 1.000000e-01 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
  %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %14 : tensor<1x1x13x57xf32>
// CHECK-LABEL:   func.func @test_convtrans_stride11_with_leakyrelu(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.LeakyRelu"(%[[VAL_20]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_23]] : tensor<1x1x13x57xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride11_with_leakyrelu(
// CONSTPROP-SAME:                                                      %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CONSTPROP-SAME:                                                      %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.LeakyRelu"(%[[VAL_13]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_16]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

func.func @test_convtrans_stride11_with_leakyrelu_with_default(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<1xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %12 = "onnx.LeakyRelu"(%11) {} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
  %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %14 : tensor<1x1x13x57xf32>
// CHECK-LABEL:   func.func @test_convtrans_stride11_with_leakyrelu_with_default(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                                   %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.LeakyRelu"(%[[VAL_20]]) {alpha = 0.00999999977 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_23]] : tensor<1x1x13x57xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride11_with_leakyrelu_with_default(
// CONSTPROP-SAME:                                                                   %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CONSTPROP-SAME:                                                                   %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.LeakyRelu"(%[[VAL_13]]) {alpha = 0.00999999977 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_16]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

func.func @test_convtrans_stride11_with_qdq_relu(%arg0: tensor<1x1x12x44xf32>, %arg1: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<1xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  %14 = "onnx.Relu"(%13) {} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>  
  %15 = "onnx.QuantizeLinear"(%14, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %16 = "onnx.DequantizeLinear"(%15, %0, %3) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %16 : tensor<1x1x13x57xf32>

// CHECK-LABEL:   func.func @test_convtrans_stride11_with_qdq_relu(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<16> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<4> : tensor<16xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.QuantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Relu"(%[[VAL_22]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.QuantizeLinear"(%[[VAL_23]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_24]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           onnx.Return %[[VAL_25]] : tensor<1x1x13x57xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride11_with_qdq_relu(
// CONSTPROP-SAME:                                                     %[[VAL_0:.*]]: tensor<1x1x12x44xf32>,
// CONSTPROP-SAME:                                                     %[[VAL_1:.*]]: tensor<1x1x4x16xf32>) -> tensor<1x1x13x57xf32> {
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x4x16xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.QuantizeLinear"(%[[VAL_13]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.DequantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.Relu"(%[[VAL_15]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.QuantizeLinear"(%[[VAL_16]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.DequantizeLinear"(%[[VAL_17]], %[[VAL_3]], %[[VAL_6]]) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_18]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

 func.func @test_convtrans_stide22(%arg0: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<256xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<512x256x6x6xi8>, tensor<f32>, tensor<i8>) -> tensor<512x256x6x6xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x5x21xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x10x42xf32>    
    %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
    %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
    onnx.Return %13 : tensor<1x256x10x42xf32>

  // CHECK-LABEL:   func.func @test_convtrans_stide22(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<7> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[6, 7]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_20]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK:           %[[VAL_24:.*]] = "onnx.DequantizeLinear"(%[[VAL_23]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_26:.*]] = "onnx.ReverseSequence"(%[[VAL_25]], %[[VAL_14]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_27:.*]] = "onnx.ReverseSequence"(%[[VAL_26]], %[[VAL_14]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_11]], %[[VAL_10]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_9]], %[[VAL_8]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_7]], %[[VAL_6]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_5]], %[[VAL_4]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_33]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_34]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_36]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.DequantizeLinear"(%[[VAL_31]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_38]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_40]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_37]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Concat"(%[[VAL_42]], %[[VAL_44]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Concat"(%[[VAL_45]], %[[VAL_43]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Reshape"(%[[VAL_46]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_47]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Concat"(%[[VAL_48]], %[[VAL_49]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.QuantizeLinear"(%[[VAL_51]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_53:.*]] = "onnx.DequantizeLinear"(%[[VAL_52]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return %[[VAL_53]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stide22(
// CONSTPROP-SAME:                                      %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_13]], %[[VAL_12]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_17]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_19]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_21]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_23]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.Reshape"(%[[VAL_18]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Reshape"(%[[VAL_20]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.Reshape"(%[[VAL_24]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Concat"(%[[VAL_25]], %[[VAL_27]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.Concat"(%[[VAL_28]], %[[VAL_26]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Reshape"(%[[VAL_29]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.Reshape"(%[[VAL_30]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Concat"(%[[VAL_31]], %[[VAL_32]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_5]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.QuantizeLinear"(%[[VAL_34]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.DequantizeLinear"(%[[VAL_35]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_36]] : tensor<1x256x10x42xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride22_with_relu(%arg0: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<256xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<512x256x6x6xi8>, tensor<f32>, tensor<i8>) -> tensor<512x256x6x6xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x5x21xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x10x42xf32>
    %12 = "onnx.Relu"(%11) {} : (tensor<1x256x10x42xf32>) -> tensor<1x256x10x42xf32>    
    %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
    %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
    onnx.Return %14 : tensor<1x256x10x42xf32>

  // CHECK-LABEL:   func.func @test_convtrans_stride22_with_relu(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<7> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[6, 7]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_20]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK:           %[[VAL_24:.*]] = "onnx.DequantizeLinear"(%[[VAL_23]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_26:.*]] = "onnx.ReverseSequence"(%[[VAL_25]], %[[VAL_14]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_27:.*]] = "onnx.ReverseSequence"(%[[VAL_26]], %[[VAL_14]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_11]], %[[VAL_10]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_9]], %[[VAL_8]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_7]], %[[VAL_6]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_5]], %[[VAL_4]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_33]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_34]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_37]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Relu"(%[[VAL_38]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_31]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_40]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Relu"(%[[VAL_41]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_43]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Relu"(%[[VAL_44]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Reshape"(%[[VAL_42]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_45]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Concat"(%[[VAL_46]], %[[VAL_48]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Concat"(%[[VAL_49]], %[[VAL_47]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.Reshape"(%[[VAL_51]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.Concat"(%[[VAL_52]], %[[VAL_53]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.Reshape"(%[[VAL_54]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.QuantizeLinear"(%[[VAL_55]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_57:.*]] = "onnx.DequantizeLinear"(%[[VAL_56]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return %[[VAL_57]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride22_with_relu(
// CONSTPROP-SAME:                                                 %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_13]], %[[VAL_12]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_17]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.Relu"(%[[VAL_18]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_20]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.Relu"(%[[VAL_21]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_23]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.Relu"(%[[VAL_24]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_26]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.Relu"(%[[VAL_27]]) : (tensor<1x256x5x21xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Reshape"(%[[VAL_19]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.Reshape"(%[[VAL_28]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Concat"(%[[VAL_29]], %[[VAL_31]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.Concat"(%[[VAL_32]], %[[VAL_30]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.Concat"(%[[VAL_35]], %[[VAL_36]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_37]], %[[VAL_5]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.QuantizeLinear"(%[[VAL_38]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_40]] : tensor<1x256x10x42xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride22_with_qdq_relu(%arg0: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<256xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<512x256x6x6xi8>, tensor<f32>, tensor<i8>) -> tensor<512x256x6x6xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 6], pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x512x5x21xf32>, tensor<512x256x6x6xf32>, tensor<256xf32>) -> tensor<1x256x10x42xf32>
    %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
    %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
    %14 = "onnx.Relu"(%13) {} : (tensor<1x256x10x42xf32>) -> tensor<1x256x10x42xf32>    
    %15 = "onnx.QuantizeLinear"(%14, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
    %16 = "onnx.DequantizeLinear"(%15, %0, %3) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
    onnx.Return %16 : tensor<1x256x10x42xf32>
// CHECK-LABEL:   func.func @test_convtrans_stride22_with_qdq_relu(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<7> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[6, 7]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_20]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK:           %[[VAL_24:.*]] = "onnx.DequantizeLinear"(%[[VAL_23]], %[[VAL_16]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_26:.*]] = "onnx.ReverseSequence"(%[[VAL_25]], %[[VAL_14]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_27:.*]] = "onnx.ReverseSequence"(%[[VAL_26]], %[[VAL_14]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_11]], %[[VAL_10]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_9]], %[[VAL_8]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_7]], %[[VAL_6]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_29]], %[[VAL_5]], %[[VAL_4]], %[[VAL_13]], %[[VAL_12]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_33]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_34]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.QuantizeLinear"(%[[VAL_35]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_36]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Relu"(%[[VAL_37]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_39]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.QuantizeLinear"(%[[VAL_40]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_42:.*]] = "onnx.DequantizeLinear"(%[[VAL_41]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Relu"(%[[VAL_42]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.DequantizeLinear"(%[[VAL_31]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_44]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.QuantizeLinear"(%[[VAL_45]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_47:.*]] = "onnx.DequantizeLinear"(%[[VAL_46]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Relu"(%[[VAL_47]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_17]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Conv"(%[[VAL_24]], %[[VAL_49]], %[[VAL_22]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.QuantizeLinear"(%[[VAL_50]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_52:.*]] = "onnx.DequantizeLinear"(%[[VAL_51]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.Relu"(%[[VAL_52]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.Reshape"(%[[VAL_43]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.Reshape"(%[[VAL_48]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_57:.*]] = "onnx.Reshape"(%[[VAL_53]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CHECK:           %[[VAL_58:.*]] = "onnx.Concat"(%[[VAL_54]], %[[VAL_56]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_59:.*]] = "onnx.Concat"(%[[VAL_57]], %[[VAL_55]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CHECK:           %[[VAL_60:.*]] = "onnx.Reshape"(%[[VAL_58]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_61:.*]] = "onnx.Reshape"(%[[VAL_59]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CHECK:           %[[VAL_62:.*]] = "onnx.Concat"(%[[VAL_60]], %[[VAL_61]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CHECK:           %[[VAL_63:.*]] = "onnx.Reshape"(%[[VAL_62]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           %[[VAL_64:.*]] = "onnx.QuantizeLinear"(%[[VAL_63]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           %[[VAL_65:.*]] = "onnx.DequantizeLinear"(%[[VAL_64]], %[[VAL_15]], %[[VAL_18]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return %[[VAL_65]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride22_with_qdq_relu(
// CONSTPROP-SAME:                                                     %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<256x512x3x3xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<[1, 256, 5, 1, 42]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 256, 5, 21, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_13]], %[[VAL_12]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_9]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_17]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.QuantizeLinear"(%[[VAL_18]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.DequantizeLinear"(%[[VAL_19]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.Relu"(%[[VAL_20]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_22]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.QuantizeLinear"(%[[VAL_23]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_24]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Relu"(%[[VAL_25]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_27]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.QuantizeLinear"(%[[VAL_28]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.DequantizeLinear"(%[[VAL_29]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Relu"(%[[VAL_30]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_10]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<256x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<256x512x3x3xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_16]], %[[VAL_32]], %[[VAL_14]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<256x512x3x3xf32>, tensor<256xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.QuantizeLinear"(%[[VAL_33]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.DequantizeLinear"(%[[VAL_34]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x256x10x42xf32>) -> tensor<1x256x5x21xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_21]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_7]]) {allowzero = 0 : si64} : (tensor<1x256x5x21xf32>, tensor<5xi64>) -> tensor<1x256x5x21x1xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Concat"(%[[VAL_37]], %[[VAL_39]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.Concat"(%[[VAL_40]], %[[VAL_38]]) {axis = -1 : si64} : (tensor<1x256x5x21x1xf32>, tensor<1x256x5x21x1xf32>) -> tensor<1x256x5x21x2xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_42]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x256x5x21x2xf32>, tensor<5xi64>) -> tensor<1x256x5x1x42xf32>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.Concat"(%[[VAL_43]], %[[VAL_44]]) {axis = -2 : si64} : (tensor<1x256x5x1x42xf32>, tensor<1x256x5x1x42xf32>) -> tensor<1x256x5x2x42xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.Reshape"(%[[VAL_45]], %[[VAL_5]]) {allowzero = 0 : si64} : (tensor<1x256x5x2x42xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_47:.*]] = "onnx.QuantizeLinear"(%[[VAL_46]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_48:.*]] = "onnx.DequantizeLinear"(%[[VAL_47]], %[[VAL_8]], %[[VAL_11]]) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_48]] : tensor<1x256x10x42xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride33(%arg0: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<1xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x3x3xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>   
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [3, 3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
    %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
    %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
    onnx.Return %13 : tensor<1x1x54x222xf32>

  // CHECK-LABEL:   func.func @test_convtrans_stride33(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<[5, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[4, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_24:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_25:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_28:.*]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_29:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_29]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_35:.*]] = "onnx.ReverseSequence"(%[[VAL_34]], %[[VAL_23]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_36:.*]] = "onnx.ReverseSequence"(%[[VAL_35]], %[[VAL_23]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_36]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_37]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_40:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_19]], %[[VAL_18]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_41:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_17]], %[[VAL_16]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_42:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_15]], %[[VAL_14]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_43:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_13]], %[[VAL_12]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_44:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_11]], %[[VAL_10]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_45:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_9]], %[[VAL_8]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_46:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_7]], %[[VAL_6]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_47:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_5]], %[[VAL_4]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_48:.*]] = "onnx.DequantizeLinear"(%[[VAL_47]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_48]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.DequantizeLinear"(%[[VAL_44]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_50]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.DequantizeLinear"(%[[VAL_45]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_52]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.DequantizeLinear"(%[[VAL_46]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_54]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.DequantizeLinear"(%[[VAL_43]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_57:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_56]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_58:.*]] = "onnx.DequantizeLinear"(%[[VAL_40]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_59:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_58]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_60:.*]] = "onnx.DequantizeLinear"(%[[VAL_41]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_61:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_60]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_62:.*]] = "onnx.DequantizeLinear"(%[[VAL_42]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_63:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_62]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_64:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_65:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_64]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_66:.*]] = "onnx.Reshape"(%[[VAL_49]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_67:.*]] = "onnx.Reshape"(%[[VAL_51]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_68:.*]] = "onnx.Reshape"(%[[VAL_53]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_69:.*]] = "onnx.Reshape"(%[[VAL_55]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_70:.*]] = "onnx.Reshape"(%[[VAL_57]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_71:.*]] = "onnx.Reshape"(%[[VAL_59]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_72:.*]] = "onnx.Reshape"(%[[VAL_61]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_73:.*]] = "onnx.Reshape"(%[[VAL_63]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_74:.*]] = "onnx.Reshape"(%[[VAL_65]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_75:.*]] = "onnx.Concat"(%[[VAL_66]], %[[VAL_67]], %[[VAL_72]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_76:.*]] = "onnx.Concat"(%[[VAL_69]], %[[VAL_70]], %[[VAL_71]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_77:.*]] = "onnx.Concat"(%[[VAL_68]], %[[VAL_73]], %[[VAL_74]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_78:.*]] = "onnx.Reshape"(%[[VAL_75]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_79:.*]] = "onnx.Reshape"(%[[VAL_76]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_80:.*]] = "onnx.Reshape"(%[[VAL_77]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_81:.*]] = "onnx.Concat"(%[[VAL_78]], %[[VAL_79]], %[[VAL_80]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           %[[VAL_82:.*]] = "onnx.Reshape"(%[[VAL_81]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_83:.*]] = "onnx.QuantizeLinear"(%[[VAL_82]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_84:.*]] = "onnx.DequantizeLinear"(%[[VAL_83]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return %[[VAL_84]] : tensor<1x1x54x222xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride33(
// CONSTPROP-SAME:                                       %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_14:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_15:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_16:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_17:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_24]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_26]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_28]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_30]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_32]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_34]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_36]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_38]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_23]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_27]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_29]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_47:.*]] = "onnx.Reshape"(%[[VAL_37]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_48:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_49:.*]] = "onnx.Concat"(%[[VAL_40]], %[[VAL_41]], %[[VAL_46]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_50:.*]] = "onnx.Concat"(%[[VAL_43]], %[[VAL_44]], %[[VAL_45]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_51:.*]] = "onnx.Concat"(%[[VAL_42]], %[[VAL_47]], %[[VAL_48]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_52:.*]] = "onnx.Reshape"(%[[VAL_49]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_53:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_54:.*]] = "onnx.Reshape"(%[[VAL_51]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_55:.*]] = "onnx.Concat"(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CONSTPROP:           %[[VAL_56:.*]] = "onnx.Reshape"(%[[VAL_55]], %[[VAL_10]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_57:.*]] = "onnx.QuantizeLinear"(%[[VAL_56]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_58:.*]] = "onnx.DequantizeLinear"(%[[VAL_57]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           onnx.Return %[[VAL_58]] : tensor<1x1x54x222xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride33_with_relu(%arg0: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<1xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x3x3xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>   
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [3, 3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
    %12 = "onnx.Relu"(%11) {} : (tensor<1x1x54x222xf32>) -> tensor<1x1x54x222xf32>    
    %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
    %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
    onnx.Return %14 : tensor<1x1x54x222xf32>

  // CHECK-LABEL:   func.func @test_convtrans_stride33_with_relu(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<[5, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[4, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_24:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_25:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_28:.*]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_29:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_29]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_35:.*]] = "onnx.ReverseSequence"(%[[VAL_34]], %[[VAL_23]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_36:.*]] = "onnx.ReverseSequence"(%[[VAL_35]], %[[VAL_23]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_36]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_37]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_40:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_19]], %[[VAL_18]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_41:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_17]], %[[VAL_16]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_42:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_15]], %[[VAL_14]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_43:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_13]], %[[VAL_12]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_44:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_11]], %[[VAL_10]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_45:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_9]], %[[VAL_8]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_46:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_7]], %[[VAL_6]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_47:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_5]], %[[VAL_4]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_48:.*]] = "onnx.DequantizeLinear"(%[[VAL_47]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_48]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.Relu"(%[[VAL_49]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.DequantizeLinear"(%[[VAL_44]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_51]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.Relu"(%[[VAL_52]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.DequantizeLinear"(%[[VAL_45]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_54]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.Relu"(%[[VAL_55]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_57:.*]] = "onnx.DequantizeLinear"(%[[VAL_46]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_58:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_57]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_59:.*]] = "onnx.Relu"(%[[VAL_58]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_60:.*]] = "onnx.DequantizeLinear"(%[[VAL_43]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_61:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_60]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_62:.*]] = "onnx.Relu"(%[[VAL_61]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_63:.*]] = "onnx.DequantizeLinear"(%[[VAL_40]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_64:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_63]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_65:.*]] = "onnx.Relu"(%[[VAL_64]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_66:.*]] = "onnx.DequantizeLinear"(%[[VAL_41]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_67:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_66]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_68:.*]] = "onnx.Relu"(%[[VAL_67]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_69:.*]] = "onnx.DequantizeLinear"(%[[VAL_42]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_70:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_69]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_71:.*]] = "onnx.Relu"(%[[VAL_70]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_72:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_73:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_72]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_74:.*]] = "onnx.Relu"(%[[VAL_73]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_75:.*]] = "onnx.Reshape"(%[[VAL_50]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_76:.*]] = "onnx.Reshape"(%[[VAL_53]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_77:.*]] = "onnx.Reshape"(%[[VAL_56]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_78:.*]] = "onnx.Reshape"(%[[VAL_59]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_79:.*]] = "onnx.Reshape"(%[[VAL_62]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_80:.*]] = "onnx.Reshape"(%[[VAL_65]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_81:.*]] = "onnx.Reshape"(%[[VAL_68]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_82:.*]] = "onnx.Reshape"(%[[VAL_71]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_83:.*]] = "onnx.Reshape"(%[[VAL_74]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_84:.*]] = "onnx.Concat"(%[[VAL_75]], %[[VAL_76]], %[[VAL_81]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_85:.*]] = "onnx.Concat"(%[[VAL_78]], %[[VAL_79]], %[[VAL_80]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_86:.*]] = "onnx.Concat"(%[[VAL_77]], %[[VAL_82]], %[[VAL_83]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_87:.*]] = "onnx.Reshape"(%[[VAL_84]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_88:.*]] = "onnx.Reshape"(%[[VAL_85]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_89:.*]] = "onnx.Reshape"(%[[VAL_86]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_90:.*]] = "onnx.Concat"(%[[VAL_87]], %[[VAL_88]], %[[VAL_89]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           %[[VAL_91:.*]] = "onnx.Reshape"(%[[VAL_90]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_92:.*]] = "onnx.QuantizeLinear"(%[[VAL_91]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_93:.*]] = "onnx.DequantizeLinear"(%[[VAL_92]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return %[[VAL_93]] : tensor<1x1x54x222xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride33_with_relu(
// CONSTPROP-SAME:                                                 %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_14:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_15:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_16:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_17:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.Relu"(%[[VAL_23]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_25]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Relu"(%[[VAL_26]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_28]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.Relu"(%[[VAL_29]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_31]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Relu"(%[[VAL_32]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_34]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_37]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.Relu"(%[[VAL_38]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_40]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.Relu"(%[[VAL_41]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_43]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.Relu"(%[[VAL_44]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_47:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_46]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_48:.*]] = "onnx.Relu"(%[[VAL_47]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_49:.*]] = "onnx.Reshape"(%[[VAL_24]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_50:.*]] = "onnx.Reshape"(%[[VAL_27]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_51:.*]] = "onnx.Reshape"(%[[VAL_30]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_52:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_53:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_54:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_55:.*]] = "onnx.Reshape"(%[[VAL_42]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_56:.*]] = "onnx.Reshape"(%[[VAL_45]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_57:.*]] = "onnx.Reshape"(%[[VAL_48]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_58:.*]] = "onnx.Concat"(%[[VAL_49]], %[[VAL_50]], %[[VAL_55]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_59:.*]] = "onnx.Concat"(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_60:.*]] = "onnx.Concat"(%[[VAL_51]], %[[VAL_56]], %[[VAL_57]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_61:.*]] = "onnx.Reshape"(%[[VAL_58]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_62:.*]] = "onnx.Reshape"(%[[VAL_59]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_63:.*]] = "onnx.Reshape"(%[[VAL_60]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_64:.*]] = "onnx.Concat"(%[[VAL_61]], %[[VAL_62]], %[[VAL_63]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CONSTPROP:           %[[VAL_65:.*]] = "onnx.Reshape"(%[[VAL_64]], %[[VAL_10]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_66:.*]] = "onnx.QuantizeLinear"(%[[VAL_65]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_67:.*]] = "onnx.DequantizeLinear"(%[[VAL_66]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           onnx.Return %[[VAL_67]] : tensor<1x1x54x222xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride33_with_qdq_relu(%arg0: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
    %3 = onnx.Constant dense<2> : tensor<i8>
    %4 = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
    %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
    %6 = onnx.Constant dense<2> : tensor<1xi8>
    %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
    %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<1x1x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x3x3xf32>
    %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
    %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>   
    %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [3, 3]} : (tensor<1x1x18x74xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x54x222xf32>
    %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
    %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
    %14 = "onnx.Relu"(%13) {} : (tensor<1x1x54x222xf32>) -> tensor<1x1x54x222xf32>    
    %15 = "onnx.QuantizeLinear"(%14, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
    %16 = "onnx.DequantizeLinear"(%15, %0, %3) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
    onnx.Return %16 : tensor<1x1x54x222xf32>
// CHECK-LABEL:   func.func @test_convtrans_stride33_with_qdq_relu(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<5> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<[5, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<[3, 4]> : tensor<2xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<[5, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<[4, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK:           %[[VAL_22:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK:           %[[VAL_24:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_25:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_26:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK:           %[[VAL_27:.*]] = onnx.Constant dense<2> : tensor<i8>
// CHECK:           %[[VAL_28:.*]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_29:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK:           %[[VAL_30:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK:           %[[VAL_31:.*]] = "onnx.DequantizeLinear"(%[[VAL_30]], %[[VAL_29]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           %[[VAL_33:.*]] = "onnx.DequantizeLinear"(%[[VAL_32]], %[[VAL_25]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_28]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_35:.*]] = "onnx.ReverseSequence"(%[[VAL_34]], %[[VAL_23]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_36:.*]] = "onnx.ReverseSequence"(%[[VAL_35]], %[[VAL_23]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_36]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_37]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           %[[VAL_39:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_40:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_19]], %[[VAL_18]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_41:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_17]], %[[VAL_16]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_42:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_15]], %[[VAL_14]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_43:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_13]], %[[VAL_12]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_44:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_11]], %[[VAL_10]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_45:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_9]], %[[VAL_8]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_46:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_7]], %[[VAL_6]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_47:.*]] = "onnx.Slice"(%[[VAL_38]], %[[VAL_5]], %[[VAL_4]], %[[VAL_22]], %[[VAL_21]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           %[[VAL_48:.*]] = "onnx.DequantizeLinear"(%[[VAL_47]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_49:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_48]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_50:.*]] = "onnx.QuantizeLinear"(%[[VAL_49]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_51:.*]] = "onnx.DequantizeLinear"(%[[VAL_50]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_52:.*]] = "onnx.Relu"(%[[VAL_51]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_53:.*]] = "onnx.DequantizeLinear"(%[[VAL_44]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_54:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_53]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_55:.*]] = "onnx.QuantizeLinear"(%[[VAL_54]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_56:.*]] = "onnx.DequantizeLinear"(%[[VAL_55]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_57:.*]] = "onnx.Relu"(%[[VAL_56]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_58:.*]] = "onnx.DequantizeLinear"(%[[VAL_45]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_59:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_58]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_60:.*]] = "onnx.QuantizeLinear"(%[[VAL_59]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_61:.*]] = "onnx.DequantizeLinear"(%[[VAL_60]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_62:.*]] = "onnx.Relu"(%[[VAL_61]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_63:.*]] = "onnx.DequantizeLinear"(%[[VAL_46]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_64:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_63]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_65:.*]] = "onnx.QuantizeLinear"(%[[VAL_64]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_66:.*]] = "onnx.DequantizeLinear"(%[[VAL_65]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_67:.*]] = "onnx.Relu"(%[[VAL_66]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_68:.*]] = "onnx.DequantizeLinear"(%[[VAL_43]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_69:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_68]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_70:.*]] = "onnx.QuantizeLinear"(%[[VAL_69]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_71:.*]] = "onnx.DequantizeLinear"(%[[VAL_70]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_72:.*]] = "onnx.Relu"(%[[VAL_71]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_73:.*]] = "onnx.DequantizeLinear"(%[[VAL_40]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_74:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_73]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_75:.*]] = "onnx.QuantizeLinear"(%[[VAL_74]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_76:.*]] = "onnx.DequantizeLinear"(%[[VAL_75]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_77:.*]] = "onnx.Relu"(%[[VAL_76]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_78:.*]] = "onnx.DequantizeLinear"(%[[VAL_41]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_79:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_78]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_80:.*]] = "onnx.QuantizeLinear"(%[[VAL_79]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_81:.*]] = "onnx.DequantizeLinear"(%[[VAL_80]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_82:.*]] = "onnx.Relu"(%[[VAL_81]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_83:.*]] = "onnx.DequantizeLinear"(%[[VAL_42]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_84:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_83]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_85:.*]] = "onnx.QuantizeLinear"(%[[VAL_84]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_86:.*]] = "onnx.DequantizeLinear"(%[[VAL_85]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_87:.*]] = "onnx.Relu"(%[[VAL_86]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_88:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_26]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_89:.*]] = "onnx.Conv"(%[[VAL_33]], %[[VAL_88]], %[[VAL_31]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_90:.*]] = "onnx.QuantizeLinear"(%[[VAL_89]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_91:.*]] = "onnx.DequantizeLinear"(%[[VAL_90]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_92:.*]] = "onnx.Relu"(%[[VAL_91]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           %[[VAL_93:.*]] = "onnx.Reshape"(%[[VAL_52]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_94:.*]] = "onnx.Reshape"(%[[VAL_57]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_95:.*]] = "onnx.Reshape"(%[[VAL_62]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_96:.*]] = "onnx.Reshape"(%[[VAL_67]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_97:.*]] = "onnx.Reshape"(%[[VAL_72]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_98:.*]] = "onnx.Reshape"(%[[VAL_77]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_99:.*]] = "onnx.Reshape"(%[[VAL_82]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_100:.*]] = "onnx.Reshape"(%[[VAL_87]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_101:.*]] = "onnx.Reshape"(%[[VAL_92]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK:           %[[VAL_102:.*]] = "onnx.Concat"(%[[VAL_93]], %[[VAL_94]], %[[VAL_99]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_103:.*]] = "onnx.Concat"(%[[VAL_96]], %[[VAL_97]], %[[VAL_98]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_104:.*]] = "onnx.Concat"(%[[VAL_95]], %[[VAL_100]], %[[VAL_101]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK:           %[[VAL_105:.*]] = "onnx.Reshape"(%[[VAL_102]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_106:.*]] = "onnx.Reshape"(%[[VAL_103]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_107:.*]] = "onnx.Reshape"(%[[VAL_104]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           %[[VAL_108:.*]] = "onnx.Concat"(%[[VAL_105]], %[[VAL_106]], %[[VAL_107]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           %[[VAL_109:.*]] = "onnx.Reshape"(%[[VAL_108]], %[[VAL_1]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           %[[VAL_110:.*]] = "onnx.QuantizeLinear"(%[[VAL_109]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           %[[VAL_111:.*]] = "onnx.DequantizeLinear"(%[[VAL_110]], %[[VAL_24]], %[[VAL_27]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return %[[VAL_111]] : tensor<1x1x54x222xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride33_with_qdq_relu(
// CONSTPROP-SAME:                                                     %[[VAL_0:.*]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<1x1x1x1xi8>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_13:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_14:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_15:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_16:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_17:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_18:.*]] = onnx.Constant dense<2> : tensor<1xi8>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.QuantizeLinear"(%[[VAL_23]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_24]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Relu"(%[[VAL_25]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_27]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.QuantizeLinear"(%[[VAL_28]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.DequantizeLinear"(%[[VAL_29]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Relu"(%[[VAL_30]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_32]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.QuantizeLinear"(%[[VAL_33]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.DequantizeLinear"(%[[VAL_34]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_37]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.QuantizeLinear"(%[[VAL_38]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Relu"(%[[VAL_40]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_42]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.QuantizeLinear"(%[[VAL_43]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.DequantizeLinear"(%[[VAL_44]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.Relu"(%[[VAL_45]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_47:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_48:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_47]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_49:.*]] = "onnx.QuantizeLinear"(%[[VAL_48]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_50:.*]] = "onnx.DequantizeLinear"(%[[VAL_49]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_51:.*]] = "onnx.Relu"(%[[VAL_50]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_52:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_53:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_52]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_54:.*]] = "onnx.QuantizeLinear"(%[[VAL_53]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_55:.*]] = "onnx.DequantizeLinear"(%[[VAL_54]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_56:.*]] = "onnx.Relu"(%[[VAL_55]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_57:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_58:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_57]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_59:.*]] = "onnx.QuantizeLinear"(%[[VAL_58]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_60:.*]] = "onnx.DequantizeLinear"(%[[VAL_59]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_61:.*]] = "onnx.Relu"(%[[VAL_60]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_62:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_63:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_62]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_64:.*]] = "onnx.QuantizeLinear"(%[[VAL_63]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_65:.*]] = "onnx.DequantizeLinear"(%[[VAL_64]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_66:.*]] = "onnx.Relu"(%[[VAL_65]]) : (tensor<1x1x54x222xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_67:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_68:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_69:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_70:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_71:.*]] = "onnx.Reshape"(%[[VAL_46]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_72:.*]] = "onnx.Reshape"(%[[VAL_51]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_73:.*]] = "onnx.Reshape"(%[[VAL_56]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_74:.*]] = "onnx.Reshape"(%[[VAL_61]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_75:.*]] = "onnx.Reshape"(%[[VAL_66]], %[[VAL_12]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CONSTPROP:           %[[VAL_76:.*]] = "onnx.Concat"(%[[VAL_67]], %[[VAL_68]], %[[VAL_73]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_77:.*]] = "onnx.Concat"(%[[VAL_70]], %[[VAL_71]], %[[VAL_72]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_78:.*]] = "onnx.Concat"(%[[VAL_69]], %[[VAL_74]], %[[VAL_75]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CONSTPROP:           %[[VAL_79:.*]] = "onnx.Reshape"(%[[VAL_76]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_80:.*]] = "onnx.Reshape"(%[[VAL_77]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_81:.*]] = "onnx.Reshape"(%[[VAL_78]], %[[VAL_11]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CONSTPROP:           %[[VAL_82:.*]] = "onnx.Concat"(%[[VAL_79]], %[[VAL_80]], %[[VAL_81]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CONSTPROP:           %[[VAL_83:.*]] = "onnx.Reshape"(%[[VAL_82]], %[[VAL_10]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           %[[VAL_84:.*]] = "onnx.QuantizeLinear"(%[[VAL_83]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_85:.*]] = "onnx.DequantizeLinear"(%[[VAL_84]], %[[VAL_13]], %[[VAL_16]]) {axis = 1 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           onnx.Return %[[VAL_85]] : tensor<1x1x54x222xf32>
// CONSTPROP:         }
}