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
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.QuantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.QuantizeLinear"(%[[VAL_13]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.DequantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           onnx.Return %[[VAL_15]] : tensor<1x1x13x57xf32>
// CONSTPROP:         }
}

// -----

func.func @test_convtrans_4phase_pads_0011(%arg0: tensor<1x128x10x16xf32>) -> tensor<1x32x20x32xf32> {      
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %3 = onnx.Constant dense<2> : tensor<i8>
  %4 = onnx.Constant dense<2> : tensor<128x32x3x3xi8>
  %5 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %6 = onnx.Constant dense<2> : tensor<32xi8>  
  %7 = "onnx.DequantizeLinear"(%6, %5, %3) {axis = 1 : si64} : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>
  %8 = "onnx.DequantizeLinear"(%4, %2, %3) {axis = 1 : si64} : (tensor<128x32x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<128x32x3x3xf32>
  %9 = "onnx.QuantizeLinear"(%arg0, %1, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x128x10x16xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xi8>
  %10 = "onnx.DequantizeLinear"(%9, %1, %3) {axis = 1 : si64} : (tensor<1x128x10x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xf32>    
  %11 = "onnx.ConvTranspose"(%10, %8, %7) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x128x10x16xf32>, tensor<128x32x3x3xf32>, tensor<32xf32>) -> tensor<1x32x20x32xf32>
  %12 = "onnx.QuantizeLinear"(%11, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x32x20x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xi8>
  %13 = "onnx.DequantizeLinear"(%12, %0, %3) {axis = 1 : si64} : (tensor<1x32x20x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xf32>
  onnx.Return %13 : tensor<1x32x20x32xf32>
// CHECK-LABEL:  func.func @test_convtrans_4phase_pads_0011
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x10x16xf32>) -> tensor<1x32x20x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 32, 10, 16]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0> : tensor<i8>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<2> : tensor<128x32x3x3xi8>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<2> : tensor<32xi8>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.DequantizeLinear"([[VAR_19_]], [[VAR_18_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_14_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x10x16xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xi8>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.DequantizeLinear"([[VAR_21_]], [[VAR_14_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x10x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_17_]]) {perm = [2, 3, 0, 1]} : (tensor<128x32x3x3xi8>) -> tensor<3x3x128x32xi8>
// CHECK:           [[VAR_24_:%.+]] = "onnx.ReverseSequence"([[VAR_23_]], [[VAR_12_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x128x32xi8>, tensor<3xi64>) -> tensor<3x3x128x32xi8>
// CHECK:           [[VAR_25_:%.+]] = "onnx.ReverseSequence"([[VAR_24_]], [[VAR_12_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x128x32xi8>, tensor<3xi64>) -> tensor<3x3x128x32xi8>
// CHECK:           [[VAR_26_:%.+]] = "onnx.Transpose"([[VAR_25_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x128x32xi8>) -> tensor<128x32x3x3xi8>
// CHECK:           [[VAR_27_:%.+]] = "onnx.Transpose"([[VAR_26_]]) {perm = [1, 0, 2, 3]} : (tensor<128x32x3x3xi8>) -> tensor<32x128x3x3xi8>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Pad"([[VAR_27_]], [[VAR_11_]], [[VAR_10_]], [[VAR_9_]]) {mode = "constant"} : (tensor<32x128x3x3xi8>, tensor<8xi64>, tensor<i8>, none) -> tensor<32x128x4x4xi8>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xi8>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xi8>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<32x128x4x4xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<32x128x2x2xi8>
// CHECK:           [[VAR_33_:%.+]] = "onnx.DequantizeLinear"([[VAR_32_]], [[VAR_15_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Conv"([[VAR_22_]], [[VAR_33_]], [[VAR_20_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.DequantizeLinear"([[VAR_29_]], [[VAR_15_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Conv"([[VAR_22_]], [[VAR_35_]], [[VAR_20_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.DequantizeLinear"([[VAR_30_]], [[VAR_15_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Conv"([[VAR_22_]], [[VAR_37_]], [[VAR_20_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.DequantizeLinear"([[VAR_31_]], [[VAR_15_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Conv"([[VAR_22_]], [[VAR_39_]], [[VAR_20_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CHECK:           [[VAR_41_:%.+]] = "onnx.Concat"([[VAR_36_]], [[VAR_40_]], [[VAR_38_]], [[VAR_34_]]) {axis = 1 : si64} : (tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>) -> tensor<1x128x10x16xf32>
// CHECK:           [[VAR_42_:%.+]] = "onnx.Reshape"([[VAR_41_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x128x10x16xf32>, tensor<5xi64>) -> tensor<2x2x32x10x16xf32>
// CHECK:           [[VAR_43_:%.+]] = "onnx.Transpose"([[VAR_42_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x10x16xf32>) -> tensor<32x10x2x16x2xf32>
// CHECK:           [[VAR_44_:%.+]] = "onnx.Reshape"([[VAR_43_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<32x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CHECK:           [[VAR_45_:%.+]] = "onnx.QuantizeLinear"([[VAR_44_]], [[VAR_13_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x32x20x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xi8>
// CHECK:           [[VAR_46_:%.+]] = "onnx.DequantizeLinear"([[VAR_45_]], [[VAR_13_]], [[VAR_16_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x32x20x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xf32>
// CHECK:           onnx.Return [[VAR_46_]] : tensor<1x32x20x32xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_4phase_pads_0011(
// CONSTPROP-SAME:                                               %[[VAL_0:.*]]: tensor<1x128x10x16xf32>) -> tensor<1x32x20x32xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<{{.*}}> : tensor<32x128x2x2xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<{{.*}}> : tensor<32x128x2x2xi8>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<{{.*}}> : tensor<32x128x2x2xi8>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<2> : tensor<32x128x2x2xi8>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<[1, 32, 20, 32]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<[2, 2, 32, 10, 16]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_11:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_12:.*]] = onnx.Constant dense<2> : tensor<32xi8>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_11]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_8]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x10x16xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xi8>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.DequantizeLinear"(%[[VAL_14]], %[[VAL_8]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x10x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x10x16xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_9]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.Conv"(%[[VAL_15]], %[[VAL_16]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_9]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.Conv"(%[[VAL_15]], %[[VAL_18]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 1, 0, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_9]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.Conv"(%[[VAL_15]], %[[VAL_20]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 1, 1, 0], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_9]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<32x128x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128x2x2xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_15]], %[[VAL_22]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 1], strides = [1, 1]} : (tensor<1x128x10x16xf32>, tensor<32x128x2x2xf32>, tensor<32xf32>) -> tensor<1x32x10x16xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.Concat"(%[[VAL_19]], %[[VAL_23]], %[[VAL_21]], %[[VAL_17]]) {axis = 1 : si64} : (tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>, tensor<1x32x10x16xf32>) -> tensor<1x128x10x16xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.Reshape"(%[[VAL_24]], %[[VAL_6]]) {allowzero = 0 : si64} : (tensor<1x128x10x16xf32>, tensor<5xi64>) -> tensor<2x2x32x10x16xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_25]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x32x10x16xf32>) -> tensor<32x10x2x16x2xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Reshape"(%[[VAL_26]], %[[VAL_5]]) {allowzero = 0 : si64} : (tensor<32x10x2x16x2xf32>, tensor<4xi64>) -> tensor<1x32x20x32xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.QuantizeLinear"(%[[VAL_27]], %[[VAL_7]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x32x20x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xi8>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.DequantizeLinear"(%[[VAL_28]], %[[VAL_7]], %[[VAL_10]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x32x20x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x32x20x32xf32>
// CONSTPROP:           onnx.Return %[[VAL_29]] : tensor<1x32x20x32xf32>
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
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Relu"(%[[VAL_20]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.Relu"(%[[VAL_13]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.LeakyRelu"(%[[VAL_20]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.LeakyRelu"(%[[VAL_13]]) {alpha = 1.000000e-01 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.LeakyRelu"(%[[VAL_20]]) {alpha = 0.00999999977 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.LeakyRelu"(%[[VAL_13]]) {alpha = 0.00999999977 : f32} : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.QuantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.DequantizeLinear"(%[[VAL_15]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CHECK:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_9]], %[[VAL_7]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CHECK:           %[[VAL_13:.*]] = "onnx.DequantizeLinear"(%[[VAL_12]], %[[VAL_5]], %[[VAL_7]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x4x16xi8>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_15:.*]] = "onnx.ReverseSequence"(%[[VAL_14]], %[[VAL_3]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x16x1x1xi8>, tensor<16xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_16:.*]] = "onnx.ReverseSequence"(%[[VAL_15]], %[[VAL_2]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<4x16x1x1xi8>, tensor<4xi64>) -> tensor<4x16x1x1xi8>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 1]} : (tensor<4x16x1x1xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x4x16xi8>) -> tensor<1x1x4x16xi8>
// CHECK:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_6]], %[[VAL_7]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_13]], %[[VAL_19]], %[[VAL_11]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.QuantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Relu"(%[[VAL_22]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.QuantizeLinear"(%[[VAL_23]], %[[VAL_4]], %[[VAL_7]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CHECK:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_24]], %[[VAL_4]], %[[VAL_7]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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
// CONSTPROP:           %[[VAL_9:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_7]], %[[VAL_6]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_4]], %[[VAL_6]]) {{.*}}: (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.DequantizeLinear"(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) {{.*}} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {{.*}} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Conv"(%[[VAL_11]], %[[VAL_12]], %[[VAL_9]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [2, 14, 2, 14], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.QuantizeLinear"(%[[VAL_13]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.DequantizeLinear"(%[[VAL_14]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.Relu"(%[[VAL_15]]) : (tensor<1x1x13x57xf32>) -> tensor<1x1x13x57xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.QuantizeLinear"(%[[VAL_16]], %[[VAL_3]], %[[VAL_6]]) {{.*}}: (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.DequantizeLinear"(%[[VAL_17]], %[[VAL_3]], %[[VAL_6]]) {{.*}} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
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

// CHECK-LABEL:  func.func @test_convtrans_stide22
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.DequantizeLinear"([[VAR_16_]], [[VAR_15_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.DequantizeLinear"([[VAR_18_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_21_:%.+]] = "onnx.ReverseSequence"([[VAR_20_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_22_:%.+]] = "onnx.ReverseSequence"([[VAR_21_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_28_]], [[VAR_26_]], [[VAR_27_]], [[VAR_25_]]) {axis = 0 : si64} : (tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>) -> tensor<1024x512x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.DequantizeLinear"([[VAR_29_]], [[VAR_12_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Conv"([[VAR_19_]], [[VAR_31_]], [[VAR_30_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.Reshape"([[VAR_32_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CHECK:           [[VAR_34_:%.+]] = "onnx.Transpose"([[VAR_33_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_34_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.QuantizeLinear"([[VAR_35_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           [[VAR_37_:%.+]] = "onnx.DequantizeLinear"([[VAR_36_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return [[VAR_37_]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stide22(
// CONSTPROP-SAME:                                      %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1024x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_8]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_11]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Concat"(%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.Conv"(%[[VAL_12]], %[[VAL_14]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.Reshape"(%[[VAL_15]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Reshape"(%[[VAL_17]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.QuantizeLinear"(%[[VAL_18]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.DequantizeLinear"(%[[VAL_19]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_20]] : tensor<1x256x10x42xf32>
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

// CHECK-LABEL:  func.func @test_convtrans_stride22_with_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.DequantizeLinear"([[VAR_16_]], [[VAR_15_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.DequantizeLinear"([[VAR_18_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_21_:%.+]] = "onnx.ReverseSequence"([[VAR_20_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_22_:%.+]] = "onnx.ReverseSequence"([[VAR_21_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_28_]], [[VAR_26_]], [[VAR_27_]], [[VAR_25_]]) {axis = 0 : si64} : (tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>) -> tensor<1024x512x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.DequantizeLinear"([[VAR_29_]], [[VAR_12_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Conv"([[VAR_19_]], [[VAR_31_]], [[VAR_30_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.Relu"([[VAR_32_]]) : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Transpose"([[VAR_34_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Reshape"([[VAR_35_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.QuantizeLinear"([[VAR_36_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           [[VAR_38_:%.+]] = "onnx.DequantizeLinear"([[VAR_37_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return [[VAR_38_]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride22_with_relu(
// CONSTPROP-SAME:                                                 %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1024x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_8]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_11]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Concat"(%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.Conv"(%[[VAL_12]], %[[VAL_14]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.Relu"(%[[VAL_15]]) : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.Reshape"(%[[VAL_16]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.Reshape"(%[[VAL_18]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_19]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_21]] : tensor<1x256x10x42xf32>
// CONSTPROP:         }
  }

  // -----

  func.func @test_convtrans_stride22_with_lrelu(%arg0: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
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
    %12 = "onnx.LeakyRelu"(%11) {} : (tensor<1x256x10x42xf32>) -> tensor<1x256x10x42xf32>    
    %13 = "onnx.QuantizeLinear"(%12, %0, %3) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
    %14 = "onnx.DequantizeLinear"(%13, %0, %3) {axis = 1 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
    onnx.Return %14 : tensor<1x256x10x42xf32>
// CHECK-LABEL:  func.func @test_convtrans_stride22_with_lrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.DequantizeLinear"([[VAR_16_]], [[VAR_15_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.DequantizeLinear"([[VAR_18_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_21_:%.+]] = "onnx.ReverseSequence"([[VAR_20_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_22_:%.+]] = "onnx.ReverseSequence"([[VAR_21_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_28_]], [[VAR_26_]], [[VAR_27_]], [[VAR_25_]]) {axis = 0 : si64} : (tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>) -> tensor<1024x512x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.DequantizeLinear"([[VAR_29_]], [[VAR_12_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Conv"([[VAR_19_]], [[VAR_31_]], [[VAR_30_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.LeakyRelu"([[VAR_32_]]) {alpha = 0.00999999977 : f32} : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Transpose"([[VAR_34_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Reshape"([[VAR_35_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.QuantizeLinear"([[VAR_36_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           [[VAR_38_:%.+]] = "onnx.DequantizeLinear"([[VAR_37_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return [[VAR_38_]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride22_with_lrelu(
// CONSTPROP-SAME:                                                  %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1024x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_8]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_11]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Concat"(%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.Conv"(%[[VAL_12]], %[[VAL_14]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.LeakyRelu"(%[[VAL_15]]) {alpha = 0.00999999977 : f32} : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.Reshape"(%[[VAL_16]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.Reshape"(%[[VAL_18]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_19]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_21]] : tensor<1x256x10x42xf32>
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
// CHECK-LABEL:  func.func @test_convtrans_stride22_with_qdq_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<6> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<6> : tensor<6xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2> : tensor<512x256x6x6xi8>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<256xi8>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.DequantizeLinear"([[VAR_16_]], [[VAR_15_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.DequantizeLinear"([[VAR_18_]], [[VAR_11_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [2, 3, 0, 1]} : (tensor<512x256x6x6xi8>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_21_:%.+]] = "onnx.ReverseSequence"([[VAR_20_]], [[VAR_9_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_22_:%.+]] = "onnx.ReverseSequence"([[VAR_21_]], [[VAR_9_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<6x6x512x256xi8>, tensor<6xi64>) -> tensor<6x6x512x256xi8>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [2, 3, 0, 1]} : (tensor<6x6x512x256xi8>) -> tensor<512x256x6x6xi8>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_23_]]) {perm = [1, 0, 2, 3]} : (tensor<512x256x6x6xi8>) -> tensor<256x512x6x6xi8>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_6_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_4_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_2_]], [[VAR_5_]], [[VAR_8_]], [[VAR_7_]]) : (tensor<256x512x6x6xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<256x512x3x3xi8>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Concat"([[VAR_28_]], [[VAR_26_]], [[VAR_27_]], [[VAR_25_]]) {axis = 0 : si64} : (tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>, tensor<256x512x3x3xi8>) -> tensor<1024x512x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Concat"([[VAR_17_]], [[VAR_17_]], [[VAR_17_]], [[VAR_17_]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.DequantizeLinear"([[VAR_29_]], [[VAR_12_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Conv"([[VAR_19_]], [[VAR_31_]], [[VAR_30_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.QuantizeLinear"([[VAR_32_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1024x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1024x5x21xi8>
// CHECK:           [[VAR_34_:%.+]] = "onnx.DequantizeLinear"([[VAR_33_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1024x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Relu"([[VAR_34_]]) : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Reshape"([[VAR_35_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.Transpose"([[VAR_36_]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CHECK:           [[VAR_38_:%.+]] = "onnx.Reshape"([[VAR_37_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CHECK:           [[VAR_39_:%.+]] = "onnx.QuantizeLinear"([[VAR_38_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CHECK:           [[VAR_40_:%.+]] = "onnx.DequantizeLinear"([[VAR_39_]], [[VAR_10_]], [[VAR_13_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CHECK:           onnx.Return [[VAR_40_]] : tensor<1x256x10x42xf32>
// CHECK:         }
// CONSTPROP-LABEL:   func.func @test_convtrans_stride22_with_qdq_relu(
// CONSTPROP-SAME:                                                     %[[VAL_0:.*]]: tensor<1x512x5x21xf32>) -> tensor<1x256x10x42xf32> {
// CONSTPROP:           %[[VAL_1:.*]] = onnx.Constant dense<2> : tensor<1024x512x3x3xi8>
// CONSTPROP:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 256, 10, 42]> : tensor<4xi64>
// CONSTPROP:           %[[VAL_3:.*]] = onnx.Constant dense<[2, 2, 256, 5, 21]> : tensor<5xi64>
// CONSTPROP:           %[[VAL_4:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CONSTPROP:           %[[VAL_5:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CONSTPROP:           %[[VAL_6:.*]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CONSTPROP:           %[[VAL_7:.*]] = onnx.Constant dense<2> : tensor<i8>
// CONSTPROP:           %[[VAL_8:.*]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CONSTPROP:           %[[VAL_9:.*]] = onnx.Constant dense<2> : tensor<256xi8>
// CONSTPROP:           %[[VAL_10:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_8]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<256xi8>, tensor<f32>, tensor<i8>) -> tensor<256xf32>
// CONSTPROP:           %[[VAL_11:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x512x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xi8>
// CONSTPROP:           %[[VAL_12:.*]] = "onnx.DequantizeLinear"(%[[VAL_11]], %[[VAL_5]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x512x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x512x5x21xf32>
// CONSTPROP:           %[[VAL_13:.*]] = "onnx.Concat"(%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]) {axis = 0 : si64} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1024xf32>
// CONSTPROP:           %[[VAL_14:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_6]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1024x512x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1024x512x3x3xf32>
// CONSTPROP:           %[[VAL_15:.*]] = "onnx.Conv"(%[[VAL_12]], %[[VAL_14]], %[[VAL_13]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x5x21xf32>, tensor<1024x512x3x3xf32>, tensor<1024xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_16:.*]] = "onnx.QuantizeLinear"(%[[VAL_15]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1024x5x21xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1024x5x21xi8>
// CONSTPROP:           %[[VAL_17:.*]] = "onnx.DequantizeLinear"(%[[VAL_16]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1024x5x21xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_18:.*]] = "onnx.Relu"(%[[VAL_17]]) : (tensor<1x1024x5x21xf32>) -> tensor<1x1024x5x21xf32>
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.Reshape"(%[[VAL_18]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x1024x5x21xf32>, tensor<5xi64>) -> tensor<2x2x256x5x21xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [2, 3, 0, 4, 1]} : (tensor<2x2x256x5x21xf32>) -> tensor<256x5x2x21x2xf32>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.Reshape"(%[[VAL_20]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<256x5x2x21x2xf32>, tensor<4xi64>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.QuantizeLinear"(%[[VAL_21]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x256x10x42xf32>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xi8>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.DequantizeLinear"(%[[VAL_22]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x256x10x42xi8>, tensor<f32>, tensor<i8>) -> tensor<1x256x10x42xf32>
// CONSTPROP:           onnx.Return %[[VAL_23]] : tensor<1x256x10x42xf32>
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
// CHECK-LABEL:  func.func @test_convtrans_stride33
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.DequantizeLinear"([[VAR_21_]], [[VAR_20_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.DequantizeLinear"([[VAR_23_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_26_:%.+]] = "onnx.ReverseSequence"([[VAR_25_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_27_:%.+]] = "onnx.ReverseSequence"([[VAR_26_]], [[VAR_14_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_27_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_28_]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_10_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_8_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_7_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_6_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_5_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_4_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_3_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           [[VAR_39_:%.+]] = "onnx.DequantizeLinear"([[VAR_38_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_39_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.DequantizeLinear"([[VAR_35_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_41_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.DequantizeLinear"([[VAR_36_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_43_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = "onnx.DequantizeLinear"([[VAR_37_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_45_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = "onnx.DequantizeLinear"([[VAR_34_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_47_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.DequantizeLinear"([[VAR_31_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_49_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = "onnx.DequantizeLinear"([[VAR_32_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_51_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.DequantizeLinear"([[VAR_33_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_53_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_55_:%.+]] = "onnx.DequantizeLinear"([[VAR_30_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_55_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = "onnx.Reshape"([[VAR_40_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = "onnx.Reshape"([[VAR_42_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = "onnx.Reshape"([[VAR_44_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_60_:%.+]] = "onnx.Reshape"([[VAR_46_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_61_:%.+]] = "onnx.Reshape"([[VAR_48_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_62_:%.+]] = "onnx.Reshape"([[VAR_50_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_63_:%.+]] = "onnx.Reshape"([[VAR_52_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_64_:%.+]] = "onnx.Reshape"([[VAR_54_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_65_:%.+]] = "onnx.Reshape"([[VAR_56_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_66_:%.+]] = "onnx.Concat"([[VAR_57_]], [[VAR_58_]], [[VAR_63_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_67_:%.+]] = "onnx.Concat"([[VAR_60_]], [[VAR_61_]], [[VAR_62_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Concat"([[VAR_59_]], [[VAR_64_]], [[VAR_65_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = "onnx.Reshape"([[VAR_66_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_70_:%.+]] = "onnx.Reshape"([[VAR_67_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_71_:%.+]] = "onnx.Reshape"([[VAR_68_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_72_:%.+]] = "onnx.Concat"([[VAR_69_]], [[VAR_70_]], [[VAR_71_]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           [[VAR_73_:%.+]] = "onnx.Reshape"([[VAR_72_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           [[VAR_74_:%.+]] = "onnx.QuantizeLinear"([[VAR_73_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           [[VAR_75_:%.+]] = "onnx.DequantizeLinear"([[VAR_74_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return [[VAR_75_]] : tensor<1x1x54x222xf32>
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
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_24]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_26]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_28]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_30]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_32]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_34]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_36]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
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
// CONSTPROP:           %[[VAL_57:.*]] = "onnx.QuantizeLinear"(%[[VAL_56]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_58:.*]] = "onnx.DequantizeLinear"(%[[VAL_57]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
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

// CHECK-LABEL:  func.func @test_convtrans_stride33_with_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.DequantizeLinear"([[VAR_21_]], [[VAR_20_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.DequantizeLinear"([[VAR_23_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_26_:%.+]] = "onnx.ReverseSequence"([[VAR_25_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_27_:%.+]] = "onnx.ReverseSequence"([[VAR_26_]], [[VAR_14_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_27_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_28_]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_10_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_8_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_7_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_6_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_5_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_4_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_3_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           [[VAR_39_:%.+]] = "onnx.DequantizeLinear"([[VAR_38_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_40_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_39_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.Relu"([[VAR_40_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = "onnx.DequantizeLinear"([[VAR_35_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_43_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_42_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Relu"([[VAR_43_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = "onnx.DequantizeLinear"([[VAR_36_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_46_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_45_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = "onnx.Relu"([[VAR_46_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.DequantizeLinear"([[VAR_37_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_49_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_48_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = "onnx.Relu"([[VAR_49_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = "onnx.DequantizeLinear"([[VAR_34_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_52_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_51_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Relu"([[VAR_52_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.DequantizeLinear"([[VAR_31_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_54_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = "onnx.Relu"([[VAR_55_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = "onnx.DequantizeLinear"([[VAR_32_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_58_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_57_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = "onnx.Relu"([[VAR_58_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_60_:%.+]] = "onnx.DequantizeLinear"([[VAR_33_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_61_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_60_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_62_:%.+]] = "onnx.Relu"([[VAR_61_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_63_:%.+]] = "onnx.DequantizeLinear"([[VAR_30_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_64_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_63_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_65_:%.+]] = "onnx.Relu"([[VAR_64_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_66_:%.+]] = "onnx.Reshape"([[VAR_41_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_67_:%.+]] = "onnx.Reshape"([[VAR_44_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Reshape"([[VAR_47_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = "onnx.Reshape"([[VAR_50_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_70_:%.+]] = "onnx.Reshape"([[VAR_53_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_71_:%.+]] = "onnx.Reshape"([[VAR_56_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_72_:%.+]] = "onnx.Reshape"([[VAR_59_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_73_:%.+]] = "onnx.Reshape"([[VAR_62_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_74_:%.+]] = "onnx.Reshape"([[VAR_65_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_75_:%.+]] = "onnx.Concat"([[VAR_66_]], [[VAR_67_]], [[VAR_72_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_76_:%.+]] = "onnx.Concat"([[VAR_69_]], [[VAR_70_]], [[VAR_71_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_77_:%.+]] = "onnx.Concat"([[VAR_68_]], [[VAR_73_]], [[VAR_74_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_78_:%.+]] = "onnx.Reshape"([[VAR_75_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_79_:%.+]] = "onnx.Reshape"([[VAR_76_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_80_:%.+]] = "onnx.Reshape"([[VAR_77_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_81_:%.+]] = "onnx.Concat"([[VAR_78_]], [[VAR_79_]], [[VAR_80_]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           [[VAR_82_:%.+]] = "onnx.Reshape"([[VAR_81_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           [[VAR_83_:%.+]] = "onnx.QuantizeLinear"([[VAR_82_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           [[VAR_84_:%.+]] = "onnx.DequantizeLinear"([[VAR_83_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return [[VAR_84_]] : tensor<1x1x54x222xf32>
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
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.Relu"(%[[VAL_23]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_25]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.Relu"(%[[VAL_26]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_28]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.Relu"(%[[VAL_29]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_31]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Relu"(%[[VAL_32]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_34]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_37]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.Relu"(%[[VAL_38]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_40]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.Relu"(%[[VAL_41]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_43]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.Relu"(%[[VAL_44]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
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
// CONSTPROP:           %[[VAL_66:.*]] = "onnx.QuantizeLinear"(%[[VAL_65]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_67:.*]] = "onnx.DequantizeLinear"(%[[VAL_66]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
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
// CHECK-LABEL:  func.func @test_convtrans_stride33_with_qdq_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x18x74xf32>) -> tensor<1x1x54x222xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 54, 222]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 1, 18, 1, 222]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 1, 18, 74, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<3> : tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<3> : tensor<3xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.22070313E-4> : tensor<f32>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<2> : tensor<i8>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<2> : tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<2> : tensor<1xi8>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.DequantizeLinear"([[VAR_21_]], [[VAR_20_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.DequantizeLinear"([[VAR_23_]], [[VAR_16_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [2, 3, 0, 1]} : (tensor<1x1x3x3xi8>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_26_:%.+]] = "onnx.ReverseSequence"([[VAR_25_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_27_:%.+]] = "onnx.ReverseSequence"([[VAR_26_]], [[VAR_14_]]) {batch_axis = 0 : si64, time_axis = 1 : si64} : (tensor<3x3x1x1xi8>, tensor<3xi64>) -> tensor<3x3x1x1xi8>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_27_]]) {perm = [2, 3, 0, 1]} : (tensor<3x3x1x1xi8>) -> tensor<1x1x3x3xi8>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_28_]]) {perm = [1, 0, 2, 3]} : (tensor<1x1x3x3xi8>) -> tensor<1x1x3x3xi8>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_11_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_10_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_8_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_7_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_6_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_5_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_4_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_3_]], [[VAR_12_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<1x1x3x3xi8>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x1x1xi8>
// CHECK:           [[VAR_39_:%.+]] = "onnx.DequantizeLinear"([[VAR_38_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_40_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_39_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_41_:%.+]] = "onnx.QuantizeLinear"([[VAR_40_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_42_:%.+]] = "onnx.DequantizeLinear"([[VAR_41_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.Relu"([[VAR_42_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.DequantizeLinear"([[VAR_35_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_45_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_44_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_46_:%.+]] = "onnx.QuantizeLinear"([[VAR_45_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_47_:%.+]] = "onnx.DequantizeLinear"([[VAR_46_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Relu"([[VAR_47_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.DequantizeLinear"([[VAR_36_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_50_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_49_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_51_:%.+]] = "onnx.QuantizeLinear"([[VAR_50_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_52_:%.+]] = "onnx.DequantizeLinear"([[VAR_51_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Relu"([[VAR_52_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.DequantizeLinear"([[VAR_37_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_54_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_56_:%.+]] = "onnx.QuantizeLinear"([[VAR_55_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_57_:%.+]] = "onnx.DequantizeLinear"([[VAR_56_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = "onnx.Relu"([[VAR_57_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = "onnx.DequantizeLinear"([[VAR_34_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_60_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_59_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_61_:%.+]] = "onnx.QuantizeLinear"([[VAR_60_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_62_:%.+]] = "onnx.DequantizeLinear"([[VAR_61_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_63_:%.+]] = "onnx.Relu"([[VAR_62_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_64_:%.+]] = "onnx.DequantizeLinear"([[VAR_31_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_65_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_64_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_66_:%.+]] = "onnx.QuantizeLinear"([[VAR_65_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_67_:%.+]] = "onnx.DequantizeLinear"([[VAR_66_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Relu"([[VAR_67_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = "onnx.DequantizeLinear"([[VAR_32_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_70_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_69_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_71_:%.+]] = "onnx.QuantizeLinear"([[VAR_70_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_72_:%.+]] = "onnx.DequantizeLinear"([[VAR_71_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_73_:%.+]] = "onnx.Relu"([[VAR_72_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_74_:%.+]] = "onnx.DequantizeLinear"([[VAR_33_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_75_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_74_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_76_:%.+]] = "onnx.QuantizeLinear"([[VAR_75_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_77_:%.+]] = "onnx.DequantizeLinear"([[VAR_76_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_78_:%.+]] = "onnx.Relu"([[VAR_77_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_79_:%.+]] = "onnx.DequantizeLinear"([[VAR_30_]], [[VAR_17_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CHECK:           [[VAR_80_:%.+]] = "onnx.Conv"([[VAR_24_]], [[VAR_79_]], [[VAR_22_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CHECK:           [[VAR_81_:%.+]] = "onnx.QuantizeLinear"([[VAR_80_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CHECK:           [[VAR_82_:%.+]] = "onnx.DequantizeLinear"([[VAR_81_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_83_:%.+]] = "onnx.Relu"([[VAR_82_]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CHECK-DAG:       [[VAR_84_:%.+]] = "onnx.Reshape"([[VAR_43_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_85_:%.+]] = "onnx.Reshape"([[VAR_48_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_86_:%.+]] = "onnx.Reshape"([[VAR_53_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_87_:%.+]] = "onnx.Reshape"([[VAR_58_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_88_:%.+]] = "onnx.Reshape"([[VAR_63_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_89_:%.+]] = "onnx.Reshape"([[VAR_68_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_90_:%.+]] = "onnx.Reshape"([[VAR_73_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_91_:%.+]] = "onnx.Reshape"([[VAR_78_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_92_:%.+]] = "onnx.Reshape"([[VAR_83_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74xf32>, tensor<5xi64>) -> tensor<1x1x18x74x1xf32>
// CHECK-DAG:       [[VAR_93_:%.+]] = "onnx.Concat"([[VAR_84_]], [[VAR_85_]], [[VAR_90_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_94_:%.+]] = "onnx.Concat"([[VAR_87_]], [[VAR_88_]], [[VAR_89_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_95_:%.+]] = "onnx.Concat"([[VAR_86_]], [[VAR_91_]], [[VAR_92_]]) {axis = -1 : si64} : (tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>, tensor<1x1x18x74x1xf32>) -> tensor<1x1x18x74x3xf32>
// CHECK-DAG:       [[VAR_96_:%.+]] = "onnx.Reshape"([[VAR_93_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK-DAG:       [[VAR_97_:%.+]] = "onnx.Reshape"([[VAR_94_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_98_:%.+]] = "onnx.Reshape"([[VAR_95_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x18x74x3xf32>, tensor<5xi64>) -> tensor<1x1x18x1x222xf32>
// CHECK:           [[VAR_99_:%.+]] = "onnx.Concat"([[VAR_96_]], [[VAR_97_]], [[VAR_98_]]) {axis = -2 : si64} : (tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>, tensor<1x1x18x1x222xf32>) -> tensor<1x1x18x3x222xf32>
// CHECK:           [[VAR_100_:%.+]] = "onnx.Reshape"([[VAR_99_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1x18x3x222xf32>, tensor<4xi64>) -> tensor<1x1x54x222xf32>
// CHECK:           [[VAR_101_:%.+]] = "onnx.QuantizeLinear"([[VAR_100_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CHECK:           [[VAR_102_:%.+]] = "onnx.DequantizeLinear"([[VAR_101_]], [[VAR_15_]], [[VAR_18_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CHECK:           onnx.Return [[VAR_102_]] : tensor<1x1x54x222xf32>
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
// CONSTPROP:           %[[VAL_19:.*]] = "onnx.DequantizeLinear"(%[[VAL_18]], %[[VAL_17]], %[[VAL_16]]) {{.*}} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
// CONSTPROP:           %[[VAL_20:.*]] = "onnx.QuantizeLinear"(%[[VAL_0]], %[[VAL_14]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_21:.*]] = "onnx.DequantizeLinear"(%[[VAL_20]], %[[VAL_14]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_22:.*]] = "onnx.DequantizeLinear"(%[[VAL_1]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_23:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_24:.*]] = "onnx.QuantizeLinear"(%[[VAL_23]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_25:.*]] = "onnx.DequantizeLinear"(%[[VAL_24]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_26:.*]] = "onnx.Relu"(%[[VAL_25]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_27:.*]] = "onnx.DequantizeLinear"(%[[VAL_4]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_27]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_29:.*]] = "onnx.QuantizeLinear"(%[[VAL_28]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_30:.*]] = "onnx.DequantizeLinear"(%[[VAL_29]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_31:.*]] = "onnx.Relu"(%[[VAL_30]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_32:.*]] = "onnx.DequantizeLinear"(%[[VAL_3]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_32]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_34:.*]] = "onnx.QuantizeLinear"(%[[VAL_33]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_35:.*]] = "onnx.DequantizeLinear"(%[[VAL_34]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_36:.*]] = "onnx.Relu"(%[[VAL_35]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_37:.*]] = "onnx.DequantizeLinear"(%[[VAL_2]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_38:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_37]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_39:.*]] = "onnx.QuantizeLinear"(%[[VAL_38]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_40:.*]] = "onnx.DequantizeLinear"(%[[VAL_39]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_41:.*]] = "onnx.Relu"(%[[VAL_40]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_42:.*]] = "onnx.DequantizeLinear"(%[[VAL_5]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_43:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_42]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_44:.*]] = "onnx.QuantizeLinear"(%[[VAL_43]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_45:.*]] = "onnx.DequantizeLinear"(%[[VAL_44]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_46:.*]] = "onnx.Relu"(%[[VAL_45]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_47:.*]] = "onnx.DequantizeLinear"(%[[VAL_8]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_48:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_47]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_49:.*]] = "onnx.QuantizeLinear"(%[[VAL_48]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_50:.*]] = "onnx.DequantizeLinear"(%[[VAL_49]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_51:.*]] = "onnx.Relu"(%[[VAL_50]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_52:.*]] = "onnx.DequantizeLinear"(%[[VAL_7]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_53:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_52]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_54:.*]] = "onnx.QuantizeLinear"(%[[VAL_53]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_55:.*]] = "onnx.DequantizeLinear"(%[[VAL_54]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_56:.*]] = "onnx.Relu"(%[[VAL_55]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_57:.*]] = "onnx.DequantizeLinear"(%[[VAL_6]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_58:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_57]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_59:.*]] = "onnx.QuantizeLinear"(%[[VAL_58]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_60:.*]] = "onnx.DequantizeLinear"(%[[VAL_59]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_61:.*]] = "onnx.Relu"(%[[VAL_60]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_62:.*]] = "onnx.DequantizeLinear"(%[[VAL_9]], %[[VAL_15]], %[[VAL_16]]) {{.*}} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
// CONSTPROP:           %[[VAL_63:.*]] = "onnx.Conv"(%[[VAL_21]], %[[VAL_62]], %[[VAL_19]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x18x74xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_64:.*]] = "onnx.QuantizeLinear"(%[[VAL_63]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x18x74xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xi8>
// CONSTPROP:           %[[VAL_65:.*]] = "onnx.DequantizeLinear"(%[[VAL_64]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x18x74xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x18x74xf32>
// CONSTPROP:           %[[VAL_66:.*]] = "onnx.Relu"(%[[VAL_65]]) : (tensor<1x1x18x74xf32>) -> tensor<1x1x18x74xf32>
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
// CONSTPROP:           %[[VAL_84:.*]] = "onnx.QuantizeLinear"(%[[VAL_83]], %[[VAL_13]], %[[VAL_16]]) {{.*}}: (tensor<1x1x54x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xi8>
// CONSTPROP:           %[[VAL_85:.*]] = "onnx.DequantizeLinear"(%[[VAL_84]], %[[VAL_13]], %[[VAL_16]]) {{.*}} : (tensor<1x1x54x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x54x222xf32>
// CONSTPROP:           onnx.Return %[[VAL_85]] : tensor<1x1x54x222xf32>
// CONSTPROP:         }
}