// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-1d-phased %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

func.func @test_convtrans_stride_2_kernel_shape_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_2_kernel_shape_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_8_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_15_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_14_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_4_]], [[VAR_3_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_6_]], [[VAR_2_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_19_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_20_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x24x400xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_4_b(%arg0: tensor<1x64x400xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x400xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_2_kernel_shape_4_b
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x400xf32>, [[PARAM_1_:%.+]]: tensor<64x24x4xf32>) -> tensor<1x24x800xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 400, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<400> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<401> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_8_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_15_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x400xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x401xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_14_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x400xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x401xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_4_]], [[VAR_3_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x401xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x400xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_6_]], [[VAR_2_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x401xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x400xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x400xf32>, tensor<4xi64>) -> tensor<1x24x400x1xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_19_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x400xf32>, tensor<4xi64>) -> tensor<1x24x400x1xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_20_]]) {axis = -1 : si64} : (tensor<1x24x400x1xf32>, tensor<1x24x400x1xf32>) -> tensor<1x24x400x2xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x400x2xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4_b
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_8_unsupported(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x404xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x404xf32>
  onnx.Return %1 : tensor<1x24x404xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_kernel_shape_8_unsupported
// CHECK:           onnx.ConvTranspose
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_8_unsupported
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_4_nobias(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {      
  %0 = "onnx.NoValue"() { value} : () -> none
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, none) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_2_kernel_shape_4_nobias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_8_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_15_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, none) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_14_]], [[VAR_9_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, none) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_4_]], [[VAR_3_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_6_]], [[VAR_2_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_19_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_20_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x24x400xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4_nobias
// DISABLED: onnx.ConvTranspose
// -----

func.func @test_convtrans_stride_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_10_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_7_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_9_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_22_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_23_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Reshape"([[VAR_27_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Concat"([[VAR_28_]], [[VAR_29_]], [[VAR_30_]], [[VAR_31_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.Reshape"([[VAR_32_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return [[VAR_33_]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_5(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x10xf32>) -> tensor<1x24x1001xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [10],  pads = [2,2], strides = [5]} : (tensor<1x64x200xf32>, tensor<64x24x10xf32>, tensor<24xf32>) -> tensor<1x24x1001xf32>
  onnx.Return %1 : tensor<1x24x1001xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_5
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x10xf32>) -> tensor<1x24x1001xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1001> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 1005]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 24, 201, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 1]> : tensor<6xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<10> : tensor<64xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x10xf32>) -> tensor<10x64x24xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.ReverseSequence"([[VAR_16_]], [[VAR_14_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x64x24xf32>, tensor<64xi64>) -> tensor<10x64x24xf32>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_17_]]) {perm = [1, 2, 0]} : (tensor<10x64x24xf32>) -> tensor<64x24x10xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_18_]]) {perm = [1, 0, 2]} : (tensor<64x24x10xf32>) -> tensor<24x64x10xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_11_]], [[VAR_10_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_9_]], [[VAR_10_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_13_]], [[VAR_10_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_8_]], [[VAR_10_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_7_]], [[VAR_10_]], [[VAR_13_]], [[VAR_12_]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_22_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_21_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_20_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_24_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_23_]], [[VAR_15_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_9_]], [[VAR_6_]], [[VAR_13_]], [[VAR_9_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_29_]], [[VAR_9_]], [[VAR_6_]], [[VAR_13_]], [[VAR_9_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Pad"([[VAR_30_]], [[VAR_5_]], [[VAR_4_]], [[VAR_3_]]) {mode = "constant"} : (tensor<1x24x200xf32>, tensor<6xi64>, tensor<f32>, none) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Pad"([[VAR_31_]], [[VAR_5_]], [[VAR_4_]], [[VAR_3_]]) {mode = "constant"} : (tensor<1x24x200xf32>, tensor<6xi64>, tensor<f32>, none) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "onnx.Reshape"([[VAR_27_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_32_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           [[VAR_39_:%.+]] = "onnx.Concat"([[VAR_34_]], [[VAR_35_]], [[VAR_36_]], [[VAR_37_]], [[VAR_38_]]) {axis = -1 : si64} : (tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>) -> tensor<1x24x201x5xf32>
// CHECK:           [[VAR_40_:%.+]] = "onnx.Reshape"([[VAR_39_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x201x5xf32>, tensor<3xi64>) -> tensor<1x24x1005xf32>
// CHECK:           [[VAR_41_:%.+]] = "onnx.Slice"([[VAR_40_]], [[VAR_11_]], [[VAR_0_]], [[VAR_13_]], [[VAR_9_]]) : (tensor<1x24x1005xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x1001xf32>
// CHECK:           onnx.Return [[VAR_41_]] : tensor<1x24x1001xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_5
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_dilation2(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x403xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [2], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x403xf32>
  onnx.Return %1 : tensor<1x24x403xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_dilation2
// CHECK:           onnx.ConvTranspose
// DISABLED-LABEL: test_convtrans_stride_2_dilation2
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_nodilation(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_2_nodilation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.ReverseSequence"([[VAR_10_]], [[VAR_8_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_6_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_4_]], [[VAR_5_]], [[VAR_7_]], [[VAR_7_]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_15_]], [[VAR_9_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_14_]], [[VAR_9_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_4_]], [[VAR_3_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_6_]], [[VAR_2_]], [[VAR_7_]], [[VAR_4_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Reshape"([[VAR_19_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_20_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_22_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return [[VAR_23_]] : tensor<1x24x400xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_nodilation
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_lrelu(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 1.000000e-01 : f32} : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_4_lrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_10_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_7_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_9_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.LeakyRelu"([[VAR_20_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.LeakyRelu"([[VAR_22_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.LeakyRelu"([[VAR_24_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.LeakyRelu"([[VAR_26_]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_23_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_25_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Reshape"([[VAR_29_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_31_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Concat"([[VAR_32_]], [[VAR_33_]], [[VAR_34_]], [[VAR_35_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_36_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return [[VAR_37_]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4_lrelu
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_relu(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.Relu"(%1) : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_4_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_10_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_7_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_9_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Relu"([[VAR_20_]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Relu"([[VAR_22_]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Relu"([[VAR_24_]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Relu"([[VAR_26_]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_23_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_25_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Reshape"([[VAR_29_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_31_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Concat"([[VAR_32_]], [[VAR_33_]], [[VAR_34_]], [[VAR_35_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_36_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return [[VAR_37_]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4_relu
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_lrelu_default_value(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.LeakyRelu"(%1) : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:  func.func @test_convtrans_stride_4_lrelu_default_value
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x200xf32>, [[PARAM_1_:%.+]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.ReverseSequence"([[VAR_12_]], [[VAR_10_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_7_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_5_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_9_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]], [[VAR_8_]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_17_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.LeakyRelu"([[VAR_20_]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.LeakyRelu"([[VAR_22_]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_19_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.LeakyRelu"([[VAR_24_]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_18_]], [[VAR_11_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.LeakyRelu"([[VAR_26_]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_23_]], [[VAR_7_]], [[VAR_3_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_25_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_5_]], [[VAR_2_]], [[VAR_9_]], [[VAR_5_]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "onnx.Reshape"([[VAR_28_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Reshape"([[VAR_29_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_31_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Concat"([[VAR_32_]], [[VAR_33_]], [[VAR_34_]], [[VAR_35_]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_36_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return [[VAR_37_]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4_lrelu
// DISABLED: onnx.ConvTranspose
