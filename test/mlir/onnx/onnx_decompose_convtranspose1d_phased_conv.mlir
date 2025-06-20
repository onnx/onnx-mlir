// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-1d-phased %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

func.func @test_convtrans_stride_2_kernel_shape_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:   func.func @test_convtrans_stride_2_kernel_shape_4(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.ReverseSequence"(%[[VAL_13]], %[[VAL_11]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_9]], %[[VAL_8]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_7]], %[[VAL_6]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_18]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_17]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_19]], %[[VAL_7]], %[[VAL_5]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_9]], %[[VAL_4]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Reshape"(%[[VAL_21]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Concat"(%[[VAL_24]], %[[VAL_23]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return %[[VAL_26]] : tensor<1x24x400xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_4_b(%arg0: tensor<1x64x400xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x400xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   func.func @test_convtrans_stride_2_kernel_shape_4_b(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: tensor<1x64x400xf32>,
// CHECK-SAME:                                                        %[[VAL_1:.*]]: tensor<64x24x4xf32>) -> tensor<1x24x800xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 400, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<400> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<401> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.ReverseSequence"(%[[VAL_13]], %[[VAL_11]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_9]], %[[VAL_8]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_7]], %[[VAL_6]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_18]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x400xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x401xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_17]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x400xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x401xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_19]], %[[VAL_7]], %[[VAL_5]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x401xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x400xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_9]], %[[VAL_4]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x401xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x400xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Reshape"(%[[VAL_21]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x400xf32>, tensor<4xi64>) -> tensor<1x24x400x1xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x400xf32>, tensor<4xi64>) -> tensor<1x24x400x1xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Concat"(%[[VAL_24]], %[[VAL_23]]) {axis = -1 : si64} : (tensor<1x24x400x1xf32>, tensor<1x24x400x1xf32>) -> tensor<1x24x400x2xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x400x2xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return %[[VAL_26]] : tensor<1x24x800xf32>
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
// CHECK-LABEL:   func.func @test_convtrans_stride_2_kernel_shape_4_nobias(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                                             %[[VAL_1:.*]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK:           %[[VAL_12:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_13:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.ReverseSequence"(%[[VAL_13]], %[[VAL_11]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_9]], %[[VAL_8]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_7]], %[[VAL_6]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_18]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, none) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_17]], %[[VAL_12]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, none) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_19]], %[[VAL_7]], %[[VAL_5]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_9]], %[[VAL_4]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Reshape"(%[[VAL_21]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Concat"(%[[VAL_24]], %[[VAL_23]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return %[[VAL_26]] : tensor<1x24x400xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4_nobias
// DISABLED: onnx.ConvTranspose
// -----

func.func @test_convtrans_stride_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   func.func @test_convtrans_stride_4(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<11> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<9> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_14]], %[[VAL_8]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_7]], %[[VAL_6]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_21]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_26]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Slice"(%[[VAL_27]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Slice"(%[[VAL_28]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Reshape"(%[[VAL_29]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Reshape"(%[[VAL_30]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Reshape"(%[[VAL_32]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Concat"(%[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_37]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return %[[VAL_38]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_5(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x10xf32>) -> tensor<1x24x1001xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [10],  pads = [2,2], strides = [5]} : (tensor<1x64x200xf32>, tensor<64x24x10xf32>, tensor<24xf32>) -> tensor<1x24x1001xf32>
  onnx.Return %1 : tensor<1x24x1001xf32>
}
// CHECK-LABEL:   func.func @test_convtrans_stride_5(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: tensor<64x24x10xf32>) -> tensor<1x24x1001xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<1001> : tensor<1xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 1005]> : tensor<3xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<[1, 24, 201, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_5:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<[0, 0, 0, 0, 0, 1]> : tensor<6xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<14> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<13> : tensor<1xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<12> : tensor<1xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<11> : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK:           %[[VAL_17:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_18:.*]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK:           %[[VAL_19:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_20:.*]] = onnx.Constant dense<10> : tensor<64xi64>
// CHECK:           %[[VAL_21:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x10xf32>) -> tensor<10x64x24xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.ReverseSequence"(%[[VAL_22]], %[[VAL_20]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x64x24xf32>, tensor<64xi64>) -> tensor<10x64x24xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_23]]) {perm = [1, 2, 0]} : (tensor<10x64x24xf32>) -> tensor<64x24x10xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_24]]) {perm = [1, 0, 2]} : (tensor<64x24x10xf32>) -> tensor<24x64x10xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_17]], %[[VAL_16]], %[[VAL_19]], %[[VAL_18]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_15]], %[[VAL_14]], %[[VAL_19]], %[[VAL_18]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_19]], %[[VAL_13]], %[[VAL_19]], %[[VAL_18]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_12]], %[[VAL_11]], %[[VAL_19]], %[[VAL_18]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Slice"(%[[VAL_25]], %[[VAL_10]], %[[VAL_9]], %[[VAL_19]], %[[VAL_18]]) : (tensor<24x64x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_28]], %[[VAL_21]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_27]], %[[VAL_21]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_26]], %[[VAL_21]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_30]], %[[VAL_21]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_29]], %[[VAL_21]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Slice"(%[[VAL_34]], %[[VAL_15]], %[[VAL_8]], %[[VAL_19]], %[[VAL_15]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Slice"(%[[VAL_35]], %[[VAL_15]], %[[VAL_8]], %[[VAL_19]], %[[VAL_15]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Pad"(%[[VAL_36]], %[[VAL_7]], %[[VAL_6]], %[[VAL_5]]) {mode = "constant"} : (tensor<1x24x200xf32>, tensor<6xi64>, tensor<f32>, none) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Pad"(%[[VAL_37]], %[[VAL_7]], %[[VAL_6]], %[[VAL_5]]) {mode = "constant"} : (tensor<1x24x200xf32>, tensor<6xi64>, tensor<f32>, none) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_31]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Reshape"(%[[VAL_32]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Reshape"(%[[VAL_38]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           %[[VAL_44:.*]] = "onnx.Reshape"(%[[VAL_39]], %[[VAL_4]]) {allowzero = 0 : si64} : (tensor<1x24x201xf32>, tensor<4xi64>) -> tensor<1x24x201x1xf32>
// CHECK:           %[[VAL_45:.*]] = "onnx.Concat"(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_44]]) {axis = -1 : si64} : (tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>, tensor<1x24x201x1xf32>) -> tensor<1x24x201x5xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Reshape"(%[[VAL_45]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x201x5xf32>, tensor<3xi64>) -> tensor<1x24x1005xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Slice"(%[[VAL_46]], %[[VAL_17]], %[[VAL_2]], %[[VAL_19]], %[[VAL_15]]) : (tensor<1x24x1005xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x1001xf32>
// CHECK:           onnx.Return %[[VAL_47]] : tensor<1x24x1001xf32>
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
// CHECK-LABEL:   func.func @test_convtrans_stride_2_nodilation(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 400]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<4> : tensor<64xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x4xf32>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.ReverseSequence"(%[[VAL_13]], %[[VAL_11]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x64x24xf32>, tensor<64xi64>) -> tensor<4x64x24xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]) {perm = [1, 2, 0]} : (tensor<4x64x24xf32>) -> tensor<64x24x4xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0, 2]} : (tensor<64x24x4xf32>) -> tensor<24x64x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_9]], %[[VAL_8]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Slice"(%[[VAL_16]], %[[VAL_7]], %[[VAL_6]], %[[VAL_10]], %[[VAL_10]]) : (tensor<24x64x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_18]], %[[VAL_12]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_17]], %[[VAL_12]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_19]], %[[VAL_7]], %[[VAL_5]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_9]], %[[VAL_4]], %[[VAL_10]], %[[VAL_7]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Reshape"(%[[VAL_21]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Reshape"(%[[VAL_22]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Concat"(%[[VAL_24]], %[[VAL_23]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x2xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Reshape"(%[[VAL_25]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x2xf32>, tensor<3xi64>) -> tensor<1x24x400xf32>
// CHECK:           onnx.Return %[[VAL_26]] : tensor<1x24x400xf32>
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
// CHECK-LABEL:   func.func @test_convtrans_stride_4_lrelu(
// CHECK-SAME:                                             %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<11> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<9> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_14]], %[[VAL_8]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_7]], %[[VAL_6]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.LeakyRelu"(%[[VAL_25]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_21]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.LeakyRelu"(%[[VAL_27]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.LeakyRelu"(%[[VAL_29]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.LeakyRelu"(%[[VAL_31]]) {alpha = 1.000000e-01 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_26]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Slice"(%[[VAL_28]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Slice"(%[[VAL_32]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Concat"(%[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_40]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return %[[VAL_42]] : tensor<1x24x800xf32>
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
// CHECK-LABEL:   func.func @test_convtrans_stride_4_relu(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<11> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<9> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_14]], %[[VAL_8]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_7]], %[[VAL_6]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Relu"(%[[VAL_25]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_21]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Relu"(%[[VAL_27]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Relu"(%[[VAL_29]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Relu"(%[[VAL_31]]) : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_26]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Slice"(%[[VAL_28]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Slice"(%[[VAL_32]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Concat"(%[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_40]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return %[[VAL_42]] : tensor<1x24x800xf32>
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
// CHECK-LABEL:   func.func @test_convtrans_stride_4_lrelu_default_value(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: tensor<1x64x200xf32>,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<[1, 24, 800]> : tensor<3xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<[1, 24, 200, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<201> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = onnx.Constant dense<200> : tensor<1xi64>
// CHECK:           %[[VAL_6:.*]] = onnx.Constant dense<11> : tensor<1xi64>
// CHECK:           %[[VAL_7:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_8:.*]] = onnx.Constant dense<10> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = onnx.Constant dense<9> : tensor<1xi64>
// CHECK:           %[[VAL_10:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_11:.*]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK:           %[[VAL_12:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_13:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_14:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = onnx.Constant dense<8> : tensor<64xi64>
// CHECK:           %[[VAL_16:.*]] = onnx.Constant dense<2.000000e-02> : tensor<24xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_1]]) {perm = [2, 0, 1]} : (tensor<64x24x8xf32>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.ReverseSequence"(%[[VAL_17]], %[[VAL_15]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<8x64x24xf32>, tensor<64xi64>) -> tensor<8x64x24xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 2, 0]} : (tensor<8x64x24xf32>) -> tensor<64x24x8xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0, 2]} : (tensor<64x24x8xf32>) -> tensor<24x64x8xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_12]], %[[VAL_11]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_10]], %[[VAL_9]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_14]], %[[VAL_8]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Slice"(%[[VAL_20]], %[[VAL_7]], %[[VAL_6]], %[[VAL_14]], %[[VAL_13]]) : (tensor<24x64x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x64x2xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_22]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.LeakyRelu"(%[[VAL_25]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_21]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.LeakyRelu"(%[[VAL_27]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_24]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.LeakyRelu"(%[[VAL_29]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Conv"(%[[VAL_0]], %[[VAL_23]], %[[VAL_16]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [2], pads = [1, 1], strides = [1]} : (tensor<1x64x200xf32>, tensor<24x64x2xf32>, tensor<24xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.LeakyRelu"(%[[VAL_31]]) {alpha = 0.00999999977 : f32} : (tensor<1x24x201xf32>) -> tensor<1x24x201xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Slice"(%[[VAL_26]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Slice"(%[[VAL_28]], %[[VAL_12]], %[[VAL_5]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Slice"(%[[VAL_30]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Slice"(%[[VAL_32]], %[[VAL_10]], %[[VAL_4]], %[[VAL_14]], %[[VAL_10]]) : (tensor<1x24x201xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24x200xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Reshape"(%[[VAL_33]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Reshape"(%[[VAL_34]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Reshape"(%[[VAL_35]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Reshape"(%[[VAL_36]], %[[VAL_3]]) {allowzero = 0 : si64} : (tensor<1x24x200xf32>, tensor<4xi64>) -> tensor<1x24x200x1xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Concat"(%[[VAL_37]], %[[VAL_38]], %[[VAL_39]], %[[VAL_40]]) {axis = -1 : si64} : (tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>, tensor<1x24x200x1xf32>) -> tensor<1x24x200x4xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Reshape"(%[[VAL_41]], %[[VAL_2]]) {allowzero = 0 : si64} : (tensor<1x24x200x4xf32>, tensor<3xi64>) -> tensor<1x24x800xf32>
// CHECK:           onnx.Return %[[VAL_42]] : tensor<1x24x800xf32>
// CHECK:         }
// DISABLED-LABEL: test_convtrans_stride_4_lrelu
// DISABLED: onnx.ConvTranspose
