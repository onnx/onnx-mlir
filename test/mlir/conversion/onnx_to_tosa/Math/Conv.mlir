// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32>
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_stride_13(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<5x3x256x256xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<2x3x64x64xf32>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: tensor<2xf32>) -> tensor<5x2x15x15xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_3]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_3]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_4]], %[[VAL_5]], %[[VAL_2]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x15x15x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>) ->  tensor<5x2x197x199xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x197x199xf32>
  return %0 : tensor<5x2x197x199xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_novalue(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<5x3x256x256xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<2x3x64x64xf32>) -> tensor<5x2x197x199xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_2]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_2]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 3, 2, 4>, stride = array<i64: 1, 1>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x197x199x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x197x199x2xf32>, tensor<4xi32>) -> tensor<5x2x197x199xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x197x199xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<7x3x64x64xf32>) ->   tensor<5x7x15x15xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<5x7x15x15xf32>
  return %0 :  tensor<5x7x15x15xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_no_dilation_pad(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<5x3x256x256xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<7x3x64x64xf32>) -> tensor<5x7x15x15xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_2]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_2]] : (tensor<7x3x64x64xf32>, tensor<4xi32>) -> tensor<7x64x64x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xf32>}> : () -> tensor<7xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x256x256x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>) -> tensor<5x15x15x7xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x15x15x7xf32>, tensor<4xi32>) -> tensor<5x7x15x15xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x7x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x256x260xf32>, %arg1 : tensor<2x3x60x64xf32>) ->  tensor<5x2x197x197xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) : (tensor<5x3x256x260xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x197x197xf32>
  return %0 : tensor<5x2x197x197xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_no_dilation_pad_stride(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: tensor<5x3x256x260xf32>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: tensor<2x3x60x64xf32>) -> tensor<5x2x197x197xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_2]] : (tensor<5x3x256x260xf32>, tensor<4xi32>) -> tensor<5x256x260x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_2]] : (tensor<2x3x60x64xf32>, tensor<4xi32>) -> tensor<2x60x64x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>) -> tensor<5x197x197x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x197x197x2xf32>, tensor<4xi32>) -> tensor<5x2x197x197xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x197x197xf32>
}

// -----
func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) ->  tensor<5x12x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 1, 1, 1], strides = [13, 13], group = 4 : si64} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) ->  tensor<5x12x17x17xf32>
  return %0 : tensor<5x12x17x17xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_group(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<5x64x256x256xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<12x16x45x45xf32>,
// CHECK-SAME:                                      %[[VAL_2:.*]]: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_3]] : (tensor<5x64x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x64xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_3]] : (tensor<12x16x45x45xf32>, tensor<4xi32>) -> tensor<12x45x45x16xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 0>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_9:.*]] = tosa.conv2d %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK:           %[[VAL_10:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 16>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK:           %[[VAL_11:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 3, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK:           %[[VAL_12:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 3>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_13:.*]] = tosa.conv2d %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK:           %[[VAL_14:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 32>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK:           %[[VAL_15:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 6, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK:           %[[VAL_16:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 6>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_17:.*]] = tosa.conv2d %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK:           %[[VAL_18:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 48>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK:           %[[VAL_19:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 9, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK:           %[[VAL_20:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 9>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_21:.*]] = tosa.conv2d %[[VAL_18]], %[[VAL_19]], %[[VAL_20]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.concat %[[VAL_9]], %[[VAL_13]], %[[VAL_17]], %[[VAL_21]] {axis = 3 : i32} : (tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>) -> tensor<5x17x17x12xf32>
// CHECK:           %[[VAL_23:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_24:.*]] = tosa.transpose %[[VAL_22]], %[[VAL_23]] : (tensor<5x17x17x12xf32>, tensor<4xi32>) -> tensor<5x12x17x17xf32>
// CHECK:           return %[[VAL_24]] : tensor<5x12x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_autopad(%arg0: tensor<5x3x125x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x125x256xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER"} : (tensor<5x3x125x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x125x256xf32>
  return %0 : tensor<5x2x125x256xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_autopad(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<5x3x125x256xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<2x3x64x64xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: tensor<2xf32>) -> tensor<5x2x125x256xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_3]] : (tensor<5x3x125x256xf32>, tensor<4xi32>) -> tensor<5x125x256x3xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_3]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_4]], %[[VAL_5]], %[[VAL_2]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 32, 31, 32, 31>, stride = array<i64: 1, 1>} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x125x256x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x125x256x2xf32>, tensor<4xi32>) -> tensor<5x2x125x256xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x125x256xf32>
}