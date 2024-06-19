// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa=grouped-conv-threshold=4 -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32>
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_stride_13(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<5x3x256x256xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<2x3x64x64xf32>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: tensor<2xf32>) -> tensor<5x2x15x15xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_3]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_3]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 245, 245, 3>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x256x256x3xf32>) -> tensor<5x245x245x3xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = tosa.conv2d %[[VAL_6]], %[[VAL_5]], %[[VAL_2]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x245x245x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x15x15x2xf32>
// CHECK-DAG:       %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_7]], %[[VAL_8]] : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32>
// CHECK:           return %[[VAL_9]] : tensor<5x2x15x15xf32>
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
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 3, 2, 4>, stride = array<i64: 1, 1>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x197x199x2xf32>
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
// CHECK-DAG:       %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_2]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_2]] : (tensor<7x3x64x64xf32>, tensor<4xi32>) -> tensor<7x64x64x3xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xf32>}> : () -> tensor<7xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_3]] {size = array<i64: 5, 246, 246, 3>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x256x256x3xf32>) -> tensor<5x246x246x3xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.conv2d %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x246x246x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>) -> tensor<5x15x15x7xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.transpose %[[VAL_7]], %[[VAL_8]] : (tensor<5x15x15x7xf32>, tensor<4xi32>) -> tensor<5x7x15x15xf32>
// CHECK:           return %[[VAL_9]] : tensor<5x7x15x15xf32>
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
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>) -> tensor<5x197x197x2xf32>
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
// CHECK-DAG:       %[[VAL_4:.*]] = tosa.transpose %[[VAL_0]], %[[VAL_3]] : (tensor<5x64x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x64xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]], %[[VAL_3]] : (tensor<12x16x45x45xf32>, tensor<4xi32>) -> tensor<12x45x45x16xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.slice %[[VAL_4]] {size = array<i64: 5, 252, 252, 64>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x256x256x64xf32>) -> tensor<5x252x252x64xf32>
// CHECK:           %[[VAL_7:.*]] = tosa.slice %[[VAL_6]] {size = array<i64: 5, 252, 252, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x252x252x64xf32>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       %[[VAL_8:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       %[[VAL_9:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 0>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       %[[VAL_10:.*]] = tosa.conv2d %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       %[[VAL_11:.*]] = tosa.slice %[[VAL_6]] {size = array<i64: 5, 252, 252, 16>, start = array<i64: 0, 0, 0, 16>} : (tensor<5x252x252x64xf32>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       %[[VAL_12:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 3, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       %[[VAL_13:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 3>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       %[[VAL_14:.*]] = tosa.conv2d %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       %[[VAL_15:.*]] = tosa.slice %[[VAL_6]] {size = array<i64: 5, 252, 252, 16>, start = array<i64: 0, 0, 0, 32>} : (tensor<5x252x252x64xf32>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       %[[VAL_16:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 6, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       %[[VAL_17:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 6>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       %[[VAL_18:.*]] = tosa.conv2d %[[VAL_15]], %[[VAL_16]], %[[VAL_17]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       %[[VAL_19:.*]] = tosa.slice %[[VAL_6]] {size = array<i64: 5, 252, 252, 16>, start = array<i64: 0, 0, 0, 48>} : (tensor<5x252x252x64xf32>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       %[[VAL_20:.*]] = tosa.slice %[[VAL_5]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 9, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       %[[VAL_21:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 3>, start = array<i64: 9>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           %[[VAL_22:.*]] = tosa.conv2d %[[VAL_19]], %[[VAL_20]], %[[VAL_21]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       %[[VAL_23:.*]] = tosa.concat %[[VAL_10]], %[[VAL_14]], %[[VAL_18]], %[[VAL_22]] {axis = 3 : i32} : (tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>) -> tensor<5x17x17x12xf32>
// CHECK-DAG:       %[[VAL_24:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_25:.*]] = tosa.transpose %[[VAL_23]], %[[VAL_24]] : (tensor<5x17x17x12xf32>, tensor<4xi32>) -> tensor<5x12x17x17xf32>
// CHECK:           return %[[VAL_25]] : tensor<5x12x17x17xf32>
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
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_4]], %[[VAL_5]], %[[VAL_2]] {dilation = array<i64: 1, 1>, pad = array<i64: 32, 31, 32, 31>, stride = array<i64: 1, 1>} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x125x256x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]], %[[VAL_7]] : (tensor<5x125x256x2xf32>, tensor<4xi32>) -> tensor<5x2x125x256xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x125x256xf32>
}

// -----
func.func @test_onnx_conv2d_group_higher_4(%arg0: tensor<5x128x256x256xf32>, %arg1 : tensor<16x16x45x45xf32>, %arg2: tensor<16xf32>) ->  tensor<5x16x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 8 : si64, pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x128x256x256xf32>, tensor<16x16x45x45xf32>, tensor<16xf32>) ->  tensor<5x16x17x17xf32>
  return %0 : tensor<5x16x17x17xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_higher_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x128x256x256xf32>, [[PARAM_1_:%.+]]: tensor<16x16x45x45xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<5x16x17x17xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {auto_pad = "NOTSET", group = 8 : si64, pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x128x256x256xf32>, tensor<16x16x45x45xf32>, tensor<16xf32>) -> tensor<5x16x17x17xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x16x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_group_to_depthwise(%arg0: tensor<32x48x112x112xf32>, %arg1 : tensor<48x1x3x3xf32>, %arg2: tensor<48xf32>) ->  tensor<32x48x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 48 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1395", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<32x48x112x112xf32>, tensor<48x1x3x3xf32>, tensor<48xf32>) -> tensor<32x48x112x112xf32>
  return %0 : tensor<32x48x112x112xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_to_depthwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32x48x112x112xf32>, [[PARAM_1_:%.+]]: tensor<48x1x3x3xf32>, [[PARAM_2_:%.+]]: tensor<48xf32>) -> tensor<32x48x112x112xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<32x48x112x112xf32>, tensor<4xi32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_2_]] : (tensor<48x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x48x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 3, 3, 48, 1>} : (tensor<3x3x48x1xf32>) -> tensor<3x3x48x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.depthwise_conv2d [[VAR_1_]], [[VAR_4_]], [[PARAM_2_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<32x112x112x48xf32>, tensor<3x3x48x1xf32>, tensor<48xf32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_7_:%.+]] = tosa.transpose [[VAR_5_]], [[VAR_6_]] : (tensor<32x112x112x48xf32>, tensor<4xi32>) -> tensor<32x48x112x112xf32>
// CHECK:           return [[VAR_7_]] : tensor<32x48x112x112xf32>
}

// -----

func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple(%arg0: tensor<32x24x112x112xf32>, %arg1 : tensor<48x1x3x3xf32>, %arg2: tensor<48xf32>) ->  tensor<32x48x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 24 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1395", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<32x24x112x112xf32>, tensor<48x1x3x3xf32>, tensor<48xf32>) -> tensor<32x48x112x112xf32>
  return %0 : tensor<32x48x112x112xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32x24x112x112xf32>, [[PARAM_1_:%.+]]: tensor<48x1x3x3xf32>, [[PARAM_2_:%.+]]: tensor<48xf32>) -> tensor<32x48x112x112xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<32x24x112x112xf32>, tensor<4xi32>) -> tensor<32x112x112x24xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_2_]] : (tensor<48x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x48x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_3_]] {new_shape = array<i64: 3, 3, 24, 2>} : (tensor<3x3x48x1xf32>) -> tensor<3x3x24x2xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.depthwise_conv2d [[VAR_1_]], [[VAR_4_]], [[PARAM_2_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<32x112x112x24xf32>, tensor<3x3x24x2xf32>, tensor<48xf32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_7_:%.+]] = tosa.transpose [[VAR_5_]], [[VAR_6_]] : (tensor<32x112x112x48xf32>, tensor<4xi32>) -> tensor<32x48x112x112xf32>
// CHECK:           return [[VAR_7_]] : tensor<32x48x112x112xf32>
}

// -----

func.func @test_onnx_conv2d_dyn_shapes(%arg0: tensor<?x?x?x?xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<?x?x?x?xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<?x?x?x?xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_dyn_shapes
// CHECK: onnx.Conv
}

// -----

func.func @test_onnx_conv2d_dyn_shapes_with_shape_inference(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<?x?x?x?xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_dyn_shapes_with_shape_inference
// CHECK: tosa.conv
}