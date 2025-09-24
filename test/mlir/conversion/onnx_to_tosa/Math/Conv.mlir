// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_stride_13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x98x98xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x3x256x256xf32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_1_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x3x64x64xf32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.conv2d [[VAR_0_]], [[VAR_1_]], [[PARAM_2_]], [[VAR_2_]], [[VAR_2_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x98x98x2xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_3_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x98x98x2xf32>) -> tensor<5x2x98x98xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x2x98x98xf32>
// CHECK:         }
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>) ->  tensor<5x2x197x199xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x197x199xf32>
  return %0 : tensor<5x2x197x199xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_onnx_conv2d_novalue
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>) -> tensor<5x2x197x199xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x3x256x256xf32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_1_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x3x64x64xf32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.conv2d [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_3_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 3, 2, 4>, stride = array<i64: 1, 1>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x197x199x2xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x197x199x2xf32>) -> tensor<5x2x197x199xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x2x197x199xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<7x3x64x64xf32>) ->   tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [12, 12]} : (tensor<5x3x256x256xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<*xf32>
  return %0 :  tensor<*xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_no_dilation_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<7x3x64x64xf32>) -> tensor<5x7x17x17xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x3x256x256xf32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_1_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<7x3x64x64xf32>) -> tensor<7x64x64x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<7xf32>}> : () -> tensor<7xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.conv2d [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_3_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 12, 12>} : (tensor<5x256x256x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x17x17x7xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_4_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x17x17x7xf32>) -> tensor<5x7x17x17xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x7x17x17xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x256x260xf32>, %arg1 : tensor<2x3x60x64xf32>) ->  tensor<5x2x197x197xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) : (tensor<5x3x256x260xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x197x197xf32>
  return %0 : tensor<5x2x197x197xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_no_dilation_pad_stride(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: tensor<5x3x256x260xf32>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: tensor<2x3x60x64xf32>) -> tensor<5x2x197x197xf32> {
// CHECK:           %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x3x256x260xf32>) -> tensor<5x256x260x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x3x60x64xf32>) -> tensor<2x60x64x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_9]], %[[VAL_9]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x197x197x2xf32>
// CHECK:           %[[VAL_8:.*]] = tosa.transpose %[[VAL_6]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x197x197x2xf32>) -> tensor<5x2x197x197xf32>
// CHECK:           return %[[VAL_8]] : tensor<5x2x197x197xf32>
}

// -----
func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) ->  tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 1, 1, 1], strides = [3, 3], group = 4 : si64} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) ->  tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x64x256x256xf32>, [[PARAM_1_:%.+]]: tensor<12x16x45x45xf32>, [[PARAM_2_:%.+]]: tensor<12xf32>) -> tensor<5x12x72x72xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.transpose [[PARAM_0_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x64x256x256xf32>) -> tensor<5x256x256x64xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_1_]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<12x16x45x45xf32>) -> tensor<12x45x45x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<[5, 256, 256, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.slice [[VAR_0_]], [[VAR_2_]], [[VAR_3_]] : (tensor<5x256x256x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {values = dense<[3, 45, 45, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_2_]], [[VAR_5_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {values = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_7_]], [[VAR_8_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.conv2d [[VAR_4_]], [[VAR_6_]], [[VAR_9_]], [[VAR_10_]], [[VAR_10_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x72x72x3xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.slice [[VAR_0_]], [[VAR_12_]], [[VAR_3_]] : (tensor<5x256x256x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.const_shape  {values = dense<[3, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_1_]]4, [[VAR_5_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_8_]], [[VAR_8_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = tosa.conv2d [[VAR_13_]], [[VAR_15_]], [[VAR_16_]], [[VAR_10_]], [[VAR_10_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x72x72x3xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 32]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = tosa.slice [[VAR_0_]], [[VAR_18_]], [[VAR_3_]] : (tensor<5x256x256x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = tosa.const_shape  {values = dense<[6, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_20_]], [[VAR_5_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = tosa.const_shape  {values = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_23_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_22_]], [[VAR_8_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = tosa.conv2d [[VAR_19_]], [[VAR_21_]], [[VAR_23_]], [[VAR_10_]], [[VAR_10_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x72x72x3xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = tosa.const_shape  {values = dense<[0, 0, 0, 48]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = tosa.slice [[VAR_0_]], [[VAR_25_]], [[VAR_3_]] : (tensor<5x256x256x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = tosa.const_shape  {values = dense<[9, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_27_]], [[VAR_5_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = tosa.const_shape  {values = dense<9> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_30_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_29_]], [[VAR_8_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK:           [[VAR_31_:%.+]] = tosa.conv2d [[VAR_26_]], [[VAR_28_]], [[VAR_30_]], [[VAR_10_]], [[VAR_10_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x72x72x3xf32>
// CHECK:           [[VAR_32_:%.+]] = tosa.concat [[VAR_11_]], [[VAR_17_]], [[VAR_24_]], [[VAR_31_]] {axis = 3 : i32} : (tensor<5x72x72x3xf32>, tensor<5x72x72x3xf32>, tensor<5x72x72x3xf32>, tensor<5x72x72x3xf32>) -> tensor<5x72x72x12xf32>
// CHECK:           [[VAR_33_:%.+]] = tosa.transpose [[VAR_32_]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x72x72x12xf32>) -> tensor<5x12x72x72xf32>
// CHECK:           return [[VAR_33_]] : tensor<5x12x72x72xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_autopad(%arg0: tensor<5x3x125x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x125x256xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER"} : (tensor<5x3x125x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x125x256xf32>
  return %0 : tensor<5x2x125x256xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_autopad(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<5x3x125x256xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<2x3x64x64xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: tensor<2xf32>) -> tensor<5x2x125x256xf32> {
// CHECK:           %[[VAL_3:.*]] = tosa.transpose %[[VAL_0]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<5x3x125x256xf32>) -> tensor<5x125x256x3xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x3x64x64xf32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.conv2d %[[VAL_3]], %[[VAL_4]], %[[VAL_2]], %[[ZERO]], %[[ZERO]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 32, 31, 32, 31>, stride = array<i64: 1, 1>} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x125x256x2xf32>
// CHECK:           %[[VAL_6:.*]] = tosa.transpose %[[VAL_5]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<5x125x256x2xf32>) -> tensor<5x2x125x256xf32>
// CHECK:           return %[[VAL_6]] : tensor<5x2x125x256xf32>
}
