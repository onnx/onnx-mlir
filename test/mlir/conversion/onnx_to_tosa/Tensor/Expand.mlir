// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_expand(%arg0: tensor<1x64x1x1xf32>) -> tensor<1x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 64, 64, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<1x64x64x64xf32>
  return %1 : tensor<1x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x1x1xf32>) -> tensor<1x64x64x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 64, 64, 64]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 64, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_2_:%.+]] = tosa.tile [[PARAM_0_]], [[VAR_1_]] : (tensor<1x64x1x1xf32>, !tosa.shape<4>) -> tensor<1x64x64x64xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x64x64x64xf32>

// -----

func.func @test_expand_splat(%arg0: tensor<1x64x1x1xf32>) -> tensor<64x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<64> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<64x64x64x64xf32>
  return %1 : tensor<64x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x1x1xf32>) -> tensor<64x64x64x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<64> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[64, 1, 64, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_2_:%.+]] = tosa.tile [[PARAM_0_]], [[VAR_1_]] : (tensor<1x64x1x1xf32>, !tosa.shape<4>) -> tensor<64x64x64x64xf32>
// CHECK:           return [[VAR_2_]] : tensor<64x64x64x64xf32>

// -----

func.func @test_expand_new_dims_out(%arg0: tensor<1x64x1xf32>) -> tensor<64x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[64, 64, 64, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1xf32>, tensor<4xi64>) -> tensor<64x64x64x64xf32>
  return %1 : tensor<64x64x64x64xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_out
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x1xf32>) -> tensor<64x64x64x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<64> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[64, 1, 64, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<1x64x1x1xf32>, !tosa.shape<4>) -> tensor<64x64x64x64xf32>
// CHECK:           return [[VAR_3_]] : tensor<64x64x64x64xf32>

// -----

func.func @test_expand_new_dims_start(%arg0: tensor<256x256x16xf32>) -> tensor<1x512x256x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 512, 256, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<256x256x16xf32>, tensor<4xi64>) -> tensor<1x512x256x16xf32>
  return %1 : tensor<1x512x256x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_start
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<256x256x16xf32>) -> tensor<1x512x256x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 512, 256, 16]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 256, 256, 16>} : (tensor<256x256x16xf32>) -> tensor<1x256x256x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<1x256x256x16xf32>, !tosa.shape<4>) -> tensor<1x512x256x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x512x256x16xf32>

// -----

func.func @test_expand_new_dims_mix(%arg0: tensor<128x64xf32>) -> tensor<1x128x16x128x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 128, 16, 128, 16]> : tensor<5xi64>} : () -> tensor<5xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<128x64xf32>, tensor<5xi64>) -> tensor<1x128x16x128x16xf32>
  return %1 : tensor<1x128x16x128x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_new_dims_mix
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x64xf32>) -> tensor<1x128x16x128x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 128, 16, 128, 16]> : tensor<5xi64>}> : () -> tensor<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 128, 1, 64, 1>} : (tensor<128x64xf32>) -> tensor<1x128x1x64x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 16, 2, 16]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<1x128x1x64x1xf32>, !tosa.shape<5>) -> tensor<1x128x16x128x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x128x16x128x16xf32>

// -----

func.func @test_expand_no_tile(%arg0: tensor<128x16xf32>) -> tensor<1x1x128x16xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 1, 128, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<128x16xf32>, tensor<4xi64>) -> tensor<1x1x128x16xf32>
  return %1 : tensor<1x1x128x16xf32>
}

// CHECK-LABEL:  func.func @test_expand_no_tile
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x16xf32>) -> tensor<1x1x128x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 1, 128, 16]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 1, 128, 16>} : (tensor<128x16xf32>) -> tensor<1x1x128x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<1x1x128x16xf32>, !tosa.shape<4>) -> tensor<1x1x128x16xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x1x128x16xf32>

// -----
func.func @test_expand_tile_one_dim_big(%arg0: tensor<1x6x1x1xf32>) -> tensor<1x6x576x672xf32> {
  %0 =  onnx.Constant dense<[1, 1, 576, 672]> : tensor<4xi64> 
  %1 = "onnx.Expand"(%arg0, %0) {onnx_node_name = "Expand_1417"} : (tensor<1x6x1x1xf32>, tensor<4xi64>) -> tensor<1x6x576x672xf32>
  return %1 : tensor<1x6x576x672xf32>
}
// CHECK-LABEL:  func.func @test_expand_tile_one_dim_big
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x6x1x1xf32>) -> tensor<1x6x576x672xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 1, 576, 672]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 576, 672]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_2_:%.+]] = tosa.tile [[PARAM_0_]], [[VAR_1_]] : (tensor<1x6x1x1xf32>, !tosa.shape<4>) -> tensor<1x6x576x672xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x6x576x672xf32>

// -----

func.func @test_expand_smaller_dims(%arg0: tensor<128x64x1x1xf32>) -> tensor<1x128x64x64x128xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 64, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<128x64x1x1xf32>, tensor<3xi64>) -> tensor<1x128x64x64x128xf32>
  return %1 : tensor<1x128x64x64x128xf32>
}

// CHECK-LABEL:  func.func @test_expand_smaller_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x64x1x1xf32>) -> tensor<1x128x64x64x128xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[1, 64, 128]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 128, 64, 1, 1>} : (tensor<128x64x1x1xf32>) -> tensor<1x128x64x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 1, 64, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<1x128x64x1x1xf32>, !tosa.shape<5>) -> tensor<1x128x64x64x128xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x128x64x64x128xf32>

// -----

func.func @test_expand_mixed_smaller(%arg0 : tensor<2x1x6x1xbf16>) -> tensor<2x7x6x5xbf16> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xbf16>, tensor<3xi64>) -> tensor<2x7x6x5xbf16>
  func.return %1 : tensor<2x7x6x5xbf16>
}

// CHECK-LABEL:  func.func @test_expand_mixed_smaller
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x6x1xbf16>) -> tensor<2x7x6x5xbf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[7, 1, 5]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 2, 1, 6, 1>} : (tensor<2x1x6x1xbf16>) -> tensor<2x1x6x1xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 7, 1, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_3_:%.+]] = tosa.tile [[VAR_1_]], [[VAR_2_]] : (tensor<2x1x6x1xbf16>, !tosa.shape<4>) -> tensor<2x7x6x5xbf16>
// CHECK:           return [[VAR_3_]] : tensor<2x7x6x5xbf16>

// -----

func.func @test_expand_no_legalization(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<4xi64>) -> tensor<1x64x64x64xf32> {
  %0 = "onnx.Expand"(%arg0, %arg1) : (tensor<1x64x1x1xf32>, tensor<4xi64>) -> tensor<1x64x64x64xf32>
  return %0 : tensor<1x64x64x64xf32>
}
// CHECK-LABEL:  func @test_expand_no_legalization
// CHECK: onnx.Expand
