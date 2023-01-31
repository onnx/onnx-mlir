// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32>
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_stride_13
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], %arg2) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x15x15x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>) ->  tensor<5x2x197x199xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x197x199xf32>
  return %0 : tensor<5x2x197x199xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_novalue
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad =  [1, 3, 2, 4], stride = [1, 1]} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x197x199x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x197x199x2xf32>, tensor<4xi32>) -> tensor<5x2x197x199xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<7x3x64x64xf32>) ->   tensor<5x7x15x15xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<5x7x15x15xf32>
  return %0 :  tensor<5x7x15x15xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_no_dilation_pad
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<7x3x64x64xf32>, tensor<4xi32>) -> tensor<7x64x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<7xf32>} : () -> tensor<7xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [13, 13]} : (tensor<5x256x256x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>) -> tensor<5x15x15x7xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x15x15x7xf32>, tensor<4xi32>) -> tensor<5x7x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x256x260xf32>, %arg1 : tensor<2x3x60x64xf32>) ->  tensor<5x2x197x197xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) : (tensor<5x3x256x260xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x197x197xf32>
  return %0 : tensor<5x2x197x197xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_no_dilation_pad_stride
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x256x260xf32>, tensor<4xi32>) -> tensor<5x256x260x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x60x64xf32>, tensor<4xi32>) -> tensor<2x60x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>) -> tensor<5x197x197x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x197x197x2xf32>, tensor<4xi32>) -> tensor<5x2x197x197xf32>
}

// -----
func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) ->  tensor<5x12x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 1, 1, 1], strides = [13, 13], group = 4 : si64} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) ->  tensor<5x12x17x17xf32>
  return %0 : tensor<5x12x17x17xf32>
//  CHECK-LABEL: func @test_onnx_conv2d_group
//  CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//  CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x64x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x64xf32>
//  CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//  CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<12x16x45x45xf32>, tensor<4xi32>) -> tensor<12x45x45x16xf32>
//  CHECK: [[INPUTSLICE1:%.+]] = "tosa.slice"([[TRANSINPUT]]) {size = [5, 256, 256, 16], start = [0, 0, 0, 0]} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
//  CHECK: [[KERNELSLICE1:%.+]] = "tosa.slice"([[TRANSKERNEL]]) {size = [3, 45, 45, 16], start = [0, 0, 0, 0]} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
//  CHECK: [[BIASSLICE1:%.+]] = "tosa.slice"(%arg2) {size = [3], start = [0]} : (tensor<12xf32>) -> tensor<3xf32>
//  CHECK: [[OUTPUTSLICE1:%.+]] = "tosa.conv2d"([[INPUTSLICE1]], [[KERNELSLICE1]], [[BIASSLICE1]]) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
//  CHECK: [[INPUTSLICE2:%.+]] = "tosa.slice"([[TRANSINPUT]]) {size = [5, 256, 256, 16], start = [0, 0, 0, 16]} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
//  CHECK: [[KERNELSLICE2:%.+]] = "tosa.slice"([[TRANSKERNEL]]) {size = [3, 45, 45, 16], start = [3, 0, 0, 0]} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
//  CHECK: [[BIASSLICE2:%.+]] = "tosa.slice"(%arg2) {size = [3], start = [3]} : (tensor<12xf32>) -> tensor<3xf32>
//  CHECK: [[OUTPUTSLICE2:%.+]] = "tosa.conv2d"([[INPUTSLICE2]], [[KERNELSLICE2]], [[BIASSLICE2]]) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
//  CHECK: [[INPUTSLICE3:%.+]] = "tosa.slice"([[TRANSINPUT]]) {size = [5, 256, 256, 16], start = [0, 0, 0, 32]} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
//  CHECK: [[KERNELSLICE3:%.+]] = "tosa.slice"([[TRANSKERNEL]]) {size = [3, 45, 45, 16], start = [6, 0, 0, 0]} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
//  CHECK: [[BIASSLICE3:%.+]] = "tosa.slice"(%arg2) {size = [3], start = [6]} : (tensor<12xf32>) -> tensor<3xf32>
//  CHECK: [[OUTPUTSLICE3:%.+]] = "tosa.conv2d"([[INPUTSLICE3]], [[KERNELSLICE3]], [[BIASSLICE3]]) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
//  CHECK: [[INPUTSLICE4:%.+]] = "tosa.slice"([[TRANSINPUT]]) {size = [5, 256, 256, 16], start = [0, 0, 0, 48]} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
//  CHECK: [[KERNELSLICE4:%.+]] = "tosa.slice"([[TRANSKERNEL]]) {size = [3, 45, 45, 16], start = [9, 0, 0, 0]} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
//  CHECK: [[BIASSLICE4:%.+]] = "tosa.slice"(%arg2) {size = [3], start = [9]} : (tensor<12xf32>) -> tensor<3xf32>
//  CHECK: [[OUTPUTSLICE4:%.+]] = "tosa.conv2d"([[INPUTSLICE4]], [[KERNELSLICE4]], [[BIASSLICE4]]) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
//  CHECK: [[OUTPUT:%.+]] = "tosa.concat"([[OUTPUTSLICE1]], [[OUTPUTSLICE2]], [[OUTPUTSLICE3]], [[OUTPUTSLICE4]]) {axis = 3 : i64} : (tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>) -> tensor<5x17x17x12xf32>
//  CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
//  CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x17x17x12xf32>, tensor<4xi32>) -> tensor<5x12x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_autopad(%arg0: tensor<5x3x125x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x125x256xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER"} : (tensor<5x3x125x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x125x256xf32>
  return %0 : tensor<5x2x125x256xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_autopad
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x125x256xf32>, tensor<4xi32>) -> tensor<5x125x256x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], %arg2) {dilation = [1, 1], pad = [32, 31, 32, 31], stride = [1, 1]} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x125x256x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x125x256x2xf32>, tensor<4xi32>) -> tensor<5x2x125x256xf32>
}