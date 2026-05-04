// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
/// Test lowering of UpsampleAndPad operation to Krnl dialect.
//===----------------------------------------------------------------------===//

// COM: Test basic 2D upsampling and padding.
func.func @test_upsample_and_pad_2d(%arg0 : tensor<1x1x3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<1x1x3x3xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP0:#.+]] = affine_map<(d0) -> (d0 * 2)>
  // CHECK-DAG: [[MAP1:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
  // CHECK-DAG: [[MAP2:#.+]] = affine_map<(d0, d1) -> (d1 * 2)>
  // CHECK-DAG: [[MAP3:#.+]] = affine_map<(d0, d1) -> (d1 * 2 + 1)>
  // CHECK-LABEL: test_upsample_and_pad_2d
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<1x1x7x7xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<1x1x7x7xf32>
  // CHECK: [[LOOP:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3) with ([[LOOP]]#0 -> %arg1 = 0 to 1, [[LOOP]]#1 -> %arg2 = 0 to 1, [[LOOP]]#2 -> %arg3 = 0 to 3, [[LOOP]]#3 -> %arg4 = 0 to 3){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3)
  // CHECK:   [[IDX2:%.+]] = affine.apply [[MAP1]]([[IV]]#2)
  // CHECK:   [[IDX3:%.+]] = affine.apply [[MAP3]]([[IV]]#2, [[IV]]#3)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{\]}} : memref<1x1x3x3xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IV]]#1, [[IDX2]], [[IDX3]]{{\]}} : memref<1x1x7x7xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x1x7x7xf32>
}

// -----

// COM: Test with no upsampling (stride=1), only padding.
func.func @test_pad_only(%arg0 : tensor<2x2x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [1, 1], pads = [2, 2, 2, 2]} : (tensor<2x2x4x4xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP0:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-DAG: [[MAP1:#.+]] = affine_map<(d0, d1) -> (d1 + 2)>
  // CHECK-LABEL: test_pad_only
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<2x2x8x8xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<2x2x8x8xf32>
  // CHECK: [[LOOP:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3) with ([[LOOP]]#0 -> %arg1 = 0 to 2, [[LOOP]]#1 -> %arg2 = 0 to 2, [[LOOP]]#2 -> %arg3 = 0 to 4, [[LOOP]]#3 -> %arg4 = 0 to 4){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3)
  // CHECK:   [[IDX2:%.+]] = affine.apply [[MAP0]]([[IV]]#2)
  // CHECK:   [[IDX3:%.+]] = affine.apply [[MAP1]]([[IV]]#2, [[IV]]#3)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{\]}} : memref<2x2x4x4xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IV]]#1, [[IDX2]], [[IDX3]]{{\]}} : memref<2x2x8x8xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<2x2x8x8xf32>
}

// -----

// COM: Test with upsampling only (no padding).
func.func @test_upsample_only(%arg0 : tensor<1x2x3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [3, 3], pads = [0, 0, 0, 0]} : (tensor<1x2x3x3xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP0:#.+]] = affine_map<(d0) -> (d0 * 3)>
  // CHECK-DAG: [[MAP1:#.+]] = affine_map<(d0, d1) -> (d1 * 3)>
  // CHECK-LABEL: test_upsample_only
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<1x2x7x7xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<1x2x7x7xf32>
  // CHECK: [[LOOP:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3) with ([[LOOP]]#0 -> %arg1 = 0 to 1, [[LOOP]]#1 -> %arg2 = 0 to 2, [[LOOP]]#2 -> %arg3 = 0 to 3, [[LOOP]]#3 -> %arg4 = 0 to 3){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3)
  // CHECK:   [[IDX2:%.+]] = affine.apply [[MAP0]]([[IV]]#2)
  // CHECK:   [[IDX3:%.+]] = affine.apply [[MAP1]]([[IV]]#2, [[IV]]#3)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{\]}} : memref<1x2x3x3xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IV]]#1, [[IDX2]], [[IDX3]]{{\]}} : memref<1x2x7x7xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x2x7x7xf32>
}

// -----

// COM: Test with 1D (k=1).
func.func @test_upsample_and_pad_1d(%arg0 : tensor<4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2], pads = [1, 1]} : (tensor<4x5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
  // CHECK-LABEL: test_upsample_and_pad_1d
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<4x11xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<4x11xf32>
  // CHECK: [[LOOP:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1) with ([[LOOP]]#0 -> %arg1 = 0 to 4, [[LOOP]]#1 -> %arg2 = 0 to 5){
  // CHECK:   [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1)
  // CHECK:   [[IDX1:%.+]] = affine.apply [[MAP]]([[IV]]#1)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1{{\]}} : memref<4x5xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IDX1]]{{\]}} : memref<4x11xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<4x11xf32>
}

// -----

// COM: Test with 5D input and k=3.
func.func @test_upsample_and_pad_5d_k3(%arg0 : tensor<2x3x2x2x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2, 2], pads = [1, 1, 1, 1, 1, 1]} : (tensor<2x3x2x2x2xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP0:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
  // CHECK-DAG: [[MAP1:#.+]] = affine_map<(d0, d1) -> (d1 * 2 + 1)>
  // CHECK-DAG: [[MAP2:#.+]] = affine_map<(d0, d1, d2) -> (d2 * 2 + 1)>
  // CHECK-LABEL: test_upsample_and_pad_5d_k3
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<2x3x5x5x5xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<2x3x5x5x5xf32>
  // CHECK: [[LOOP:%.+]]:5 = krnl.define_loops 5
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3, [[LOOP]]#4) with ([[LOOP]]#0 -> %arg1 = 0 to 2, [[LOOP]]#1 -> %arg2 = 0 to 3, [[LOOP]]#2 -> %arg3 = 0 to 2, [[LOOP]]#3 -> %arg4 = 0 to 2, [[LOOP]]#4 -> %arg5 = 0 to 2){
  // CHECK:   [[IV:%.+]]:5 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3, [[LOOP]]#4)
  // CHECK:   [[IDX2:%.+]] = affine.apply [[MAP0]]([[IV]]#2)
  // CHECK:   [[IDX3:%.+]] = affine.apply [[MAP1]]([[IV]]#2, [[IV]]#3)
  // CHECK:   [[IDX4:%.+]] = affine.apply [[MAP2]]([[IV]]#2, [[IV]]#3, [[IV]]#4)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3, [[IV]]#4{{\]}} : memref<2x3x2x2x2xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IV]]#1, [[IDX2]], [[IDX3]], [[IDX4]]{{\]}} : memref<2x3x5x5x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<2x3x5x5x5xf32>
}

// -----

// COM: Test with asymmetric padding.
func.func @test_upsample_and_pad_asymmetric(%arg0 : tensor<1x1x2x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 2, 3, 4]} : (tensor<1x1x2x2xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: [[MAP0:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
  // CHECK-DAG: [[MAP1:#.+]] = affine_map<(d0, d1) -> (d1 * 2 + 2)>
  // CHECK-LABEL: test_upsample_and_pad_asymmetric
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}} : memref<1x1x7x9xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.memset [[RES]], [[CST]] : memref<1x1x7x9xf32>
  // CHECK: [[LOOP:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3) with ([[LOOP]]#0 -> %arg1 = 0 to 1, [[LOOP]]#1 -> %arg2 = 0 to 1, [[LOOP]]#2 -> %arg3 = 0 to 2, [[LOOP]]#3 -> %arg4 = 0 to 2){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2, [[LOOP]]#3)
  // CHECK:   [[IDX2:%.+]] = affine.apply [[MAP0]]([[IV]]#2)
  // CHECK:   [[IDX3:%.+]] = affine.apply [[MAP1]]([[IV]]#2, [[IV]]#3)
  // CHECK:   [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{\]}} : memref<1x1x2x2xf32>
  // CHECK:   krnl.store [[LOAD]], [[RES]]{{\[}}[[IV]]#0, [[IV]]#1, [[IDX2]], [[IDX3]]{{\]}} : memref<1x1x7x9xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x1x7x9xf32>
}