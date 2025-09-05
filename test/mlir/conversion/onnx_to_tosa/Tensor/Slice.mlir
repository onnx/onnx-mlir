// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="convert-slice-only-when-step-one=true" -cse %s -split-input-file | FileCheck %s --check-prefix=ONLY-STEP1


func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_constant_default_steps
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_negative
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_end_outofbound
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

// -----

func.func @slice_all_dynamic(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<1xi64>,
                             %arg2: tensor<1xi64>,
                             %arg3: tensor<1xi64>,
                             %arg4: tensor<1xi64>)
                              -> tensor<20x9x5xf32> {
  %0 = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<20x10x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<20x9x5xf32>
  return %0 : tensor<20x9x5xf32>
}

// -----

func.func @slice_default_axes(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<3xi64>,
                             %arg2: tensor<3xi64>)
                              -> tensor<20x10x1xf32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %1 = onnx.Constant dense<1> : tensor<3xi64>
  %2 = "onnx.Slice"(%arg0, %arg1, %arg2, %0, %1) {onnx_node_name = "onnx.Slice_0"} : (tensor<20x10x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<20x10x1xf32>
  return %2 : tensor<20x10x1xf32>
}

// -----

func.func @slice_just_steps(%arg0: tensor<100x200xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL: func @slice_just_steps
// CHECK: %0 = tosa.reshape %arg0 {new_shape = array<i64: 20, 5, 20, 10>} : (tensor<100x200xf32>) -> tensor<20x5x20x10xf32>
// CHECK: %1 = tosa.slice %0 {size = array<i64: 20, 1, 20, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<20x5x20x10xf32>) -> tensor<20x1x20x1xf32>
// CHECK: %2 = tosa.reshape %1 {new_shape = array<i64: 20, 20>} : (tensor<20x1x20x1xf32>) -> tensor<20x20xf32>
// CHECK: return %2 : tensor<20x20xf32>

// ONLY-STEP1-LABEL: func @slice_just_steps
// ONLY-STEP1: tosa.slice

// -----

func.func @slice_steps_and_edges(%arg0: tensor<100x200xf32>) -> tensor<16x17xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[82, 178]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x17xf32>
  return %1 : tensor<16x17xf32> 
}
// CHECK-LABEL: func @slice_steps_and_edges
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 80, 170>, start = array<i64: 5, 10>} : (tensor<100x200xf32>) -> tensor<80x170xf32>
// CHECK: %1 = tosa.reshape %0 {new_shape = array<i64: 16, 5, 17, 10>} : (tensor<80x170xf32>) -> tensor<16x5x17x10xf32>
// CHECK: %2 = tosa.slice %1 {size = array<i64: 16, 1, 17, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<16x5x17x10xf32>) -> tensor<16x1x17x1xf32>
// CHECK: %3 = tosa.reshape %2 {new_shape = array<i64: 16, 17>} : (tensor<16x1x17x1xf32>) -> tensor<16x17xf32>
// CHECK: return %3 : tensor<16x17xf32>

// ONLY-STEP1-LABEL: func @slice_steps_and_edges
// ONLY-STEP1: tosa.slice

// -----

func.func @slice_steps_and_edges_with_padding(%arg0: tensor<99x195xf32>) -> tensor<19x19xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[97, 192]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<99x195xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<19x19xf32>
  return %1 : tensor<19x19xf32> 
}
// CHECK-LABEL: func @slice_steps_and_edges_with_padding
// CHECK: %0 = tosa.const_shape {value = dense<[0, 1, 0, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %1 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: %2 = tosa.pad %arg0, %0, %1 : (tensor<99x195xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<100x200xf32>
// CHECK: %3 = tosa.slice %2 {size = array<i64: 95, 190>, start = array<i64: 5, 10>} : (tensor<100x200xf32>) -> tensor<95x190xf32>
// CHECK: %4 = tosa.reshape %3 {new_shape = array<i64: 19, 5, 19, 10>} : (tensor<95x190xf32>) -> tensor<19x5x19x10xf32>
// CHECK: %5 = tosa.slice %4 {size = array<i64: 19, 1, 19, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<19x5x19x10xf32>) -> tensor<19x1x19x1xf32>
// CHECK: %6 = tosa.reshape %5 {new_shape = array<i64: 19, 19>} : (tensor<19x1x19x1xf32>) -> tensor<19x19xf32>
// CHECK: return %6 : tensor<19x19xf32>

// ONLY-STEP1-LABEL: func @slice_steps_and_edges_with_padding
// ONLY-STEP1: tosa.slice

// -----

func.func @slice_just_steps_with_padding(%arg0: tensor<99x195xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[99, 195]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<99x195xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL: func @slice_just_steps_with_padding
// CHECK: %0 = tosa.const_shape {value = dense<[0, 1, 0, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %1 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: %2 = tosa.pad %arg0, %0, %1 : (tensor<99x195xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<100x200xf32>
// CHECK: %3 = tosa.reshape %2 {new_shape = array<i64: 20, 5, 20, 10>} : (tensor<100x200xf32>) -> tensor<20x5x20x10xf32>
// CHECK: %4 = tosa.slice %3 {size = array<i64: 20, 1, 20, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<20x5x20x10xf32>) -> tensor<20x1x20x1xf32>
// CHECK: %5 = tosa.reshape %4 {new_shape = array<i64: 20, 20>} : (tensor<20x1x20x1xf32>) -> tensor<20x20xf32>
// CHECK: return %5 : tensor<20x20xf32>

// ONLY-STEP1-LABEL: func @slice_just_steps_with_padding
// ONLY-STEP1: tosa.slice

// -----

func.func @slice_negative_steps(%arg0: tensor<100x200xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[-5, -10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL: func @slice_negative_steps
// CHECK: onnx.Slice

// -----

func.func @slice_start_greater_than_dim(%arg0: tensor<10x30xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[20, 20]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[21,21]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<0> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<10x30xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32> 
}
// CHECK-LABEL: func @slice_start_greater_than_dim
// CHECK: onnx.Slice

// -----

func.func @slice_step_greater_than_dim(%arg0: tensor<9x30xf32>) -> tensor<1x2xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[8, 25]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[10, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<9x30xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32> 
}
// CHECK-LABEL: func @slice_step_greater_than_dim
// CHECK: %0 = tosa.const_shape {value = dense<[0, 6, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %1 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: %2 = tosa.pad %arg0, %0, %1 : (tensor<9x30xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<15x30xf32>
// CHECK: %3 = tosa.slice %2 {size = array<i64: 10, 20>, start = array<i64: 5, 5>} : (tensor<15x30xf32>) -> tensor<10x20xf32>
// CHECK: %4 = tosa.reshape %3 {new_shape = array<i64: 1, 10, 2, 10>} : (tensor<10x20xf32>) -> tensor<1x10x2x10xf32>
// CHECK: %5 = tosa.slice %4 {size = array<i64: 1, 1, 2, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<1x10x2x10xf32>) -> tensor<1x1x2x1xf32>
// CHECK: %6 = tosa.reshape %5 {new_shape = array<i64: 1, 2>} : (tensor<1x1x2x1xf32>) -> tensor<1x2xf32>
// CHECK: return %6 : tensor<1x2xf32>

// -----

func.func @slice_4d(%arg0: tensor<1x56x56x92xf32>) -> tensor<1x28x28x92xf32> {
  %axes = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[56,56]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x56x56x92xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x28x28x92xf32>
  return %1 : tensor<1x28x28x92xf32> 
}
// CHECK-LABEL: func @slice_4d
// CHECK: %0 = tosa.const_shape {value = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK: %1 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK: %2 = tosa.pad %arg0, %0, %1 : (tensor<1x56x56x92xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<1x57x57x92xf32>
// CHECK: %3 = tosa.slice %2 {size = array<i64: 1, 56, 56, 92>, start = array<i64: 0, 1, 1, 0>} : (tensor<1x57x57x92xf32>) -> tensor<1x56x56x92xf32>
// CHECK: %4 = tosa.reshape %3 {new_shape = array<i64: 1, 28, 2, 28, 2, 92>} : (tensor<1x56x56x92xf32>) -> tensor<1x28x2x28x2x92xf32>
// CHECK: %5 = tosa.slice %4 {size = array<i64: 1, 28, 1, 28, 1, 92>, start = array<i64: 0, 0, 0, 0, 0, 0>} : (tensor<1x28x2x28x2x92xf32>) -> tensor<1x28x1x28x1x92xf32>
// CHECK: %6 = tosa.reshape %5 {new_shape = array<i64: 1, 28, 28, 92>} : (tensor<1x28x1x28x1x92xf32>) -> tensor<1x28x28x92xf32>
// CHECK: return %6 : tensor<1x28x28x92xf32>

// ONLY-STEP1-LABEL: func @slice_4d
// ONLY-STEP1: tosa.slice

// -----

func.func @slice_4d_step2_1sliced_dim(%arg0: tensor<1x3x640x640xbf16>) -> tensor<1x3x320x640xbf16> {
  %starts = "tosa.const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
  %ends = "tosa.const"() <{value = dense<9223372036854775807> : tensor<1xi64>}> : () -> tensor<1xi64>
  %axes_steps = "tosa.const"() <{value = dense<2> : tensor<1xi64>}> : () -> tensor<1xi64>
  %3 = "onnx.Slice"(%arg0, %starts, %ends, %axes_steps, %axes_steps) : (tensor<1x3x640x640xbf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x320x640xbf16>
  return %3 : tensor<1x3x320x640xbf16>
}
// CHECK-LABEL: func @slice_4d_step2_1sliced_dim
// CHECK-SAME:                (%[[ARG_0:.*]]: tensor<1x3x640x640xbf16>) -> tensor<1x3x320x640xbf16>
// CHECK: %[[VAL_0:.*]] = tosa.reshape %[[ARG_0]] {new_shape = array<i64: 1, 3, 320, 2, 640>} : (tensor<1x3x640x640xbf16>) -> tensor<1x3x320x2x640xbf16>
// CHECK: %[[VAL_1:.*]] = tosa.slice %[[VAL_0]] {size = array<i64: 1, 3, 320, 1, 640>, start = array<i64: 0, 0, 0, 0, 0>} : (tensor<1x3x320x2x640xbf16>) -> tensor<1x3x320x1x640xbf16>
// CHECK: %[[VAL_2:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 1, 3, 320, 640>} : (tensor<1x3x320x1x640xbf16>) -> tensor<1x3x320x640xbf16>
// CHECK: return %[[VAL_2]] : tensor<1x3x320x640xbf16>

// ONLY-STEP1-LABEL: func @slice_4d_step2_1sliced_dim
// ONLY-STEP1-NOT: tosa.slice