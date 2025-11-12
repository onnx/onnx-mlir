// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-split-to-slice %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// -----

// Test Split with constant split sizes along axis 0
func.func @test_split_constant_sizes_axis0(%arg0: tensor<10x20xf32>) -> (tensor<3x20xf32>, tensor<7x20xf32>) {
  %split_sizes = onnx.Constant dense<[3, 7]> : tensor<2xi64>
  %0:2 = "onnx.Split"(%arg0, %split_sizes) {axis = 0 : si64} : (tensor<10x20xf32>, tensor<2xi64>) -> (tensor<3x20xf32>, tensor<7x20xf32>)
  return %0#0, %0#1 : tensor<3x20xf32>, tensor<7x20xf32>

// CHECK-LABEL:  func.func @test_split_constant_sizes_axis0
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<10x20xf32>) -> (tensor<3x20xf32>, tensor<7x20xf32>) {
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[10, 20]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[3, 0]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<[3, 20]> : tensor<2xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<10x20xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x20xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<10x20xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<7x20xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]] : tensor<3x20xf32>, tensor<7x20xf32>

// DISABLED-LABEL:  func.func @test_split_constant_sizes_axis0
// DISABLED:        "onnx.Split"
}

// -----

// Test Split with constant split sizes along axis 1
func.func @test_split_constant_sizes_axis1(%arg0: tensor<4x199x63x256xf32>) -> (tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x7x63x256xf32>) {
  %split_sizes = onnx.Constant dense<[32, 32, 32, 32, 32, 32, 7]> : tensor<7xi64>
  %0:7 = "onnx.Split"(%arg0, %split_sizes) {axis = 1 : si64} : (tensor<4x199x63x256xf32>, tensor<7xi64>) -> (tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x7x63x256xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6 : tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x7x63x256xf32>

// CHECK-LABEL:  func.func @test_split_constant_sizes_axis1
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<4x199x63x256xf32>) -> (tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x32x63x256xf32>, tensor<4x7x63x256xf32>) {
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<4xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<[4, 32, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[0, 32, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[4, 64, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS3:%.+]] = onnx.Constant dense<[0, 64, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS3:%.+]] = onnx.Constant dense<[4, 96, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS4:%.+]] = onnx.Constant dense<[0, 96, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS4:%.+]] = onnx.Constant dense<[4, 128, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS5:%.+]] = onnx.Constant dense<[0, 128, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS5:%.+]] = onnx.Constant dense<[4, 160, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS6:%.+]] = onnx.Constant dense<[0, 160, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS6:%.+]] = onnx.Constant dense<[4, 192, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[STARTS7:%.+]] = onnx.Constant dense<[0, 192, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[ENDS7:%.+]] = onnx.Constant dense<[4, 199, 63, 256]> : tensor<4xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<4xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE3:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS3]], [[ENDS3]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE4:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS4]], [[ENDS4]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE5:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS5]], [[ENDS5]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE6:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS6]], [[ENDS6]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x32x63x256xf32>
// CHECK:           [[SLICE7:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS7]], [[ENDS7]], [[AXES]], [[STEPS]]) : (tensor<4x199x63x256xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<4x7x63x256xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]], [[SLICE3]], [[SLICE4]], [[SLICE5]], [[SLICE6]], [[SLICE7]]

// DISABLED-LABEL:  func.func @test_split_constant_sizes_axis1
// DISABLED:        "onnx.Split"
}

// -----

// Test Split with equal sizes (no split parameter)
func.func @test_split_equal_sizes(%arg0: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:2 = "onnx.Split"(%arg0, %none) {axis = 0 : si64} : (tensor<8x4xf32>, none) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>

// CHECK-LABEL:  func.func @test_split_equal_sizes
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[8, 4]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[4, 0]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<4> : tensor<2xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<8x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x4xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<8x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x4xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]] : tensor<4x4xf32>, tensor<4x4xf32>

// DISABLED-LABEL:  func.func @test_split_equal_sizes
// DISABLED:        "onnx.Split"
}

// -----

// Test Split with negative axis
func.func @test_split_negative_axis(%arg0: tensor<10x20x30xf32>) -> (tensor<10x20x10xf32>, tensor<10x20x10xf32>, tensor<10x20x10xf32>) {
  %split_sizes = onnx.Constant dense<[10, 10, 10]> : tensor<3xi64>
  %0:3 = "onnx.Split"(%arg0, %split_sizes) {axis = -1 : si64} : (tensor<10x20x30xf32>, tensor<3xi64>) -> (tensor<10x20x10xf32>, tensor<10x20x10xf32>, tensor<10x20x10xf32>)
  return %0#0, %0#1, %0#2 : tensor<10x20x10xf32>, tensor<10x20x10xf32>, tensor<10x20x10xf32>

// CHECK-LABEL:  func.func @test_split_negative_axis
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<10x20x30xf32>) -> (tensor<10x20x10xf32>, tensor<10x20x10xf32>, tensor<10x20x10xf32>) {
// CHECK-DAG:       [[ENDS3:%.+]] = onnx.Constant dense<[10, 20, 30]> : tensor<3xi64>
// CHECK-DAG:       [[STARTS3:%.+]] = onnx.Constant dense<[0, 0, 20]> : tensor<3xi64>
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[10, 20, 20]> : tensor<3xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[0, 0, 10]> : tensor<3xi64>
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<3xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<[10, 20, 10]> : tensor<3xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<10x20x30xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<10x20x10xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<10x20x30xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<10x20x10xf32>
// CHECK:           [[SLICE3:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS3]], [[ENDS3]], [[AXES]], [[STEPS]]) : (tensor<10x20x30xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<10x20x10xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]], [[SLICE3]]

// DISABLED-LABEL:  func.func @test_split_negative_axis
// DISABLED:        "onnx.Split"
}

// -----

// Test Split with 2D tensor along axis 1
func.func @test_split_2d_axis1(%arg0: tensor<1x64x256xf32>) -> (tensor<1x64x128xf32>, tensor<1x64x128xf32>) {
  %split_sizes = onnx.Constant dense<[128, 128]> : tensor<2xi64>
  %0:2 = "onnx.Split"(%arg0, %split_sizes) {axis = 2 : si64} : (tensor<1x64x256xf32>, tensor<2xi64>) -> (tensor<1x64x128xf32>, tensor<1x64x128xf32>)
  return %0#0, %0#1 : tensor<1x64x128xf32>, tensor<1x64x128xf32>

// CHECK-LABEL:  func.func @test_split_2d_axis1
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<1x64x256xf32>) -> (tensor<1x64x128xf32>, tensor<1x64x128xf32>) {
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[1, 64, 256]> : tensor<3xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[0, 0, 128]> : tensor<3xi64>
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<3xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<[1, 64, 128]> : tensor<3xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<1x64x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x64x128xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<1x64x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x64x128xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]] : tensor<1x64x128xf32>, tensor<1x64x128xf32>

// DISABLED-LABEL:  func.func @test_split_2d_axis1
// DISABLED:        "onnx.Split"
}

// -----

// Test Split with uneven equal sizes (dimension not divisible by number of outputs)
func.func @test_split_uneven(%arg0: tensor<10x20xf32>) -> (tensor<4x20xf32>, tensor<3x20xf32>, tensor<3x20xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "onnx.Split"(%arg0, %none) {axis = 0 : si64} : (tensor<10x20xf32>, none) -> (tensor<4x20xf32>, tensor<3x20xf32>, tensor<3x20xf32>)
  return %0#0, %0#1, %0#2 : tensor<4x20xf32>, tensor<3x20xf32>, tensor<3x20xf32>

// CHECK-LABEL:  func.func @test_split_uneven
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<10x20xf32>) -> (tensor<4x20xf32>, tensor<3x20xf32>, tensor<3x20xf32>) {
// CHECK-DAG:       [[ENDS3:%.+]] = onnx.Constant dense<[10, 20]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS3:%.+]] = onnx.Constant dense<[7, 0]> : tensor<2xi64>
// CHECK-DAG:       [[ENDS2:%.+]] = onnx.Constant dense<[7, 20]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS2:%.+]] = onnx.Constant dense<[4, 0]> : tensor<2xi64>
// CHECK-DAG:       [[STARTS1:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK-DAG:       [[ENDS1:%.+]] = onnx.Constant dense<[4, 20]> : tensor<2xi64>
// CHECK-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           [[SLICE1:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS1]], [[ENDS1]], [[AXES]], [[STEPS]]) : (tensor<10x20xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x20xf32>
// CHECK:           [[SLICE2:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS2]], [[ENDS2]], [[AXES]], [[STEPS]]) : (tensor<10x20xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x20xf32>
// CHECK:           [[SLICE3:%.+]] = "onnx.Slice"([[INPUT]], [[STARTS3]], [[ENDS3]], [[AXES]], [[STEPS]]) : (tensor<10x20xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x20xf32>
// CHECK:           return [[SLICE1]], [[SLICE2]], [[SLICE3]] : tensor<4x20xf32>, tensor<3x20xf32>, tensor<3x20xf32>

// DISABLED-LABEL:  func.func @test_split_uneven
// DISABLED:        "onnx.Split"
}

