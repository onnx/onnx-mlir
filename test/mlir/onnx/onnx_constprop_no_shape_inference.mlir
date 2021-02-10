// RUN: onnx-mlir-opt --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Split tests

// -----

// COM: split attribute is not specified

// CHECK-LABEL: @test_split_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func @test_split_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0) { axis = 0 : si64} : (tensor<2x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "std.return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// COM: split attribute is not specified

// CHECK-LABEL: @test_split_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func @test_split_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>} : () -> tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0) { axis = 1 : si64} : (tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "std.return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK: {{.*}}  = "onnx.Constant"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}
