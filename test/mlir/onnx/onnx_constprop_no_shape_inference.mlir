// RUN: onnx-mlir-opt --decompose-onnx --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Split tests

// -----

// COM: split input is not specified

// CHECK-LABEL: @test_split_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func.func @test_split_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %cst) { axis = 0 : si64} : (tensor<2x10xf32>, none) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "onnx.Return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// COM: split input is not specified

// CHECK-LABEL: @test_split_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func.func @test_split_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.Split"(%0, %cst) { axis = 1 : si64} : (tensor<2x10xf32>, none) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "onnx.Return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>
  // CHECK: {{.*}}  = onnx.Constant dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.Split"{{.*}}
}

// -----

// COM: split attribute is not specified

// CHECK-LABEL: @test_splitv11_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
func.func @test_splitv11_axis_0_no_splitattr() -> (tensor<1x10xf32>, tensor<1x10xf32>) {
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 0 : si64} : (tensor<2x10xf32>) -> (tensor<1x10xf32>, tensor<1x10xf32>)
  "onnx.Return"(%1, %2) : (tensor<1x10xf32>, tensor<1x10xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<1x10xf32>
  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<1x10xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}

// -----

// COM: split attribute is not specified

// CHECK-LABEL: @test_splitv11_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
func.func @test_splitv11_axis_1_no_splitattr() -> (tensor<2x5xf32>, tensor<2x5xf32>) {
  %0 = onnx.Constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]> : tensor<2x10xf32>
  %1, %2 = "onnx.SplitV11"(%0) { axis = 1 : si64} : (tensor<2x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>)
  "onnx.Return"(%1, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> ()

  // CHECK: {{.*}} = onnx.Constant dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01]]> : tensor<2x5xf32>
  // CHECK: {{.*}}  = onnx.Constant dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01]]> : tensor<2x5xf32>
  // CHECK-NOT: {{.*}} = "onnx.SplitV11"{{.*}}
}

// -----

//===----------------------------------------------------------------------===//
/// scatternd tests

// CHECK-LABEL: @test_scatternd_f32()
func.func @test_scatternd_f32() -> (tensor<8xf32>) {
  %0 = onnx.Constant { name = "constant.0", value = dense<[1., 2., 3., 4., 5., 6., 7., 8.]>:tensor<8xf32> } : tensor<8xf32>
  %1 = onnx.Constant { name = "constant.1", value = dense< [[4], [3], [1], [7]]>:tensor<4x1xi64> } : tensor<4x1xi64>
  %2 = onnx.Constant { name = "constant.2", value = dense<[9., 10., 11., 12.]>:tensor<4xf32> } : tensor<4xf32>
  // CHECK : [[R1:%.+]] = "onnx.Constant"{{.*}} dense<{{\[}}1.000000e+00, 1.100000e+01, 3.000000e+00, 1.000000e+01, 9.000000e+00, 6.000000e+00, 7.000000e+00, 1.200000e+01],
  // CHECK-NEXT : onnx.Return [[R1]] : tensor<8xf32>
  %3 = "onnx.ScatterND"(%0, %1, %2) {node_name = "ScatterND_6467", node_type = "ScatterND"} : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>
  onnx.Return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @test_scatternd_i32()
func.func @test_scatternd_i32() -> (tensor<4x4x4xi32>) {
  %0 = "onnx.Constant"() { name = "constant.0", value = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]>:tensor<4x4x4xi32> } : () -> tensor<4x4x4xi32>
  %1 = onnx.Constant { name = "constant.1", value = dense<[[0], [2]]>:tensor<2x1xi64> } : tensor<2x1xi64>
  %2 = "onnx.Constant"() { name = "constant.2", value = dense<[[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]>:tensor<2x4x4xi32> } : () -> tensor<2x4x4xi32>
  // CHECK : [[R1:%.+]] = "onnx.Constant"{{.*}} dense<{{\[\[\[}}5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]], [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]>
  // CHECK-NEXT : onnx.Return [[R1]] : tensor<4x4x4xi32>
  %3 = "onnx.ScatterND"(%0, %1, %2) {node_name = "ScatterND_6467", node_type = "ScatterND"} : (tensor<4x4x4xi32>, tensor<2x1xi64>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>
  onnx.Return %3 : tensor<4x4x4xi32>
}

// -----
