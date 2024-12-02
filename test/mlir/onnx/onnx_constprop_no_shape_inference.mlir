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

//===----------------------------------------------------------------------===//
/// Checks to ensure that constprop does not crash on non static shapes.
/// This does only checks the absence of crashes, not that constants get folded

// binary ops
// CHECK-LABEL: @test_add_dynamic_result
func.func @test_add_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Add"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_sub_dynamic_result
func.func @test_sub_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Sub"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_mul_dynamic_result
func.func @test_mul_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Mul"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_div_dynamic_result
func.func @test_div_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Div"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_bitwise_and_dynamic_result
func.func @test_bitwise_and_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.BitwiseAnd"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_bitwise_or_dynamic_result
func.func @test_bitwise_or_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.BitwiseOr"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_and_dynamic_result
func.func @test_and_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi1> } : tensor<4x4x4xi1>
  %1 = "onnx.And"(%0, %0) : (tensor<4x4x4xi1>, tensor<4x4x4xi1>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_or_dynamic_result
func.func @test_or_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi1> } : tensor<4x4x4xi1>
  %1 = "onnx.Or"(%0, %0) : (tensor<4x4x4xi1>, tensor<4x4x4xi1>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_xor_dynamic_result
func.func @test_xor_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi1> } : tensor<4x4x4xi1>
  %1 = "onnx.And"(%0, %0) : (tensor<4x4x4xi1>, tensor<4x4x4xi1>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_eq_dynamic_result
func.func @test_eq_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi1> } : tensor<4x4x4xi1>
  %1 = "onnx.Equal"(%0, %0) : (tensor<4x4x4xi1>, tensor<4x4x4xi1>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_less_dynamic_result
func.func @test_less_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Less"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_greater_dynamic_result
func.func @test_greater_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Greater"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_less_or_equal_dynamic_result
func.func @test_less_or_equal_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.LessOrEqual"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_greater_or_equal_dynamic_result
func.func @test_greater_or_equal_dynamic_result() -> (tensor<*xi1>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.GreaterOrEqual"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi1>
  onnx.Return %1 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_mod_dynamic_result
func.func @test_mod_dynamic_result() -> (tensor<*xi32>) {
  %0 = onnx.Constant { value = dense<1>: tensor<4x4x4xi32> } : tensor<4x4x4xi32>
  %1 = "onnx.Mod"(%0, %0) : (tensor<4x4x4xi32>, tensor<4x4x4xi32>) -> tensor<*xi32>
  onnx.Return %1 : tensor<*xi32>
}

// -----
// misc ops

// CHECK-LABEL: @test_where_dynamic_result()
func.func @test_where_dynamic_result() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[true, false]> : tensor<2xi1>
  %1 = onnx.Constant dense<[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]> : tensor<3x2xf32>
  %2 = onnx.Constant dense<[[2.0]]> : tensor<1x1xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<1x1xf32>) -> tensor<*xf32>
  "onnx.Return"(%3) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_matmul_2d_dynamic_result()
func.func @test_matmul_2d_dynamic_result() -> (tensor<*xf32>) {
  %0 = "onnx.Constant"() {value = dense<1.> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "onnx.Constant"() {value = dense<1.> : tensor<3x1xf32>} : () -> tensor<3x1xf32>
  %3 = "onnx.MatMul"(%0, %1) : (tensor<2x3xf32>, tensor<3x1xf32>) -> tensor<*xf32>
  onnx.Return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_gemm_dynamic_result()
func.func @test_gemm_dynamic_result() -> (tensor<*xi32>) {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Constant"() {value = dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %2 = "onnx.Constant"() {value = dense<[[1000, 2000], [3000, 4000]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %3 = "onnx.Gemm"(%0, %1, %2) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<*xi32>
  onnx.Return %3 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_squeeze_dynamic_result()
func.func @test_squeeze_dynamic_result() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Squeeze"(%0, %1) : (tensor<2x1x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_unsqueeze_dynamic_result()
func.func @test_unsqueeze_dynamic_result() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[4.0, 16.0]> : tensor<2xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%0, %1) : (tensor<2xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL:  @test_pad_dynamic_result()
func.func @test_pad_dynamic_result() -> tensor<*xf32> {
  %data = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %pads = onnx.Constant dense<[0, 2, 0, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<3x2xf32>, tensor<4xi64>, none, none) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL:  @test_concat_negative_axis_dynamic_result()
func.func @test_concat_negative_axis_dynamic_result() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]> : tensor<3x2xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = -1 : si64} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL:  @test_gather_axis_0_dynamic_result()
func.func @test_gather_axis_0_dynamic_result() -> tensor<*xf32>{
  %0 = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %1 = onnx.Constant dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
  %2 = "onnx.Gather"(%0, %1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL:  @test_nonzero_dynamic_result()
func.func @test_nonzero_dynamic_result() -> tensor<*xi64> {
  %0 = "onnx.Constant"() {value = dense<[[2, 1], [0, 2], [0, 1]]> : tensor<3x2xi8>} : () -> tensor<3x2xi8>
  %1 = "onnx.NonZero"(%0) : (tensor<3x2xi8>) -> tensor<*xi64>
  onnx.Return %1 : tensor<*xi64>
}

// -----

// CHECK-LABEL: @test_scatternd_f32_dynamic_result()
func.func @test_scatternd_f32_dynamic_result() -> (tensor<*xf32>) {
  %0 = onnx.Constant { name = "constant.0", value = dense<[1., 2., 3., 4., 5., 6., 7., 8.]>:tensor<8xf32> } : tensor<8xf32>
  %1 = onnx.Constant { name = "constant.1", value = dense< [[4], [3], [1], [7]]>:tensor<4x1xi64> } : tensor<4x1xi64>
  %2 = onnx.Constant { name = "constant.2", value = dense<[9., 10., 11., 12.]>:tensor<4xf32> } : tensor<4xf32>
  %3 = "onnx.ScatterND"(%0, %1, %2) {node_name = "ScatterND_6467", node_type = "ScatterND"} : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<*xf32>
  onnx.Return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_pow_dynamic_result()
func.func @test_pow_dynamic_result() -> tensor<*xf32> {
  %0 = onnx.Constant dense<64.0> : tensor<2x2xf32>
  %1 = onnx.Constant dense<0.5> : tensor<f32>
  %2 = "onnx.Pow"(%0, %1) : (tensor<2x2xf32> , tensor<f32>) -> tensor<*xf32>
  onnx.Return %2 : tensor<*xf32>
}


// -----

/// variadic ops

// CHECK-LABEL: @test_max_dynamic_result
func.func @test_max_dynamic_result() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Max"(%0) : (tensor<2x2xi32>) -> tensor<*xi32>
  "onnx.Return"(%1) : (tensor<*xi32>) -> ()
}

// -----

// CHECK-LABEL: @test_min_dynamic_result
func.func @test_min_dynamic_result() -> tensor<*xi32> {
  %0 = "onnx.Constant"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "onnx.Min"(%0) : (tensor<2x2xi32>) -> tensor<*xi32>
  "onnx.Return"(%1) : (tensor<*xi32>) -> ()
}

// -----

// CHECK-LABEL: @test_sum_dynamic_result
func.func @test_sum_dynamic_result() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<0.5> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "onnx.Sum"(%0) : (tensor<2x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
}

// -----
