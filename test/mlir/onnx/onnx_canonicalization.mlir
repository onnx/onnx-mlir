// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @test_matmul_add_fused(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32> {
func.func @test_matmul_add_fused(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%1) : (tensor<10x10xf32>) -> ()
}

// -----

// onnx.MatMul ops for non 2-D matrices should not get fused because Gemm only supports 2-D matrices.
// CHECK-LABEL: func @test_matmul_add_not_fused(%{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
func.func @test_matmul_add_not_fused(%a0: tensor<10x10x10xf32>, %a1: tensor<10x10x10xf32>, %a2: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  "func.return"(%1) : (tensor<10x10x10xf32>) -> ()
}

// -----

// onnx.MatMul ops with more than one result uses should not get fused.
// CHECK-LABEL: func @test_sigmoid_add(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func.func @test_sigmoid_add(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %3 = "onnx.Add"(%1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%3) : (tensor<10x10xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_identity_identity(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func.func @test_identity_identity(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.Identity"(%a0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Identity"(%a1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%2) : (tensor<10x10xf32>) -> ()
}

// -----

func.func @test_dropout(%arg: tensor<10x10xf32>) -> (tensor<10x10xf32>, none) {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1:2 = "onnx.Dropout"(%arg, %0, %0) : (tensor<10x10xf32>, none, none) -> (tensor<10x10xf32>, none)
  "func.return"(%1#0, %1#1) : (tensor<10x10xf32>, none) -> ()
  // CHECK-LABEL: test_dropout
  // CHECK-NOT: "onnx.Dropout"
  // CHECK-NEXT: [[NONE:%.+]] = "onnx.NoValue"
  // CHECK-NEXT: return %arg0, [[NONE]] : tensor<10x10xf32>, none
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion_beta_zero(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion_beta_zero(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) {beta = 0.0 : f32}: (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion_rank3(%{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<256xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion_rank3(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @cast_elimination(%{{.*}}: tensor<2xf32>) -> tensor<2xf32> {
func.func @cast_elimination(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>

  // CHECK-NEXT: return %arg0 : tensor<2xf32>
}

// -----

func.func @test_conv_batchnormtestmode_fusion_nobias(%arg0: tensor<1x3x224x224xf32>, %0: tensor<64x3x7x7xf32>, %2: tensor<64xf32>, %3: tensor<64xf32>, %4: tensor<64xf32>, %5: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %0, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none) -> tensor<1x64x112x112xf32>
    %6 = "onnx.BatchNormalizationInferenceMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %6 :  tensor<1x64x112x112xf32>

    // CHECK-LABEL:  func.func @test_conv_batchnormtestmode_fusion_nobias
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x224x224xf32>, [[PARAM_1_:%.+]]: tensor<64x3x7x7xf32>, [[PARAM_2_:%.+]]: tensor<64xf32>, [[PARAM_3_:%.+]]: tensor<64xf32>, [[PARAM_4_:%.+]]: tensor<64xf32>, [[PARAM_5_:%.+]]: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.00000007E-5> : tensor<1xf32>
    // CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_5_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    // CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_2_]], [[VAR_2_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_4_:%.+]] = "onnx.UnsqueezeV11"([[VAR_3_]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_4_]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Neg"([[PARAM_4_]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_6_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_8_:%.+]] = "onnx.Add"([[PARAM_3_]], [[VAR_7_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_9_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_5_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>
    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationInferenceMode"{{.*}}
    // CHECK:           return [[VAR_9_]] : tensor<1x64x112x112xf32>
}

// -----

func.func @test_conv_batchnormtestmode_fusion(%arg0 : tensor<1x3x224x224xf32>, %arg1 : tensor<64xf32>, %0 : tensor<64x3x7x7xf32>, %2 : tensor<64xf32>, %3 : tensor<64xf32>, %4 : tensor<64xf32>, %5 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %0, %arg1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %6 = "onnx.BatchNormalizationInferenceMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %6 :  tensor<1x64x112x112xf32>

    // CHECK-LABEL:  func.func @test_conv_batchnormtestmode_fusion
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x224x224xf32>, [[PARAM_1_:%.+]]: tensor<64xf32>, [[PARAM_2_:%.+]]: tensor<64x3x7x7xf32>, [[PARAM_3_:%.+]]: tensor<64xf32>, [[PARAM_4_:%.+]]: tensor<64xf32>, [[PARAM_5_:%.+]]: tensor<64xf32>, [[PARAM_6_:%.+]]: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.00000007E-5> : tensor<1xf32>
    // CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_6_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    // CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_3_]], [[VAR_2_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_4_:%.+]] = "onnx.UnsqueezeV11"([[VAR_3_]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[PARAM_2_]], [[VAR_4_]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[PARAM_5_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    // CHECK:           [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_6_]]) : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_8_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_7_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:           [[VAR_9_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_5_]], [[VAR_8_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>
    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationInferenceMode"{{.*}}
    // CHECK:           return [[VAR_9_]] : tensor<1x64x112x112xf32>
}

// -----

// Check the removal of identity transposes.
// CHECK-LABEL: func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [0, 1, 2, 3]} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%0) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the fusion of transposes when transposes at the output side are moved
// to the input side. This is only done when there are transposes at the input side.
// CHECK-LABEL: func @test_transpose_concat_reversed
func.func @test_transpose_concat_reversed(%arg0: tensor<?x5x5x1xf32>, %arg1: tensor<?x5x5x2xf32>) -> tensor<?x5x5x3xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<?x5x5x1xf32>) -> tensor<?x1x5x5xf32>
    %1 = "onnx.Transpose"(%arg1) {perm = [0, 3, 1, 2]} : (tensor<?x5x5x2xf32>) -> tensor<?x2x5x5xf32>
    %2 = "onnx.Concat"(%0, %1) {axis = 1 : si64} : (tensor<?x1x5x5xf32>, tensor<?x2x5x5xf32>) -> tensor<?x3x5x5xf32>
    %3 = "onnx.Transpose"(%2) {perm = [0, 2, 3, 1]} : (tensor<?x3x5x5xf32>) -> tensor<?x5x5x3xf32>
    return %3 : tensor<?x5x5x3xf32>

    // CHECK-NEXT: "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<?x5x5x1xf32>, tensor<?x5x5x2xf32>) -> tensor<?x5x5x3xf32>
    // CHECK-NOT: "onnx.Transpose"
}

// -----

// Check the removal of identity reshapes.
// CHECK-LABEL: func @test_reshape_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_reshape_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = onnx.Constant dense<[10, 11, 12, 13]> : tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the removal of reshapes that are used for matmul's input and output.
// The 1st reshape is moved down then fused with the 2nd reshape to become an identity.

// CHECK-LABEL: func @test_reshape_removal_with_matmul_4D(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<3x5x10x1xf32> {
func.func @test_reshape_removal_with_matmul_4D(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<3x5x10x1xf32> {
  %shape1 = onnx.Constant dense<[150, 20]> : tensor<2xi64>
  %shape2 = onnx.Constant dense<[3, 5, 10, 1]> : tensor<4xi64>
  %0 = "onnx.Reshape"(%arg0, %shape1) : (tensor<3x5x10x20xf32>, tensor<2xi64>) -> tensor<150x20xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<150x20xf32>, tensor<20x1xf32>) -> tensor<150x1xf32>
  %2 = "onnx.Reshape"(%1, %shape2)  : (tensor<150x1xf32>, tensor<4xi64>) -> tensor<3x5x10x1xf32>
  return %2 : tensor<3x5x10x1xf32>
  // CHECK-NEXT: "onnx.MatMul"(%arg0, %arg1) : (tensor<3x5x10x20xf32>, tensor<20x1xf32>) -> tensor<3x5x10x1xf32>
  // CHECK-NOT: "onnx.Reshape"
}

// -----

// Check the removal of reshapes that are used for matmul's input and output.
// The 1st reshape is moved down then fused with the 2nd reshape to become a single reshape.

// CHECK-LABEL: func @test_reshape_should_not_remove(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<15x10x1xf32> {
func.func @test_reshape_should_not_remove(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<15x10x1xf32> {
  %shape1 = onnx.Constant dense<[150, 20]> : tensor<2xi64>
  %shape2 = onnx.Constant dense<[15, 10, 1]> : tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape1) : (tensor<3x5x10x20xf32>, tensor<2xi64>) -> tensor<150x20xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<150x20xf32>, tensor<20x1xf32>) -> tensor<150x1xf32>
  %2 = "onnx.Reshape"(%1, %shape2)  : (tensor<150x1xf32>, tensor<3xi64>) -> tensor<15x10x1xf32>
  return %2 : tensor<15x10x1xf32>
  // CHECK: [[CST:%.+]] = onnx.Constant dense<[15, 10, 1]> : tensor<3xi64>
  // CHECK: [[MATMUL:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x5x10x20xf32>, tensor<20x1xf32>) -> tensor<3x5x10x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[MATMUL]], [[CST]]) {allowzero = 0 : si64} : (tensor<3x5x10x1xf32>, tensor<3xi64>) -> tensor<15x10x1xf32>
  // CHECK: return [[RES]] : tensor<15x10x1xf32>
}

// -----

// Check the combining of transposes into a simple transpose.
// CHECK-LABEL: func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
func.func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [2, 3, 0, 1]} : (tensor<13x12x11x10xf32>) -> tensor<11x10x13x12xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.Transpose"(%arg0) {perm = [1, 0, 3, 2]} : (tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32>
  "func.return"(%1) : (tensor<11x10x13x12xf32>) -> ()
}

// -----

// Check the combining of two transposes besides Atan op into a simple transpose and the removing of combined transpose. (No attribute case)
//
// CHECK-LABEL: func @test_transpose_besides_atan_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_transpose_besides_atan_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x11x12x13xf32>) -> tensor<10x13x11x12xf32>
  %1 = "onnx.Atan"(%0) : (tensor<10x13x11x12xf32>) -> tensor<10x13x11x12xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 3, 1]} : (tensor<10x13x11x12xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.Atan"(%arg0) : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  "func.return"(%2) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of two transposes besides leaky relu op into a simple transpose and the removing of combined transpose. (One attribute case)
//
// CHECK-LABEL: func @test_transpose_besides_leakyrelu_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_transpose_besides_leakyrelu_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x11x12x13xf32>) -> tensor<10x13x11x12xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha = 1.000000e-01 : f32} : (tensor<10x13x11x12xf32>) -> tensor<10x13x11x12xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 3, 1]} : (tensor<10x13x11x12xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.LeakyRelu"(%arg0) {alpha = 1.000000e-01 : f32} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  "func.return"(%2) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of two transposes besides HardSigmoid op into a simple transpose and the removing of combined transpose. (Two attribute case)
//
// CHECK-LABEL: func @test_transpose_besides_hardsigmoid_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_transpose_besides_hardsigmoid_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [0, 3, 1, 2]} : (tensor<10x11x12x13xf32>) -> tensor<10x13x11x12xf32>
  %1 = "onnx.HardSigmoid"(%0)  {alpha = 1.000000e-01 : f32, beta = 2.000000e-01 : f32} : (tensor<10x13x11x12xf32>) -> tensor<10x13x11x12xf32>
  %2 = "onnx.Transpose"(%1)  {perm = [0, 2, 3, 1]} : (tensor<10x13x11x12xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.HardSigmoid"(%arg0) {alpha = 1.000000e-01 : f32, beta = 2.000000e-01 : f32} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  "func.return"(%2) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of reshape into a simple reshape.
// CHECK-LABEL: func @test_reshape_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
func.func @test_reshape_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
  %0 = onnx.Constant dense<[10, 12, 11, 13]> : tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
  %2 = onnx.Constant dense<[11, 10, 13, 12]> : tensor<4xi64>
  %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<11x10x13x12xf32>
  // CHECK-NEXT: [[RES:%.+]] = onnx.Constant dense<[11, 10, 13, 12]> : tensor<4xi64>
  // CHECK-NEXT: %{{.*}} = "onnx.Reshape"(%arg0, [[RES]]) {allowzero = 0 : si64} : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<11x10x13x12xf32>
  "func.return"(%3) : (tensor<11x10x13x12xf32>) -> ()
}

// -----

// Check the combining of transposes into an identity transpose, which in turns is removed.
// CHECK-LABEL: func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [3, 2, 1, 0]} : (tensor<13x12x11x10xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of reshape into an identity reshape, which in turns is removed.
// CHECK-LABEL: func @test_reshape_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func.func @test_reshape_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = onnx.Constant dense<[10, 12, 11, 13]> : tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
  %2 = onnx.Constant dense<[10, 11, 12, 13]> : tensor<4xi64>
  %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%3) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

func.func @test_shape1(%arg0 : tensor<2x4x8x16xf32>) -> tensor<4xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
  // CHECK-LABEL:  func.func @test_shape1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x16xf32>) -> tensor<4xi64> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 4, 8, 16]> : tensor<4xi64>
  // CHECK:           return [[VAR_0_]] : tensor<4xi64>
}

// -----

func.func @test_shape2(%arg0 : tensor<?x4x8x16xf32>) -> tensor<*xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<?x4x8x16xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_shape2
  // CHECK-NEXT: %0 = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<?x4x8x16xf32>) -> tensor<*xi64>
  // CHECK-NEXT: return %0 : tensor<*xi64>
}


// -----

func.func @test_size1(%arg0 : tensor<2x4x8x16xf32>) -> tensor<i64> {
  %0 = "onnx.Size"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<i64>
  return %0 : tensor<i64>

  // CHECK-LABEL: @test_size1
  // CHECK-NEXT: %0 = onnx.Constant dense<1024> : tensor<i64>
  // CHECK-NEXT: %0 : tensor<i64>
}

// -----

func.func @test_size2(%arg0 : tensor<*xf32>) -> tensor<*xi64> {
  %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_size2
  // CHECK-NEXT: %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
  // CHECK-NEXT: return %0 : tensor<*xi64>
}

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMeanV13
func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_average_pool
  // CHECK: [[RES:%.+]] = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: return [[RES]] : tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMeanV13 with dynamic dimensions
func.func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  return %0 : tensor<1x?x?x1xf32>
  // CHECK-LABEL: test_global_average_pool_dyn_dims
  // CHECK: [[RES:%.+]] = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  // CHECK: return [[RES]] : tensor<1x?x?x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMaxV13
func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_average_pool
  // CHECK: [[RES:%.+]] = "onnx.ReduceMaxV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: return [[RES]] : tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMaxV13 with dynamic dimensions
func.func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  return %0 : tensor<1x?x?x1xf32>
  // CHECK-LABEL: test_global_average_pool_dyn_dims
  // CHECK: [[RES:%.+]] = "onnx.ReduceMaxV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  // CHECK: return [[RES]] : tensor<1x?x?x1xf32>
}

// -----

// COM: Test removing squeeze/unsqueeze pairs when they use the same axes.

func.func @test_remove_unsqueeze_squeeze(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %1 = onnx.Constant dense<[0, -2]> : tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<2xi64>) -> tensor<10x10xf32>
  return %3: tensor<10x10xf32>

  // CHECK-LABEL: test_remove_unsqueeze_squeeze
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK-NOT: {{.*}} = onnx.Squeeze"{{.*}}
  // CHECK: return {{.*}}

}

// -----

func.func @test_remove_unsqueezev11_squeezev11(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes=[0, -2]} : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
  return %1: tensor<10x10xf32>

  // CHECK-LABEL: test_remove_unsqueezev11_squeezev11
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK-NOT: {{.*}} = onnx.SqueezeV11"{{.*}}
  // CHECK: return {{.*}}

}

// -----

// COM: Test removing squeeze/cast/unsqueeze pairs when they use the same axes.

func.func @test_remove_unsqueeze_cast_squeeze(%arg0 : tensor<10x10xf32>) -> tensor<10x10xi64> {
  %0 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %1 = onnx.Constant dense<[0, -2]> : tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Cast"(%2) {to = i64}: (tensor<1x10x1x10xf32>) -> tensor<1x10x1x10xi64>
  %4 = "onnx.Squeeze"(%3, %1) : (tensor<1x10x1x10xi64>, tensor<2xi64>) -> tensor<10x10xi64>
  return %4: tensor<10x10xi64>

  // CHECK-LABEL: test_remove_unsqueeze_cast_squeeze
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: [[RES:%.+]] = "onnx.Cast"{{.*}}
  // CHECK: return [[RES]]
}

// -----

func.func @test_should_not_remove_unsqueeze_squeeze(%arg0 : tensor<10x10xf32>) -> tensor<10x1x10xf32> {
  %0 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %1 = onnx.Constant dense<[0]> : tensor<1xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  return %3: tensor<10x1x10xf32>
  // CHECK-LABEL: test_should_not_remove_unsqueeze_squeeze
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_should_not_remove_unsqueezev11_squeezev11(%arg0 : tensor<10x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes=[0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
  return %1: tensor<10x1x10xf32>
  // CHECK-LABEL: test_should_not_remove_unsqueezev11_squeezev11
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_remove_squeeze_unsqueeze(%arg0 : tensor<10x1x10xf32>) -> tensor<10x1x10xf32> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x10xf32>
  %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  return %3: tensor<10x1x10xf32>
  // CHECK-LABEL: test_remove_squeeze_unsqueeze
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_remove_squeeze_cast_unsqueeze(%arg0 : tensor<10x1x10xf32>) -> tensor<10x1x10xi64> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x10xf32>
  %3 = "onnx.Cast"(%2) { to = i64 } : (tensor<10x10xf32>) -> tensor<10x10xi64>
  %4 = "onnx.Unsqueeze"(%3, %1) : (tensor<10x10xi64>, tensor<1xi64>) -> tensor<10x1x10xi64>
  return %4: tensor<10x1x10xi64>
  // CHECK-LABEL: test_remove_squeeze_cast_unsqueeze
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: [[RES:%.+]] = "onnx.Cast"{{.*}}
  // CHECK: return [[RES]]
}

// -----

func.func @test_remove_squeezev11_unsqueezev11(%arg0 : tensor<10x1x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes=[1]} : (tensor<10x1x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[1]} : (tensor<10x10xf32>) -> tensor<10x1x10xf32>
  return %1: tensor<10x1x10xf32>
  // CHECK-LABEL: test_remove_squeezev11_unsqueezev11
  // CHECK-NOT: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_should_not_remove_squeeze_unsqueeze(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %1 = onnx.Constant dense<[3]> : tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10x1xf32>
  return %3: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_squeeze_unsqueeze
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_should_not_remove_squeezev11_unsqueezev11(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes=[0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[3]} : (tensor<10x1x10xf32>) -> tensor<10x1x10x1xf32>
  return %1: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_squeezev11_unsqueezev11
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_should_not_remove_null_axes_squeeze_unsqueeze(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %cst) : (tensor<1x10x1x10xf32>, none) -> tensor<10x10xf32>
  %2 = "onnx.Unsqueeze"(%1, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<10x1x10x1xf32>
  return %2: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_null_axes_squeeze_unsqueeze
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func.func @test_should_not_remove_null_axes_squeezev11_unsqueezev11(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[1, 3]} : (tensor<10x10xf32>) -> tensor<10x1x10x1xf32>
  return %1: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_null_axes_squeezev11_unsqueezev11
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

// COM: Test removing DepthToSpace/SpaceToDepth pairs when blocksize of the two operators are the same.

func.func @test_remove_depth_to_space_space_to_depth(%arg0 : tensor<1x16x32x64xf32>) -> tensor<1x16x32x64xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<1x256x8x16xf32>
  %1 = "onnx.DepthToSpace"(%0) {blocksize = 4 : si64, mode = "CRD"} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  "func.return"(%1) : (tensor<1x16x32x64xf32>) -> ()

  // CHECK-LABEL: test_remove_depth_to_space_space_to_depth
  // CHECK: return %arg0 : tensor<1x16x32x64xf32>
}

// -----

func.func @test_remove_space_to_depth_depth_to_space(%arg0 : tensor<1x256x8x16xf32>) -> tensor<1x256x8x16xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64, mode = "CRD"} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  %1 = "onnx.SpaceToDepth"(%0) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<1x256x8x16xf32>
  "func.return"(%1) : (tensor<1x256x8x16xf32>) -> ()

  // CHECK-LABEL: test_remove_space_to_depth_depth_to_space
  // CHECK: return %arg0 : tensor<1x256x8x16xf32>
}

// -----

func.func @test_constant_1() -> tensor<i64> {
  %0 = onnx.Constant {value_int = 1 : si64} : tensor<i64>
  return %0 : tensor<i64>
// CHECK-LABEL:       func @test_constant_1
// CHECK:           [[VAR_0:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK:           return [[VAR_0]] : tensor<i64>
}


// -----

func.func @test_constant_2() -> tensor<f32> {
  %0 = onnx.Constant {value_float = 2.0 : f32 } : tensor<f32>
  return %0 : tensor<f32>
// CHECK-LABEL:     func @test_constant_2
// CHECK: [[VAR_0:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK: return [[VAR_0]] : tensor<f32>
}

// -----

func.func @test_constant_3() -> tensor<3xi64> {
  %0 = onnx.Constant {value_ints = [1, 2, 3] } : tensor<3xi64>
  return %0 : tensor<3xi64>
// CHECK-LABEL:       func @test_constant_3
// CHECK-SAME:     () -> tensor<3xi64> {
// CHECK:           [[VAR_0:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK:           return [[VAR_0]] : tensor<3xi64>
}

// -----

func.func @test_rewrite_batchnormtestmode_Nd(%arg0 : tensor<1x64x112x112xf32>, %scale : tensor<64xf32>, %bias : tensor<64xf32>, %mean : tensor<64xf32>, %var : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %scale, %bias, %mean, %var) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %0 :  tensor<1x64x112x112xf32>

  // CHECK-LABEL:  func.func @test_rewrite_batchnormtestmode_Nd
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x112x112xf32>, [[PARAM_1_:%.+]]: tensor<64xf32>, [[PARAM_2_:%.+]]: tensor<64xf32>, [[PARAM_3_:%.+]]: tensor<64xf32>, [[PARAM_4_:%.+]]: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.00000007E-5> : tensor<1xf32>
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<64xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_1_]], [[VAR_2_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_4_:%.+]] = "onnx.UnsqueezeV11"([[VAR_3_]]) {axes = [1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_4_]]) : (tensor<1x64x112x112xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Mul"([[PARAM_3_]], [[VAR_3_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_7_:%.+]] = "onnx.Sub"([[PARAM_2_]], [[VAR_6_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_8_:%.+]] = "onnx.UnsqueezeV11"([[VAR_7_]]) {axes = [1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_9_:%.+]] = "onnx.Add"([[VAR_5_]], [[VAR_8_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>
  // CHECK:           return [[VAR_9_]] : tensor<1x64x112x112xf32>
}

// -----

func.func @test_rewrite_batchnormtestmode_1d(%arg0 : tensor<64xf32>, %scale : tensor<1xf32>, %bias : tensor<1xf32>, %mean : tensor<1xf32>, %var : tensor<1xf32>) -> tensor<64xf32> {
    %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %scale, %bias, %mean, %var) {epsilon = 1.00000007E-5 : f32} : (tensor<64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<64xf32>
    return %0 :  tensor<64xf32>

// CHECK-LABEL:  func.func @test_rewrite_batchnormtestmode_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<64xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>, [[PARAM_3_:%.+]]: tensor<1xf32>, [[PARAM_4_:%.+]]: tensor<1xf32>) -> tensor<64xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.00000007E-5> : tensor<1xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_0_]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<1xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_1_]], [[VAR_2_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_3_]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[PARAM_3_]], [[VAR_3_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[PARAM_2_]], [[VAR_5_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_6_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<64xf32>
// CHECK:           return [[VAR_7_]] : tensor<64xf32>
}

// -----

func.func @test_normalize_add(%arg0 : tensor<2xf32>) -> tensor<2xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = onnx.Constant dense<[0.0, 1.0]> : tensor<2xf32>
    %1 = "onnx.Add"(%0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
    // CHECK-LABEL: test_normalize_add
    // CHECK: [[CONSTANT:%.+]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
    // CHECK: [[RES:%.+]] = "onnx.Add"(%arg0, [[CONSTANT]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    // CHECK: return [[RES]] : tensor<2xf32>
}

// -----

func.func @test_fuse_add_conv(%arg0 : tensor<1x1x28x28xf32>, %arg1 : tensor<8x1x5x5xf32>) -> tensor<1x8x28x28xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Convolution28", strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, none) -> tensor<1x8x28x28xf32>
    %1 = onnx.Constant dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<1x8x28x28xf32>, tensor<8x1x1xf32>) -> tensor<1x8x28x28xf32>
    return %2 : tensor<1x8x28x28xf32>
// CHECK-LABEL:  func.func @test_fuse_add_conv
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x28x28xf32>, [[PARAM_1_:%.+]]: tensor<8x1x5x5xf32>) -> tensor<1x8x28x28xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[-0.161539719, -0.433835655, 0.091641359, -0.0168522168, -0.0650264397, -0.131737873, 0.0204175506, -0.121110231]> : tensor<8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, tensor<8xf32>) -> tensor<1x8x28x28xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x8x28x28xf32>

}

// -----

func.func @test_fuse_mul_conv(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
    %0 = onnx.Constant dense<[[[[0.0234164055, 0.0228030644], [2.442580e-02, 0.0237577036]]], [[[-0.0410864502, 0.0488203131], [0.164448678, -0.0200194642]]], [[[-4.34581793E-9, 0.025325032], [0.0373019315, 0.165243402]]], [[[-0.0198689923, 0.131284416], [0.0572107285, 2.33985098E-8]]], [[[0.0187684372, -0.148515195], [0.0154875498, 0.019133633]]], [[[0.0176953916, -0.0154658081], [0.0233727545, -0.274110436]]], [[[-0.021181887, 0.0936150252], [0.135688141, -0.0202601217]]], [[[-0.0201558527, 0.0192655921], [0.227748245, -0.196346223]]]]> : tensor<8x1x2x2xf32>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<*xf32>
    %3 = onnx.Constant dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>
    %4 = "onnx.Mul"(%2, %3) : (tensor<*xf32>, tensor<8x1x1xf32>) -> tensor<*xf32>
    return %4 : tensor<*xf32>

  // CHECK-LABEL:  func.func @test_fuse_mul_conv
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[-0.161539719]{{.}}, {{.}}[-0.433835655]{{.}}, {{.}}[0.091641359]{{.}}, {{.}}[-0.0168522168]{{.}}, {{.}}[-0.0650264397]{{.}}, {{.}}[-0.131737873]{{.}}, {{.}}[0.0204175506]{{.}}, {{.}}[-0.121110231]{{.}}{{.}}> : tensor<8x1x1xf32>
  // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}[{{.}}[0.0234164055, 0.0228030644], [2.442580e-02, 0.0237577036]{{.}}{{.}}, {{.}}{{.}}[-0.0410864502, 0.0488203131], [0.164448678, -0.0200194642]{{.}}{{.}}, {{.}}{{.}}[-4.34581793E-9, 0.025325032], [0.0373019315, 0.165243402]{{.}}{{.}}, {{.}}{{.}}[-0.0198689923, 0.131284416], [0.0572107285, 2.33985098E-8]{{.}}{{.}}, {{.}}{{.}}[0.0187684372, -0.148515195], [0.0154875498, 0.019133633]{{.}}{{.}}, {{.}}{{.}}[0.0176953916, -0.0154658081], [0.0233727545, -0.274110436]{{.}}{{.}}, {{.}}{{.}}[-0.021181887, 0.0936150252], [0.135688141, -0.0202601217]{{.}}{{.}}, {{.}}{{.}}[-0.0201558527, 0.0192655921], [0.227748245, -0.196346223]{{.}}{{.}}]> : tensor<8x1x2x2xf32>
  // CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK:           [[VAR_3_:%.+]] = "onnx.UnsqueezeV11"([[VAR_0_]]) {axes = [3]} : (tensor<8x1x1xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_1_]]) : (tensor<*xf32>, tensor<8x1x2x2xf32>) -> tensor<*xf32>
  // CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_4_]], [[VAR_2_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<*xf32>, none) -> tensor<*xf32>
  // CHECK:           return [[VAR_5_]] : tensor<*xf32>
}

// -----

func.func @test_less(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<i32>) -> tensor<f32>
  %1 = "onnx.Cast"(%arg1) {to = f32} : (tensor<i32>) -> tensor<f32>
  %2 = "onnx.Less"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
  // CHECK-LABEL: test_less
  // CHECK: [[RES:%.]] = "onnx.Less"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: return [[RES]] : tensor<i1>
}

// -----

// Cast is not removed because of unsigned integers.
func.func @test_less_should_not_remove_cast(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = ui32} : (tensor<f32>) -> tensor<ui32>
  %1 = "onnx.Cast"(%arg1) {to = ui32} : (tensor<f32>) -> tensor<ui32>
  %2 = "onnx.Less"(%0, %1) : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
  return %2 : tensor<i1>
  // CHECK-LABEL: test_less_should_not_remove_cast
  // CHECK: "onnx.Cast"
  // CHECK: "onnx.Cast"
  // CHECK: "onnx.Less"
}

// -----

// Check deriving a new maximum trip count from the break condition of the loop.
// In this test, the new maximum trip count is a constant.
func.func @test_loop_derive_max_trip_count(%arg0: tensor<?x30xf32>) -> tensor<?x?x30xf32> {
  %0 = onnx.Constant dense<9223372036854775807> : tensor<i64>
  %1 = onnx.Constant dense<true> : tensor<i1>
  %2 = onnx.Constant dense<0> : tensor<i32>
  %3 = onnx.Constant dense<30> : tensor<i32>
  %4:4 = "onnx.Loop"(%0, %1, %2, %3, %arg0) ({
  ^bb0(%arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<?x30xf32>):
    %5 = onnx.Constant dense<4> : tensor<i32>
    %6 = "onnx.Add"(%arg3, %5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = "onnx.Relu"(%arg5) : (tensor<?x30xf32>) -> tensor<?x30xf32>
    %8 = "onnx.Less"(%6, %arg4) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    onnx.Return %8, %6, %arg4, %7 : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
  }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
  return %4#3 : tensor<?x?x30xf32>
// CHECK-LABEL:  func.func @test_loop_derive_max_trip_count
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x30xf32>) -> tensor<?x?x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<8> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4> : tensor<i32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0> : tensor<i32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<30> : tensor<i32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.Loop"([[VAR_0_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[PARAM_0_]]) ({
// CHECK:           ^bb0([[arg1_:%.+]]: tensor<i64>, [[arg2_:%.+]]: tensor<i1>, [[arg3_:%.+]]: tensor<i32>, [[arg4_:%.+]]: tensor<i32>, [[arg5_:%.+]]: tensor<?x30xf32>):
// CHECK-DAG:         [[VAR_6_:%.+]] = "onnx.Add"([[arg3_]], [[VAR_1_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:         [[VAR_7_:%.+]] = "onnx.Relu"([[arg5_]]) : (tensor<?x30xf32>) -> tensor<?x30xf32>
// CHECK:             onnx.Return [[arg2_]], [[VAR_6_]], [[arg4_]], [[VAR_7_]] : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
// CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
// CHECK:           return [[VAR_5_]]#3 : tensor<?x?x30xf32>

}

// -----

// Check deriving a new maximum trip count from the break condition of the loop.
// In this test, the new maximum trip count is not a constant.
func.func @test_loop_derive_max_trip_count_non_constant_ub(%arg0: tensor<?x30xf32>, %arg1: tensor<i32>) -> tensor<?x?x30xf32> {
  %0 = onnx.Constant dense<9223372036854775807> : tensor<i64>
  %1 = onnx.Constant dense<true> : tensor<i1>
  %2 = onnx.Constant dense<0> : tensor<i32>
  %3:4 = "onnx.Loop"(%0, %1, %2, %arg1, %arg0) ({
  ^bb0(%arg2: tensor<i64>, %arg3: tensor<i1>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<?x30xf32>):
    %4 = onnx.Constant dense<1> : tensor<i32>
    %5 = "onnx.Add"(%arg4, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = "onnx.Relu"(%arg6) : (tensor<?x30xf32>) -> tensor<?x30xf32>
    %7 = "onnx.Less"(%5, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    onnx.Return %7, %5, %arg5, %6 : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
  }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
  return %3#3 : tensor<?x?x30xf32>
// CHECK-LABEL:  func @test_loop_derive_max_trip_count_non_constant_ub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x30xf32>, [[PARAM_1_:%.+]]: tensor<i32>) -> tensor<?x?x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<i64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0> : tensor<i32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_1_]]) {to = i64} : (tensor<i32>) -> tensor<i64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Cast"([[VAR_3_]]) {to = i64} : (tensor<i32>) -> tensor<i64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_4_]], [[VAR_5_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_6_]]) {to = f32} : (tensor<i64>) -> tensor<f32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_0_]]) {to = f32} : (tensor<i32>) -> tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Div"([[VAR_7_]], [[VAR_8_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Ceil"([[VAR_9_]]) : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Cast"([[VAR_10_]]) {to = i64} : (tensor<f32>) -> tensor<i64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Min"([[VAR_1_]], [[VAR_1_]]1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:           [[VAR_13_:%.+]]:4 = "onnx.Loop"([[VAR_12_]], [[VAR_2_]], [[VAR_3_]], [[PARAM_1_]], [[PARAM_0_]]) ({
// CHECK:           ^bb0([[arg2_:%.+]]: tensor<i64>, [[arg3_:%.+]]: tensor<i1>, [[arg4_:%.+]]: tensor<i32>, [[arg5_:%.+]]: tensor<i32>, [[arg6_:%.+]]: tensor<?x30xf32>):
// CHECK-DAG:         [[VAR_14_:%.+]] = "onnx.Add"([[arg4_]], [[VAR_0_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.Relu"([[arg6_]]) : (tensor<?x30xf32>) -> tensor<?x30xf32>
// CHECK:             onnx.Return [[arg3_]], [[VAR_14_]], [[arg5_]], [[VAR_15_]] : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
// CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
// CHECK:           return [[VAR_13_]]#3 : tensor<?x?x30xf32>

}

// -----

func.func @test_rnn_layout1(%arg0: tensor<5x4x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>, %arg3: tensor<5x1x3xf32>) -> tensor<5x1x3xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %arg3) {layout = 1 : si64} : (tensor<5x4x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, tensor<5x1x3xf32>) -> (tensor<5x4x1x3xf32>, tensor<5x1x3xf32>)
  return %Y_h : tensor<5x1x3xf32>
// CHECK-LABEL:  func.func @test_rnn_layout1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x4x2xf32>, [[PARAM_1_:%.+]]: tensor<1x3x2xf32>, [[PARAM_2_:%.+]]: tensor<1x3x3xf32>, [[PARAM_3_:%.+]]: tensor<5x1x3xf32>) -> tensor<5x1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0, 2]} : (tensor<5x4x2xf32>) -> tensor<4x5x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_3_]]) {perm = [1, 0, 2]} : (tensor<5x1x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           %Y, %Y_h = "onnx.RNN"([[VAR_1_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_0_]], [[VAR_0_]], [[VAR_2_]]) {activations = ["Tanh", "Tanh"], direction = "forward", hidden_size = 3 : si64, layout = 0 : si64} : (tensor<4x5x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, tensor<1x5x3xf32>) -> (tensor<4x1x5x3xf32>, tensor<1x5x3xf32>)
// CHECK:           [[VAR_3_:%.+]] = "onnx.Transpose"(%Y_h) {perm = [1, 0, 2]} : (tensor<1x5x3xf32>) -> tensor<5x1x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x1x3xf32>
// CHECK:         }
}

// -----

func.func @test_gru_layout1(%arg0: tensor<5x4x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> (tensor<5x4x1x3xf32>, tensor<5x1x3xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {layout = 1 : si64} : (tensor<5x4x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (tensor<5x4x1x3xf32>, tensor<5x1x3xf32>)
  return %Y, %Y_h : tensor<5x4x1x3xf32>, tensor<5x1x3xf32>
// CHECK-LABEL:  func.func @test_gru_layout1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x4x2xf32>, [[PARAM_1_:%.+]]: tensor<1x9x2xf32>, [[PARAM_2_:%.+]]: tensor<1x9x3xf32>) -> (tensor<5x4x1x3xf32>, tensor<5x1x3xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0, 2]} : (tensor<5x4x2xf32>) -> tensor<4x5x2xf32>
// CHECK:           %Y, %Y_h = "onnx.GRU"([[VAR_1_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]]) {direction = "forward", hidden_size = 3 : si64, layout = 0 : si64, linear_before_reset = 0 : si64} : (tensor<4x5x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (tensor<4x1x5x3xf32>, tensor<1x5x3xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"(%Y) {perm = [2, 0, 1, 3]} : (tensor<4x1x5x3xf32>) -> tensor<5x4x1x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Transpose"(%Y_h) {perm = [1, 0, 2]} : (tensor<1x5x3xf32>) -> tensor<5x1x3xf32>
// CHECK:           return [[VAR_2_]], [[VAR_3_]] : tensor<5x4x1x3xf32>, tensor<5x1x3xf32>
// CHECK:         }
}

// -----

func.func @test_lstm_layout1(%arg0: tensor<5x4x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>, %arg3: tensor<5x1x3xf32>) -> tensor<5x1x3xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %arg3, %cst) {layout = 1 : si64} : (tensor<5x4x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, tensor<5x1x3xf32>, none) -> (tensor<5x4x1x3xf32>, none, tensor<5x1x3xf32>)
  return %Y_c : tensor<5x1x3xf32>
// CHECK-LABEL:  func.func @test_lstm_layout1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x4x2xf32>, [[PARAM_1_:%.+]]: tensor<1x12x2xf32>, [[PARAM_2_:%.+]]: tensor<1x12x3xf32>, [[PARAM_3_:%.+]]: tensor<5x1x3xf32>) -> tensor<5x1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0, 2]} : (tensor<5x4x2xf32>) -> tensor<4x5x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_3_]]) {perm = [1, 0, 2]} : (tensor<5x1x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           %Y, %Y_h, %Y_c = "onnx.LSTM"([[VAR_1_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]], [[VAR_2_]], [[VAR_0_]]) {direction = "forward", hidden_size = 3 : si64, input_forget = 0 : si64, layout = 0 : si64} : (tensor<4x5x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, tensor<1x5x3xf32>, none) -> (tensor<4x1x5x3xf32>, none, tensor<1x5x3xf32>)
// CHECK:           [[VAR_3_:%.+]] = "onnx.Transpose"(%Y_c) {perm = [1, 0, 2]} : (tensor<1x5x3xf32>) -> tensor<5x1x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x1x3xf32>
// CHECK:         }
}

// -----

func.func @test_dim_to_constant(%arg0: tensor<?x256xi64>) -> (tensor<1xi64>) {
  %0 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x256xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>

// CHECK-LABEL: test_dim_to_constant
// CHECK-NOT: "onnx.Dim"
// CHECK:     [[RES:%.+]] = onnx.Constant dense<256> : tensor<1xi64>
// CHECK:     return [[RES]] : tensor<1xi64>
}

// -----

func.func @test_layout_transform(%arg0: tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>> {
    %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<5x3x32x32xf32,#onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
    return %0 : tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>

// CHECK-LABEL: test_layout_transform
// CHECK-NOT: "onnx.LayoutTransform"
// CHECK: return
}

// -----

func.func @test_softmax_v11_ranked(%arg0 : tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis = 2 : si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
  return %0 : tensor<10x20x30xf32>

// CHECK-LABEL:  func.func @test_softmax_v11_ranked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Softmax"([[PARAM_0_]]) {axis = 2 : si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x20x30xf32>
// CHECK:         }
}

// -----

func.func @test_softmax_v11_unranked_unchanged(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis = 2 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_softmax_v11_unranked_unchanged
// CHECK: "onnx.SoftmaxV11"
}

// -----

func.func @test_softmax_v11_unranked(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis = -1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_softmax_v11_unranked
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Softmax"([[PARAM_0_]]) {axis = -1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

#transpose = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
#reshape =  affine_map<(d0, d1) -> (d0 floordiv 32, d0 mod 32, d1 floordiv 64, d1 mod 64)>
func.func @shape_transform_compose(%arg0: tensor<128x128xf32>) -> tensor<2x4x32x64xf32> {
  %0 = "onnx.ShapeTransform"(%arg0) {index_map = #reshape} : (tensor<128x128xf32>) -> tensor<4x32x2x64xf32>
  %1 = "onnx.ShapeTransform"(%0) {index_map = #transpose} : (tensor<4x32x2x64xf32>) -> tensor<2x4x32x64xf32>
  return %1 : tensor<2x4x32x64xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d1 floordiv 64, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @shape_transform_compose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x128xf32>) -> tensor<2x4x32x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.ShapeTransform"([[PARAM_0_]]) {index_map = #map} : (tensor<128x128xf32>) -> tensor<2x4x32x64xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x4x32x64xf32>
// CHECK:         }
}

// -----

// Remove ShapeTransform with identity map.
#identity = affine_map<(d0, d1) -> (d0, d1)>
func.func @shape_transform_identity_map(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = "onnx.ShapeTransform"(%arg0) {index_map = #identity} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>

// CHECK-LABEL:  func.func @shape_transform_identity_map
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK:           return [[PARAM_0_]] : tensor<128x128xf32>
// CHECK:         }
}
