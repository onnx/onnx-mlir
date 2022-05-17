// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @test_matmul_add_fused(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32> {
func @test_matmul_add_fused(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%1) : (tensor<10x10xf32>) -> ()
}

// -----

// onnx.MatMul ops for non 2-D matrices should not get fused because Gemm only supports 2-D matrices.
// CHECK-LABEL: func @test_matmul_add_not_fused(%{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
func @test_matmul_add_not_fused(%a0: tensor<10x10x10xf32>, %a1: tensor<10x10x10xf32>, %a2: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  "func.return"(%1) : (tensor<10x10x10xf32>) -> ()
}

// -----

// onnx.MatMul ops with more than one result uses should not get fused.
// CHECK-LABEL: func @test_sigmoid_add(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_sigmoid_add(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %3 = "onnx.Add"(%1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%3) : (tensor<10x10xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_identity_identity(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_identity_identity(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.Identity"(%a0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Identity"(%a1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%2) : (tensor<10x10xf32>) -> ()
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion_rank3(%{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<256xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion_rank3(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @cast_elimination(%{{.*}}: tensor<2xf32>) -> tensor<2xf32> {
func @cast_elimination(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>

  // CHECK-NEXT: return %arg0 : tensor<2xf32>
}

// -----

func @test_conv_batchnormtestmode_fusion_nobias(%arg0 : tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Conv"(%arg0, %0, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none) -> tensor<1x64x112x112xf32>
    %2 = "onnx.Constant"() : () -> tensor<64xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.BatchNormalizationInferenceMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %6 :  tensor<1x64x112x112xf32>

    // CHECK-LABEL: test_conv_batchnormtestmode_fusion_nobias
    // CHECK: [[WEIGHT:%.+]] = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    // CHECK: [[SCALE:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[B:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[MEAN:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[VARIANCE:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[EPSILON:%.+]] = "onnx.Constant"() {value = dense<1.00000007E-5> : tensor<1xf32>} : () -> tensor<1xf32>

    // CHECK: [[VAR_EPSILON:%.+]] = "onnx.Add"([[VARIANCE]], [[EPSILON]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    // CHECK: [[SQRT:%.+]] = "onnx.Sqrt"([[VAR_EPSILON]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[COEFFICIENT_W:%.+]] = "onnx.Div"([[SCALE]], [[SQRT]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[UNSQUEEZE:%.+]] = "onnx.UnsqueezeV11"([[COEFFICIENT_W]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_WEIGHT:%.+]] = "onnx.Mul"([[WEIGHT]], [[UNSQUEEZE]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[NEG_MEAN:%.+]] = "onnx.Neg"([[MEAN]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[MUL:%.+]] = "onnx.Mul"([[COEFFICIENT_W]], [[NEG_MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_BIAS:%.+]] = "onnx.Add"([[MUL]], [[B]]) : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.+]] = "onnx.Conv"(%arg0, [[NEW_WEIGHT]], [[NEW_BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>

    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationInferenceMode"{{.*}}

    // CHECK: return [[RES]] : tensor<1x64x112x112xf32>
}

// -----

func @test_conv_batchnormtestmode_fusion(%arg0 : tensor<1x3x224x224xf32>, %arg1 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Conv"(%arg0, %0, %arg1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %2 = "onnx.Constant"() : () -> tensor<64xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.BatchNormalizationInferenceMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %6 :  tensor<1x64x112x112xf32>

    // CHECK-LABEL: test_conv_batchnormtestmode_fusion
    // CHECK: [[WEIGHT:%.+]] = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    // CHECK: [[SCALE:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[B:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[MEAN:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[VARIANCE:%.+]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[EPSILON:%.+]] = "onnx.Constant"() {value = dense<1.00000007E-5> : tensor<1xf32>} : () -> tensor<1xf32>

    // CHECK: [[VAR_EPSILON:%.+]] = "onnx.Add"([[VARIANCE]], [[EPSILON]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    // CHECK: [[SQRT:%.+]] = "onnx.Sqrt"([[VAR_EPSILON]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[COEFFICIENT_W:%.+]] = "onnx.Div"([[SCALE]], [[SQRT]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[UNSQUEEZE:%.+]] = "onnx.UnsqueezeV11"([[COEFFICIENT_W]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_WEIGHT:%.+]] = "onnx.Mul"([[WEIGHT]], [[UNSQUEEZE]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[SUB:%.+]] = "onnx.Sub"(%arg1, [[MEAN]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    // CHECK: [[MUL:%.+]] = "onnx.Mul"([[COEFFICIENT_W]], [[SUB]]) : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_BIAS:%.+]] = "onnx.Add"([[MUL]], [[B]]) : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.+]] = "onnx.Conv"(%arg0, [[NEW_WEIGHT]], [[NEW_BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>

    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationInferenceMode"{{.*}}

    // CHECK: return [[RES]] : tensor<1x64x112x112xf32>
}

// -----

// Check the removal of identity transposes.
// CHECK-LABEL: func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [0, 1, 2, 3]} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%0) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the fusion of transposes when transposes at the output side are moved
// to the input side. This is only done when there are transposes at the input side.
// CHECK-LABEL: func @test_transpose_concat_reversed
func @test_transpose_concat_reversed(%arg0: tensor<?x5x5x1xf32>, %arg1: tensor<?x5x5x2xf32>) -> tensor<?x5x5x3xf32> {
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
func @test_reshape_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Constant"() {value = dense<[10, 11, 12, 13]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the removal of reshapes that are used for matmul's input and output.
// The 1st reshape is moved down then fused with the 2nd reshape to become an identity.

// CHECK-LABEL: func @test_reshape_removal_with_matmul_4D(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<3x5x10x1xf32> {
func @test_reshape_removal_with_matmul_4D(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<3x5x10x1xf32> {
  %shape1 = "onnx.Constant"() { value = dense<[150, 20]> : tensor<2xi64> } : () -> tensor<2xi64>
  %shape2 = "onnx.Constant"() { value = dense<[3, 5, 10, 1]> : tensor<4xi64> } : () -> tensor<4xi64>
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
func @test_reshape_should_not_remove(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>) -> tensor<15x10x1xf32> {
  %shape1 = "onnx.Constant"() { value = dense<[150, 20]> : tensor<2xi64> } : () -> tensor<2xi64>
  %shape2 = "onnx.Constant"() { value = dense<[15, 10, 1]> : tensor<3xi64> } : () -> tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape1) : (tensor<3x5x10x20xf32>, tensor<2xi64>) -> tensor<150x20xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<150x20xf32>, tensor<20x1xf32>) -> tensor<150x1xf32>
  %2 = "onnx.Reshape"(%1, %shape2)  : (tensor<150x1xf32>, tensor<3xi64>) -> tensor<15x10x1xf32>
  return %2 : tensor<15x10x1xf32>
  // CHECK: [[CST:%.+]] = "onnx.Constant"() {value = dense<[15, 10, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK: [[MATMUL:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x5x10x20xf32>, tensor<20x1xf32>) -> tensor<3x5x10x1xf32>
  // CHECK: [[RES:%.+]] = "onnx.Reshape"([[MATMUL]], [[CST]]) : (tensor<3x5x10x1xf32>, tensor<3xi64>) -> tensor<15x10x1xf32>
  // CHECK: return [[RES]] : tensor<15x10x1xf32>
}

// -----

// Check the combining of transposes into a simple transpose.
// CHECK-LABEL: func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [2, 3, 0, 1]} : (tensor<13x12x11x10xf32>) -> tensor<11x10x13x12xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.Transpose"(%arg0) {perm = [1, 0, 3, 2]} : (tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32>
  "func.return"(%1) : (tensor<11x10x13x12xf32>) -> ()
}

// -----

// Check the combining of reshape into a simple reshape.
// CHECK-LABEL: func @test_reshape_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
func @test_reshape_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
  %0 = "onnx.Constant"() {value = dense<[10, 12, 11, 13]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
  %2 = "onnx.Constant"() {value = dense<[11, 10, 13, 12]> : tensor<4xi64> } : () -> tensor<4xi64>
  %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<11x10x13x12xf32>
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Constant"() {value = dense<[11, 10, 13, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
  // CHECK-NEXT: %{{.*}} = "onnx.Reshape"(%arg0, [[RES]]) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<11x10x13x12xf32>
  "func.return"(%3) : (tensor<11x10x13x12xf32>) -> ()
}

// -----

// Check the combining of transposes into an identity transpose, which in turns is removed.
// CHECK-LABEL: func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [3, 2, 1, 0]} : (tensor<13x12x11x10xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of reshape into an identity reshape, which in turns is removed.
// CHECK-LABEL: func @test_reshape_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func @test_reshape_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Constant"() {value = dense<[10, 12, 11, 13]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
  %2 = "onnx.Constant"() {value = dense<[10, 11, 12, 13]> : tensor<4xi64> } : () -> tensor<4xi64>
  %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "func.return"(%3) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

func @test_shape1(%arg0 : tensor<2x4x8x16xf32>) -> tensor<*xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_shape1
  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<[2, 4, 8, 16]> : tensor<4xi64>} : () -> tensor<*xi64>
  // CHECK-NEXT: %0 : tensor<*xi64>
}

// -----

func @test_shape2(%arg0 : tensor<?x4x8x16xf32>) -> tensor<*xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<?x4x8x16xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_shape2
  // CHECK-NEXT: %0 = "onnx.Shape"(%arg0) : (tensor<?x4x8x16xf32>) -> tensor<*xi64>
  // CHECK-NEXT: return %0 : tensor<*xi64>
}


// -----

func @test_size1(%arg0 : tensor<2x4x8x16xf32>) -> tensor<*xi64> {
  %0 = "onnx.Size"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_size1
  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<1024> : tensor<1xi64>} : () -> tensor<*xi64>
  // CHECK-NEXT: %0 : tensor<*xi64>
}

// -----

func @test_size2(%arg0 : tensor<*xf32>) -> tensor<*xi64> {
  %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: @test_size2
  // CHECK-NEXT: %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
  // CHECK-NEXT: return %0 : tensor<*xi64>
}

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMean
func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_average_pool
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [2, 3]} : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: return [[RES]] : tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMean with dynamic dimensions
func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  return %0 : tensor<1x?x?x1xf32>
  // CHECK-LABEL: test_global_average_pool_dyn_dims
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [2, 3]} : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  // CHECK: return [[RES]] : tensor<1x?x?x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax
func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
  // CHECK-LABEL: test_global_average_pool
  // CHECK: [[RES:%.+]] = "onnx.ReduceMax"(%arg0) {axes = [2, 3]} : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: return [[RES]] : tensor<1x3x1x1xf32>
}

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax with dynamic dimensions
func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  return %0 : tensor<1x?x?x1xf32>
  // CHECK-LABEL: test_global_average_pool_dyn_dims
  // CHECK: [[RES:%.+]] = "onnx.ReduceMax"(%arg0) {axes = [2, 3]} : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
  // CHECK: return [[RES]] : tensor<1x?x?x1xf32>
}

// -----

// COM: Test removing squeeze/unsqueeze pairs when they use the same axes.

func @test_remove_unsqueeze_squeeze(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Constant"() {value = dense<[0, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<2xi64>) -> tensor<10x10xf32>
  return %3: tensor<10x10xf32>

  // CHECK-LABEL: test_remove_unsqueeze_squeeze
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK-NOT: {{.*}} = onnx.Squeeze"{{.*}}
  // CHECK: return {{.*}}

}

// -----

func @test_remove_unsqueezev11_squeezev11(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes=[0, -2]} : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
  return %1: tensor<10x10xf32>

  // CHECK-LABEL: test_remove_unsqueezev11_squeezev11
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK-NOT: {{.*}} = onnx.SqueezeV11"{{.*}}
  // CHECK: return {{.*}}

}

// -----

func @test_should_not_remove_unsqueeze_squeeze(%arg0 : tensor<10x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
  %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  return %3: tensor<10x1x10xf32>
  // CHECK-LABEL: test_should_not_remove_unsqueeze_squeeze
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_should_not_remove_unsqueezev11_squeezev11(%arg0 : tensor<10x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes=[0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
  return %1: tensor<10x1x10xf32>
  // CHECK-LABEL: test_should_not_remove_unsqueezev11_squeezev11
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_remove_squeeze_unsqueeze(%arg0 : tensor<10x1x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x10xf32>
  %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  return %3: tensor<10x1x10xf32>
  // CHECK-LABEL: test_remove_squeeze_unsqueeze
  // CHECK-NOT: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_remove_squeezev11_unsqueezev11(%arg0 : tensor<10x1x10xf32>) -> tensor<10x1x10xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes=[1]} : (tensor<10x1x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[1]} : (tensor<10x10xf32>) -> tensor<10x1x10xf32>
  return %1: tensor<10x1x10xf32>
  // CHECK-LABEL: test_remove_squeezev11_unsqueezev11
  // CHECK-NOT: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK-NOT: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_should_not_remove_squeeze_unsqueeze(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
  %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10x1xf32>
  return %3: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_squeeze_unsqueeze
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_should_not_remove_squeezev11_unsqueezev11(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes=[0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[3]} : (tensor<10x1x10xf32>) -> tensor<10x1x10x1xf32>
  return %1: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_squeezev11_unsqueezev11
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_should_not_remove_null_axes_squeeze_unsqueeze(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %cst) : (tensor<1x10x1x10xf32>, none) -> tensor<10x10xf32>
  %2 = "onnx.Unsqueeze"(%1, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<10x1x10x1xf32>
  return %2: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_null_axes_squeeze_unsqueeze
  // CHECK: {{.*}} = "onnx.Squeeze"{{.*}}
  // CHECK: {{.*}} = "onnx.Unsqueeze"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_should_not_remove_null_axes_squeezev11_unsqueezev11(%arg0 : tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[1, 3]} : (tensor<10x10xf32>) -> tensor<10x1x10x1xf32>
  return %1: tensor<10x1x10x1xf32>
  // CHECK-LABEL: test_should_not_remove_null_axes_squeezev11_unsqueezev11
  // CHECK: {{.*}} = "onnx.SqueezeV11"{{.*}}
  // CHECK: {{.*}} = "onnx.UnsqueezeV11"{{.*}}
  // CHECK: return {{.*}}
}

// -----

func @test_constant_1() -> tensor<i64> {
  %0 = "onnx.Constant"() {value_int = 1 : si64} : () -> tensor<i64>
  return %0 : tensor<i64>
// CHECK-LABEL:       func @test_constant_1
// CHECK:           [[VAR_0:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<i64>
// CHECK:           return [[VAR_0]] : tensor<i64>
}


// -----

func @test_constant_2() -> tensor<f32> {
  %0 = "onnx.Constant"() {value_float = 2.0 : f32 } : () -> tensor<f32>
  return %0 : tensor<f32>
// CHECK-LABEL:     func @test_constant_2 
// CHECK: [[VAR_0:%.+]] = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<1xf32>} : () -> tensor<f32>
// CHECK: return [[VAR_0]] : tensor<f32>
}

// -----

func @test_constant_1() -> tensor<?xi64> {
  %0 = "onnx.Constant"() {value_ints = [1, 2, 3] } : () -> tensor<?xi64>
  return %0 : tensor<?xi64>
// CHECK-LABEL:       func @test_constant_1       
// CHECK-SAME:     () -> tensor<?xi64> {
// CHECK:           [[VAR_0:%.+]] = "onnx.Constant"() {value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> tensor<?xi64>
// CHECK:           return [[VAR_0]] : tensor<?xi64>
}

// -----

func @test_rewrite_batchnormtestmode_Nd(%arg0 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    %scale = "onnx.Constant"() : () -> tensor<64xf32>
    %bias = "onnx.Constant"() : () -> tensor<64xf32>
    %mean = "onnx.Constant"() : () -> tensor<64xf32>
    %var = "onnx.Constant"() : () -> tensor<64xf32>
    %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %scale, %bias, %mean, %var) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %0 :  tensor<1x64x112x112xf32>

    // CHECK-LABEL: test_rewrite_batchnormtestmode_Nd
    // CHECK: [[SCALE:%.*]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[BIAS:%.*]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[MEAN:%.*]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[VAR:%.*]] = "onnx.Constant"() : () -> tensor<64xf32>
    // CHECK: [[EPSILON:%.*]] = "onnx.Constant"() {value = dense<1.00000007E-5> : tensor<1xf32>} : () -> tensor<1xf32>

    // CHECK: [[ADD:%.*]] = "onnx.Add"([[VAR]], [[EPSILON]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    // CHECK: [[SQRT:%.*]] = "onnx.Sqrt"([[ADD]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[A:%.*]] = "onnx.Div"([[SCALE]], [[SQRT]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[A_UNSQUEEZE:%.*]] = "onnx.UnsqueezeV11"([[A]]) {axes = [1, 2]} : (tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[X_A:%.*]] = "onnx.Mul"(%arg0, [[A_UNSQUEEZE]]) : (tensor<1x64x112x112xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[SUB:%.*]] = "onnx.Mul"([[MEAN]], [[A]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[B:%.*]] = "onnx.Sub"([[BIAS]], [[SUB]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[B_UNSQUEEZE:%.*]] = "onnx.UnsqueezeV11"([[B]]) {axes = [1, 2]} : (tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.*]] = "onnx.Add"([[X_A]], [[B_UNSQUEEZE]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32>
    // CHECK: return [[RES]] : tensor<1x64x112x112xf32>
}

// -----

func @test_rewrite_batchnormtestmode_1d(%arg0 : tensor<64xf32>) -> tensor<64xf32> {
    %scale = "onnx.Constant"() : () -> tensor<1xf32>
    %bias = "onnx.Constant"() : () -> tensor<1xf32>
    %mean = "onnx.Constant"() : () -> tensor<1xf32>
    %var = "onnx.Constant"() : () -> tensor<1xf32>
    %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %scale, %bias, %mean, %var) {epsilon = 1.00000007E-5 : f32} : (tensor<64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<64xf32>
    return %0 :  tensor<64xf32>

    // CHECK-LABEL: test_rewrite_batchnormtestmode_1d
    // CHECK: [[SCALE:%.*]] = "onnx.Constant"() : () -> tensor<1xf32>
    // CHECK: [[BIAS:%.*]] = "onnx.Constant"() : () -> tensor<1xf32>
    // CHECK: [[MEAN:%.*]] = "onnx.Constant"() : () -> tensor<1xf32>
    // CHECK: [[VAR:%.*]] = "onnx.Constant"() : () -> tensor<1xf32>
    // CHECK: [[EPSILON:%.*]] = "onnx.Constant"() {value = dense<1.00000007E-5> : tensor<1xf32>} : () -> tensor<1xf32>

    // CHECK: [[ADD:%.*]] = "onnx.Add"([[VAR]], [[EPSILON]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    // CHECK: [[SQRT:%.*]] = "onnx.Sqrt"([[ADD]]) : (tensor<1xf32>) -> tensor<*xf32>
    // CHECK: [[A:%.*]] = "onnx.Div"([[SCALE]], [[SQRT]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[X_A:%.*]] = "onnx.Mul"(%arg0, [[A]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[SUB:%.*]] = "onnx.Mul"([[MEAN]], [[A]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[B:%.*]] = "onnx.Sub"([[BIAS]], [[SUB]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.*]] = "onnx.Add"([[X_A]], [[B]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<64xf32>
    // CHECK: return [[RES]] : tensor<64xf32>
}

// -----

func @test_normalize_add(%arg0 : tensor<2xf32>) -> tensor<2xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0.0, 1.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %1 = "onnx.Add"(%0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> 
    return %1 : tensor<2xf32> 
    // CHECK-LABEL: test_normalize_add
    // CHECK: [[CONSTANT:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    // CHECK: [[RES:%.+]] = "onnx.Add"(%arg0, [[CONSTANT]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    // CHECK: return [[RES]] : tensor<2xf32>
}

// -----

func @test_fuse_add_conv(%arg0 : tensor<1x1x28x28xf32>, %arg1 : tensor<8x1x5x5xf32>) -> tensor<1x8x28x28xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Convolution28", strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, none) -> tensor<1x8x28x28xf32> 
    %1 = "onnx.Constant"() {value = dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<1x8x28x28xf32>, tensor<8x1x1xf32>) -> tensor<1x8x28x28xf32> 
    return %2 : tensor<1x8x28x28xf32> 

    // CHECK-LABEL: test_fuse_add_conv
    // CHECK-SAME:  ([[X:%.+]]: tensor<1x1x28x28xf32>, [[W:%.+]]: tensor<8x1x5x5xf32>) -> tensor<1x8x28x28xf32> {
    // CHECK: [[CONSTANT:%.+]] = "onnx.Constant"() {value = dense<{{.}}{{.}}[-0.161539719]{{.}}, {{.}}[-0.433835655]{{.}}, {{.}}[0.091641359]{{.}}, {{.}}[-0.0168522168]{{.}}, {{.}}[-0.0650264397]{{.}}, {{.}}[-0.131737873]{{.}}, {{.}}[0.0204175506]{{.}}, {{.}}[-0.121110231]{{.}}{{.}}> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    // CHECK: [[BIAS:%.+]] = "onnx.SqueezeV11"([[CONSTANT]]) {axes = [1, 2]} : (tensor<8x1x1xf32>) -> tensor<*xf32>
    // CHECK: [[RES:%.+]] = "onnx.Conv"([[X]], [[W]], [[BIAS]]) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, tensor<*xf32>) -> tensor<1x8x28x28xf32>
    // CHECK: return [[RES]] : tensor<1x8x28x28xf32>
    // CHECK: }
}
