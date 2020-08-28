// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @test_matmul_add_fused(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32> {
func @test_matmul_add_fused(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%1) : (tensor<10x10xf32>) -> ()
}

// -----

// onnx.MatMul ops for non 2-D matrices should not get fused because Gemm only supports 2-D matrices.
// CHECK-LABEL: func @test_matmul_add_not_fused(%{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
func @test_matmul_add_not_fused(%a0: tensor<10x10x10xf32>, %a1: tensor<10x10x10xf32>, %a2: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  "std.return"(%1) : (tensor<10x10x10xf32>) -> ()
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
  "std.return"(%3) : (tensor<10x10xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_identity_identity(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_identity_identity(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.Identity"(%a0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Identity"(%a1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%2) : (tensor<10x10xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_constant_pad(%{{.*}}: tensor<?x?xf32>) -> tensor<*xf32> {
func @test_constant_pad(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: [[SQUARE:%.+]] = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 2, 0, 0]} : (tensor<?x?xf32>) -> tensor<*xf32> 
  %0 ="onnx.Constant"() {value=[0, 2, 0, 0]} : ()-> tensor<?xi64>
  %2 = "onnx.PadConstantValue"(%arg0, %0) {constant_value=0. : f32, mode = "constant"} : (tensor<?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()
}

// -----

// CHECK-LABEL: @test_conv_split(%{{.*}}: tensor<1x9x32x64xf32>, %{{.*}}: tensor<5x9x6x7xf32>) -> tensor<*xf32> {
func @test_conv_split(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 3, 4, 5]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: %cst = constant unit
  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<[0, 0, 2, 3, 0, 0, 4, 5]> : tensor<8xi64>} : () -> tensor<8xi64>
  // CHECK-NEXT: %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK-NEXT: %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<1x9x32x64xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %3 = "onnx.Conv"(%2, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, pads = [0, 0, 0, 0]} : (tensor<*xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  // CHECK-NEXT: return %3 : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion_rank3(%{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<256xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion_rank3(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @cast_elimination(%{{.*}}: tensor<2xf32>) -> tensor<2xf32> {
func @cast_elimination(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : i64} : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>

  // CHECK-NEXT: return %arg0 : tensor<2xf32>
}

// -----

func @test_conv_batchnormtestmode_fusion_nobias(%arg0 : tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %cst = constant unit
    %0 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Conv"(%arg0, %0, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none) -> tensor<1x64x112x112xf32>
    %2 = "onnx.Constant"() : () -> tensor<64xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.BatchNormalizationTestMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
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
    // CHECK: [[UNSQUEEZE:%.+]] = "onnx.Unsqueeze"([[COEFFICIENT_W]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_WEIGHT:%.+]] = "onnx.Mul"([[WEIGHT]], [[UNSQUEEZE]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[NEG_MEAN:%.+]] = "onnx.Neg"([[MEAN]]) : (tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[MUL:%.+]] = "onnx.Mul"([[COEFFICIENT_W]], [[NEG_MEAN]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_BIAS:%.+]] = "onnx.Add"([[B]], [[MUL]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[PAD_ARG1:%.+]] = "onnx.Constant"() {value = dense<[0, 0, 3, 3, 0, 0, 3, 3]> : tensor<8xi64>} : () -> tensor<8xi64>
    // CHECK: [[PAD_ARG2:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: [[PADDED_INPUT:%.+]] = "onnx.Pad"(%arg0, [[PAD_ARG1]], [[PAD_ARG2]]) {mode = "constant"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.+]] = "onnx.Conv"([[PADDED_INPUT]], [[NEW_WEIGHT]], [[NEW_BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32> 

    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationTestMode"{{.*}}

    // CHECK: return [[RES]] : tensor<1x64x112x112xf32>
}

// -----

func @test_conv_batchnormtestmode_fusion(%arg0 : tensor<1x3x224x224xf32>, %arg1 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %cst = constant unit
    %0 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %1 = "onnx.Conv"(%arg0, %0, %arg1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %2 = "onnx.Constant"() : () -> tensor<64xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.BatchNormalizationTestMode"(%1, %2, %3, %4, %5) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
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
    // CHECK: [[UNSQUEEZE:%.+]] = "onnx.Unsqueeze"([[COEFFICIENT_W]]) {axes = [1, 2, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_WEIGHT:%.+]] = "onnx.Mul"([[WEIGHT]], [[UNSQUEEZE]]) : (tensor<64x3x7x7xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[SUB:%.+]] = "onnx.Sub"(%arg1, [[MEAN]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    // CHECK: [[MUL:%.+]] = "onnx.Mul"([[COEFFICIENT_W]], [[SUB]]) : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    // CHECK: [[NEW_BIAS:%.+]] = "onnx.Add"([[B]], [[MUL]]) : (tensor<64xf32>, tensor<*xf32>) -> tensor<*xf32>

    // CHECK: [[PAD_ARG1:%.+]] = "onnx.Constant"() {value = dense<[0, 0, 3, 3, 0, 0, 3, 3]> : tensor<8xi64>} : () -> tensor<8xi64>
    // CHECK: [[PAD_ARG2:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: [[PADDED_INPUT:%.+]] = "onnx.Pad"(%arg0, [[PAD_ARG1]], [[PAD_ARG2]]) {mode = "constant"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<*xf32>

    // CHECK: [[RES:%.+]] = "onnx.Conv"([[PADDED_INPUT]], [[NEW_WEIGHT]], [[NEW_BIAS]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<1x64x112x112xf32> 

    // CHECK-NOT: {{.*}} = "onnx.BatchNormalizationTestMode"{{.*}}

    // CHECK: return [[RES]] : tensor<1x64x112x112xf32>
}

// -----

// Check the removal of identity transposes.
// CHECK-LABEL: func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func @test_transpose_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [0, 1, 2, 3]} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "std.return"(%0) : (tensor<10x11x12x13xf32>) -> ()
}

// -----

// Check the combining of transposes into a simple transpose.
// CHECK-LABEL: func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
func @test_transpose_fusion(%arg0: tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [2, 3, 0, 1]} : (tensor<13x12x11x10xf32>) -> tensor<11x10x13x12xf32>
  // CHECK-NEXT: %{{.*}} = "onnx.Transpose"(%arg0) {perm = [1, 0, 3, 2]} : (tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32>
  "std.return"(%1) : (tensor<11x10x13x12xf32>) -> ()
}

// -----

// Check the combining of transposes into an identity transpose, which in turns is removed.
// CHECK-LABEL: func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
func @test_transpose_fusion_removal(%arg0: tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %0 = "onnx.Transpose"(%arg0)  {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
  %1 = "onnx.Transpose"(%0)  {perm = [3, 2, 1, 0]} : (tensor<13x12x11x10xf32>) -> tensor<10x11x12x13xf32>
  // CHECK-NEXT: return %arg0 : tensor<10x11x12x13xf32>
  "std.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
}
