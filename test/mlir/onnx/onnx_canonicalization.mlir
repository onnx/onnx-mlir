// RUN: onnf-opt --canonicalize %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_matmul_add_fused(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32> {
func @test_matmul_add_fused(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%1) : (tensor<10x10xf32>) -> ()
}

// onnx.MatMul ops for non 2-D matrices should not get fused because Gemm only supports 2-D matrices.
// CHECK-LABEL: func @test_matmul_add_not_fused(%{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>, %{{.*}}: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
func @test_matmul_add_not_fused(%a0: tensor<10x10x10xf32>, %a1: tensor<10x10x10xf32>, %a2: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  "std.return"(%1) : (tensor<10x10x10xf32>) -> ()
}


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

// CHECK-LABEL: @test_identity_identity(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_identity_identity(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.Identity"(%a0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Identity"(%a1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%2) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL: @test_constant_pad(%{{.*}}: tensor<?x?xf32>) -> tensor<*xf32> {
func @test_constant_pad(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: [[SQUARE:%.+]] = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 2, 0, 0]} : (tensor<?x?xf32>) -> tensor<*xf32> 
  %0 ="onnx.Constant"() {value=[0, 2, 0, 0]} : ()-> tensor<?xi64>
  %2 = "onnx.PadConstantValue"(%arg0, %0) {constant_value=0. : f32, mode = "constant"} : (tensor<?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: @test_conv_split(%{{.*}}: tensor<1x9x32x64xf32>, %{{.*}}: tensor<5x9x6x7xf32>) -> tensor<*xf32> {
func @test_conv_split(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 3, 4, 5]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NEXT: %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 0, 2, 3, 0, 0, 4, 5]} : (tensor<1x9x32x64xf32>) -> tensor<1x9x38x72xf32>
  // CHECK-NEXT: %1 = "onnx.ConvNoBias"(%0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [0, 0, 0, 0]} : (tensor<1x9x38x72xf32>, tensor<5x9x6x7xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return %1 : tensor<*xf32>
}

//CHECK-LABEL: @test_gemm_add_fusion(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

//CHECK-LABEL: @test_gemm_add_fusion_rank3(%{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<256xf32>) -> tensor<*xf32> {
func @test_gemm_add_fusion_rank3(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64} : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>
  // return [[GEMM]] : tensor<*xf32>
}

//CHECK-LABEL: @test_maxpoolsingleout_split(%{{.*}}: tensor<5x5x32x32xf32>) -> tensor<5x8x32x39xf32> {
func @test_maxpoolsingleout_split(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x8x32x39xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [5,3], pads = [1, 2, 3, 4] } : (tensor<5x5x32x32xf32>) -> tensor<5x8x32x39xf32>
  "std.return"(%0) : (tensor<5x8x32x39xf32>) -> ()

  // CHECK-NEXT: %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0xFF800000 : f32, mode = "constant", pads = [0, 0, 1, 2, 0, 0, 3, 4]} : (tensor<5x5x32x32xf32>) -> tensor<5x8x32x39xf32>
  // CHECK-NEXT: %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [5, 3], pads = [0, 0, 0, 0], storage_order = 0 : i64} : (tensor<5x8x32x39xf32>) -> tensor<5x8x32x39xf32>
  // CHECK-NEXT: return %1 : tensor<5x8x32x39xf32>
}

//CHECK-LABEL: @test_maxpoolsingleout_split_unknown_dims(%{{.*}}: tensor<*xf32>) -> tensor<*xf32> {
func @test_maxpoolsingleout_split_unknown_dims(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [5,3], pads = [1, 2, 3, 4] } : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0xFF800000 : f32, mode = "constant", pads = [0, 0, 1, 2, 0, 0, 3, 4]} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [5, 3], pads = [0, 0, 0, 0], storage_order = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return %1 : tensor<*xf32>
}

