// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: @test_reducel1(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel1(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL1"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[ABS:%.+]] =  "onnx.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[ABS]]) {axes = [1], keepdims = 0 : si64} : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducel2(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel2(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL2"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[MUL]]) {axes = [1], keepdims = 0 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQRT:%.+]] =  "onnx.Sqrt"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsum(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsum(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSum"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"(%arg0) {axes = [1], keepdims = 0 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsumexp(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsumexp(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[EXP:%.+]] =  "onnx.Exp"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]]) {axes = [1], keepdims = 0 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducesumsquare(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducesumsquare(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumSquare"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[SQUARE:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[SQUARE]]) {axes = [1], keepdims = 0 : si64} : (tensor<*xf32>) -> tensor<*xf32>
}

// -----
// Scaler Pattern test
// -----

// null
// CHECK-LABEL: func @test_scaler_null_float(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func @test_scaler_null_float(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: return %arg0 : tensor<3xf32>
}

// -----

// null not float
// CHECK-LABEL: func @test_scaler_null(%{{.*}}: tensor<3xi32>) -> tensor<3xf32> {
func @test_scaler_null(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) : (tensor<3xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<3xi32>) -> tensor<3xf32>
  // CHECK-NEXT: return %0 : tensor<3xf32>
}

// -----

// scaler no offset
// CHECK-LABEL: func @test_scaler_no_offset(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func @test_scaler_no_offset(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %1 = "onnx.Mul"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %1 : tensor<3xf32>
}

// -----

// scaler no offset, int input
// CHECK-LABEL: func @test_scaler_no_offset2(%{{.*}}: tensor<3xi32>) -> tensor<3xf32> {
func @test_scaler_no_offset2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<3xi32>) -> tensor<*xf32>
  // CHECK-NEXT: %1 = "onnx.Constant"() {value = dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %2 = "onnx.Mul"(%0, %1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %2 : tensor<3xf32>
}

// -----

// scaler no scale
// CHECK-LABEL: func @test_scaler_no_scale(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func @test_scaler_no_scale(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %1 = "onnx.Sub"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %1 : tensor<3xf32>
}

// -----

// scaler no scale, int input
// CHECK-LABEL: func @test_scaler_no_scale2(%{{.*}}: tensor<3xi32>) -> tensor<3xf32> {
func @test_scaler_no_scale2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  //CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<3xi32>) -> tensor<*xf32>
  //CHECK-NEXT: %1 = "onnx.Constant"() {value = dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>} : () -> tensor<3xf32>
  //CHECK-NEXT: %2 = "onnx.Sub"(%0, %1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
  //CHECK-NEXT: return %2 : tensor<3xf32>
}

// -----

// normal scaler
// CHECK-LABEL: func @test_scaler_normal(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func @test_scaler_normal(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32], scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %1 = "onnx.Sub"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %2 = "onnx.Constant"() {value = dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %3 = "onnx.Mul"(%1, %2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %3 : tensor<3xf32>
}

// -----

// normal scaler, int input
// CHECK-LABEL: func @test_scaler_normal2(%{{.*}}: tensor<3xi32>) -> tensor<3xf32> {
func @test_scaler_normal2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32], scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<3xi32>) -> tensor<*xf32>
  // CHECK-NEXT: %1 = "onnx.Constant"() {value = dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %2 = "onnx.Sub"(%0, %1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %3 = "onnx.Constant"() {value = dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %4 = "onnx.Mul"(%2, %3) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %4 : tensor<3xf32>
}

// -----

// normal scaler with constant offset and scale
// CHECK-LABEL: func @test_scaler_constant(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func @test_scaler_constant(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32], scale = [3.125000e-02 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<1986.99939> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK-NEXT: %1 = "onnx.Sub"(%arg0, %0) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %2 = "onnx.Constant"() {value = dense<3.125000e-02> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK-NEXT: %3 = "onnx.Mul"(%1, %2) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %3 : tensor<3xf32>
}

// -----

// Rewrite LogSoftmax using Log and Softmax.
func @test_logsoftmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LogSoftmax"(%arg0) {axis=1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_logsoftmax
  // CHECK: [[SOFTMAX:%.+]] = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  // CHECK: [[RES:%.+]] = "onnx.Log"([[SOFTMAX]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return [[RES]] : tensor<*xf32>
}

