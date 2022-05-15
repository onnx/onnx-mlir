// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: @test_reducel1(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel1(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL1"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[ABS:%.+]] =  "onnx.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[ABS]], [[CST]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducel2(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel2(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL2"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[ABS]], [[CST]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQRT:%.+]] =  "onnx.Sqrt"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsum(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsum(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSum"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"(%arg0, [[CST]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x?xf32>, tensor<1xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsumexp(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsumexp(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_MAX:%.+]] = "onnx.ReduceMax"(%arg0) {axes = [1], keepdims = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SUB:%.+]] = "onnx.Sub"(%arg0, [[REDUCE_MAX]]) : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[EXP:%.+]] = "onnx.Exp"([[SUB]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]], [[CST]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32> 
  // CHECK-NEXT: [[LOG:%.+]] = "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQUEEZE:%.+]] = "onnx.SqueezeV11"([[REDUCE_MAX]]) {axes = [1]} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Add"([[LOG]], [[SQUEEZE]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return [[RES]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsumexp_keepdims(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsumexp_keepdims(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0) {axes=[1], keepdims = 1 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_MAX:%.+]] = "onnx.ReduceMax"(%arg0) {axes = [1], keepdims = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SUB:%.+]] = "onnx.Sub"(%arg0, [[REDUCE_MAX]]) : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[EXP:%.+]] = "onnx.Exp"([[SUB]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]], [[CST]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] = "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Add"([[LOG]], [[REDUCE_MAX]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return [[RES]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducesumsquare(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducesumsquare(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumSquare"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[SQUARE:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[CST:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[SQUARE]], [[CST]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
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

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi32>) -> tensor<3xf32>
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

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi32>) -> tensor<*xf32>
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

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi32>) -> tensor<*xf32>
  // CHECK-NEXT: %1 = "onnx.Constant"() {value = dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK-NEXT: %2 = "onnx.Sub"(%0, %1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %2 : tensor<3xf32>
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

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi32>) -> tensor<*xf32>
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
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_logsoftmax
  // CHECK: [[SOFTMAX:%.+]] = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  // CHECK: [[RES:%.+]] = "onnx.Log"([[SOFTMAX]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return [[RES]] : tensor<*xf32>
}

// -----

func @test_upsample(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<4xf32>) -> tensor<1x1x4x6xf32> {
  %0 = "onnx.Upsample"(%arg0, %arg1) {mode = "nearest"} : (tensor<1x1x2x2xf32>, tensor<4xf32>) -> tensor<1x1x4x6xf32>
  return %0 : tensor<1x1x4x6xf32>
  // CHECK-LABEL: test_upsample
  // CHECK: [[NONE_0:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[NONE_1:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[RES:%.+]] = "onnx.Resize"(%arg0, [[NONE_0]], %arg1, [[NONE_1]]) {mode = "nearest"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x6xf32>
  // CHECK: return [[RES]] : tensor<1x1x4x6xf32>
}

// -----

func @test_upsamplev9(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<4xf32>) -> tensor<1x1x4x6xf32> {
  %0 = "onnx.UpsampleV9"(%arg0, %arg1) {mode = "nearest"} : (tensor<1x1x2x2xf32>, tensor<4xf32>) -> tensor<1x1x4x6xf32>
  return %0 : tensor<1x1x4x6xf32>
  // CHECK-LABEL: test_upsamplev9
  // CHECK: [[NONE_0:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[NONE_1:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[RES:%.+]] = "onnx.Resize"(%arg0, [[NONE_0]], %arg1, [[NONE_1]]) {mode = "nearest"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x6xf32>
  // CHECK: return [[RES]] : tensor<1x1x4x6xf32>
}

// -----

func @test_upsamplev7(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x6xf32> {
  %0 = "onnx.UpsampleV7"(%arg0) {mode = "nearest", scales = [0.1 : f32, 0.2 : f32, 0.3 : f32, 0.4 : f32]} : (tensor<1x1x2x2xf32>) -> tensor<1x1x4x6xf32>
  return %0 : tensor<1x1x4x6xf32>
  // CHECK-LABEL: test_upsamplev7
  // CHECK: [[NONE_0:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[SCALES:%.+]] = "onnx.Constant"() {value = dense<[1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01]> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK: [[NONE_1:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[RES:%.+]] = "onnx.Resize"(%arg0, [[NONE_0]], [[SCALES]], [[NONE_1]]) {mode = "nearest"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x6xf32>
  // CHECK: return [[RES]] : tensor<1x1x4x6xf32>
}

// -----

func @test_padv2(%arg0: tensor<1x3x224x224xf32>) -> tensor<*xf32> {
    %0 = "onnx.PadV2"(%arg0) {mode = "reflect", pads = [0, 0, 4, 4, 0, 0, 4, 4]} : (tensor<1x3x224x224xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
    // CHECK-LABEL: test_padv2
    // CHECK: [[PAD:%.+]] = "onnx.Constant"() {value = dense<[0, 0, 4, 4, 0, 0, 4, 4]> : tensor<8xi64>} : () -> tensor<8xi64>
    // CHECK: [[CONSTANT_VALUE:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    // CHECK: [[RES:%.+]] = "onnx.Pad"(%arg0, [[PAD]], [[CONSTANT_VALUE]]) {mode = "reflect"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<*xf32>
    // CHECK: return [[RES]] : tensor<*xf32>
}

// -----

  func @test_resizev10(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
    %0 = "onnx.ResizeV10"(%arg0, %arg1) {mode = "nearest"} : (tensor<1x2x3x4xf32>, tensor<4xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

    // CHECK-LABEL:  func @test_resizev10
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<4xf32>) -> tensor<*xf32> {
    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
    // CHECK:           [[VAR_2_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]], [[VAR_1_]]) {mode = "nearest"} : (tensor<1x2x3x4xf32>, tensor<0xf32>, tensor<4xf32>, tensor<0xi64>) -> tensor<*xf32>
    // CHECK:           return [[VAR_2_]] : tensor<*xf32>
  }

  func @test_resizev11(%arg0: tensor<*xf32>, %arg1: tensor<*xi64>) -> tensor<*xf32> {
    %0 = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    %1 = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    %2 = "onnx.Resize"(%arg0, %0, %0, %arg1) {coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "floor", onnx_node_name = "Resize__697"} : (tensor<*xf32>, tensor<0xf32>, tensor<0xf32>, tensor<*xi64>) -> tensor<*xf32>
    return %2 : tensor<*xf32>

    // CHECK-LABEL:  func @test_resizev11
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xi64>) -> tensor<*xf32> {
    // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    // CHECK:           [[VAR_2_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[VAR_0_]], [[PARAM_1_]]) {coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "floor", onnx_node_name = "Resize__697"} : (tensor<*xf32>, tensor<0xf32>, tensor<0xf32>, tensor<*xi64>) -> tensor<*xf32>
    // CHECK:           return [[VAR_2_]] : tensor<*xf32>
  }

func @test_seqence_construct_1(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>> {
  %0 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>>
  return %0 : !onnx.Seq<tensor<*xf32>>

  // CHECK-LABEL:  func @test_seqence_construct_1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
  // CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<*xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.SequenceInsert"([[VAR_1_]], [[PARAM_1_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<*xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  // CHECK:           return [[VAR_2_]] : !onnx.Seq<tensor<*xf32>>
}

// -----

func @test_clipv6(%arg0 : tensor<*xf32>) -> () {
  %0 = "onnx.ClipV6"(%arg0) {max = 6.000000e+00 : f32, min = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  return 

  // CHECK-LABEL:  func @test_clipv6
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.Clip"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<*xf32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  // CHECK:           return
}

// -----

func @test_scatter(%arg0: tensor<64x25600xf32>, %arg1: tensor<64x100xi64>, %arg2: tensor<64x100xf32>) -> tensor<*xf32> {
  %0 = "onnx.Scatter"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL:  func @test_scatter
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<64x25600xf32>, [[PARAM_1:%.+]]: tensor<64x100xi64>, [[PARAM_2:%.+]]: tensor<64x100xf32>) -> tensor<*xf32> {
  // CHECK-NEXT:      [[RES:%.+]] = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<*xf32>
  // CHECK-NEXT:      return [[RES]] : tensor<*xf32>
}

