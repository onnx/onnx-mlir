// RUN: onnf-opt --decompose-onnx %s -split-input-file | FileCheck %s

// CHECK-LABEL: @test_reducel1(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel1(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL1"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[ABS:%.+]] =  "onnx.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[ABS]]) {axes = [1], keepdims = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducel2(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel2(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL2"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[MUL]]) {axes = [1], keepdims = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQRT:%.+]] =  "onnx.Sqrt"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducelogsum(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsum(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSum"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"(%arg0) {axes = [1], keepdims = 0 : i64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducelogsumexp(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsumexp(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[EXP:%.+]] =  "onnx.Exp"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]]) {axes = [1], keepdims = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducesumsquare(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducesumsquare(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumSquare"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[SQUARE:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[SQUARE]]) {axes = [1], keepdims = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
}

//CHECK-LABEL: @test_maxpoolsingleout_split(%{{.*}}: tensor<5x5x32x32xf32>) -> tensor<*xf32> {
func @test_maxpoolsingleout_split(%arg0: tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [5,3], pads = [1, 2, 3, 4] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 0, 1, 2, 0, 0, 3, 4]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %1 = "onnx.MaxPoolSingleOutNoPads"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [5, 3], pads = [0, 0, 0, 0, 0, 0, 0, 0], storage_order = 0 : i64} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return %1 : tensor<*xf32>
}

//CHECK-LABEL: @test_maxpoolsingleout_no_split(%{{.*}}: tensor<5x5x32x32xf32>) -> tensor<*xf32> {
func @test_maxpoolsingleout_no_split(%arg0: tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [5,3]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: %0 = "onnx.MaxPoolSingleOutNoPads"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [5, 3], storage_order = 0 : i64} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return %0 : tensor<*xf32>
}
