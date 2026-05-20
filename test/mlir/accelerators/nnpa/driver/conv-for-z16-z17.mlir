// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZHighIR --printIR %s | FileCheck %s --check-prefix=CHECK-Z17
// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZHighIR --printIR %s | FileCheck %s --check-prefix=CHECK-Z16

// -----

func.func @conv_3x3_dyn(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<?x?x?x?xf32>, tensor<64x3x3x3xf32>, none) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>

// COM: Cannot use conv because of the dynamic image shape, but matmul with broadcast is fine.
// CHECK-Z17-LABEL:  func.func @conv_3x3_dyn
// CHECK-Z17:        onnx.Im2Col
// CHECK-Z17:        zhigh.MatMul

// COM: Cannot use conv because of the dynamic image shape, cannot use matmul with broadcast, leave as is.
// CHECK-Z16-LABEL:  func.func @conv_3x3_dyn
// CHECK-Z16:        onnx.Conv
}

// -----

func.func @conv_3x3_dyn_bs1(%arg0: tensor<1x?x?x?xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<1x?x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x?x?x?xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x?x?x?xf32>
  return %1 : tensor<1x?x?x?xf32>

// COM: Cannot use conv because of the dynamic image shape, but matmul with broadcast is fine.
// CHECK-Z17-LABEL:  func.func @conv_3x3_dyn_bs1
// CHECK-Z17:        onnx.Im2Col
// CHECK-Z17:        zhigh.MatMul

// COM: Cannot use conv because of the dynamic image shape, but matmul with broadcast is fine.
// CHECK-Z16-LABEL:  func.func @conv_3x3_dyn_bs1
// CHECK-Z16:        onnx.Im2Col
// CHECK-Z16:        zhigh.MatMul
}

// -----

func.func @conv_3x3_static(%arg0: tensor<2x3x256x256xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<2x3x256x256xf32>, tensor<64x3x3x3xf32>, none) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>

// COM: Can use conv.
// CHECK-Z17-LABEL:  func.func @conv_3x3_static
// CHECK-Z17:        zhigh.Conv2D

// COM: Can use conv.
// CHECK-Z16-LABEL:  func.func @conv_3x3_static
// CHECK-Z16:        zhigh.Conv2D
}

// -----

func.func @conv_1x1_dyn(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<?x?x?x?xf32>, tensor<64x3x1x1xf32>, none) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>

// COM: Cannot use conv, but matmul is good (broadcast ok).
// CHECK-Z17-LABEL:  func.func @conv_1x1_dyn
// CHECK-Z17-NOT:    onnx.Im2Col
// CHECK-Z17:        zhigh.MatMul

// COM: Cannot use conv, but matmul is not good (broadcast 1xN), matmul better?
// CHECK-Z16-LABEL:  func.func @conv_1x1_dyn
// CHECK-Z16-NOT:    onnx.Im2Col
// CHECK-Z16:        onnx.MatMul
}

// -----

func.func @conv_1x1_dyn_bs1(%arg0: tensor<1x3x?x?xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<1x64x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x?x?xf32>, tensor<64x3x1x1xf32>, none) -> tensor<1x64x?x?xf32>
  return %1 : tensor<1x64x?x?xf32>

// COM: Cannot use conv, but matmul is good (broadcast ok).
// CHECK-Z17-LABEL:  func.func @conv_1x1_dyn_bs1
// CHECK-Z17-NOT:    onnx.Im2Col
// CHECK-Z17:        zhigh.MatMul

// COM: Cannot use conv, but matmul is not good (broadcast 1xN), matmul better?
// CHECK-Z16-LABEL:  func.func @conv_1x1_dyn_bs1
// CHECK-Z16-NOT:    onnx.Im2Col
// CHECK-Z16:        zhigh.MatMul
}

// -----

func.func @conv_1x1_static(%arg0: tensor<1x3x128x128xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<1x64x128x128xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x128x128xf32>, tensor<64x3x1x1xf32>, none) -> tensor<1x64x128x128xf32>
  return %1 : tensor<1x64x128x128xf32>

// COM: Can use conv.
// CHECK-Z17-LABEL:  func.func @conv_1x1_static
// CHECK-Z17:        zhigh.Conv2D

// COM: Can use conv.
// CHECK-Z16-LABEL:  func.func @conv_1x1_static
// CHECK-Z16:        zhigh.Conv2D
}
