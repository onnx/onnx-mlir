// RUN: onnx-mlir-opt --fuse-pad-into-avgpool %s | FileCheck %s

func.func @test_fuse_pad_avgpool(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x8x8xf32> {
    %0 = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 2, 2]> : tensor<8xi64>
    %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x1x7x7xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      pads = [1, 1, 1, 1],
      strides = [1, 1]} : (tensor<1x1x7x7xf32>) -> tensor<1x1x8x8xf32>
    return %4 : tensor<1x1x8x8xf32>
  }


// CHECK-LABEL: func.func @test_fuse_pad_avgpool
// CHECK-NOT: onnx.Pad
// CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
// CHECK-SAME: kernel_shape = [2, 2]
// CHECK-SAME: pads = [2, 2, 3, 3]
// CHECK-SAME: strides = [1, 1]
// CHECK: return %[[POOL]]