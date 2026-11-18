// Baselines for the negative-pad cases in UpsampleAndPad.mlir: the same
// computation as an onnx.UpsampleAndPad with only non-negative pads, followed
// by an onnx.Slice that removes the cropped elements.


func.func @test_upsample_and_mixed_pads(%arg0 : tensor<1x1x3x4xf32> {onnx.name = "x"}) -> (tensor<1x1x4x9xf32> {onnx.name = "y"}) {
  %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 3], pads = [0, 1, 2, 0]} : (tensor<1x1x3x4xf32>) -> tensor<1x1x7x11xf32>
  %starts = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[7, 9]> : tensor<2xi64>
  %axes = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<1x1x7x11xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x1x4x9xf32>
  "func.return"(%1) : (tensor<1x1x4x9xf32>) -> ()
}
