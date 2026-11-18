// Baselines for the negative-pad cases in Pad_with_canonicalize.mlir: the
// same computation as an onnx.Slice that removes the cropped elements,
// followed by an onnx.Pad with only non-negative pads.


func.func @pad_constant_mode_negative_pads(%arg0: tensor<4x5xf32> {onnx.name = "x"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[4, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x3xf32>
  %pads = onnx.Constant dense<[0, 1, 0, 0]> : tensor<4xi64>
  %1 = "onnx.Pad"(%0, %pads, %cst, %cst) {mode = "constant"} : (tensor<3x3xf32>, tensor<4xi64>, none, none) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

func.func @pad_constant_mode_negative_pads_dynamic(%arg0: tensor<?x?xf32> {onnx.name = "x"}) -> (tensor<?x?xf32> {onnx.name = "y"}) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[9223372036854775807, -2]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x?xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  %pads = onnx.Constant dense<[0, 1, 0, 0]> : tensor<4xi64>
  %1 = "onnx.Pad"(%0, %pads, %cst, %cst) {mode = "constant"} : (tensor<?x?xf32>, tensor<4xi64>, none, none) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
