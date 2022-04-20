// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @mod(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{fmod must be 1 when the input type is floating point}}
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @test_depth_to_space_default(%arg0 : tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{The input tensor depth must be divisible by the (blocksize * blocksize)}}  
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 7 : si64} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  "std.return"(%0) : (tensor<1x16x32x64xf32>) -> ()
}

// -----

func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  // expected-error @+1 {{onnx.ArgMax 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ArgMax"(%arg0) { axis = 4 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "std.return"(%1) : (tensor<*xi64>) -> ()
}

// -----

func @test_argmin_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  // expected-error @+1 {{onnx.ArgMin 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ArgMin"(%arg0) { axis = 4 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>  
  "std.return"(%1) : (tensor<*xi64>) -> ()
}

// -----

func @test_compress_verifier_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x1x32xi1>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Compress 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.Compress"(%arg0, %arg1) { axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi1>)  -> tensor<*xf32>  
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_concat_verifier_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Concat 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_concat_verifier_2(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{Concat inputs must all have the same rank}}  
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_concat_verifier_3(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{Concat input dimensions must be all identical, except for dimension on the axis of the concatenation. Expected something compatible with: 'tensor<5x5x1x32xf32>' but got 'tensor<5x5x3x32xf32>' instead.}}  
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 1 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_constantofshape_verifier_1(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must be a 1D tensor}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
  "std.return"(%1) : (tensor<2x2xi64>) -> ()
}

// -----

func @test_constantofshape_verifier_2(%arg0: tensor<2x2x2x2xi64>) -> tensor<2x2x2x2xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must be a 1D tensor}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<2x2x2x2xi64>) -> tensor<2x2x2x2xi64>
  "std.return"(%1) : (tensor<2x2x2x2xi64>) -> ()
}

// -----

func @test_constantofshape_verifier_3(%arg0: tensor<?xi64>) -> tensor<?xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must have static shape}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<?xi64>) -> tensor<?xi64>
  "std.return"(%1) : (tensor<?xi64>) -> ()
}

// -----

func @test_constantofshape_verifier_4() -> tensor<2xi64> {
   // expected-error @+2 {{'onnx.ConstantOfShape' op All values of the input tensor must be >=0}}
   %0 = "onnx.Constant"(){ value = dense<[-1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
   %1 = "onnx.ConstantOfShape"(%0) : (tensor<2xi64>) -> tensor<2xi64>
  "std.return"(%1) : (tensor<2xi64>) -> ()
}

// -----

func @test_flatten_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{ONNXFlattenOP: axis() value is out of range}}
  %1 = "onnx.Flatten"(%arg0) { axis = 5 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_onehotencoder_verifier_1(%arg0: tensor<2x2xf32>) -> tensor<*xf32> {
   // expected-error @+1 {{'onnx.OneHotEncoder' op input is a tensor of float, int32, or double, but no cats_int64s attribute}}
   %1 = "onnx.OneHotEncoder"(%arg0) { cats_string = ["a","b","c"]} : (tensor<2x2xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_onehotencoder_verifier_2(%arg0: tensor<2x2x!onnx.String>) -> tensor<*x!onnx.String> {
   // expected-error @+1 {{'onnx.OneHotEncoder' op input is not a tensor of float, int32, or double, but no cats_strings attribute}}
   %1 = "onnx.OneHotEncoder"(%arg0) { cats_int64s = [1,2,3]} : (tensor<2x2x!onnx.String>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_pow_verifier_1(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @test_scatterelements_verifier_1(%arg0 : tensor<f32>, %arg1 : tensor<5xi64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: operand '<block argument> of type 'tensor<f32>' at index: 0' has rank 0, rank should be > 0}}  
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<f32>, tensor<5xi64>, tensor<5xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_scatterelements_verifier_2(%arg0 : tensor<5xf32>, %arg1 : tensor<5x3xi64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: operand '<block argument> of type 'tensor<5x3xi64>' at index: 1' has rank 2, rank should be 1}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<5xf32>, tensor<5x3xi64>, tensor<5xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_scatterelements_verifier_3(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x1x32xi64>, %arg2 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<5x5x1x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_scatterelements_verifier_4(%arg0 : tensor<3xf32>, %arg1 : tensor<3xf32>) -> tensor<*xf32> {
  // expected-error @+2 {{onnx.ScatterElements 'indices' value is 3, accepted range is [-3, 2]}}
  %indices = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ScatterElements"(%arg0, %indices, %arg1) {axis = 0 : si64} : (tensor<3xf32>, tensor<1xi64>, tensor<3xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func @test_sequence_empty() -> none {
  // expected-error @+1 {{SequenceEmpty dtype() does not match the output type}}
  %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xi32>>
  %2 = "onnx.NoValue"() {value} : () -> none
  return %2 : none
}
