// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @mod(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{fmod must be 1 when the input type is floating point}}
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_depth_to_space_default(%arg0 : tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{The input tensor depth must be divisible by the (blocksize * blocksize)}}  
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 7 : si64} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  "func.return"(%0) : (tensor<1x16x32x64xf32>) -> ()
}

// -----

func.func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  // expected-error @+1 {{onnx.ArgMax: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ArgMax"(%arg0) { axis = 4 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// -----

func.func @test_argmin_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  // expected-error @+1 {{onnx.ArgMin: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ArgMin"(%arg0) { axis = 4 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>  
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// -----

func.func @test_compress_verifier_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x1x32xi1>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Compress: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.Compress"(%arg0, %arg1) { axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi1>)  -> tensor<*xf32>  
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_concat_verifier_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Concat: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_concat_verifier_2(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Concat: operand '<block argument> of type 'tensor<5x5x32xf32>' at index: 2' has rank 3, rank should be 4}}  
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_concat_verifier_3(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Concat: operand '<block argument> of type 'tensor<5x5x3x32xf32>' at index: 1' has dimension at index 2 with value 3, value should be 1}}
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_concat_from_sequence_verifier_1(%arg0 : !onnx.Seq<tensor<5x5x1x32xf32>>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ConcatFromSequence: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ConcatFromSequence"(%arg0) {axis = 4 : si64} : (!onnx.Seq<tensor<5x5x1x32xf32>>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_concat_from_sequence_verifier_2(%arg0 : !onnx.Seq<tensor<5x5x1x32xf32>>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ConcatFromSequence: 'axis' value is -6, accepted range is [-5, 4]}}
  %1 = "onnx.ConcatFromSequence"(%arg0) {axis = -6 : si64, new_axis = 1 : si64} : (!onnx.Seq<tensor<5x5x1x32xf32>>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_dequantize_linear_verifier_1(%arg0 : tensor<5x5x1xi32>, %arg1 : tensor<3xf32>, %arg2 : tensor<3xi32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.DequantizeLinear: 'axis' value is 3, accepted range is [-3, 2]}}
  %1 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 3 : si64} : (tensor<5x5x1xi32>, tensor<3xf32>, tensor<3xi32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_constantofshape_verifier_1(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must be a 1D tensor}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
  "func.return"(%1) : (tensor<2x2xi64>) -> ()
}

// -----

func.func @test_constantofshape_verifier_2(%arg0: tensor<2x2x2x2xi64>) -> tensor<2x2x2x2xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must be a 1D tensor}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<2x2x2x2xi64>) -> tensor<2x2x2x2xi64>
  "func.return"(%1) : (tensor<2x2x2x2xi64>) -> ()
}

// -----

func.func @test_constantofshape_verifier_3(%arg0: tensor<?xi64>) -> tensor<?xi64> {
   // expected-error @+1 {{'onnx.ConstantOfShape' op Input tensor must have static shape}}
   %1 = "onnx.ConstantOfShape"(%arg0) : (tensor<?xi64>) -> tensor<?xi64>
  "func.return"(%1) : (tensor<?xi64>) -> ()
}

// -----

func.func @test_constantofshape_verifier_4() -> tensor<2xi64> {
   // expected-error @+2 {{'onnx.ConstantOfShape' op All values of the input tensor must be >=0}}
   %0 = "onnx.Constant"(){ value = dense<[-1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
   %1 = "onnx.ConstantOfShape"(%0) : (tensor<2xi64>) -> tensor<2xi64>
  "func.return"(%1) : (tensor<2xi64>) -> ()
}

// -----

func.func @test_flatten_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Flatten: 'axis' value is 5, accepted range is [-4, 4]}}
  %1 = "onnx.Flatten"(%arg0) {axis = 5 : si64} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_gather_verifier_1(%data: tensor<2x2xf32>, %indices: tensor<2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.Gather: 'axis' value is -3, accepted range is [-2, 1]}}
  %1 = "onnx.Gather"(%data, %indices) {axis = -3 : si64 } : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_gatherElements_verifier_1(%data: tensor<2x2xf32>, %indices: tensor<2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherElements: operand '<block argument> of type 'tensor<2xi64>' at index: 1' has rank 1, rank should be 2}}
  %1 = "onnx.GatherElements"(%data, %indices) { } : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_gatherElements_verifier_2(%data: tensor<2x2xf32>, %indices: tensor<2x2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherElements: 'axis' value is 2, accepted range is [-2, 1]}}
  %1 = "onnx.GatherElements"(%data, %indices) {axis = 2 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_hardmax_verifier_1(%arg0: tensor<2x2xf32>) -> tensor<*xf32> {
   // expected-error @+1 {{onnx.Hardmax: 'axis' value is 3, accepted range is [-2, 1]}}
   %1 = "onnx.Hardmax"(%arg0) {axis = 3: si64} : (tensor<2x2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'data' has to be >=1
func.func @test_gather_elements_verifier_1(%arg0 : tensor<f32>, %arg1 : tensor<5xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherElements: operand '<block argument> of type 'tensor<f32>' at index: 0' has rank 0, rank should be > 0}}  
  %1 = "onnx.GatherElements"(%arg0, %arg1) {axis = 4 : si64} : (tensor<f32>, tensor<5xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'indices' must be equal to the rank of `data`.
func.func @test_gather_elements_verifier_2(%arg0 : tensor<5xf32>, %arg1 : tensor<5x3xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherElements: operand '<block argument> of type 'tensor<5x3xi64>' at index: 1' has rank 2, rank should be 1}}
  %1 = "onnx.GatherElements"(%arg0, %arg1) {axis = 4 : si64} : (tensor<5xf32>, tensor<5x3xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: 'axis' valid range is [-r, r-1], where r = rank(data).
func.func @test_gather_elements_verifier_3(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x1x32xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherElements: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.GatherElements"(%arg0, %arg1) {axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM:  All index values in 'indices' are expected to be within bounds [-s, s-1] along axis of size s.
func.func @test_gather_elements_verifier_4(%arg0 : tensor<3xf32>, %arg1 : tensor<3xf32>) -> tensor<*xf32> {
  // expected-error @+2 {{onnx.GatherElements: 'indices' value is 3, accepted range is [-3, 2]}}
  %indices = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3xf32>, tensor<1xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'data' has to be >=1
func.func @test_gatherND_verifier_1(%arg0 : tensor<f32>, %arg1 : tensor<5xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherND: operand '<block argument> of type 'tensor<f32>' at index: 0' has rank 0, rank should be > 0}}  
  %1 = "onnx.GatherND"(%arg0, %arg1) : (tensor<f32>, tensor<5xi64>)  -> tensor<*xf32>
}

// -----

// COM: Rank of 'indices' has to be >=1
func.func @test_gatherND_verifier_2(%arg0 : tensor<2xf32>, %arg1 : tensor<i64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherND: operand '<block argument> of type 'tensor<i64>' at index: 1' has rank 0, rank should be > 0}}  
  %1 = "onnx.GatherND"(%arg0, %arg1) : (tensor<2xf32>, tensor<i64>)  -> tensor<*xf32>
}

// -----

// COM: The value batch_dims must be smaller than the minimum of rank(data) and rank(indices).
func.func @test_gatherND_verifier_3(%arg0 : tensor<1x2x3xf32>, %arg1 : tensor<2x2x2x2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherND: 'batch_dims' value is 3, accepted range is [0, 2]}}
  %1 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 3 : si64}: (tensor<1x2x3xf32>, tensor<2x2x2x2xi64>)  -> tensor<*xf32>
}

// -----

// COM: The first 'batchDims' dimensions of the shape of the 'indices' and 'data' tensors must be equal.
func.func @test_gatherND_verifier_4(%arg0 : tensor<2x2x3x4xf32>, %arg1 : tensor<2x3x2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherND: operand '<block argument> of type 'tensor<2x3x2xi64>' at index: 1' has dimension at index 1 with value 3, value should be 2}}
  %1 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 2 : si64} : (tensor<2x2x3x4xf32>, tensor<2x3x2xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: The last dimension of the 'indices' shape must be a value in the range [1, rank(data)-batch_dims].
func.func @test_gatherND_verifier_5(%arg0 : tensor<1x2x3x4xf32>, %arg1 : tensor<1x4xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.GatherND: operand '<block argument> of type 'tensor<1x4xi64>' at index: 1' has dimension at index 1 with value 4, value should be <= 3}}
  %1 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64} : (tensor<1x2x3x4xf32>, tensor<1x4xi64>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: All values in 'indices' are expected to satisfy the inequality:
// COM:   -data.shape[i] <= indices[...,i] <= (data.shape[i]-1)].
func.func @test_gatherND_verifier_6(%arg0 : tensor<3x4x4x4xf32>) -> tensor<*xf32> {
  // expected-error @+2 {{onnx.GatherND: 'indices[0]' value is 3, accepted range is [-3, 2]}}
  %indices = "onnx.Constant"() {value = dense<[3,2,2]> : tensor<3xi64>} : () -> tensor<3x3x2xi64>
  %1 = "onnx.GatherND"(%arg0, %indices) : (tensor<3x4x4x4xf32>, tensor<3x3x2xi64>)  -> tensor<*xf32>
}

// -----

func.func @test_if_verifier_1(%arg0: tensor<i1>) -> tensor<2xf32> {
  // expected-error @+1 {{'onnx.If' op then branch #results=2 differ from if #results=1}}
  %0 = "onnx.If"(%arg0) ({
    %1 = "onnx.Constant"() {value = dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.NoValue"() {value} : () -> none
    onnx.Return %1, %2 : tensor<2xf32>, none
  }, {
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    onnx.Return %1 : tensor<2xf32>
  }) : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @test_if_verifier_2(%arg0: tensor<i1>) -> !onnx.Seq<tensor<*xf32>> {
  // expected-error @+1 {{'onnx.If' op else branch #results=2 differ from if #results=1}}
  %0 = "onnx.If"(%arg0) ({
    %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
    onnx.Return %1 : !onnx.Seq<tensor<*xf32>>
  }, {
    %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
    %2 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    onnx.Return %1, %2 : !onnx.Seq<tensor<*xf32>>, tensor<2xf32>
  }) : (tensor<i1>) -> !onnx.Seq<tensor<*xf32>>
  return %0 : !onnx.Seq<tensor<*xf32>>
}

// -----

func.func @test_if_verifier_3(%arg0: tensor<i1>) -> tensor<2xf32> {
  // expected-error @+1 {{'onnx.If' op then branch disagrees on result type #1 of 1}}
  %0 = "onnx.If"(%arg0) ({
    %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
    onnx.Return %1 : !onnx.Seq<tensor<*xf32>>
  }, {
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    onnx.Return %1 : tensor<2xf32>
  }) : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @test_if_verifier_4(%arg0: tensor<i1>) -> !onnx.Seq<tensor<*xf32>> {
  // expected-error @+1 {{'onnx.If' op else branch disagrees on result type #1 of 1}}
  %0 = "onnx.If"(%arg0) ({
    %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
    onnx.Return %1 : !onnx.Seq<tensor<*xf32>>
  }, {
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    onnx.Return %1 : tensor<2xf32>
  }) : (tensor<i1>) -> !onnx.Seq<tensor<*xf32>>
  return %0 : !onnx.Seq<tensor<*xf32>>
}

// -----

func.func @test_onehotencoder_verifier_1(%arg0: tensor<2x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{'onnx.OneHotEncoder' op input is a tensor of float, int32, or double, but no cats_int64s attribute}}
  %1 = "onnx.OneHotEncoder"(%arg0) { cats_string = ["a","b","c"]} : (tensor<2x2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_onehotencoder_verifier_2(%arg0: tensor<2x2x!onnx.String>) -> tensor<*x!onnx.String> {
  // expected-error @+1 {{'onnx.OneHotEncoder' op input is not a tensor of float, int32, or double, but no cats_strings attribute}}
  %1 = "onnx.OneHotEncoder"(%arg0) { cats_int64s = [1,2,3]} : (tensor<2x2x!onnx.String>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatterelements_verifier_1(%arg0: tensor<2xf32>, %arg1: tensor<2x2xi64>, %arg2: tensor<2x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: operand '<block argument> of type 'tensor<2x2xi64>' at index: 1' has rank 2, rank should be 1}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<2xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatterelements_verifier_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi64>, %arg2: tensor<2x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: 'axis' value is -3, accepted range is [-2, 1]}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = -3 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_logsoftmax_verifier_1(%arg0: tensor<2x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.LogSoftmax: 'axis' value is 3, accepted range is [-2, 1]}}
  %1 = "onnx.LogSoftmax"(%arg0) {axis = 3 : si64} : (tensor<2x2xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_pow_verifier_1(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatter_elements_verifier_1(%arg0 : tensor<f32>, %arg1 : tensor<5xi64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: operand '<block argument> of type 'tensor<f32>' at index: 0' has rank 0, rank should be > 0}}  
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<f32>, tensor<5xi64>, tensor<5xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatter_elements_verifier_2(%arg0 : tensor<5xf32>, %arg1 : tensor<5x3xi64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: operand '<block argument> of type 'tensor<5x3xi64>' at index: 1' has rank 2, rank should be 1}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<5xf32>, tensor<5x3xi64>, tensor<5xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatterelements_verifier_3(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x1x32xi64>, %arg2 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterElements: 'axis' value is 4, accepted range is [-4, 3]}}
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<5x5x1x32xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_scatterelements_verifier_4(%arg0 : tensor<3xf32>, %arg1 : tensor<3xf32>) -> tensor<*xf32> {
  // expected-error @+2 {{onnx.ScatterElements: 'indices' value is 3, accepted range is [-3, 2]}}
  %indices = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ScatterElements"(%arg0, %indices, %arg1) {axis = 0 : si64} : (tensor<3xf32>, tensor<1xi64>, tensor<3xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'data' has to be >=1
func.func @test_scatterND_verifier_1(%arg0 : tensor<f32>, %arg1 : tensor<5xi64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<f32>' at index: 0' has rank 0, rank should be > 0}}  
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<f32>, tensor<5xi64>, tensor<5xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'indices' has to be >=1
func.func @test_scatterND_verifier_2(%arg0 : tensor<2xf32>, %arg1 : tensor<i64>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<i64>' at index: 1' has rank 0, rank should be > 0}}  
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<2xf32>, tensor<i64>, tensor<5xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Rank of 'updates' has to rank(data) + rank(indices) + indices.shape[-1] -1
func.func @test_scatterND_verifier_3(%arg0 : tensor<2x3xf32>, %arg1 : tensor<1xi64>, %arg2 : tensor<2x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<2x2xf32>' at index: 2' has rank 2, rank should be 1}}  
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<2x3xf32>, tensor<1xi64>, tensor<2x2xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: The last dimension of 'indices' shape can be a value at most equal to rank(data).
func.func @test_scatterND_verifier_4(%arg0 : tensor<1x2x3x4xf32>, %arg1 : tensor<2x5xi64>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<2x5xi64>' at index: 1' has dimension at index 1 with value 5, value should be <= 4}}
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<1x2x3x4xf32>, tensor<2x5xi64>, tensor<f32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Let q = rank(indices). The first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
func.func @test_scatterND_verifier_5(%arg0 : tensor<1x2x3x4xf32>, %arg1 : tensor<2x2xi64>, %arg2 : tensor<1x2x3xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<1x2x3xf32>' at index: 2' has dimension at index 0 with value 1, value should be 2}}
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<1x2x3x4xf32>, tensor<2x2xi64>, tensor<1x2x3xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

// COM: Let r = rank(data), q = rank(indices), and k = indices.shape[-1] --> updates.shape[q:] must match data.shape[k:r-1].
func.func @test_scatterND_verifier_5(%arg0 : tensor<1x2x3x4xf32>, %arg1 : tensor<2x2xi64>, %arg2 : tensor<2x3x3xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{onnx.ScatterND: operand '<block argument> of type 'tensor<2x3x3xf32>' at index: 2' has dimension at index 2 with value 3, value should be 4}}
  %1 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {axis = 4 : si64} : (tensor<1x2x3x4xf32>, tensor<2x2xi64>, tensor<2x3x3xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

func.func @test_sequence_empty() -> none {
  // expected-error @+1 {{SequenceEmpty dtype() does not match the output type}}
  %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xi32>>
  %2 = "onnx.NoValue"() {value} : () -> none
  return %2 : none
}

// -----

func.func @test_split_verifier_1(%arg0: tensor<2x2xi64>, %arg1: tensor<2xi64>) -> tensor<*xi64> {
  // expected-error @+1 {{onnx.Split: 'axis' value is 2, accepted range is [-2, 1]}}
  %1 = "onnx.Split"(%arg0, %arg1) {axis = 2 : si64} : (tensor<2x2xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// -----

func.func @test_splitToSequence_verifier_1(%arg0: tensor<2x2xi64>, %arg1: tensor<2xi64>) -> !onnx.Seq<tensor<*xi64>> {
  // expected-error @+1 {{onnx.SplitToSequence: 'axis' value is -3, accepted range is [-2, 1]}}
  %1 = "onnx.SplitToSequence"(%arg0, %arg1) {axis = -3 : si64} : (tensor<2x2xi64>, tensor<2xi64>) -> !onnx.Seq<tensor<*xi64>>
  "func.return"(%1) : (!onnx.Seq<tensor<*xi64>>) -> ()
}

// -----

func.func @test_topK_verifier_1(%arg0: tensor<3x4xi64>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  // expected-error @+1 {{onnx.TopK: 'axis' value is 2, accepted range is [-2, 1]}}
  %1, %2 = "onnx.TopK"(%arg0, %arg1) {axis = 2 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xi64>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  "func.return"(%1,%2) : (tensor<*xf32>, tensor<*xi64>) -> ()
}

// -----

func.func @test_topK_verifier_2(%arg0: tensor<3x4xi64>, %arg1: tensor<1x1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  // expected-error @+1 {{onnx.TopK: operand '<block argument> of type 'tensor<1x1xi64>' at index: 1' has rank 2, rank should be < 2}} 
  %1, %2 = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xi64>, tensor<1x1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  "func.return"(%1,%2) : (tensor<*xf32>, tensor<*xi64>) -> ()
}

// -----

func.func @test_unique_verifier_1(%arg0: tensor<3x4xi64>) -> (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>) {
  // expected-error @+1 {{onnx.Unique: 'axis' value is 2, accepted range is [-2, 1]}}  
  %1,%2,%3,%4 = "onnx.Unique"(%arg0) {axis = 2 : si64, sorted = 1 : si64} : (tensor<3x4xi64>) -> (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>)
  "func.return"(%1,%2,%3,%4) : (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>) -> ()
}

// -----

func.func @test_unique_verifier_2(%arg0: tensor<3x4xi64>) -> (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>) {
  // expected-error @+1 {{onnx.Unique: 'sorted' value is 2, accepted range is [0, 1]}}  
  %1,%2,%3,%4 = "onnx.Unique"(%arg0) {axis = 0 : si64, sorted = 2 : si64} : (tensor<3x4xi64>) -> (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>)
  "func.return"(%1,%2,%3,%4) : (tensor<*xf32>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>) -> ()
}
