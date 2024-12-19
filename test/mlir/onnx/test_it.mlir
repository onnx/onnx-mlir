// COM: Full reduction over all dimensions to a scalar value.
module {
  func.func @main_graph(%arg0: tensor<?x64x?xf32> {onnx.name = "x"}) -> (tensor<*xf32> {onnx.name = "y"}) {
    %axes = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x64x?xf32>, none) -> tensor<*xf32>
    return %0: tensor<*xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

module {
  func.func @main_graph(%arg0: tensor<3x2x2xf32> {onnx.name = "x"}) -> (tensor<*xf32> {onnx.name = "y"}) {
   %cst = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64> } : () -> tensor<1xi64>
   %0 ="onnx.ReduceMax"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
   "func.return"(%0) : (tensor<*xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

func.func @reduce_min_axes_defined_noop_0(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>> { 
module {
   func.func @main_graph(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>> {onnx.name = "x"}) -> (tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>> {onnx.name = "y"}) {
   %0 = "zhigh.ReduceMin"(%arg0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
   return %0 : tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

func.func @should_lower_to_zlow(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> { 

module {
  func.func @main_graph(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>> {onnx.name = "x"}) -> (tensor<*xf16> {onnx.name = "y"}) {
    %0 = "zhigh.Softmax"(%arg0) {act_func = "ACT_NONE"} : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
    return %0 : tensor<*xf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}


func.func @test_zhigh_quantized_matmul(%arg0: tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, %arg7: tensor<f32>, %arg8: tensor<f32>) -> tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>> {
module {
  func.func @main_graph(%arg0: tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, %arg7: tensor<f32>, %arg8: tensor<f32> {onnx.name = "x"}) -> (tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>> {onnx.name = "y"}) {
    %none = "onnx.NoValue"() {value} : () -> none
    %Out, %Out_RecScale, %Out_Offset = "zhigh.QuantizedMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %none, %none) {DequantizeOutput = 0 : si64, DisableClipping = 0 : si64, PreComputedBias = 0 : si64} : (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, none, none) -> (tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
    return %Out : tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}