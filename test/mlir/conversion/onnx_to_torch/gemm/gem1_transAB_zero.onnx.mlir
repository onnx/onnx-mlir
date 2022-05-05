//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module  {
  func @main_graph(%arg0: tensor<?x3xf32>) -> tensor<?x4xf32> attributes {input_names = ["dense_input"], output_names = ["dense"]} {
    %0 = "onnx.Constant"() {value = dense<[[0.970539271, 0.323187172, 0.396023512, 0.344399512], [0.393198192, 0.737876773, 0.374971896, 1.04072165], [0.626369476, 0.987294554, 0.391298711, 0.35478127]]> : tensor<3x4xf32>} : () -> tensor<3x4xf32>
    %1 = "onnx.Constant"() {value = dense<[0.199155629, 0.185304552, 0.36959976, 0.261764377]> : tensor<4xf32>} : () -> tensor<4xf32>
//CHECK: %int[[AVAL:.]] = torch.constant.int 0
//CHECK: %int[[BVAL:.]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %arg0, %int[[AVAL]], %int[[BVAL]] :
//XFAIL: torch.aten.transpose.inta    
%2 = "onnx.Gemm"(%arg0, %0, %1) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
    return %2 : tensor<?x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22float\22 , \22dims\22 : [-1 , 3]  }\0A\0A]\00@[   { \22type\22 : \22float\22 , \22dims\22 : [-1 , 4]  }\0A\0A]\00"} : () -> ()
}
