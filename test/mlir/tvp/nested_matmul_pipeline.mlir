// RUN: onnx-mlir --EmitApollo %s -o %t
// This test is checking that nested matmuls (i.e. matmuls with > 2 dimensions) don't cause crashes in the EmitApollo pipeline

module  {
  func @main_graph(%arg0: tensor<1x512x3072xbf16>, %arg2: tensor<3072x3072xbf16>, %arg3: tensor<3072xbf16>) -> tensor<1x512x3072xbf16> {
    %6 = "onnx.MatMul"(%arg0, %arg2) {onnx_node_name = "MatMul0"} : (tensor<1x512x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<1x512x3072xbf16>
    
    // ISSUE-TODO: workaround for bug #374 (direct return of ONNX.matmul causes crashes)
    %7 = "onnx.Add"(%6, %arg3) {onnx_node_name = "bias0"} : (tensor<1x512x3072xbf16>, tensor<3072xbf16>) -> tensor<1x512x3072xbf16>
    return %7 : tensor<1x512x3072xbf16>
  }
}