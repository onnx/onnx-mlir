// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --npu %s | FileCheck %s

func private @test_softmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  // CHECK: func private @apollo_tvp_Softmax_[[ID:.+]](memref<10x10xf32>) -> memref<10x10xf32> attributes {axis = 1 : si64, tvp.Softmax = true}
  // CHECK: call @apollo_tvp_Softmax_[[ID]](%arg0) : (memref<10x10xf32>) -> memref<10x10xf32>
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}
