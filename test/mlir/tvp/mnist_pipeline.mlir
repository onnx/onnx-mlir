// RUN: onnx-mlir --EmitApollo %s -o %t 
// RUN: FileCheck %s --input-file %t.s -check-prefix=CHECK-ASM
// RUN: FileCheck %s --input-file %t.tcp.cpp -check-prefix=CHECK-CPP

// This is the template for e2e test for MNIST pipeline. It currently checks that
// pipeline executes without crashes. This can be extended to perform more
// checks later.

module  {
// CHECK-ASM: Execute:
// CHECK-CPP: void main_graph::Execute(const Arguments &args)
  func @main_graph(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x256xbf16>, %arg4: tensor<256xbf16>) -> tensor<256x256xbf16> attributes {input_names = ["X", "weight_0", "bias_0", "weight_1", "bias_1"], output_names = ["add_1_output"]} {
    %0 = "onnx.Constant"() {value = dense<-1.184680e-38> : tensor<bf16>} : () -> tensor<bf16>
    %1 = "onnx.Mul"(%arg0, %0) : (tensor<256x1024xbf16>, tensor<bf16>) -> tensor<*xbf16>
    %2 = "onnx.MatMul"(%1, %arg1) : (tensor<*xbf16>, tensor<1024x256xbf16>) -> tensor<*xbf16>
    %3 = "onnx.Add"(%2, %arg2) : (tensor<*xbf16>, tensor<256xbf16>) -> tensor<*xbf16>
    %4 = "onnx.Relu"(%3) : (tensor<*xbf16>) -> tensor<*xbf16>
    %5 = "onnx.MatMul"(%4, %arg3) : (tensor<*xbf16>, tensor<256x256xbf16>) -> tensor<*xbf16>
    %6 = "onnx.Add"(%5, %arg4) : (tensor<*xbf16>, tensor<256xbf16>) -> tensor<256x256xbf16>
    return %6 : tensor<256x256xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 5 : i32, numOutputs = 1 : i32} : () -> ()
}
