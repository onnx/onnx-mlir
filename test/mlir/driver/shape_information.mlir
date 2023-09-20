// RUN: onnx-mlir --EmitONNXIR --shapeInformation=0:3x-1 --printIR %s | FileCheck %s

module {
func.func @main_graph(%arg0: tensor<3x2xi64>, %arg1: tensor<3x2xi64>) -> tensor<3x2xi64> { 
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x2xi64>, tensor<3x2xi64>) -> tensor<3x2xi64>
  onnx.Return %0 : tensor<3x2xi64>

// CHECK-LABEL main_graph
// CHECK: "onnx.Add"(%arg0, %arg1) {onnx_node_name = {{.*}}} : (tensor<3x?xi64>, tensor<3x2xi64>) -> tensor<3x2xi64
}
}
