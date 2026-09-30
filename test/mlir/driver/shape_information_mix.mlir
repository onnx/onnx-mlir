// RUN: onnx-mlir --EmitONNXIR --shapeInformation=0:5x1,3:5x7,-1:5 --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<5x1xi64>, %arg1: tensor<5x?x?xi64>, %arg2: tensor<5x?x?xi64>, %arg3: tensor<5x7xi64>) -> (tensor<5x1xi64>, tensor<5x?x?xi64>, tensor<5x?x?xi64>, tensor<5x7xi64>) {
    onnx.Return %arg0, %arg1, %arg2, %arg3 : tensor<5x1xi64>, tensor<5x?x?xi64>, tensor<5x?x?xi64>, tensor<5x7xi64>
  }

// CHECK-LABEL:  func.func @main_graph
// CHECK:         return {{.*}} : tensor<5x1xi64>, tensor<5x?x?xi64>, tensor<5x?x?xi64>, tensor<5x7xi64>
}

