// RUN: onnx-mlir --EmitONNXIR --shapeInformation=0:5x1,1-2:3x-1x1,3:5x7 --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<5x1xi64>, %arg1: tensor<3x?x1xi64>, %arg2: tensor<3x?x1xi64>, %arg3: tensor<5x7xi64>) -> (tensor<5x1xi64>, tensor<3x?x1xi64>, tensor<3x?x1xi64>, tensor<5x7xi64>) {
    onnx.Return %arg0, %arg1, %arg2, %arg3 : tensor<5x1xi64>, tensor<3x?x1xi64>, tensor<3x?x1xi64>, tensor<5x7xi64>
  }

// CHECK-LABEL:  func.func @main_graph
// CHECK:         return {{.*}} : tensor<5x1xi64>, tensor<3x?x1xi64>, tensor<3x?x1xi64>, tensor<5x7xi64>
}


