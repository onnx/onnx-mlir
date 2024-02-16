// RUN: onnx-mlir --EmitONNXIR --shapeInformation=-1:8 --printIR %s | FileCheck %s

// COM: set batchsize = 8 for all function inputs.
func.func @test_set_batch_size(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>) -> tensor<?x?xi64> { 
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  onnx.Return %0 : tensor<?x?xi64>

// CHECK-LABEL test_set_batch_size
// CHECK: "onnx.Add"(%arg0, %arg1) {onnx_node_name = {{.*}}} : (tensor<8x?xi64>, tensor<8x?xi64>) -> tensor<8x?xi64
}
