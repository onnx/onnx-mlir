// RUN: IMPORTER_FORCE_DYNAMIC=0:0%0:1 onnx-mlir --EmitONNXBasic --printIR %s | FileCheck %s

// COM: set batchsize = 8 for all function inputs.
func.func @test_set_force_dynamic(%arg0: tensor<1x2xi64>, %arg1: tensor<1x2xi64>) -> tensor<1x2xi64> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
  onnx.Return %0 : tensor<1x2xi64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_set_force_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x2xi64>, [[PARAM_1_:%.+]]: tensor<1x2xi64>) -> tensor<1x?xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x2xi64>, tensor<1x2xi64>) -> tensor<1x?xi64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<1x?xi64>
// CHECK:         }
}
