// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZHighIR --printIR %s | FileCheck %s

// This pattern is found in bert models, where the output of attention layer is passed through a dense layer, then added with the attention layer's input.
// To simplify the test we use the input of MatMul to mimic the input of attention layer.
// In this test, we expect that constant propagation does not change the ordering of the two Adds, so that the constant and onnx.Matmul are lowered to a single zhigh.MatMul.
func.func @test_matmul_add_add(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<?x?x768xf32> {
  %cst = onnx.Constant dense<5.0> : tensor<768xf32>
  %matmul = "onnx.MatMul"(%arg0, %arg1): (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
  %add1 = "onnx.Add"(%cst, %matmul) : (tensor<768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
  %add2 = "onnx.Add"(%add1, %arg0) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
  return %add2 : tensor<?x?x768xf32>

// CHECK-LABEL:  func.func @test_matmul_add_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xf32>, [[PARAM_1_:%.+]]: tensor<768x768xf32>) -> tensor<?x?x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "3DS"} : (tensor<?x?x768xf32>) -> tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<768x768xf32>) -> tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<49152xi8>} : () -> tensor<768xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.MatMul"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<768x768xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<768xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>
}
