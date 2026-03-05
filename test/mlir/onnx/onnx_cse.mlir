// RUN: onnx-mlir-opt -onnx-cse %s -split-input-file | FileCheck %s

// COM: Test ONNX CSE pass with ONNX operations to remove common sub-expressions.
func.func @test_cse(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "onnx.Tanh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.Tanh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.Add"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %3 : tensor<*xf32>
  // CHECK-LABEL: test_cse
  // CHECK-NEXT: "onnx.Log"
  // CHECK-COUNT-1: "onnx.Tanh"
  // CHECK-NEXT: "onnx.Add"
}

// -----

func.func @test_cse_noattrs() -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "c"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  return %3, %4, %5 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>

  // CHECK-LABEL: test_cse_noattrs
  // CHECK-COUNT-1: [[DQ:%[0-9]+]] = "onnx.DequantizeLinear"
  // CHECK-NEXT: return [[DQ]], [[DQ]], [[DQ]]
}

// -----

func.func @diff_attrs(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %1 = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %2 = "onnx.LeakyRelu"(%arg0) {alpha = 1.000000e-01 : f32} : (tensor<f32>) -> tensor<f32>
  %3 = "onnx.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<f32>) -> tensor<f32>
  %4 = "onnx.Add"(%2, %0) {ResultNames = ["add1_out"], onnx_node_name = "add1"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %5 = "onnx.Add"(%3, %1) {ResultNames = ["add2_out"], onnx_node_name = "add2"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %6 = "onnx.Add"(%4, %5) {ResultNames = ["final_add_out"], onnx_node_name = "final_add"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %6 : tensor<f32>

  // CHECK-LABEL: @diff_attrs
  // CHECK-COUNT-2: onnx.Constant
  // CHECK-COUNT-2: "onnx.LeakyRelu"
  // CHECK-COUNT-3: "onnx.Add"
}
