// RUN: onnx-mlir-opt -legalize-quark-quantized-ops --split-input-file %s | FileCheck %s


func.func @test_constant_dense_1d_value() -> tensor<3xf32> {
    %0 = onnx.Constant {value = dense<[-8192.0, -1.1875, 1.1875]> : tensor<3xf32>} : tensor<3xf32>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = bf16} : (tensor<3xf32>) -> tensor<3xbf16>
    %2 = "onnx.Cast"(%1) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    "onnx.Return"(%2) : (tensor<3xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_constant_dense_1d_value() -> tensor<3xf32> {
// CHECK:           %[[VAL_0:.*]] = onnx.Constant dense<[-8.192000e+03, -1.187500e+00, 1.187500e+00]> : tensor<3xbf16>
// CHECK:           %[[VAL_1:.*]] = "onnx.Cast"(%[[VAL_0]]) {saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
// CHECK:           onnx.Return %[[VAL_1]] : tensor<3xf32>
// CHECK:         }

// -----

func.func @test_constant_splat() -> tensor<3xf32> {
    %0 = onnx.Constant {value = dense<2.0> : tensor<3xf32>} : tensor<3xf32>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = bf16} : (tensor<3xf32>) -> tensor<3xbf16>
    %2 = "onnx.Cast"(%1) { saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
    "onnx.Return"(%2) : (tensor<3xf32>) -> ()
}
// CHECK-LABEL:   func.func @test_constant_splat() -> tensor<3xf32> {
// CHECK:           %[[VAL_0:.*]] = onnx.Constant dense<2.000000e+00> : tensor<3xbf16>
// CHECK:           %[[VAL_1:.*]] = "onnx.Cast"(%[[VAL_0]]) {saturate = 1 : si64, to = f32} : (tensor<3xbf16>) -> tensor<3xf32>
// CHECK:           onnx.Return %[[VAL_1]] : tensor<3xf32>
// CHECK:         }

// -----

func.func @test_constant_dense_1d_value() -> tensor<3xbf16> {
    %0 = onnx.Constant {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : tensor<3xf32>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = bf16} : (tensor<3xf32>) -> tensor<3xbf16>
    "onnx.Return"(%1) : (tensor<3xbf16>) -> ()
}
// CHECK-LABEL:   func.func @test_constant_dense_1d_value() -> tensor<3xbf16> {
// CHECK:           %[[VAL_0:.*]] = onnx.Constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xbf16>
// CHECK:           onnx.Return %[[VAL_0]] : tensor<3xbf16>
// CHECK:         }

// -----

func.func @test_constant_dense_1d_value() -> tensor<*xbf16> {
    %0 = onnx.Constant {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : tensor<*xf32>
    %1 = "onnx.Cast"(%0) { saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    "onnx.Return"(%1) : (tensor<*xbf16>) -> ()
}

// CHECK-LABEL:   func.func @test_constant_dense_1d_value() -> tensor<*xbf16> {
// CHECK:           %[[VAL_0:.*]] = onnx.Constant {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : tensor<*xf32>
// CHECK:           %[[VAL_1:.*]] = "onnx.Cast"(%[[VAL_0]]) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
// CHECK:           onnx.Return %[[VAL_1]] : tensor<*xbf16>
// CHECK:         }
