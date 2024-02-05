// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func private @test_squeezev11_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1,-2]} : (tensor<?x1x32x?x64xf32>) -> (tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeezev11_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x32x?x64xf32>
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_dim_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<?x32x64xf32>
// CHECK:         }
}

// -----

func.func private @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeeze_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x32x?x64xf32>
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_dim_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<?x32x64xf32>
// CHECK:         }
}

