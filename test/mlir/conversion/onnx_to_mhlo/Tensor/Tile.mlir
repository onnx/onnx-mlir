// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

// Test tile with constant repeats
func.func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL: func @test_tile1
// CHECK-NEXT:    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 3]> : vector<2xi64>} : (tensor<4x8xf32>) -> tensor<3x4x2x8xf32>
// CHECK-NEXT:    %1 = "mhlo.reshape"(%0) : (tensor<3x4x2x8xf32>) -> tensor<12x16xf32>
}
// -----

func.func @test_tile_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Tile"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_tile_dynamic
// CHECK-NEXT:    %0 = mhlo.constant dense<5> : tensor<1xi64>
// CHECK-NEXT:    %1 = mhlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %2 = mhlo.constant dense<32> : tensor<1xi64>
// CHECK-NEXT:    %3 = "mhlo.slice"(%arg1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-NEXT:    %4 = "mhlo.slice"(%arg1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-NEXT:    %5 = "mhlo.slice"(%arg1) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-NEXT:    %6 = "mhlo.slice"(%arg1) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<1xi64>
// CHECK-NEXT:    %7 = "mhlo.concatenate"(%3, %0, %4, %0, %5, %1, %6, %2) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<8xi64>
// CHECK-NEXT:    %8 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %7) {broadcast_dimensions = dense<[1, 3, 5, 7]> : vector<4xi64>} : (tensor<5x5x1x32xf32>, tensor<8xi64>) -> tensor<?x?x?x?x?x?x?x?xf32>
// CHECK-NEXT:    %9 = mhlo.multiply %3, %0 : tensor<1xi64>
// CHECK-NEXT:    %10 = mhlo.multiply %4, %0 : tensor<1xi64>
// CHECK-NEXT:    %11 = mhlo.multiply %6, %2 : tensor<1xi64>
// CHECK-NEXT:    %12 = "mhlo.concatenate"(%9, %10, %5, %11) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-NEXT:    %13 = "mhlo.dynamic_reshape"(%8, %12) : (tensor<?x?x?x?x?x?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
}