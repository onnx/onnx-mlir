// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_tile(%arg0 : tensor<5x5x1x32xf32>) -> tensor<5x10x30x32xf32> {
  %const = onnx.Constant dense<[1, 2, 30, 1]> : tensor<4xi64>
  %tile = "onnx.Tile"(%arg0, %const) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<5x10x30x32xf32>
  "func.return"(%tile) : (tensor<5x10x30x32xf32>) -> ()
// CHECK-LABEL: test_tile
// CHECK: tosa.tile{{.*}} <{multiples = array<i64: 1, 2, 30, 1>}> : (tensor<5x5x1x32xf32>) -> tensor<5x10x30x32xf32>
}

// -----

func.func @test_tile_dynamic_shape(%arg0 : tensor<5x5x?x32xf32>) -> tensor<5x10x?x32xf32> {
  %const = onnx.Constant dense<[1, 2, 30, 1]> : tensor<4xi64>
  %tile = "onnx.Tile"(%arg0, %const) : (tensor<5x5x?x32xf32>, tensor<4xi64>) -> tensor<5x10x?x32xf32>
  "func.return"(%tile) : (tensor<5x10x?x32xf32>) -> ()
// CHECK-LABEL: test_tile_dynamic_shape
// CHECK: tosa.tile{{.*}} <{multiples = array<i64: 1, 2, 30, 1>}> : (tensor<5x5x?x32xf32>) -> tensor<5x10x?x32xf32>
}

// -----

func.func @test_tile_input_not_ranked(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %const = onnx.Constant dense<[1, 2, 30, 1]> : tensor<4xi64>
  %tile = "onnx.Tile"(%arg0, %const) : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%tile) : (tensor<*xf32>) -> ()
// CHECK-LABEL: test_tile_input_not_ranked
// CHECK-NOT: tosa.tile
}

// -----

func.func @test_tile_non_constant_reps(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %tile = "onnx.Tile"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%tile) : (tensor<*xf32>) -> ()
// CHECK-LABEL: test_tile_non_constant_reps
// CHECK-NOT: tosa.tile
}

// -----

func.func @test_tile_no_tosa_type(%arg0 : tensor<5x5x1x32xcomplex<f32>>) -> tensor<5x10x30x32xcomplex<f32>> {
  %const = onnx.Constant dense<[1, 2, 30, 1]> : tensor<4xi64>
  %tile = "onnx.Tile"(%arg0, %const) : (tensor<5x5x1x32xcomplex<f32>>, tensor<4xi64>) -> tensor<5x10x30x32xcomplex<f32>>
  "func.return"(%tile) : (tensor<5x10x30x32xcomplex<f32>>) -> ()
// CHECK-LABEL: test_tile_no_tosa_type
// CHECK-NOT: tosa.tile
}

// -----

func.func @test_tile_no_valid_tosa_tile_type(%arg0 : tensor<5x5x1x32xf64>) -> tensor<5x10x30x32xf64> {
  %const = onnx.Constant dense<[1, 2, 30, 1]> : tensor<4xi64>
  %tile = "onnx.Tile"(%arg0, %const) : (tensor<5x5x1x32xf64>, tensor<4xi64>) -> tensor<5x10x30x32xf64>
  "func.return"(%tile) : (tensor<5x10x30x32xf64>) -> ()
// CHECK-LABEL: test_tile_no_valid_tosa_tile_type
// CHECK-NOT: tosa.tile
}
