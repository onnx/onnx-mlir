// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_f8E4M3FNUZ(%arg0: tensor<13x21x3xf8E4M3FNUZ>) -> tensor<13x21x3xf8E4M3FNUZ> {
  func.return %arg0 : tensor<13x21x3xf8E4M3FNUZ>
}
// CHECK: @test_f8E4M3FNUZ(%[[arg0:.*]]: tensor<13x21x3xf8E4M3FNUZ>) -> tensor<13x21x3xf8E4M3FNUZ>
// CHECK-NEXT: return %[[arg0]] : tensor<13x21x3xf8E4M3FNUZ>


func.func @test_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN> {
  func.return %arg0 : tensor<13x21x3xf8E4M3FN>
}
// CHECK: @test_f8E4M3FN(%[[arg0:.*]]: tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN>
// CHECK-NEXT:   return %[[arg0]] : tensor<13x21x3xf8E4M3FN>

func.func @test_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf8E5M2> {
  func.return %arg0 : tensor<13x21x3xf8E5M2>
}

// CHECK: func.func @test_f8E5M2(%[[arg0:.*]]: tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf8E5M2>
// CHECK:    return %[[arg0]] : tensor<13x21x3xf8E5M2>

func.func @test_f8E5M2FNUZ(%arg0: tensor<13x21x3xf8E5M2FNUZ>) -> tensor<13x21x3xf8E5M2FNUZ> {
  func.return %arg0 : tensor<13x21x3xf8E5M2FNUZ>
}

// CHECK: @test_f8E5M2FNUZ(%[[arg0:.*]]: tensor<13x21x3xf8E5M2FNUZ>) -> tensor<13x21x3xf8E5M2FNUZ> {
// CHECK-NEXT:   return %[[arg0]] : tensor<13x21x3xf8E5M2FNUZ>

func.func @test_f8E4M3B11FNUZ(%arg0: tensor<13x21x3xf8E4M3B11FNUZ>) -> tensor<13x21x3xf8E4M3B11FNUZ> {
  func.return %arg0 : tensor<13x21x3xf8E4M3B11FNUZ>
}

// CHECK: @test_f8E4M3B11FNUZ(%[[arg0:.*]]: tensor<13x21x3xf8E4M3B11FNUZ>) -> tensor<13x21x3xf8E4M3B11FNUZ> {
// CHECK-NEXT:   return %[[arg0]] : tensor<13x21x3xf8E4M3B11FNUZ>

