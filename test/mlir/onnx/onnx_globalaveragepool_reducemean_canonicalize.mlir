// RUN: onnx-mlir-opt -onnx-hybrid-transform="canonicalization=true enable-globalaveragepool-to-reducemean=false" %s -split-input-file | FileCheck %s

func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
   %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
   return %0 : tensor<1x3x1x1xf32>
}

// CHECK-LABEL: func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
// CHECK-NOT: onnx.ReduceMeanV13
// CHECK: return %0 : tensor<1x3x1x1xf32>

