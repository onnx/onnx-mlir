// RUN: onnf-opt --canonicalize %s -split-input-file | FileCheck %s

//CHECK: module {
module {
 func @test_sigmoid() {
   %0 = "frontend.input t1"() : () -> tensor<10x10xf32>
   %1 = "frontend.input t2"() : () -> tensor<10x10xf32>
   %2 = "frontend.input t3"() : () -> tensor<10x10xf32>
   // CHECK: %{{[0-9]+}} = "onnx.full_gemm"(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
   %3 = "onnx.matmul"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
   %4 = "onnx.add"(%3, %2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
   %5 = "frontend.output t4"(%4) : (tensor<10x10xf32>) -> tensor<10x10xf32>
 }
}