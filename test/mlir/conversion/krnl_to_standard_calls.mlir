// RUN: onnx-mlir-opt --convert-krnl-to-std %s | FileCheck %s  

// CHECK-LABEL: test_constant_call_argument
// CHECK: %[[ARG1:.*]] = constant 5.000000e-04 : f32
// CHECK: %[[ARG2:.*]] = constant 6.000000e-04 : f32
// CHECK: %[[RESULT:.*]] = call @external_func(%[[ARG0:.*]], %[[ARG1]], %[[ARG2]]) : (memref<1x512x3072xf32>, f32, f32) -> memref<1x512x3072xf32>
// CHECK: return %[[RESULT]] : memref<1x512x3072xf32>
func private @external_func(memref<1x512x3072xf32>, memref<f32>, memref<f32>) -> (memref<1x512x3072xf32>)
func @test_constant_call_argument(%arg0: memref<1x512x3072xf32>) -> memref<1x512x3072xf32> {
    %0 = "krnl.global"() {name = "constant_0", shape = [], value = dense<5.000000E-004> : tensor<f32>} : () -> memref<f32>
    %1 = "krnl.global"() {name = "constant_1", shape = [], value = dense<6.000000E-004> : tensor<f32>} : () -> memref<f32>
    %2 = call @external_func(%arg0, %0, %1) : (memref<1x512x3072xf32>, memref<f32>, memref<f32>) -> (memref<1x512x3072xf32>)
    return %2 : memref<1x512x3072xf32>
}