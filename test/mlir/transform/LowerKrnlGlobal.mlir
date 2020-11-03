// RUN: onnx-mlir-opt --convert-onnx-to-krnl --convert-krnl-to-affine --lower-krnl-global %s | FileCheck %s  

// CHECK-LABEL: func @floating_point_scalar_constant
// CHECK-NOT: %1 = "krnl.global"() {name = "constant_0", shape = [], value = dense<0.00392156886> : tensor<f32>} : () -> memref<f32>
// CHECK: %cst = constant 0.00392156886 : f32
// CHECK-NOT: %3 = affine.load %1[] : memref<f32>
// CHECK: %2 = mulf %1, %cst : f32 
func @floating_point_scalar_constant(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {input_names = ["X"], output_names = ["predictions"]} {
    %0 = "onnx.Constant"() {value = dense<0.00392156886> : tensor<f32>} : () -> tensor<f32>
    %1 = "onnx.Mul"(%arg0, %0) {onnx_node_name = "mul0"} : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
}
