// RUN: onnx-mlir-opt --convert-onnx-to-krnl --convert-krnl-to-affine --convert-krnl-to-std %s | FileCheck %s  

// CHECK: global_memref "private" constant @global_memref1 : memref<10xf32> = dense<[-0.631071984, 0.0230938029, 0.26776129, -1.28461373, -0.413267285, -1.40921199, -1.26692009, 0.772099971, 0.12288624, 0.299881667]>
// CHECK: global_memref "private" constant @global_memref0 : memref<10xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]>

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

// CHECK-LABEL: func @floating_point_vector_constant
// CHECK-NOT: "krnl.global"() {name = "constant_1", shape = [10], value = dense<[-0.631071984, 0.0230938029, 0.26776129, -1.28461373, -0.413267285, -1.40921199, -1.26692009, 0.772099971, 0.12288624, 0.299881667]> : tensor<10xf32>} : () -> memref<10xf32>
// CHECK-NOT: "krnl.global"() {name = "constant_2", shape = [10], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]> : tensor<10xf32>} : () -> memref<10xf32>
// CHECK: get_global_memref @global_memref1 : memref<10xf32>
// CHECK: get_global_memref @global_memref0 : memref<10xf32>
func @floating_point_vector_constant(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {input_names = ["X"], output_names = ["predictions"]} {
    %0 = "onnx.Constant"() {value = dense<[-0.631071984, 0.0230938029, 0.26776129, -1.28461373, -0.413267285, -1.40921199, -1.26692009, 0.772099971, 0.12288624, 0.299881667]> : tensor<10xf32>} : () -> tensor<10xf32>
    %1 = "onnx.Add"(%arg0, %0) {onnx_node_name = "add1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %2 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>} : () -> tensor<10xf32>
    %3 = "onnx.Sub"(%1, %2) {onnx_node_name = "sub1"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

    return %3 : tensor<10xf32>
}