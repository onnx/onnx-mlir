// RUN: onnx-mlir --npu --EmitMLIR %s -o=%t
// RUN: FileCheck %s --input-file %t.onnx.mlir

// CHECK: #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
                                       
// CHECK: %0 = memref.alloc() : memref<1x512x12x256xbf16>
// CHECK: %1 = memref.alloc() : memref<1x12x256x512xbf16>
// CHECK: %2 = memref.alloc() : memref<1x12x512x256xbf16>
// CHECK: %3 = linalg.reshape %arg0 {{\[}}[0], [1], [2, 3]] : memref<1x512x3072xbf16> into memref<1x512x12x256xbf16>
// CHECK: linalg.copy(%3, %2) {outputPermutation = #map0} : memref<1x512x12x256xbf16>, memref<1x12x512x256xbf16>
// CHECK: %4 = linalg.reshape %arg1 {{\[}}[0], [1], [2, 3]] : memref<1x512x3072xbf16> into memref<1x512x12x256xbf16>
// CHECK: linalg.copy(%4, %1) {outputPermutation = #map1} : memref<1x512x12x256xbf16>, memref<1x12x256x512xbf16>
// CHECK: linalg.copy(%arg2, %0) {outputPermutation = #map0} : memref<1x12x512x256xbf16>, memref<1x512x12x256xbf16
// CHECK: %5 = linalg.reshape %0 {{\[}}[0], [1], [2, 3]] : memref<1x512x12x256xbf16> into memref<1x512x3072xbf16>
// CHECK: memref.dealloc %0 : memref<1x512x12x256xbf16>
// CHECK: return %2, %1, %5 : memref<1x12x512x256xbf16>, memref<1x12x256x512xbf16>, memref<1x512x3072xbf16>

module  {

  func @main_graph(%arg0: tensor<1x512x3072xbf16>, %arg1: tensor<1x512x3072xbf16>, %arg2: tensor<1x12x512x256xbf16>) -> (tensor<1x12x512x256xbf16>, tensor<1x12x256x512xbf16>, tensor<1x512x3072xbf16>) {

    %0 = "onnx.Constant"() {value = dense<[1, 512, 12, 256]> : tensor<4xi64>} : () -> tensor<4xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<1x512x3072xbf16>, tensor<4xi64>) -> tensor<1x512x12x256xbf16>
    %2 = "onnx.Transpose"(%1) {perm = [0, 2, 1, 3]} : (tensor<1x512x12x256xbf16>) -> tensor<1x12x512x256xbf16>

    %10 = "onnx.Constant"() {value = dense<[1, 512, 12, 256]> : tensor<4xi64>} : () -> tensor<4xi64>
    %11 = "onnx.Reshape"(%arg1, %10) : (tensor<1x512x3072xbf16>, tensor<4xi64>) -> tensor<1x512x12x256xbf16>
    %12 = "onnx.Transpose"(%11) {perm = [0, 2, 3, 1]} : (tensor<1x512x12x256xbf16>) -> tensor<1x12x256x512xbf16>
    
    %20 = "onnx.Transpose"(%arg2) {perm = [0, 2, 1, 3]} : (tensor<1x12x512x256xbf16>) -> tensor<1x512x12x256xbf16>
    %21 = "onnx.Constant"() {value = dense<[1, 512, 3072]> : tensor<3xi64>} : () -> tensor<3xi64>
    %22 = "onnx.Reshape"(%20, %21) : (tensor<1x512x12x256xbf16>, tensor<3xi64>) -> tensor<1x512x3072xbf16>

    return %2, %12, %22 : tensor<1x12x512x256xbf16>, tensor<1x12x256x512xbf16>, tensor<1x512x3072xbf16>
  }

}
