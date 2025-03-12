// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @should_lower_to_zlow(%arg0: tensor<3x4x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = onnx.Constant dense<[30, 4, 5]> : tensor<3xi64>
  %1 = "zhigh.Reshape"(%arg0, %0) {layout = "3DS"} : (tensor<3x4x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<3xi64>) -> tensor<*xf16>
  return %1 : tensor<*xf16>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x50xf16, #map>) -> memref<30x4x5xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<30x4x5xf16, #map>
// CHECK:           "zlow.reshape"([[PARAM_0_]], [[RES_]]) {layout = "3DS"} : (memref<3x4x50xf16, #map>, memref<30x4x5xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<30x4x5xf16, #map>
// CHECK:         }
}

