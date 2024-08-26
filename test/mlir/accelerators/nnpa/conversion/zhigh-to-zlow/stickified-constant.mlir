// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

module  {
  func.func @remove_stick_2d() -> tensor<2x3xf32> {
    %0 = "zhigh.StickifiedConstant"() {alignment = 4096 : i64, layout = "2D", value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %1 = "zhigh.Unstick"(%0) : (tensor<2x3xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}


// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func @remove_stick_2d
// CHECK-SAME:   () -> memref<2x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zlow.stickifiedConstant"() {alignment = 4096 : i64, layout = "2D", name = "constant_stickify_0", offset = 0 : i64, shape = [1, 1, 1, 1, 32, 64], value = dense<{{.}}[0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]{{.}}> : tensor<2x3xf32>} : () -> memref<2x3xf16, #map>
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK:           "zlow.unstick"([[VAR_0_]], [[RES_]]) {layout = "2D"} : (memref<2x3xf16, [[MAP_0_]]>, memref<2x3xf32>) -> ()
// CHECK:           return [[RES_]] : memref<2x3xf32>
// CHECK:         }

