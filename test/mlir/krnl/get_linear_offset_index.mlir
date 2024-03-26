//RUN: onnx-mlir-opt --normalize-memrefs --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d2 floordiv 32, d2 mod 32, d1 mod 64)>
module {
  func.func @krnl_get_linear_offset_index(%arg0: memref<?x128x256xf32, #map>, %arg1: index, %arg2: index) -> index {
    %c5 = arith.constant 5: index
    %c10 = arith.constant 10: index
    %0 = memref.alloc(%arg1) : memref<?x128x256xf32, #map>
    %1 = krnl.get_linear_offset_index %arg0 at [%arg2, %c5, %c10] : memref<?x128x256xf32, #map>
    return %1: index 
  }

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-LABEL:  func.func @krnl_get_linear_offset_index
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x8x32x64xf32>, [[PARAM_1_:%.+]]: index, [[PARAM_2_:%.+]]: index) -> index {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[PARAM_1_]], [[CST_128_]], [[CST_256_]])
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) : memref<?x2x8x32x64xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at [symbol([[PARAM_2_]]), 0, 0, 10, 5] : memref<?x2x8x32x64xf32>
// CHECK:           return [[VAR_1_]] : index
// CHECK:         }

}
