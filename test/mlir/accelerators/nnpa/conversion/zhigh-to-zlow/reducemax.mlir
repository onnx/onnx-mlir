// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @reduce_max_axes_defined_noop_0(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>> { 
  %0 = "zhigh.ReduceMax"(%arg0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @reduce_max_axes_defined_noop_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x1xf16, #map> {
// CHECK:           [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : memref<3x4x5xf16, #map> to tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_1_:%.+]] = "zhigh.ReduceMax"([[VAR_0_]]) {op_type = "REDUCE_OP_MAXIMUM"} : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[VAR_1_]] : tensor<3x4x1xf16, #zhigh.layout<{dataLayout = "3DS"}>> to memref<3x4x1xf16, #map>
// CHECK:           return [[VAR_2_]] : memref<3x4x1xf16, #map>
// CHECK:         }
}
