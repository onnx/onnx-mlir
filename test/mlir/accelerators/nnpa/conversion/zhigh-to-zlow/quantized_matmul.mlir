// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_zhigh_quantized_matmul(%arg0: tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, %arg7: tensor<f32>, %arg8: tensor<f32>) -> tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>> {
    %none = "onnx.NoValue"() {value} : () -> none
    %Out, %Out_RecScale, %Out_Offset = "zhigh.QuantizedMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %none, %none) {DequantizeOutput = 0 : si64, DisableClipping = 0 : si64, PreComputedBias = 0 : si64} : (tensor<1x3x5xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<5x7xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<7xi8, #zhigh.layout<{dataLayout = "1D", quantizedType = "INT8"}>>, tensor<f32>, tensor<f32>, none, none) -> (tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
    return %Out : tensor<1x3x7xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 64, d0 mod 64, d1 mod 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (0, d0 floordiv 128, 0, 0, 31, d0 mod 128)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf16, #map>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<f32>, [[PARAM_3_:%.+]]: memref<5x7xi8, #map1>, [[PARAM_4_:%.+]]: memref<f32>, [[PARAM_5_:%.+]]: memref<f32>, [[PARAM_6_:%.+]]: memref<7xi8, #map2>, [[PARAM_7_:%.+]]: memref<f32>, [[PARAM_8_:%.+]]: memref<f32>) -> memref<1x3x7xf16, #map> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x7xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_1_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:           [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_2_]][] : memref<f32>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<4xi64>
// CHECK:           krnl.store [[CST_1_]], [[RES_3_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[CST_3_]], [[RES_3_]]{{.}}[[CST_1_1_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[CST_5_]], [[RES_3_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           krnl.store [[CST_7_]], [[RES_3_]]{{.}}[[CST_3_1_]]{{.}} : memref<4xi64>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<7xf16, #map3>
// CHECK:           "zlow.quantizedMatmul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]], [[PARAM_8_]], [[RES_4_]], [[RES_3_]], [[RES_]], [[RES_]]_1, [[RES_]]_2) {bias_q_type = "INT8", dequantize_output = 0 : si64, disable_clipping = 0 : si64, is_bcast = -1 : si64, is_stacked = 0 : si64, out_q_type = "DLFLOAT16", pre_computed_bias = 0 : si64, x_q_type = "DLFLOAT16", y_q_type = "WEIGHTS"} : (memref<1x3x5xf16, #map>, memref<f32>, memref<f32>, memref<5x7xi8, #map1>, memref<f32>, memref<f32>, memref<7xi8, #map2>, memref<f32>, memref<f32>, memref<7xf16, #map3>, memref<4xi64>, memref<1x3x7xf16, #map>, memref<f32>, memref<f32>) -> ()
// CHECK:           return [[RES_]] : memref<1x3x7xf16, #map>
// CHECK:         }
}
