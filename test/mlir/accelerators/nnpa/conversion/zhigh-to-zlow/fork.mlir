// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_matmul_parallel(%arg0: tensor<1x64xf32>, %arg1: tensor<64x512xf32>) -> tensor<1x512xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = onnx.Constant dense<256> : tensor<2xi64>
  %2:2 = "onnx.Split"(%arg1, %1) {axis = 1 : si64, onnx_node_name = "onnx.Split_0"} : (tensor<64x512xf32>, tensor<2xi64>) -> (tensor<64x256xf32>, tensor<64x256xf32>)
  %3 = "zhigh.Fork"() ({
    %6 = "zhigh.Stick"(%arg0) {layout = "2D"} : (tensor<1x64xf32>) -> tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %7 = "zhigh.Stick"(%2#0) {layout = "2D"} : (tensor<64x256xf32>) -> tensor<64x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %8 = "zhigh.MatMul"(%6, %7, %0) : (tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<64x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %9 = "zhigh.Unstick"(%8) : (tensor<1x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x256xf32>
    onnx.Yield {onnx_node_name = "onnx.Yield_2"} %9 : tensor<1x256xf32>
  }) {id = 0 : si64} : () -> tensor<1x256xf32>
  %4 = "zhigh.Fork"() ({
    %6 = "zhigh.Stick"(%arg0) {layout = "2D"} : (tensor<1x64xf32>) -> tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %7 = "zhigh.Stick"(%2#1) {layout = "2D"} : (tensor<64x256xf32>) -> tensor<64x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %8 = "zhigh.MatMul"(%6, %7, %0) : (tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<64x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<1x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %9 = "zhigh.Unstick"(%8) : (tensor<1x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<1x256xf32>
    onnx.Yield {onnx_node_name = "onnx.Yield_4"} %9 : tensor<1x256xf32>
  }) {id = 1 : si64} : () -> tensor<1x256xf32>
  "zhigh.Join"(%3) : (tensor<1x256xf32>) -> ()
  "zhigh.Join"(%4) : (tensor<1x256xf32>) -> ()
  %5 = "onnx.Concat"(%3, %4) {axis = 1 : si64, onnx_node_name = "onnx.Concat_5"} : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x512xf32>
  return %5 : tensor<1x512xf32>
}


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 256)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @test_matmul_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x64xf32>, [[PARAM_1_:%.+]]: memref<64x512xf32>) -> memref<1x512xf32> {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i64
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<64x256xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<64x256xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<64x512xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<64x256xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 64, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 256){
// CHECK:             [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_4_1_]]#1)
// CHECK:             [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_1_]]#0, [[LOAD_PARAM_1_MEM_1_]]{{.}} : memref<64x512xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<64x256xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x256xf32>
// CHECK-DAG:       [[VAR_token_:%.+]] = async.execute {
// CHECK:             "krnl.call"([[CST_0_]]) {funcName = "threadAffine", numOfOutput = 1 : si64} : (i64) -> ()
// CHECK:             [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x64xf16, #map1>
// CHECK:             "zlow.stick"([[PARAM_0_]], [[RES_3_]]) {layout = "2D"} : (memref<1x64xf32>, memref<1x64xf16, #map1>) -> ()
// CHECK:             [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<64x256xf16, #map1>
// CHECK:             "zlow.stick"([[RES_]], [[RES_]]_6) {layout = "2D"} : (memref<64x256xf32>, memref<64x256xf16, #map1>) -> ()
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1x256xf16, #map1>
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:             krnl.store [[CST_1_]], [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi64>
// CHECK:             krnl.store [[CST_64_]], [[RES_6_]]{{.}}[[CST_1_1_]]{{.}} : memref<3xi64>
// CHECK:             krnl.store [[CST_256_]], [[RES_6_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:             [[VAR_4_2_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_0", shape = [1, 4, 1, 1, 32, 64], value = dense_resource<zhigh> : tensor<16384xi8>} : () -> memref<1x4x1x1x32x64xf16>
// CHECK:             "zlow.matmul"([[RES_3_]], [[RES_4_]], [[VAR_4_2_]], [[RES_6_]], [[RES_5_]]) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<1x64xf16, #map1>, memref<64x256xf16, #map1>, memref<1x4x1x1x32x64xf16>, memref<3xi64>, memref<1x256xf16, #map1>) -> ()
// CHECK:             "zlow.unstick"([[RES_5_]], [[RES_2_]]) {layout = "2D"} : (memref<1x256xf16, #map1>, memref<1x256xf32>) -> ()
// CHECK:             async.yield
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1x256xf32>
// CHECK-DAG:       [[VAR_token_3_:%.+]] = async.execute {
// CHECK:             "krnl.call"([[CST_1_]]) {funcName = "threadAffine", numOfOutput = 1 : si64} : (i64) -> ()
// CHECK:             [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1x64xf16, #map1>
// CHECK:             "zlow.stick"([[PARAM_0_]], [[RES_8_]]) {layout = "2D"} : (memref<1x64xf32>, memref<1x64xf16, #map1>) -> ()
// CHECK:             [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<64x256xf16, #map1>
// CHECK:             "zlow.stick"([[RES_1_]], [[RES_9_]]) {layout = "2D"} : (memref<64x256xf32>, memref<64x256xf16, #map1>) -> ()
// CHECK-DAG:         [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1x256xf16, #map1>
// CHECK-DAG:         [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:             krnl.store [[CST_1_]], [[RES_11_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi64>
// CHECK:             krnl.store [[CST_64_]], [[RES_11_]]{{.}}[[CST_1_1_]]{{.}} : memref<3xi64>
// CHECK:             krnl.store [[CST_256_]], [[RES_11_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:             [[VAR_4_3_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_1", shape = [1, 4, 1, 1, 32, 64], value = dense_resource<zhigh_1> : tensor<16384xi8>} : () -> memref<1x4x1x1x32x64xf16>
// CHECK:             "zlow.matmul"([[RES_8_]], [[RES_9_]], [[VAR_4_3_]], [[RES_11_]], [[RES_10_]]) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<1x64xf16, #map1>, memref<64x256xf16, #map1>, memref<1x4x1x1x32x64xf16>, memref<3xi64>, memref<1x256xf16, #map1>) -> ()
// CHECK:             "zlow.unstick"([[RES_10_]], [[RES_7_]]) {layout = "2D"} : (memref<1x256xf16, #map1>, memref<1x256xf32>) -> ()
// CHECK:             async.yield
// CHECK:           }
// CHECK:           async.await [[VAR_token_]] : !async.token
// CHECK:           memref.dealloc [[RES_]] : memref<64x256xf32>
// CHECK:           async.await [[VAR_token_3_]] : !async.token
// CHECK:           memref.dealloc [[RES_1_]] : memref<64x256xf32>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1x512xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 256){
// CHECK:             [[VAR_4_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_4_4_]]#0, [[VAR_4_4_]]#1] : memref<1x256xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_12_]]{{.}}[[VAR_4_4_]]#0, [[VAR_4_4_]]#1] : memref<1x512xf32>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 256){
// CHECK:             [[VAR_4_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_4_5_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_4_5_]]#0, [[VAR_4_5_]]#1] : memref<1x256xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_2_]], [[RES_12_]]{{.}}[[VAR_4_5_]]#0, [[LOAD_PARAM_1_MEM_1_1_]]{{.}} : memref<1x512xf32>
// CHECK:           }
// CHECK:           return [[RES_12_]] : memref<1x512xf32>
// CHECK:         }

// -----