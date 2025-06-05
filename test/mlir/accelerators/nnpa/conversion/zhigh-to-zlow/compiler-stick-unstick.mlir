// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @should_lower_to_zlow(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 * 3 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 * 3 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 3)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 4)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 5)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 6)>
// CHECK-LABEL:  func.func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x7xf32>) -> memref<1x3x5x7xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x5x7x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x5x7x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 7, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<1x3x5x7xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_3_]]{{.}} : memref<1x5x7x3xf32>
// CHECK:             [[VAR_5_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_5_]], [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<1x3x5x7xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_5_]]{{.}} : memref<1x5x7x3xf32>
// CHECK:             [[VAR_7_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_2_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_7_]], [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<1x3x5x7xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_7_]]{{.}} : memref<1x5x7x3xf32>
// CHECK:           }
// CHECK:           "zlow.stick"([[RES_1_]], [[RES_]]) {layout = "NHWC"} : (memref<1x5x7x3xf32>, memref<1x5x7x3xf16, #map>) -> ()
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x5x7x3xf32>
// CHECK:           "zlow.unstick"([[RES_]], [[RES_]]_1) {layout = "NHWC"} : (memref<1x5x7x3xf16, #map>, memref<1x5x7x3xf32>) -> ()
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x7xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 5, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_3_1_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_3_1_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_3_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_3_1_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_5_1_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_5_1_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_5_1_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_7_1_:%.+]] = affine.apply [[MAP_6_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_7_1_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_7_1_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_9_:%.+]] = affine.apply [[MAP_7_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_9_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_9_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_11_:%.+]] = affine.apply [[MAP_8_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_11_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_11_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_13_:%.+]] = affine.apply [[MAP_9_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_13_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_13_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:             [[VAR_15_:%.+]] = affine.apply [[MAP_10_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_15_]], [[VAR_2_1_]]#1] : memref<1x5x7x3xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_15_]]{{.}} : memref<1x3x5x7xf32>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<1x3x5x7xf32>
// CHECK:         }
}

// -----


func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 * 7)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 4)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 5)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<()[s0] -> (s0 * 7 + 6)>
// CHECK-LABEL:  func.func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x7xf32>) -> memref<1x?x?x7xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x?x7xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x?x?x7xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) {{.*}}: memref<1x?x7x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x?x?x7xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x?x7xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<1x?x7x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_1_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 7, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_1_]], [[VAR_dim_2_]])){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#3, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<1x?x?x7xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<1x?x7x?xf32>
// CHECK:           }
// CHECK:           "zlow.stick"([[RES_1_]], [[RES_]]) {layout = "NHWC"} : (memref<1x?x7x?xf32>, memref<1x?x7x?xf16, #map>) -> ()
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) {{.*}}: memref<1x?x7x?xf32>
// CHECK:           "zlow.unstick"([[RES_]], [[RES_]]_4) {layout = "NHWC"} : (memref<1x?x7x?xf16, #map>, memref<1x?x7x?xf32>) -> ()
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<1x?x?x7xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_0), [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[LOAD_PARAM_0_MEM_1_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_5_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_5_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_5_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_7_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_7_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_2_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_7_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_9_:%.+]] = affine.apply [[MAP_6_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_9_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_9_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_11_:%.+]] = affine.apply [[MAP_7_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_11_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_4_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_11_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_13_:%.+]] = affine.apply [[MAP_8_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_13_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_5_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_13_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:             [[VAR_15_:%.+]] = affine.apply [[MAP_9_]](){{.}}[[VAR_2_1_]]#3]
// CHECK:             [[LOAD_RES_2_MEM_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2, [[VAR_15_]], [[VAR_2_1_]]#1] : memref<1x?x7x?xf32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_6_]], [[RES_3_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_15_]]{{.}} : memref<1x?x?x7xf32>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<1x?x?x7xf32>
// CHECK:         }
}

