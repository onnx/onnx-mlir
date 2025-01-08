// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zlow-stick-expansion %s -split-input-file | FileCheck %s

// -----


#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_stick_expansion_with_sat(%arg0: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
  "zlow.stick"(%arg0, %alloc) {layout = "3DS", saturation = -1 : si64} : (memref<16x8x128xf32>, memref<16x8x128xf16, #map>) -> ()
  return %alloc : memref<16x8x128xf16, #map>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + 8)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 + 16)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0)[s0] -> (d0 + 24)>
// CHECK-LABEL:  func.func @test_stick_expansion_with_sat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x8x128xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<16x8x128xf16, #map> to memref<2x64xf16>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<16x8x128xf16, #map>
// CHECK:             [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_3_]])
// CHECK:             krnl.prefetch [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}}, read, locality<1>, data : memref<16x8x128xf32>
// CHECK:             krnl.prefetch [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}}, write, locality<1>, data : memref<16x8x128xf16, #map>
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 64 step 32 {
// CHECK:               [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_5_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_7_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.addi [[VAR_5_]], [[CST_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_9_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_11_:%.+]] = arith.addi [[VAR_5_]], [[CST_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]1] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.addi [[VAR_5_]], [[CST_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]3] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.addi [[VAR_5_]], [[CST_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]5] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.addi [[VAR_5_]], [[CST_24_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]7] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addi [[VAR_5_]], [[CST_28_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]9] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.maxnumf [[VAR_22_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.maxnumf [[VAR_23_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.maxnumf [[VAR_24_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.maxnumf [[VAR_25_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.maxnumf [[VAR_26_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.maxnumf [[VAR_27_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.maxnumf [[VAR_28_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_29_]], [[VAR_30_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_38_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_31_]], [[VAR_32_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_39_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_33_]], [[VAR_34_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               [[VAR_40_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_35_]], [[VAR_36_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[I_3_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_41_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_38_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_4_]]1] : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_42_:%.+]] = affine.apply [[MAP_5_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_4_]]2] : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_43_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_40_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_4_]]3] : memref<2x64xf16>, vector<8xf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8x128xf16, #map>
// CHECK:         }
}

// -----


#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_stick_expansion_without_sat(%arg0: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
  "zlow.stick"(%arg0, %alloc) {layout = "3DS", saturation = 0 : si64} : (memref<16x8x128xf32>, memref<16x8x128xf16, #map>) -> ()
  return %alloc : memref<16x8x128xf16, #map>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + 8)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 + 16)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0)[s0] -> (d0 + 24)>
// CHECK-LABEL:  func.func @test_stick_expansion_without_sat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x8x128xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<16x8x128xf16, #map> to memref<2x64xf16>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<16x8x128xf16, #map>
// CHECK:             [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#2, [[VAR_3_]])
// CHECK:             krnl.prefetch [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}}, read, locality<1>, data : memref<16x8x128xf32>
// CHECK:             krnl.prefetch [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}}, write, locality<1>, data : memref<16x8x128xf16, #map>
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 64 step 32 {
// CHECK:               [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_5_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_7_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.addi [[VAR_5_]], [[CST_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_9_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_11_:%.+]] = arith.addi [[VAR_5_]], [[CST_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]1] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.addi [[VAR_5_]], [[CST_16_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]3] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.addi [[VAR_5_]], [[CST_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]5] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.addi [[VAR_5_]], [[CST_24_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]7] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addi [[VAR_5_]], [[CST_28_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]9] : memref<16x8x128xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_22_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_0_MEM_3_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_23_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_0_MEM_5_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               [[VAR_24_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_0_MEM_7_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               vector.store [[VAR_21_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[I_3_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_25_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_22_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_25_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_26_:%.+]] = affine.apply [[MAP_5_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_23_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_26_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_27_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]]){{.}}[[VAR_2_]]{{.}}
// CHECK:               vector.store [[VAR_24_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_27_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8x128xf16, #map>
// CHECK:         }
}

// -----


#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_unstick_expansion(%arg0: memref<16x8x128xf16, #map>) -> memref<16x8x128xf32> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf32>
  "zlow.unstick"(%arg0, %alloc) {layout = "3DS"} : (memref<16x8x128xf16, #map>, memref<16x8x128xf32>) -> ()
  return %alloc : memref<16x8x128xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0] -> (s0 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + 8)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 + 16)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0)[s0] -> (d0 + 24)>
// CHECK-LABEL:  func.func @test_unstick_expansion
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf16, #map>) -> memref<16x8x128xf32> {
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x8x128xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<16x8x128xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 16){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                 [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_6_:%.+]] = affine.apply [[MAP_1_]]([[VAR_5_]])
// CHECK:                 [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}} : memref<16x8x128xf16, #map>
// CHECK:                 [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_5_]]){{.}}[[VAR_7_]]{{.}}
// CHECK:                 krnl.prefetch [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}}, read, locality<1>, data : memref<16x8x128xf16, #map>
// CHECK:                 krnl.prefetch [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}}, write, locality<1>, data : memref<16x8x128xf32>
// CHECK:                 scf.if [[VAR_true_]] {
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_64_]] step [[CST_32_]] {
// CHECK-DAG:                 [[VAR_9_:%.+]] = affine.apply [[MAP_3_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:                 [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[I_3_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_11_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_11_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_0_:%.+]], [[VAR_output2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_13_:%.+]] = affine.apply [[MAP_5_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_13_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_15_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_9_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_17_:%.+]] = arith.addi [[VAR_9_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]7] : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_18_:%.+]] = arith.addi [[VAR_9_]], [[CST_8_]] : index
// CHECK:                     vector.store [[VAR_output1_0_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]8] : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_19_:%.+]] = arith.addi [[VAR_9_]], [[CST_12_]] : index
// CHECK:                     vector.store [[VAR_output2_1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]9] : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_20_:%.+]] = arith.addi [[VAR_9_]], [[CST_16_]] : index
// CHECK:                     vector.store [[VAR_output1_2_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_20_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_21_:%.+]] = arith.addi [[VAR_9_]], [[CST_20_]] : index
// CHECK:                     vector.store [[VAR_output2_3_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_21_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_9_]], [[CST_24_]] : index
// CHECK:                     vector.store [[VAR_output1_4_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_22_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                     [[VAR_23_:%.+]] = arith.addi [[VAR_9_]], [[CST_28_]] : index
// CHECK:                     vector.store [[VAR_output2_5_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_23_]]{{.}} : memref<16x8x128xf32>, vector<4xf32>
// CHECK:                   }
// CHECK:                 } else {
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8x128xf32>
// CHECK:         }
}

// -----


#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_unstick_expansion_127(%arg0: memref<16x8x127xf16, #map>) -> memref<16x8x127xf32> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<16x8x127xf32>
  "zlow.unstick"(%arg0, %alloc) {layout = "3DS"} : (memref<16x8x127xf16, #map>, memref<16x8x127xf32>) -> ()
  return %alloc : memref<16x8x127xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0] -> (s0 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 * -64 + 63)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0)[s0] -> (d0 + 8)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0)[s0] -> (d0 + 16)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0)[s0] -> (d0 + 24)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<()[s0] -> (-s0 + 120)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<()[s0] -> ((-s0 + 127) mod 8)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<()[s0] -> (-s0 - (-s0 + 127) mod 8 + 127)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>
// CHECK-LABEL:  func.func @test_unstick_expansion_127
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x127xf16, #map>) -> memref<16x8x127xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x8x127xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<16x8x127xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 16){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                 [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_6_:%.+]] = affine.apply [[MAP_1_]]([[VAR_5_]])
// CHECK:                 [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}} : memref<16x8x127xf16, #map>
// CHECK-DAG:             [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_5_]]){{.}}[[VAR_7_]]{{.}}
// CHECK-DAG:             [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK:                 krnl.prefetch [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}}, read, locality<1>, data : memref<16x8x127xf16, #map>
// CHECK:                 krnl.prefetch [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]]{{.}}, write, locality<1>, data : memref<16x8x127xf32>
// CHECK:                 [[VAR_9_:%.+]] = affine.apply [[MAP_3_]]([[VAR_5_]]){{.}}[[VAR_7_]]{{.}}
// CHECK:                 [[VAR_10_:%.+]] = arith.cmpi sge, [[VAR_9_]], [[CST_0_]] : index
// CHECK:                 scf.if [[VAR_10_]] {
// CHECK:                   scf.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_64_]] step [[CST_32_]] {
// CHECK-DAG:                 [[VAR_11_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:                 [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[I_3_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_13_:%.+]] = affine.apply [[MAP_5_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_13_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_1_:%.+]], [[VAR_output2_2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_15_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_17_:%.+]] = affine.apply [[MAP_7_]]([[I_3_]]){{.}}[[VAR_6_]]{{.}}
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_17_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]1] : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_19_:%.+]] = arith.addi [[VAR_11_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]9] : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_20_:%.+]] = arith.addi [[VAR_11_]], [[CST_8_]] : index
// CHECK:                     vector.store [[VAR_output1_1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_20_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_21_:%.+]] = arith.addi [[VAR_11_]], [[CST_12_]] : index
// CHECK:                     vector.store [[VAR_output2_2_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_21_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_11_]], [[CST_16_]] : index
// CHECK:                     vector.store [[VAR_output1_3_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_22_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_23_:%.+]] = arith.addi [[VAR_11_]], [[CST_20_]] : index
// CHECK:                     vector.store [[VAR_output2_4_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_23_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_24_:%.+]] = arith.addi [[VAR_11_]], [[CST_24_]] : index
// CHECK:                     vector.store [[VAR_output1_5_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_24_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_25_:%.+]] = arith.addi [[VAR_11_]], [[CST_28_]] : index
// CHECK:                     vector.store [[VAR_output2_6_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_25_]]{{.}} : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                   }
// CHECK:                 } else {
// CHECK:                   [[VAR_11_1_:%.+]] = affine.apply [[MAP_8_]](){{.}}[[VAR_6_]]{{.}}
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_11_1_]] step [[CST_8_]] {
// CHECK-DAG:                 [[VAR_15_1_:%.+]] = affine.apply [[MAP_4_]]([[I_4_]]){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:                 [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_1_1_:%.+]], [[VAR_output2_2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     vector.store [[VAR_output1_1_1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]5] : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                     [[VAR_17_1_:%.+]] = arith.addi [[VAR_15_1_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_output2_2_1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]7] : memref<16x8x127xf32>, vector<4xf32>
// CHECK:                   }
// CHECK-DAG:               [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = affine.apply [[MAP_9_]](){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:               [[VAR_13_1_:%.+]] = affine.apply [[MAP_10_]](){{.}}[[VAR_6_]]{{.}}
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_13_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_1_:%.+]], [[VAR_output2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   vector.store [[VAR_output1_1_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<4xf32>
// CHECK:                   vector.store [[VAR_output2_1_]], [[RES_1_]]{{.}}[[CST_4_]]{{.}} : memref<8xf32>, vector<4xf32>
// CHECK:                   scf.for [[I_5_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_5_]] step [[CST_1_]] {
// CHECK-DAG:                 [[VAR_15_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_5_]]{{.}} : memref<8xf32>
// CHECK-DAG:                 [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = affine.apply [[MAP_11_]]([[I_5_]]){{.}}[[VAR_6_]], [[VAR_13_1_]]{{.}}
// CHECK:                     krnl.store [[VAR_15_1_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_1_]]6] : memref<16x8x127xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8x127xf32>
// CHECK:         }
}

