// RUN: onnx-mlir-opt -O3 --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_zhigh_quantized_stick_dlfloat16_symmetric(%arg0: tensor<1x3x5xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 1 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  return %0#0: tensor<*xf16>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_dlfloat16_symmetric
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf32>) -> memref<1x3x5xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0xFF800000> : vector<12xf32>
// CHECK-DAG:       [[CST_1_dot_270000_:%.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_15_:%.+]] = arith.constant 15 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_15_]], [[RES_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_]]) : (memref<1x3x5xf32>, memref<1xindex>) -> memref<15xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<12xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>, vector<12xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 12 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_6_]]{{.}} : memref<15xf32>, vector<12xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>, vector<12xf32>
// CHECK:             [[VAR_9_:%.+]] = math.absf [[LOAD_VAR_reshape_MEM_]] : vector<12xf32>
// CHECK:             [[VAR_10_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : vector<12xf32>
// CHECK:             vector.store [[VAR_10_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>, vector<12xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 12 to 15){
// CHECK:             [[VAR_6_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_6_1_]]{{.}} : memref<15xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>
// CHECK:             [[VAR_9_1_:%.+]] = math.absf [[LOAD_VAR_reshape_MEM_1_]] : f32
// CHECK:             [[VAR_10_1_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_1_]], [[VAR_9_1_]] : f32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<12xf32>, vector<12xf32>
// CHECK:           [[VAR_3_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_1_MEM_2_]] : vector<12xf32> into f32
// CHECK:           krnl.store [[VAR_3_]], [[RES_2_]][] : memref<f32>
// CHECK:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.divf [[CST_1_dot_270000_]], [[LOAD_RES_2_MEM_]] : f32
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_5_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_4_]][] : memref<f32>
// CHECK:           [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf16, #map>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_5_]]) {layout = "3DS", saturation = -1 : si64} : (memref<1x3x5xf32>, memref<1x3x5xf16, #map>) -> ()
// CHECK:           return [[RES_5_]] : memref<1x3x5xf16, #map>
// CHECK:         }
}
