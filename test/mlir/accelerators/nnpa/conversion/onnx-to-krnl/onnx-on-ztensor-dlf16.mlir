// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test doing unary element-wise computation directly on zTensor.
// Taking ONNXSqrtOp as the example.
// Need to check that the buffer is correctly aligned to 4K.

func.func @test_onnx_sqrt_ztensor(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 7)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>) -> memref<?x3x5x7xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK:                   [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_10_:%.+]] = affine.apply [[MAP_3_]]([[VAR_9_]])
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_12_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_]])
// CHECK-DAG:               [[VAR_13_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_13_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_15_:%.+]] = math.sqrt [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_16_:%.+]] = math.sqrt [[VAR_output2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.minnumf [[VAR_15_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.minnumf [[VAR_16_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.maxnumf [[VAR_17_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.maxnumf [[VAR_18_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_21_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_19_]], [[VAR_20_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_21_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}

// -----


func.func @test_onnx_sqrt_ztensor_s8(%arg0: tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 8)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor_s8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x8xf16, #map>) -> memref<?x3x5x8xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x8xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x5x8xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x8xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x8xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK:                   [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x8xf16, #map>
// CHECK-DAG:               [[VAR_10_:%.+]] = affine.apply [[MAP_3_]]([[VAR_9_]])
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x8xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_12_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_]])
// CHECK-DAG:               [[VAR_13_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_13_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_15_:%.+]] = math.sqrt [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_16_:%.+]] = math.sqrt [[VAR_output2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.minnumf [[VAR_15_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.minnumf [[VAR_16_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.maxnumf [[VAR_17_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.maxnumf [[VAR_18_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_21_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_19_]], [[VAR_20_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_21_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x8xf16, #map>
// CHECK:         }
}

// -----


func.func @test_onnx_sqrt_ztensor_s64(%arg0: tensor<?x3x5x64xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x64xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x64xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x64xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x64xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor_s64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x64xf16, #map>) -> memref<?x3x5x64xf16, #map> {
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x64xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x5x64xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x64xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK:                   [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x64xf16, #map>
// CHECK-DAG:               [[VAR_10_:%.+]] = affine.apply [[MAP_3_]]([[VAR_9_]])
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_12_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_]])
// CHECK-DAG:               [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_14_:%.+]] = math.sqrt [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_15_:%.+]] = math.sqrt [[VAR_output2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_16_:%.+]] = arith.minnumf [[VAR_14_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_17_:%.+]] = arith.minnumf [[VAR_15_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_18_:%.+]] = arith.maxnumf [[VAR_16_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_19_:%.+]] = arith.maxnumf [[VAR_17_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_20_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_18_]], [[VAR_19_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_20_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_22_:%.+]] = math.sqrt [[VAR_output1_2_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_23_:%.+]] = math.sqrt [[VAR_output2_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_24_:%.+]] = arith.minnumf [[VAR_22_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_25_:%.+]] = arith.minnumf [[VAR_23_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_26_:%.+]] = arith.maxnumf [[VAR_24_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_27_:%.+]] = arith.maxnumf [[VAR_25_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_28_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_26_]], [[VAR_27_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_28_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_30_:%.+]] = math.sqrt [[VAR_output1_4_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_31_:%.+]] = math.sqrt [[VAR_output2_5_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_32_:%.+]] = arith.minnumf [[VAR_30_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_33_:%.+]] = arith.minnumf [[VAR_31_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_34_:%.+]] = arith.maxnumf [[VAR_32_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_35_:%.+]] = arith.maxnumf [[VAR_33_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_36_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_34_]], [[VAR_35_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_36_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_6_:%.+]], [[VAR_output2_7_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_38_:%.+]] = math.sqrt [[VAR_output1_6_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_39_:%.+]] = math.sqrt [[VAR_output2_7_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_40_:%.+]] = arith.minnumf [[VAR_38_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_41_:%.+]] = arith.minnumf [[VAR_39_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_42_:%.+]] = arith.maxnumf [[VAR_40_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_43_:%.+]] = arith.maxnumf [[VAR_41_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_44_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_42_]], [[VAR_43_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_44_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_8_:%.+]], [[VAR_output2_9_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_46_:%.+]] = math.sqrt [[VAR_output1_8_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_47_:%.+]] = math.sqrt [[VAR_output2_9_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_48_:%.+]] = arith.minnumf [[VAR_46_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_49_:%.+]] = arith.minnumf [[VAR_47_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_50_:%.+]] = arith.maxnumf [[VAR_48_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_51_:%.+]] = arith.maxnumf [[VAR_49_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_52_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_50_]], [[VAR_51_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_52_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_10_:%.+]], [[VAR_output2_11_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_54_:%.+]] = math.sqrt [[VAR_output1_10_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_55_:%.+]] = math.sqrt [[VAR_output2_11_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_56_:%.+]] = arith.minnumf [[VAR_54_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_57_:%.+]] = arith.minnumf [[VAR_55_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_58_:%.+]] = arith.maxnumf [[VAR_56_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_59_:%.+]] = arith.maxnumf [[VAR_57_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_60_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_58_]], [[VAR_59_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_60_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_12_:%.+]], [[VAR_output2_13_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_62_:%.+]] = math.sqrt [[VAR_output1_12_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_63_:%.+]] = math.sqrt [[VAR_output2_13_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_64_:%.+]] = arith.minnumf [[VAR_62_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_65_:%.+]] = arith.minnumf [[VAR_63_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_66_:%.+]] = arith.maxnumf [[VAR_64_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_67_:%.+]] = arith.maxnumf [[VAR_65_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_68_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_66_]], [[VAR_67_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_68_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_14_:%.+]], [[VAR_output2_15_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_70_:%.+]] = math.sqrt [[VAR_output1_14_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_71_:%.+]] = math.sqrt [[VAR_output2_15_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_72_:%.+]] = arith.minnumf [[VAR_70_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_73_:%.+]] = arith.minnumf [[VAR_71_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_74_:%.+]] = arith.maxnumf [[VAR_72_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_75_:%.+]] = arith.maxnumf [[VAR_73_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                   [[VAR_76_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_74_]], [[VAR_75_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                   vector.store [[VAR_76_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x64xf16, #map>
// CHECK:         }
}

// -----


func.func @test_onnx_sqrt_ztensor_s72(%arg0: tensor<?x3x5x72xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x72xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x72xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x72xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x72xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 8)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * -64 + 72)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor_s72
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x72xf16, #map>) -> memref<?x3x5x72xf16, #map> {
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x72xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x5x72xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x72xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x72xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK:                   [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x72xf16, #map>
// CHECK-DAG:               [[VAR_10_:%.+]] = affine.apply [[MAP_3_]]([[VAR_9_]])
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x72xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_12_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_]])
// CHECK-DAG:               [[VAR_13_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]])
// CHECK:                   [[VAR_14_:%.+]] = arith.cmpi sge, [[VAR_13_]], [[CST_0_]] : index
// CHECK:                   scf.if [[VAR_14_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_16_:%.+]] = math.sqrt [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = math.sqrt [[VAR_output2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.minnumf [[VAR_16_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.minnumf [[VAR_17_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.maxnumf [[VAR_18_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.maxnumf [[VAR_19_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_22_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_20_]], [[VAR_21_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_22_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_24_:%.+]] = math.sqrt [[VAR_output1_2_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = math.sqrt [[VAR_output2_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_26_:%.+]] = arith.minnumf [[VAR_24_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_27_:%.+]] = arith.minnumf [[VAR_25_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_28_:%.+]] = arith.maxnumf [[VAR_26_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_29_:%.+]] = arith.maxnumf [[VAR_27_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_30_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_28_]], [[VAR_29_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_30_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_32_:%.+]] = math.sqrt [[VAR_output1_4_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = math.sqrt [[VAR_output2_5_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_34_:%.+]] = arith.minnumf [[VAR_32_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_35_:%.+]] = arith.minnumf [[VAR_33_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_36_:%.+]] = arith.maxnumf [[VAR_34_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_37_:%.+]] = arith.maxnumf [[VAR_35_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_38_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_36_]], [[VAR_37_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_38_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_6_:%.+]], [[VAR_output2_7_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_40_:%.+]] = math.sqrt [[VAR_output1_6_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = math.sqrt [[VAR_output2_7_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.minnumf [[VAR_40_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.minnumf [[VAR_41_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.maxnumf [[VAR_43_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_46_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_44_]], [[VAR_45_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_46_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_8_:%.+]], [[VAR_output2_9_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_48_:%.+]] = math.sqrt [[VAR_output1_8_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = math.sqrt [[VAR_output2_9_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[VAR_48_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[VAR_49_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.maxnumf [[VAR_50_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.maxnumf [[VAR_51_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_54_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_52_]], [[VAR_53_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_54_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_10_:%.+]], [[VAR_output2_11_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_56_:%.+]] = math.sqrt [[VAR_output1_10_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_57_:%.+]] = math.sqrt [[VAR_output2_11_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.minnumf [[VAR_56_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.minnumf [[VAR_57_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.maxnumf [[VAR_58_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.maxnumf [[VAR_59_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_62_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_60_]], [[VAR_61_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_62_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_12_:%.+]], [[VAR_output2_13_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_64_:%.+]] = math.sqrt [[VAR_output1_12_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_65_:%.+]] = math.sqrt [[VAR_output2_13_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.minnumf [[VAR_64_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.minnumf [[VAR_65_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.maxnumf [[VAR_66_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.maxnumf [[VAR_67_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_70_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_68_]], [[VAR_69_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_70_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_14_:%.+]], [[VAR_output2_15_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_72_:%.+]] = math.sqrt [[VAR_output1_14_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_73_:%.+]] = math.sqrt [[VAR_output2_15_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_74_:%.+]] = arith.minnumf [[VAR_72_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.minnumf [[VAR_73_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.maxnumf [[VAR_74_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_77_:%.+]] = arith.maxnumf [[VAR_75_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_78_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_76_]], [[VAR_77_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_78_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   } else {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = affine.apply [[MAP_5_]]([[VAR_7_]])
// CHECK:                     scf.for [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_8_]] step [[CST_8_]] {
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                       [[VAR_output1_1_:%.+]], [[VAR_output2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                   [[VAR_17_1_:%.+]] = math.sqrt [[VAR_output1_1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_18_1_:%.+]] = math.sqrt [[VAR_output2_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_19_1_:%.+]] = arith.minnumf [[VAR_17_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_20_1_:%.+]] = arith.minnumf [[VAR_18_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_21_1_:%.+]] = arith.maxnumf [[VAR_19_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_22_1_:%.+]] = arith.maxnumf [[VAR_20_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_21_1_]], [[VAR_22_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                       vector.store [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x72xf16, #map>
// CHECK:         }
}

// -----


func.func @test_onnx_sqrt_ztensor_s75(%arg0: tensor<?x3x5x75xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x75xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x3x5x75xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x75xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x75xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 11)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * -64 + 75)>
// CHECK-LABEL:  func.func @test_onnx_sqrt_ztensor_s75
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x75xf16, #map>) -> memref<?x3x5x75xf16, #map> {
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x75xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x3x5x75xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x75xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x75xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK:                   [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x75xf16, #map>
// CHECK-DAG:               [[VAR_10_:%.+]] = affine.apply [[MAP_3_]]([[VAR_9_]])
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_5_]], [[VAR_8_]]{{.}} : memref<?x3x5x75xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_12_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_]])
// CHECK-DAG:               [[VAR_13_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]])
// CHECK:                   [[VAR_14_:%.+]] = arith.cmpi sge, [[VAR_13_]], [[CST_0_]] : index
// CHECK:                   scf.if [[VAR_14_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_16_:%.+]] = math.sqrt [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = math.sqrt [[VAR_output2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.minnumf [[VAR_16_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.minnumf [[VAR_17_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.maxnumf [[VAR_18_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.maxnumf [[VAR_19_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_22_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_20_]], [[VAR_21_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_22_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_24_:%.+]] = math.sqrt [[VAR_output1_2_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = math.sqrt [[VAR_output2_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_26_:%.+]] = arith.minnumf [[VAR_24_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_27_:%.+]] = arith.minnumf [[VAR_25_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_28_:%.+]] = arith.maxnumf [[VAR_26_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_29_:%.+]] = arith.maxnumf [[VAR_27_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_30_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_28_]], [[VAR_29_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_30_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_32_:%.+]] = math.sqrt [[VAR_output1_4_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = math.sqrt [[VAR_output2_5_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_34_:%.+]] = arith.minnumf [[VAR_32_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_35_:%.+]] = arith.minnumf [[VAR_33_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_36_:%.+]] = arith.maxnumf [[VAR_34_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_37_:%.+]] = arith.maxnumf [[VAR_35_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_38_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_36_]], [[VAR_37_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_38_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_6_:%.+]], [[VAR_output2_7_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_40_:%.+]] = math.sqrt [[VAR_output1_6_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = math.sqrt [[VAR_output2_7_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.minnumf [[VAR_40_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.minnumf [[VAR_41_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.maxnumf [[VAR_43_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_46_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_44_]], [[VAR_45_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_46_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_8_:%.+]], [[VAR_output2_9_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_48_:%.+]] = math.sqrt [[VAR_output1_8_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = math.sqrt [[VAR_output2_9_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_50_:%.+]] = arith.minnumf [[VAR_48_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[VAR_49_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.maxnumf [[VAR_50_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.maxnumf [[VAR_51_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_54_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_52_]], [[VAR_53_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_54_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_10_:%.+]], [[VAR_output2_11_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_56_:%.+]] = math.sqrt [[VAR_output1_10_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_57_:%.+]] = math.sqrt [[VAR_output2_11_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.minnumf [[VAR_56_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.minnumf [[VAR_57_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.maxnumf [[VAR_58_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.maxnumf [[VAR_59_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_62_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_60_]], [[VAR_61_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_62_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_12_:%.+]], [[VAR_output2_13_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_64_:%.+]] = math.sqrt [[VAR_output1_12_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_65_:%.+]] = math.sqrt [[VAR_output2_13_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.minnumf [[VAR_64_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.minnumf [[VAR_65_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.maxnumf [[VAR_66_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.maxnumf [[VAR_67_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_70_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_68_]], [[VAR_69_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_70_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_14_:%.+]], [[VAR_output2_15_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_72_:%.+]] = math.sqrt [[VAR_output1_14_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_73_:%.+]] = math.sqrt [[VAR_output2_15_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_74_:%.+]] = arith.minnumf [[VAR_72_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.minnumf [[VAR_73_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.maxnumf [[VAR_74_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_77_:%.+]] = arith.maxnumf [[VAR_75_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_78_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_76_]], [[VAR_77_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_78_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   } else {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = affine.apply [[MAP_5_]]([[VAR_7_]])
// CHECK:                     scf.for [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_8_]] step [[CST_8_]] {
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                       [[VAR_output1_1_:%.+]], [[VAR_output2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                   [[VAR_17_1_:%.+]] = math.sqrt [[VAR_output1_1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_18_1_:%.+]] = math.sqrt [[VAR_output2_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_19_1_:%.+]] = arith.minnumf [[VAR_17_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_20_1_:%.+]] = arith.minnumf [[VAR_18_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_21_1_:%.+]] = arith.maxnumf [[VAR_19_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_22_1_:%.+]] = arith.maxnumf [[VAR_20_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_21_1_]], [[VAR_22_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                       vector.store [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x75xf16, #map>
// CHECK:         }
}

// -----

// Test doing broadcasting binary element-wise computation directly on zTensor.
// All input/output stick

func.func @test_onnx_add_ztensor_sss(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 7)>
// CHECK-LABEL:  func.func @test_onnx_add_ztensor_sss
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x7xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_15_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_]])
// CHECK-DAG:               [[VAR_16_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_16_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.addf [[VAR_output1_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.addf [[VAR_output2_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.minnumf [[VAR_18_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.minnumf [[VAR_19_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.maxnumf [[VAR_20_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_24_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_22_]], [[VAR_23_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_24_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}

// -----

// Same; normal / stick inputs, stick output.

func.func @test_onnx_sub_ztensor_nss(%arg0: tensor<?x3x5x7xf32>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<?x3x5x7xf32>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-LABEL:  func.func @test_onnx_sub_ztensor_nss
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf32>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x7xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x7xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_11_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_11_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_12_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_13_:%.+]] = affine.apply [[MAP_3_]]([[VAR_12_]])
// CHECK-DAG:               [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_14_]] step [[CST_8_]] {
// CHECK:                     [[VAR_15_:%.+]] = affine.apply [[MAP_5_]]([[I_4_]], [[VAR_8_]])
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_15_]]{{.}} : memref<?x3x5x7xf32>, vector<4xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.addi [[VAR_15_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_17_]]{{.}} : memref<?x3x5x7xf32>, vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.minnumf [[VAR_19_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.minnumf [[VAR_20_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_24_:%.+]] = arith.maxnumf [[VAR_22_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_25_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_23_]], [[VAR_24_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_25_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_13_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}

// -----

// Same; stick / normal inputs, stick output.

func.func @test_onnx_mul_ztensor_sns(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 7)>
// CHECK-LABEL:  func.func @test_onnx_mul_ztensor_sns
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf32>) -> memref<?x3x5x7xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f32 to vector<4xf32>
// CHECK-DAG:               [[VAR_14_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_15_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_]])
// CHECK-DAG:               [[VAR_16_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_16_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.mulf [[VAR_output1_]], [[VAR_13_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.mulf [[VAR_output2_]], [[VAR_13_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.minnumf [[VAR_18_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.minnumf [[VAR_19_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.maxnumf [[VAR_20_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_24_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_22_]], [[VAR_23_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_24_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf16, #map>
// CHECK:         }
}

// -----

// Same; stick / stick inputs, normal output.

func.func @test_onnx_div_ztensor_ssn(%arg0: tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x7xf32>
  return %0 : tensor<?x3x5x7xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64)>
// CHECK-LABEL:  func.func @test_onnx_div_ztensor_ssn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x7xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x7xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x7xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x7xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x7xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x7xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK-DAG:               [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x8xf32>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[VAR_14_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_16_:%.+]] = arith.divf [[VAR_output1_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_17_:%.+]] = arith.divf [[VAR_output2_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_16_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_7_]] step [[CST_1_]] {
// CHECK:                     [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[I_4_]]{{.}} : memref<1x8xf32>
// CHECK:                     krnl.store [[LOAD_RES_1_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[I_4_]]{{.}} : memref<?x3x5x7xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x7xf32>
// CHECK:         }
}

// -----

// Same; stick / stick inputs, normal output.

func.func @test_onnx_div_ztensor_ssn_s8(%arg0: tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x8xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x3x5x8xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x8xf32>
  return %0 : tensor<?x3x5x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-LABEL:  func.func @test_onnx_div_ztensor_ssn_s8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x8xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x8xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x8xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x8xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x8xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x8xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_14_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_1_:%.+]], [[VAR_output2_2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_16_:%.+]] = arith.divf [[VAR_output1_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.divf [[VAR_output2_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_18_:%.+]] = affine.apply [[MAP_5_]]([[I_4_]], [[VAR_8_]])
// CHECK:                     vector.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_18_]]{{.}} : memref<?x3x5x8xf32>, vector<4xf32>
// CHECK:                     [[VAR_19_:%.+]] = arith.addi [[VAR_18_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_17_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_19_]]{{.}} : memref<?x3x5x8xf32>, vector<4xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x8xf32>
// CHECK:         }
}

// -----

// Same; stick / stick inputs, normal output.

func.func @test_onnx_div_ztensor_ssn_s9(%arg0: tensor<?x3x5x9xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x9xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x3x5x9xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x9xf32>
  return %0 : tensor<?x3x5x9xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * -64 + 8)>
// CHECK-LABEL:  func.func @test_onnx_div_ztensor_ssn_s9
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x9xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x9xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x9xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x9xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x9xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x9xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_14_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.divf [[VAR_output1_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.divf [[VAR_output2_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_23_:%.+]] = affine.apply [[MAP_5_]]([[I_4_]], [[VAR_8_]])
// CHECK:                     vector.store [[VAR_21_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]3] : memref<?x3x5x9xf32>, vector<4xf32>
// CHECK:                     [[VAR_24_:%.+]] = arith.addi [[VAR_23_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_22_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]4] : memref<?x3x5x9xf32>, vector<4xf32>
// CHECK:                   }
// CHECK-DAG:               [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[VAR_8_]])
// CHECK-DAG:               [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x8xf32>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[VAR_15_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_17_:%.+]] = arith.divf [[VAR_output1_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_18_:%.+]] = arith.divf [[VAR_output2_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   vector.store [[VAR_18_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>
// CHECK:                   krnl.store [[LOAD_RES_1_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_8_]]{{.}} : memref<?x3x5x9xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x9xf32>
// CHECK:         }
}

// -----

// Same; stick / stick inputs, normal output.

func.func @test_onnx_div_ztensor_ssn_s19(%arg0: tensor<?x3x5x19xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x19xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x3x5x19xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x19xf32>
  return %0 : tensor<?x3x5x19xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 12)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * -64 + 16)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-LABEL:  func.func @test_onnx_div_ztensor_ssn_s19
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x19xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x19xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x19xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x19xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x19xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x19xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   scf.for [[I_4_:%.+]] = [[CST_0_]] to [[VAR_14_]] step [[CST_8_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.divf [[VAR_output1_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.divf [[VAR_output2_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_22_:%.+]] = affine.apply [[MAP_5_]]([[I_4_]], [[VAR_8_]])
// CHECK:                     vector.store [[VAR_20_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]2] : memref<?x3x5x19xf32>, vector<4xf32>
// CHECK:                     [[VAR_23_:%.+]] = arith.addi [[VAR_22_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_21_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]3] : memref<?x3x5x19xf32>, vector<4xf32>
// CHECK:                   }
// CHECK-DAG:               [[VAR_15_:%.+]] = affine.apply [[MAP_6_]]([[VAR_8_]])
// CHECK-DAG:               [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x8xf32>
// CHECK:                   [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[VAR_15_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:               [[VAR_17_:%.+]] = arith.divf [[VAR_output1_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:               [[VAR_18_:%.+]] = arith.divf [[VAR_output2_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   vector.store [[VAR_18_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                   scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] {
// CHECK-DAG:                 [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[I_5_]]{{.}} : memref<1x8xf32>
// CHECK-DAG:                 [[VAR_20_1_:%.+]] = affine.apply [[MAP_7_]]([[I_5_]])
// CHECK:                     krnl.store [[LOAD_VAR_reinterpret_cast_MEM_2_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]0] : memref<?x3x5x19xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x19xf32>
// CHECK:         }
}

// -----

// Same; stick / stick inputs, normal output.

func.func @test_onnx_div_ztensor_ssn_s77(%arg0: tensor<?x3x5x77xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x77xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x3x5x77xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x77xf32>
  return %0 : tensor<?x3x5x77xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 13)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 64 + 8)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 64 + 16)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 64 + 24)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 64 + 40)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 * 64 + 48)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0) -> (d0 * 64 + 56)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 * -64 + 70)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0) -> (d0 * -64 + 72)>
// CHECK-DAG:   [[MAP_15_:#.+]] = affine_map<(d0) -> (d0 + 72)>
// CHECK-LABEL:  func.func @test_onnx_div_ztensor_ssn_s77
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x77xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x77xf32> {
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x77xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x77xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x77xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x77xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:                   scf.if [[VAR_15_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_1_:%.+]], [[VAR_output2_2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.divf [[VAR_output1_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_18_:%.+]] = arith.divf [[VAR_output2_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_17_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_19_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_20_:%.+]] = arith.addi [[VAR_19_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_18_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]0] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.divf [[VAR_output1_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.divf [[VAR_output2_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]] = affine.apply [[MAP_5_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_22_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]4] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_25_:%.+]] = arith.addi [[VAR_24_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_23_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]5] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_27_:%.+]] = arith.divf [[VAR_output1_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_28_:%.+]] = arith.divf [[VAR_output2_6_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_29_:%.+]] = affine.apply [[MAP_6_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_27_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]9] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_30_:%.+]] = arith.addi [[VAR_29_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_30_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_7_:%.+]], [[VAR_output2_8_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_32_:%.+]] = arith.divf [[VAR_output1_7_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = arith.divf [[VAR_output2_8_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_34_:%.+]] = affine.apply [[MAP_7_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_32_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_34_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_35_:%.+]] = arith.addi [[VAR_34_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_33_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_35_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_9_:%.+]], [[VAR_output2_10_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_37_:%.+]] = arith.divf [[VAR_output1_9_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_38_:%.+]] = arith.divf [[VAR_output2_10_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_39_:%.+]] = affine.apply [[MAP_8_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_37_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_39_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_40_:%.+]] = arith.addi [[VAR_39_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_38_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]0] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_11_:%.+]], [[VAR_output2_12_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_42_:%.+]] = arith.divf [[VAR_output1_11_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.divf [[VAR_output2_12_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_44_:%.+]] = affine.apply [[MAP_9_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_42_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]4] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_45_:%.+]] = arith.addi [[VAR_44_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_43_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]5] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_13_:%.+]], [[VAR_output2_14_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.divf [[VAR_output1_13_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.divf [[VAR_output2_14_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = affine.apply [[MAP_10_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_47_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]9] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_50_:%.+]] = arith.addi [[VAR_49_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_48_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_50_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_15_:%.+]], [[VAR_output2_16_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.divf [[VAR_output1_15_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.divf [[VAR_output2_16_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_54_:%.+]] = affine.apply [[MAP_11_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_52_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_54_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     [[VAR_55_:%.+]] = arith.addi [[VAR_54_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_53_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_55_]]{{.}} : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                   } else {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = affine.apply [[MAP_12_]]([[VAR_8_]])
// CHECK:                     scf.for [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_8_]] step [[CST_8_]] {
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                       [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                   [[VAR_22_1_:%.+]] = arith.divf [[VAR_output1_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_23_1_:%.+]] = arith.divf [[VAR_output2_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_24_1_:%.+]] = affine.apply [[MAP_13_]]([[I_4_]], [[VAR_8_]])
// CHECK:                       vector.store [[VAR_22_1_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]4] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                       [[VAR_25_1_:%.+]] = arith.addi [[VAR_24_1_]], [[CST_4_]] : index
// CHECK:                       vector.store [[VAR_23_1_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]5] : memref<?x3x5x77xf32>, vector<4xf32>
// CHECK:                     }
// CHECK-DAG:                 [[VAR_17_1_:%.+]] = affine.apply [[MAP_14_]]([[VAR_8_]])
// CHECK-DAG:                 [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x8xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_10_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_10_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_19_1_:%.+]] = arith.divf [[VAR_output1_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_20_1_:%.+]] = arith.divf [[VAR_output2_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK:                     vector.store [[VAR_19_1_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                     vector.store [[VAR_20_1_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                     scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]] {
// CHECK-DAG:                   [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[I_5_]]{{.}} : memref<1x8xf32>
// CHECK-DAG:                   [[VAR_22_2_:%.+]] = affine.apply [[MAP_15_]]([[I_5_]])
// CHECK:                       krnl.store [[LOAD_VAR_reinterpret_cast_MEM_9_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]2] : memref<?x3x5x77xf32>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x77xf32>
// CHECK:         }
}

// -----

// Same; normal / normal inputs, stick output.
func.func @test_onnx_max_ztensor_nns(%arg0: tensor<?x3x5x7xf32>, %arg1: tensor<?x3x5x1xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<?x3x5x7xf32>, tensor<?x3x5x1xf32>) -> tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x7xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-LABEL:  func.func @test_onnx_max_ztensor_nns
}

// -----

// Test where the last dim has 2 64 iters, 1 8 iter, and one 1 iter.

func.func @test_onnx_min_ztensor_sss_big(%arg0: tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 73)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * -64 + 137)>
// CHECK-LABEL:  func.func @test_onnx_min_ztensor_sss_big
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x137xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x137xf16, #map> {
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x137xf16, #map>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x137xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x137xf16, #map> to memref<2x64xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x137xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 3){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x137xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x137xf16, #map>
// CHECK-DAG:               [[VAR_15_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_]])
// CHECK-DAG:               [[VAR_16_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   [[VAR_17_:%.+]] = arith.cmpi sge, [[VAR_16_]], [[CST_0_]] : index
// CHECK:                   scf.if [[VAR_17_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.minnumf [[VAR_output1_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_20_:%.+]] = arith.minnumf [[VAR_output2_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_21_:%.+]] = arith.minnumf [[VAR_19_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.minnumf [[VAR_20_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]] = arith.maxnumf [[VAR_22_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_25_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_23_]], [[VAR_24_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_25_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_27_:%.+]] = arith.minnumf [[VAR_output1_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_28_:%.+]] = arith.minnumf [[VAR_output2_6_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_29_:%.+]] = arith.minnumf [[VAR_27_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_30_:%.+]] = arith.minnumf [[VAR_28_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_31_:%.+]] = arith.maxnumf [[VAR_29_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_32_:%.+]] = arith.maxnumf [[VAR_30_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_33_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_31_]], [[VAR_32_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_33_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_7_:%.+]], [[VAR_output2_8_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_35_:%.+]] = arith.minnumf [[VAR_output1_7_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_36_:%.+]] = arith.minnumf [[VAR_output2_8_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_37_:%.+]] = arith.minnumf [[VAR_35_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_38_:%.+]] = arith.minnumf [[VAR_36_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_39_:%.+]] = arith.maxnumf [[VAR_37_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_40_:%.+]] = arith.maxnumf [[VAR_38_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_41_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_39_]], [[VAR_40_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_41_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_9_:%.+]], [[VAR_output2_10_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_43_:%.+]] = arith.minnumf [[VAR_output1_9_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_44_:%.+]] = arith.minnumf [[VAR_output2_10_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.minnumf [[VAR_43_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.minnumf [[VAR_44_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.maxnumf [[VAR_45_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_48_:%.+]] = arith.maxnumf [[VAR_46_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_49_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_47_]], [[VAR_48_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_49_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_11_:%.+]], [[VAR_output2_12_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_51_:%.+]] = arith.minnumf [[VAR_output1_11_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_52_:%.+]] = arith.minnumf [[VAR_output2_12_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_53_:%.+]] = arith.minnumf [[VAR_51_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.minnumf [[VAR_52_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = arith.maxnumf [[VAR_53_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_57_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_55_]], [[VAR_56_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_57_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_13_:%.+]], [[VAR_output2_14_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_59_:%.+]] = arith.minnumf [[VAR_output1_13_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_60_:%.+]] = arith.minnumf [[VAR_output2_14_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.minnumf [[VAR_59_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_62_:%.+]] = arith.minnumf [[VAR_60_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.maxnumf [[VAR_61_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_64_:%.+]] = arith.maxnumf [[VAR_62_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_65_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_63_]], [[VAR_64_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_65_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_15_:%.+]], [[VAR_output2_16_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.minnumf [[VAR_output1_15_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.minnumf [[VAR_output2_16_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_69_:%.+]] = arith.minnumf [[VAR_67_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_70_:%.+]] = arith.minnumf [[VAR_68_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_71_:%.+]] = arith.maxnumf [[VAR_69_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.maxnumf [[VAR_70_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_73_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_71_]], [[VAR_72_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_73_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_17_:%.+]], [[VAR_output2_18_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                 [[VAR_75_:%.+]] = arith.minnumf [[VAR_output1_17_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.minnumf [[VAR_output2_18_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_77_:%.+]] = arith.minnumf [[VAR_75_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_78_:%.+]] = arith.minnumf [[VAR_76_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_79_:%.+]] = arith.maxnumf [[VAR_77_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.maxnumf [[VAR_78_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                     [[VAR_81_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_79_]], [[VAR_80_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                     vector.store [[VAR_81_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                   } else {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = affine.apply [[MAP_5_]]([[VAR_8_]])
// CHECK:                     scf.for [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_8_]] step [[CST_8_]] {
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                       [[VAR_output1_3_1_:%.+]], [[VAR_output2_4_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-DAG:                   [[VAR_20_1_:%.+]] = arith.minnumf [[VAR_output1_3_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_21_1_:%.+]] = arith.minnumf [[VAR_output2_4_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_22_1_:%.+]] = arith.minnumf [[VAR_20_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_23_1_:%.+]] = arith.minnumf [[VAR_21_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_24_1_:%.+]] = arith.maxnumf [[VAR_22_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_25_1_:%.+]] = arith.maxnumf [[VAR_23_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_24_1_]], [[VAR_25_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                       vector.store [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_15_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x137xf16, #map>
// CHECK:         }
}

// -----

// Same as above, but normal output

func.func @test_onnx_mod_ztensor_ssn_big(%arg0: tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>, %arg1: tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x137xf32> {
  %0 = "onnx.Mod"(%arg0, %arg1)  {fmod = 1 : si64} : (tensor<?x3x5x137xf16, #zhigh.layout<{dataLayout = "4D"}>>, tensor<?x3x5x1xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<?x3x5x137xf32>
  return %0 : tensor<?x3x5x137xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -64 + 73)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 64 + 8)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 64 + 16)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 64 + 24)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 64 + 40)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 * 64 + 48)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0) -> (d0 * 64 + 56)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 * -64 + 130)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0) -> (d0 * -64 + 136)>
// CHECK-LABEL:  func.func @test_onnx_mod_ztensor_ssn_big
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x3x5x137xf16, #map>, [[PARAM_1_:%.+]]: memref<?x3x5x1xf16, #map>) -> memref<?x3x5x137xf32> {
// CHECK-DAG:       [[CST_136_:%.+]] = arith.constant 136 : index
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x3x5x137xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x3x5x1xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x3x5x137xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x3x5x137xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:           [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 3){
// CHECK:                   [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[VAR_9_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                   [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_9_]]{{.}} : memref<?x3x5x137xf16, #map>
// CHECK-DAG:               [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:               [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3x5x1xf16, #map>
// CHECK:                   [[VAR_13_:%.+]] = vector.broadcast [[LOAD_PARAM_1_MEM_]] : f16 to vector<8xf16>
// CHECK:                   [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[VAR_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                   [[VAR_14_:%.+]] = affine.apply [[MAP_4_]]([[VAR_8_]])
// CHECK:                   [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:                   scf.if [[VAR_15_]] {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_1_:%.+]], [[VAR_output2_2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_17_:%.+]] = arith.remf [[VAR_output1_1_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_18_:%.+]] = math.copysign [[VAR_17_]], [[VAR_output1_1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_19_:%.+]] = arith.remf [[VAR_output2_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_20_:%.+]] = math.copysign [[VAR_19_]], [[VAR_output2_2_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_:%.+]] = affine.apply [[MAP_2_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_18_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]1] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_20_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]2] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_24_:%.+]] = arith.remf [[VAR_output1_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = math.copysign [[VAR_24_]], [[VAR_output1_3_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_26_:%.+]] = arith.remf [[VAR_output2_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_27_:%.+]] = math.copysign [[VAR_26_]], [[VAR_output2_4_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_28_:%.+]] = affine.apply [[MAP_5_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_25_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]8] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_29_:%.+]] = arith.addi [[VAR_28_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_27_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]9] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_31_:%.+]] = arith.remf [[VAR_output1_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_32_:%.+]] = math.copysign [[VAR_31_]], [[VAR_output1_5_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = arith.remf [[VAR_output2_6_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_34_:%.+]] = math.copysign [[VAR_33_]], [[VAR_output2_6_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_35_:%.+]] = affine.apply [[MAP_6_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_32_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_35_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_36_:%.+]] = arith.addi [[VAR_35_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_34_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_36_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_7_:%.+]], [[VAR_output2_8_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_38_:%.+]] = arith.remf [[VAR_output1_7_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_39_:%.+]] = math.copysign [[VAR_38_]], [[VAR_output1_7_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_40_:%.+]] = arith.remf [[VAR_output2_8_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_41_:%.+]] = math.copysign [[VAR_40_]], [[VAR_output2_8_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_42_:%.+]] = affine.apply [[MAP_7_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_39_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]2] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_43_:%.+]] = arith.addi [[VAR_42_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]3] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_9_:%.+]], [[VAR_output2_10_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_45_:%.+]] = arith.remf [[VAR_output1_9_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_46_:%.+]] = math.copysign [[VAR_45_]], [[VAR_output1_9_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_47_:%.+]] = arith.remf [[VAR_output2_10_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_48_:%.+]] = math.copysign [[VAR_47_]], [[VAR_output2_10_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = affine.apply [[MAP_8_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_46_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_4_]]9] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_50_:%.+]] = arith.addi [[VAR_49_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_48_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_50_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_11_:%.+]], [[VAR_output2_12_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_52_:%.+]] = arith.remf [[VAR_output1_11_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_53_:%.+]] = math.copysign [[VAR_52_]], [[VAR_output1_11_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.remf [[VAR_output2_12_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_55_:%.+]] = math.copysign [[VAR_54_]], [[VAR_output2_12_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_56_:%.+]] = affine.apply [[MAP_9_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_53_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_56_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_57_:%.+]] = arith.addi [[VAR_56_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_55_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_57_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_13_:%.+]], [[VAR_output2_14_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_59_:%.+]] = arith.remf [[VAR_output1_13_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_60_:%.+]] = math.copysign [[VAR_59_]], [[VAR_output1_13_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_61_:%.+]] = arith.remf [[VAR_output2_14_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_62_:%.+]] = math.copysign [[VAR_61_]], [[VAR_output2_14_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_63_:%.+]] = affine.apply [[MAP_10_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_60_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_6_]]3] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_64_:%.+]] = arith.addi [[VAR_63_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_62_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_6_]]4] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_15_:%.+]], [[VAR_output2_16_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_66_:%.+]] = arith.remf [[VAR_output1_15_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = math.copysign [[VAR_66_]], [[VAR_output1_15_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_68_:%.+]] = arith.remf [[VAR_output2_16_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_69_:%.+]] = math.copysign [[VAR_68_]], [[VAR_output2_16_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_70_:%.+]] = affine.apply [[MAP_11_]]([[VAR_8_]])
// CHECK:                     vector.store [[VAR_67_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_70_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     [[VAR_71_:%.+]] = arith.addi [[VAR_70_]], [[CST_4_]] : index
// CHECK:                     vector.store [[VAR_69_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_71_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                   } else {
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = affine.apply [[MAP_12_]]([[VAR_8_]])
// CHECK:                     scf.for [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_VAR_reinterpret_cast_MEM_8_]] step [[CST_8_]] {
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                       [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                       [[VAR_25_1_:%.+]] = arith.remf [[VAR_output1_4_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_26_1_:%.+]] = math.copysign [[VAR_25_1_]], [[VAR_output1_4_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_27_1_:%.+]] = arith.remf [[VAR_output2_5_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_28_1_:%.+]] = math.copysign [[VAR_27_1_]], [[VAR_output2_5_]] : vector<4xf32>
// CHECK-DAG:                   [[VAR_29_1_:%.+]] = affine.apply [[MAP_13_]]([[I_4_]], [[VAR_8_]])
// CHECK:                       vector.store [[VAR_26_1_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[VAR_2_]]9] : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                       [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = arith.addi [[VAR_29_1_]], [[CST_4_]] : index
// CHECK:                       vector.store [[VAR_28_1_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[LOAD_VAR_reinterpret_cast_MEM_2_]]{{.}} : memref<?x3x5x137xf32>, vector<4xf32>
// CHECK:                     }
// CHECK-DAG:                 [[VAR_17_1_:%.+]] = affine.apply [[MAP_14_]]([[VAR_8_]])
// CHECK-DAG:                 [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x8xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_10_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                     [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_10_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:                     [[VAR_19_1_:%.+]] = arith.remf [[VAR_output1_2_]], [[VAR_output1_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_20_1_:%.+]] = math.copysign [[VAR_19_1_]], [[VAR_output1_2_]] : vector<4xf32>
// CHECK-DAG:                 [[VAR_21_1_:%.+]] = arith.remf [[VAR_output2_3_]], [[VAR_output1_]] : vector<4xf32>
// CHECK:                     [[VAR_22_1_:%.+]] = math.copysign [[VAR_21_1_]], [[VAR_output2_3_]] : vector<4xf32>
// CHECK:                     vector.store [[VAR_20_1_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                     vector.store [[VAR_22_1_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32>, vector<4xf32>
// CHECK:                     [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32>
// CHECK:                     krnl.store [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_4_]], [[VAR_6_]], [[CST_136_]]{{.}} : memref<?x3x5x137xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3x5x137xf32>
// CHECK:         }
}

