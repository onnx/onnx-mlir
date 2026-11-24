// RUN: onnx-mlir-opt -O3 --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// GROUND-ALL: -c="-O3 --march=arm64" -r="--disable-gemm-small-n" --rtol=1e-4 --atol=1e-5

// Small-N Gemm: N (output columns) and K static, N <= 16, transA = 0. Each
// output is a dot product along K of a row of A and a row of B taken as [N,K],
// vectorized along K (z16: 4 floats per vector), R rows at a time with their
// R x N accumulators in registers. -O3 and --march are both needed: without
// either, there is no SIMD and the tiled kernel is used. Grounded on arm64,
// which has the same vector length and register count as z16, against the
// tiled kernel (--disable-gemm-small-n).

// Dynamic M, transB = 1, bias [N]. N = 2: blocks of 8 rows, then the M % 8
// remaining rows one at a time; K = 8 is two vectors, no tail; N < 4, so every
// output is reduced on its own.
// GROUND-THIS: --shape-info=0:13x8
func.func @test_gemm_small_n_dyn_m(%arg0: tensor<?x8xf32> {onnx.name = "a"}, %arg1: tensor<2x8xf32> {onnx.name = "b"}, %arg2: tensor<2xf32> {onnx.name = "c"}) -> (tensor<?x2xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<?x8xf32>, tensor<2x8xf32>, tensor<2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 floordiv 8) * 8)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_gemm_small_n_dyn_m
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x8xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<2xf32> {onnx.name = "c"}) -> (memref<?x2xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 128 : memref<?x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_3_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_3_]], [[CST_7_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]]:16 = krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[VAR_arg4_:%.+]] = [[CST_0_]] to [[CST_2_]]) iter_args([[VAR_arg5_:%.+]] = [[VAR_cst_]][[VAR_arg6_:%.+]] = [[VAR_cst_]][[VAR_arg7_:%.+]] = [[VAR_cst_]][[VAR_arg8_:%.+]] = [[VAR_cst_]][[VAR_arg9_:%.+]] = [[VAR_cst_]][[VAR_arg10_:%.+]] = [[VAR_cst_]][[VAR_arg11_:%.+]] = [[VAR_cst_]][[VAR_arg12_:%.+]] = [[VAR_cst_]][[VAR_arg13_:%.+]] = [[VAR_cst_]][[VAR_arg14_:%.+]] = [[VAR_cst_]][[VAR_arg15_:%.+]] = [[VAR_cst_]][[VAR_arg16_:%.+]] = [[VAR_cst_]][[VAR_arg17_:%.+]] = [[VAR_cst_]][[VAR_arg18_:%.+]] = [[VAR_cst_]][[VAR_arg19_:%.+]] = [[VAR_cst_]][[VAR_arg20_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[VAR_61_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_62_:%.+]] = arith.muli [[VAR_61_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_8_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_9_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_10_]], [[VAR_62_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_62_]]{{.}} : memref<2x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_72_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg5_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg7_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_74_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg9_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_75_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_76_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg13_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_77_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg15_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_78_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg17_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_79_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg19_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[VAR_62_]]{{.}} : memref<2x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_81_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg6_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_82_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg10_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_84_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg12_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_85_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg14_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_86_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg16_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_87_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg18_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_88_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg20_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_72_]], [[VAR_81_]], [[VAR_73_]], [[VAR_82_]], [[VAR_74_]], [[VAR_83_]], [[VAR_75_]], [[VAR_84_]], [[VAR_76_]], [[VAR_85_]], [[VAR_77_]], [[VAR_86_]], [[VAR_78_]], [[VAR_87_]], [[VAR_79_]], [[VAR_88_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_13_:%.+]] = vector.reduction <add>, [[VAR_12_]]#0 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[VAR_13_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.reduction <add>, [[VAR_12_]]#1 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_18_:%.+]] = arith.addf [[VAR_16_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_18_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = vector.reduction <add>, [[VAR_12_]]#2 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_21_:%.+]] = arith.addf [[VAR_19_]], [[LOAD_PARAM_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_21_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = vector.reduction <add>, [[VAR_12_]]#3 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_24_:%.+]] = arith.addf [[VAR_22_]], [[LOAD_PARAM_2_MEM_3_]] : f32
// CHECK:             krnl.store [[VAR_24_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = vector.reduction <add>, [[VAR_12_]]#4 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_27_:%.+]] = arith.addf [[VAR_25_]], [[LOAD_PARAM_2_MEM_4_]] : f32
// CHECK:             krnl.store [[VAR_27_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.reduction <add>, [[VAR_12_]]#5 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_5_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_30_:%.+]] = arith.addf [[VAR_28_]], [[LOAD_PARAM_2_MEM_5_]] : f32
// CHECK:             krnl.store [[VAR_30_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = vector.reduction <add>, [[VAR_12_]]#6 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_6_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_33_:%.+]] = arith.addf [[VAR_31_]], [[LOAD_PARAM_2_MEM_6_]] : f32
// CHECK:             krnl.store [[VAR_33_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = vector.reduction <add>, [[VAR_12_]]#7 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_7_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_36_:%.+]] = arith.addf [[VAR_34_]], [[LOAD_PARAM_2_MEM_7_]] : f32
// CHECK:             krnl.store [[VAR_36_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = vector.reduction <add>, [[VAR_12_]]#8 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_8_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_39_:%.+]] = arith.addf [[VAR_37_]], [[LOAD_PARAM_2_MEM_8_]] : f32
// CHECK:             krnl.store [[VAR_39_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = vector.reduction <add>, [[VAR_12_]]#9 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_9_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_42_:%.+]] = arith.addf [[VAR_40_]], [[LOAD_PARAM_2_MEM_9_]] : f32
// CHECK:             krnl.store [[VAR_42_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = vector.reduction <add>, [[VAR_12_]]#10 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_10_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_45_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_PARAM_2_MEM_10_]] : f32
// CHECK:             krnl.store [[VAR_45_]], [[RES_]]{{.}}[[VAR_8_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = vector.reduction <add>, [[VAR_12_]]#11 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_11_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_48_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_PARAM_2_MEM_11_]] : f32
// CHECK:             krnl.store [[VAR_48_]], [[RES_]]{{.}}[[VAR_8_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = vector.reduction <add>, [[VAR_12_]]#12 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_12_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_PARAM_2_MEM_12_]] : f32
// CHECK:             krnl.store [[VAR_51_]], [[RES_]]{{.}}[[VAR_9_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = vector.reduction <add>, [[VAR_12_]]#13 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_13_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.addf [[VAR_52_]], [[LOAD_PARAM_2_MEM_13_]] : f32
// CHECK:             krnl.store [[VAR_54_]], [[RES_]]{{.}}[[VAR_9_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = vector.reduction <add>, [[VAR_12_]]#14 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_14_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_57_:%.+]] = arith.addf [[VAR_55_]], [[LOAD_PARAM_2_MEM_14_]] : f32
// CHECK:             krnl.store [[VAR_57_]], [[RES_]]{{.}}[[VAR_10_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_58_:%.+]] = vector.reduction <add>, [[VAR_12_]]#15 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_15_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_60_:%.+]] = arith.addf [[VAR_58_]], [[LOAD_PARAM_2_MEM_15_]] : f32
// CHECK:             krnl.store [[VAR_60_]], [[RES_]]{{.}}[[VAR_10_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_1_:%.+]] = [[MAP_1_]]([[VAR_dim_]]) to [[MAP_2_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_1_:%.+]]:2 = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg4_1_:%.+]] = [[CST_0_]] to [[CST_2_]]) iter_args([[VAR_arg5_1_:%.+]] = [[VAR_cst_]][[VAR_arg6_1_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_12_1_:%.+]] = arith.muli [[LOOP_1_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_8_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_1_]], [[VAR_12_1_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_12_1_]]{{.}} : memref<2x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_15_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_8_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg5_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[VAR_12_1_]]{{.}} : memref<2x8xf32>, vector<4xf32>
// CHECK:               [[LOAD_PARAM_2_MEM_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_8_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg6_1_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_15_1_]], [[LOAD_PARAM_2_MEM_1_]] : vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_5_1_:%.+]] = vector.reduction <add>, [[VAR_4_1_]]#0 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_16_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_7_1_:%.+]] = arith.addf [[VAR_5_1_]], [[LOAD_PARAM_2_MEM_16_]] : f32
// CHECK:             krnl.store [[VAR_7_1_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_0_]]{{.}} : memref<?x2xf32>
// CHECK-DAG:         [[VAR_8_1_:%.+]] = vector.reduction <add>, [[VAR_4_1_]]#1 : vector<4xf32> into f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_17_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_10_1_:%.+]] = arith.addf [[VAR_8_1_]], [[LOAD_PARAM_2_MEM_17_]] : f32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_1_]]{{.}} : memref<?x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2xf32>
// CHECK:         }

}

// -----

// Static M, transB = 0 (B is copied transposed into an [N,K] buffer), alpha and
// beta not 1, bias [1,N] (loaded as vectors). N = 4: one group of 4 outputs,
// reduced together and stored as a vector; K = 6: one vector, then a tail of 2.
// R = 5, so M = 10 is two blocks and no remaining rows.
func.func @test_gemm_small_n_trans_b0_tail(%arg0: tensor<10x6xf32> {onnx.name = "a"}, %arg1: tensor<6x4xf32> {onnx.name = "b"}, %arg2: tensor<1x4xf32> {onnx.name = "c"}) -> (tensor<10x4xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 0.5 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x6xf32>, tensor<6x4xf32>, tensor<1x4xf32>) -> tensor<10x4xf32>
  return %0 : tensor<10x4xf32>

// CHECK-LABEL:  func.func @test_gemm_small_n_trans_b0_tail
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x6xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<6x4xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<1x4xf32> {onnx.name = "c"}) -> (memref<10x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<5.000000e-01> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 128 : memref<10x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 128 : memref<4x6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0] : memref<6x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x6xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = arith.muli [[VAR_2_1_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_4_]] : index
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]]:20 = krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_1_]]) iter_args([[VAR_arg5_:%.+]] = [[VAR_cst_1_]][[VAR_arg6_:%.+]] = [[VAR_cst_1_]][[VAR_arg7_:%.+]] = [[VAR_cst_1_]][[VAR_arg8_:%.+]] = [[VAR_cst_1_]][[VAR_arg9_:%.+]] = [[VAR_cst_1_]][[VAR_arg10_:%.+]] = [[VAR_cst_1_]][[VAR_arg11_:%.+]] = [[VAR_cst_1_]][[VAR_arg12_:%.+]] = [[VAR_cst_1_]][[VAR_arg13_:%.+]] = [[VAR_cst_1_]][[VAR_arg14_:%.+]] = [[VAR_cst_1_]][[VAR_arg15_:%.+]] = [[VAR_cst_1_]][[VAR_arg16_:%.+]] = [[VAR_cst_1_]][[VAR_arg17_:%.+]] = [[VAR_cst_1_]][[VAR_arg18_:%.+]] = [[VAR_cst_1_]][[VAR_arg19_:%.+]] = [[VAR_cst_1_]][[VAR_arg20_:%.+]] = [[VAR_cst_1_]][[VAR_arg21_:%.+]] = [[VAR_cst_1_]][[VAR_arg22_:%.+]] = [[VAR_cst_1_]][[VAR_arg23_:%.+]] = [[VAR_cst_1_]][[VAR_arg24_:%.+]] = [[VAR_cst_1_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[VAR_155_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_156_:%.+]] = arith.muli [[VAR_155_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[VAR_156_]]{{.}} : memref<10x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_156_]]{{.}} : memref<10x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_156_]]{{.}} : memref<10x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_156_]]{{.}} : memref<10x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_156_]]{{.}} : memref<10x6xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[VAR_156_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_163_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_RES_1_MEM_]], [[VAR_arg5_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_164_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_1_MEM_]], [[VAR_arg9_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_165_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_RES_1_MEM_]], [[VAR_arg13_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_166_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_RES_1_MEM_]], [[VAR_arg17_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_167_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_RES_1_MEM_]], [[VAR_arg21_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[VAR_156_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_169_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_RES_1_MEM_1_]], [[VAR_arg6_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_170_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_1_MEM_1_]], [[VAR_arg10_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_171_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_RES_1_MEM_1_]], [[VAR_arg14_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_172_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_RES_1_MEM_1_]], [[VAR_arg18_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_173_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_RES_1_MEM_1_]], [[VAR_arg22_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[VAR_156_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_175_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_RES_1_MEM_2_]], [[VAR_arg7_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_176_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_1_MEM_2_]], [[VAR_arg11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_177_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_RES_1_MEM_2_]], [[VAR_arg15_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_178_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_RES_1_MEM_2_]], [[VAR_arg19_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_179_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_RES_1_MEM_2_]], [[VAR_arg23_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_3_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[VAR_156_]]{{.}} : memref<4x6xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_181_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_RES_1_MEM_3_]], [[VAR_arg8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_182_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_1_MEM_3_]], [[VAR_arg12_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_183_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_RES_1_MEM_3_]], [[VAR_arg16_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_184_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_RES_1_MEM_3_]], [[VAR_arg20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_185_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_RES_1_MEM_3_]], [[VAR_arg24_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_163_]], [[VAR_169_]], [[VAR_175_]], [[VAR_181_]], [[VAR_164_]], [[VAR_170_]], [[VAR_176_]], [[VAR_182_]], [[VAR_165_]], [[VAR_171_]], [[VAR_177_]], [[VAR_183_]], [[VAR_166_]], [[VAR_172_]], [[VAR_178_]], [[VAR_184_]], [[VAR_167_]], [[VAR_173_]], [[VAR_179_]], [[VAR_185_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_10_:%.+]] = vector.shuffle [[VAR_9_]]#0, [[VAR_9_]]#1 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = vector.shuffle [[VAR_9_]]#0, [[VAR_9_]]#1 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addf [[VAR_10_]], [[VAR_11_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = vector.shuffle [[VAR_9_]]#2, [[VAR_9_]]#3 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = vector.shuffle [[VAR_9_]]#2, [[VAR_9_]]#3 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[VAR_13_]], [[VAR_14_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.shuffle [[VAR_12_]], [[VAR_15_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = vector.shuffle [[VAR_12_]], [[VAR_15_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addf [[VAR_16_]], [[VAR_17_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_5_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[CST_4_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_5_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_7_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_25_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_5_]], [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = vector.fma [[VAR_20_]], [[VAR_25_]], [[VAR_18_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_6_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[CST_5_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_6_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_8_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_9_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_10_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_11_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_33_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_8_]], [[LOAD_RES_1_MEM_9_]], [[LOAD_RES_1_MEM_10_]], [[LOAD_RES_1_MEM_11_]] : vector<4xf32>
// CHECK:             [[VAR_34_:%.+]] = vector.fma [[VAR_28_]], [[VAR_33_]], [[VAR_26_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.mulf [[VAR_34_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x4xf32>, vector<4xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_38_:%.+]] = arith.addf [[VAR_35_]], [[VAR_37_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_38_]], [[RES_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[CST_0_]]{{.}} : memref<10x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = vector.shuffle [[VAR_9_]]#4, [[VAR_9_]]#5 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = vector.shuffle [[VAR_9_]]#4, [[VAR_9_]]#5 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.addf [[VAR_39_]], [[VAR_40_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = vector.shuffle [[VAR_9_]]#6, [[VAR_9_]]#7 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = vector.shuffle [[VAR_9_]]#6, [[VAR_9_]]#7 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_44_:%.+]] = arith.addf [[VAR_42_]], [[VAR_43_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = vector.shuffle [[VAR_41_]], [[VAR_44_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = vector.shuffle [[VAR_41_]], [[VAR_44_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.addf [[VAR_45_]], [[VAR_46_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_7_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[CST_4_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_49_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_7_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_12_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_13_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_14_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_15_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_54_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_12_]], [[LOAD_RES_1_MEM_13_]], [[LOAD_RES_1_MEM_14_]], [[LOAD_RES_1_MEM_15_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = vector.fma [[VAR_49_]], [[VAR_54_]], [[VAR_47_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_8_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[CST_5_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_8_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_16_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_17_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_18_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_19_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_62_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_16_]], [[LOAD_RES_1_MEM_17_]], [[LOAD_RES_1_MEM_18_]], [[LOAD_RES_1_MEM_19_]] : vector<4xf32>
// CHECK:             [[VAR_63_:%.+]] = vector.fma [[VAR_57_]], [[VAR_62_]], [[VAR_55_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_64_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x4xf32>, vector<4xf32>
// CHECK:             [[VAR_66_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_67_:%.+]] = arith.addf [[VAR_64_]], [[VAR_66_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_67_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<10x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_68_:%.+]] = vector.shuffle [[VAR_9_]]#8, [[VAR_9_]]#9 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_69_:%.+]] = vector.shuffle [[VAR_9_]]#8, [[VAR_9_]]#9 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_70_:%.+]] = arith.addf [[VAR_68_]], [[VAR_69_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_71_:%.+]] = vector.shuffle [[VAR_9_]]#10, [[VAR_9_]]#11 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_72_:%.+]] = vector.shuffle [[VAR_9_]]#10, [[VAR_9_]]#11 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_73_:%.+]] = arith.addf [[VAR_71_]], [[VAR_72_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_74_:%.+]] = vector.shuffle [[VAR_70_]], [[VAR_73_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = vector.shuffle [[VAR_70_]], [[VAR_73_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_76_:%.+]] = arith.addf [[VAR_74_]], [[VAR_75_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_9_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[CST_4_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_78_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_9_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_20_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_21_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_22_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_23_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_83_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_20_]], [[LOAD_RES_1_MEM_21_]], [[LOAD_RES_1_MEM_22_]], [[LOAD_RES_1_MEM_23_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_84_:%.+]] = vector.fma [[VAR_78_]], [[VAR_83_]], [[VAR_76_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_10_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[CST_5_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_86_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_10_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_24_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_25_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_26_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_27_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_91_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_24_]], [[LOAD_RES_1_MEM_25_]], [[LOAD_RES_1_MEM_26_]], [[LOAD_RES_1_MEM_27_]] : vector<4xf32>
// CHECK:             [[VAR_92_:%.+]] = vector.fma [[VAR_86_]], [[VAR_91_]], [[VAR_84_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_93_:%.+]] = arith.mulf [[VAR_92_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x4xf32>, vector<4xf32>
// CHECK:             [[VAR_95_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_2_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_96_:%.+]] = arith.addf [[VAR_93_]], [[VAR_95_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_96_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<10x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_97_:%.+]] = vector.shuffle [[VAR_9_]]#12, [[VAR_9_]]#13 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_98_:%.+]] = vector.shuffle [[VAR_9_]]#12, [[VAR_9_]]#13 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_99_:%.+]] = arith.addf [[VAR_97_]], [[VAR_98_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_100_:%.+]] = vector.shuffle [[VAR_9_]]#14, [[VAR_9_]]#15 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_101_:%.+]] = vector.shuffle [[VAR_9_]]#14, [[VAR_9_]]#15 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_102_:%.+]] = arith.addf [[VAR_100_]], [[VAR_101_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_103_:%.+]] = vector.shuffle [[VAR_99_]], [[VAR_102_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_104_:%.+]] = vector.shuffle [[VAR_99_]], [[VAR_102_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_105_:%.+]] = arith.addf [[VAR_103_]], [[VAR_104_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_11_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[CST_4_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_107_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_11_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_28_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_29_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_30_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_31_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_112_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_28_]], [[LOAD_RES_1_MEM_29_]], [[LOAD_RES_1_MEM_30_]], [[LOAD_RES_1_MEM_31_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_113_:%.+]] = vector.fma [[VAR_107_]], [[VAR_112_]], [[VAR_105_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_12_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[CST_5_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_115_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_12_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_32_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_33_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_34_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_35_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_120_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_32_]], [[LOAD_RES_1_MEM_33_]], [[LOAD_RES_1_MEM_34_]], [[LOAD_RES_1_MEM_35_]] : vector<4xf32>
// CHECK:             [[VAR_121_:%.+]] = vector.fma [[VAR_115_]], [[VAR_120_]], [[VAR_113_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_122_:%.+]] = arith.mulf [[VAR_121_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x4xf32>, vector<4xf32>
// CHECK:             [[VAR_124_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_3_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_125_:%.+]] = arith.addf [[VAR_122_]], [[VAR_124_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_125_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_0_]]{{.}} : memref<10x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_126_:%.+]] = vector.shuffle [[VAR_9_]]#16, [[VAR_9_]]#17 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_127_:%.+]] = vector.shuffle [[VAR_9_]]#16, [[VAR_9_]]#17 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_128_:%.+]] = arith.addf [[VAR_126_]], [[VAR_127_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_129_:%.+]] = vector.shuffle [[VAR_9_]]#18, [[VAR_9_]]#19 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_130_:%.+]] = vector.shuffle [[VAR_9_]]#18, [[VAR_9_]]#19 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_131_:%.+]] = arith.addf [[VAR_129_]], [[VAR_130_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_132_:%.+]] = vector.shuffle [[VAR_128_]], [[VAR_131_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_133_:%.+]] = vector.shuffle [[VAR_128_]], [[VAR_131_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_134_:%.+]] = arith.addf [[VAR_132_]], [[VAR_133_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_13_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[CST_4_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_136_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_13_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_36_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_37_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_38_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_39_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_4_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_141_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_36_]], [[LOAD_RES_1_MEM_37_]], [[LOAD_RES_1_MEM_38_]], [[LOAD_RES_1_MEM_39_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_142_:%.+]] = vector.fma [[VAR_136_]], [[VAR_141_]], [[VAR_134_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_14_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[CST_5_]]{{.}} : memref<10x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_144_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_14_]] : f32 to vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_40_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_41_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_42_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_43_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_5_]]{{.}} : memref<4x6xf32>
// CHECK:             [[VAR_149_:%.+]] = vector.from_elements [[LOAD_RES_1_MEM_40_]], [[LOAD_RES_1_MEM_41_]], [[LOAD_RES_1_MEM_42_]], [[LOAD_RES_1_MEM_43_]] : vector<4xf32>
// CHECK:             [[VAR_150_:%.+]] = vector.fma [[VAR_144_]], [[VAR_149_]], [[VAR_142_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_151_:%.+]] = arith.mulf [[VAR_150_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x4xf32>, vector<4xf32>
// CHECK:             [[VAR_153_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_4_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_154_:%.+]] = arith.addf [[VAR_151_]], [[VAR_153_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_154_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_0_]]{{.}} : memref<10x4xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x4xf32>
// CHECK:         }

}

// -----

// A bias that differs per row ([M,1], dynamic) is loaded element by element.
// N = 3 < 4: outputs reduced one at a time.
// GROUND-THIS: --shape-info=0:5x8,2:5x1
func.func @test_gemm_small_n_bias_per_row(%arg0: tensor<?x8xf32> {onnx.name = "a"}, %arg1: tensor<3x8xf32> {onnx.name = "b"}, %arg2: tensor<?x1xf32> {onnx.name = "c"}) -> (tensor<?x3xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<?x8xf32>, tensor<3x8xf32>, tensor<?x1xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 floordiv 7) * 7)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_gemm_small_n_bias_per_row
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x8xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<3x8xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<?x1xf32> {onnx.name = "c"}) -> (memref<?x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 128 : memref<?x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_7_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_3_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_3_]], [[CST_6_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]]:21 = krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[VAR_arg4_:%.+]] = [[CST_0_]] to [[CST_2_]]) iter_args([[VAR_arg5_:%.+]] = [[VAR_cst_]][[VAR_arg6_:%.+]] = [[VAR_cst_]][[VAR_arg7_:%.+]] = [[VAR_cst_]][[VAR_arg8_:%.+]] = [[VAR_cst_]][[VAR_arg9_:%.+]] = [[VAR_cst_]][[VAR_arg10_:%.+]] = [[VAR_cst_]][[VAR_arg11_:%.+]] = [[VAR_cst_]][[VAR_arg12_:%.+]] = [[VAR_cst_]][[VAR_arg13_:%.+]] = [[VAR_cst_]][[VAR_arg14_:%.+]] = [[VAR_cst_]][[VAR_arg15_:%.+]] = [[VAR_cst_]][[VAR_arg16_:%.+]] = [[VAR_cst_]][[VAR_arg17_:%.+]] = [[VAR_cst_]][[VAR_arg18_:%.+]] = [[VAR_cst_]][[VAR_arg19_:%.+]] = [[VAR_cst_]][[VAR_arg20_:%.+]] = [[VAR_cst_]][[VAR_arg21_:%.+]] = [[VAR_cst_]][[VAR_arg22_:%.+]] = [[VAR_cst_]][[VAR_arg23_:%.+]] = [[VAR_cst_]][[VAR_arg24_:%.+]] = [[VAR_cst_]][[VAR_arg25_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[VAR_117_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_118_:%.+]] = arith.muli [[VAR_117_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_8_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_9_]], [[VAR_118_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_118_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_127_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg5_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_128_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_129_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_130_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg14_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_131_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg17_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_132_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_133_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg23_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[VAR_118_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_135_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg6_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_136_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg9_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_137_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg12_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_138_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg15_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_139_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg18_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_140_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg21_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_141_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg24_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_2_]], [[VAR_118_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_143_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg7_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_144_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg10_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_145_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg13_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_146_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg16_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_147_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg19_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_148_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg22_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_149_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg25_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_127_]], [[VAR_135_]], [[VAR_143_]], [[VAR_128_]], [[VAR_136_]], [[VAR_144_]], [[VAR_129_]], [[VAR_137_]], [[VAR_145_]], [[VAR_130_]], [[VAR_138_]], [[VAR_146_]], [[VAR_131_]], [[VAR_139_]], [[VAR_147_]], [[VAR_132_]], [[VAR_140_]], [[VAR_148_]], [[VAR_133_]], [[VAR_141_]], [[VAR_149_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_12_:%.+]] = vector.reduction <add>, [[VAR_11_]]#0 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_3_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_14_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.addf [[VAR_12_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = vector.reduction <add>, [[VAR_11_]]#1 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[VAR_3_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_19_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_21_:%.+]] = arith.addf [[VAR_17_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_21_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = vector.reduction <add>, [[VAR_11_]]#2 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[VAR_3_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_2_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_24_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_26_:%.+]] = arith.addf [[VAR_22_]], [[LOAD_PARAM_2_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_26_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.reduction <add>, [[VAR_11_]]#3 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_3_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_29_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_31_:%.+]] = arith.addf [[VAR_27_]], [[LOAD_PARAM_2_MEM_3_]] : f32
// CHECK:             krnl.store [[VAR_31_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = vector.reduction <add>, [[VAR_11_]]#4 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_34_:%.+]] = arith.select [[VAR_33_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_4_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_34_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_36_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_PARAM_2_MEM_4_]] : f32
// CHECK:             krnl.store [[VAR_36_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = vector.reduction <add>, [[VAR_11_]]#5 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_39_:%.+]] = arith.select [[VAR_38_]], [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_5_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_39_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_41_:%.+]] = arith.addf [[VAR_37_]], [[LOAD_PARAM_2_MEM_5_]] : f32
// CHECK:             krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = vector.reduction <add>, [[VAR_11_]]#6 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_44_:%.+]] = arith.select [[VAR_43_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_6_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_44_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_46_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_PARAM_2_MEM_6_]] : f32
// CHECK:             krnl.store [[VAR_46_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = vector.reduction <add>, [[VAR_11_]]#7 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_49_:%.+]] = arith.select [[VAR_48_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_7_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_49_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.addf [[VAR_47_]], [[LOAD_PARAM_2_MEM_7_]] : f32
// CHECK:             krnl.store [[VAR_51_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = vector.reduction <add>, [[VAR_11_]]#8 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_8_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_54_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_56_:%.+]] = arith.addf [[VAR_52_]], [[LOAD_PARAM_2_MEM_8_]] : f32
// CHECK:             krnl.store [[VAR_56_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_57_:%.+]] = vector.reduction <add>, [[VAR_11_]]#9 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_59_:%.+]] = arith.select [[VAR_58_]], [[VAR_6_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_9_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_59_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_61_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_PARAM_2_MEM_9_]] : f32
// CHECK:             krnl.store [[VAR_61_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_62_:%.+]] = vector.reduction <add>, [[VAR_11_]]#10 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_64_:%.+]] = arith.select [[VAR_63_]], [[VAR_6_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_10_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_64_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_66_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_PARAM_2_MEM_10_]] : f32
// CHECK:             krnl.store [[VAR_66_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_67_:%.+]] = vector.reduction <add>, [[VAR_11_]]#11 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_68_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[VAR_6_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_11_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_69_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_71_:%.+]] = arith.addf [[VAR_67_]], [[LOAD_PARAM_2_MEM_11_]] : f32
// CHECK:             krnl.store [[VAR_71_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_72_:%.+]] = vector.reduction <add>, [[VAR_11_]]#12 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_73_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_74_:%.+]] = arith.select [[VAR_73_]], [[VAR_7_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_12_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_74_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_76_:%.+]] = arith.addf [[VAR_72_]], [[LOAD_PARAM_2_MEM_12_]] : f32
// CHECK:             krnl.store [[VAR_76_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_77_:%.+]] = vector.reduction <add>, [[VAR_11_]]#13 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_78_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_79_:%.+]] = arith.select [[VAR_78_]], [[VAR_7_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_13_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_79_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_81_:%.+]] = arith.addf [[VAR_77_]], [[LOAD_PARAM_2_MEM_13_]] : f32
// CHECK:             krnl.store [[VAR_81_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_82_:%.+]] = vector.reduction <add>, [[VAR_11_]]#14 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_83_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_84_:%.+]] = arith.select [[VAR_83_]], [[VAR_7_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_14_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_84_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_86_:%.+]] = arith.addf [[VAR_82_]], [[LOAD_PARAM_2_MEM_14_]] : f32
// CHECK:             krnl.store [[VAR_86_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_87_:%.+]] = vector.reduction <add>, [[VAR_11_]]#15 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_88_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_89_:%.+]] = arith.select [[VAR_88_]], [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_15_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_89_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_91_:%.+]] = arith.addf [[VAR_87_]], [[LOAD_PARAM_2_MEM_15_]] : f32
// CHECK:             krnl.store [[VAR_91_]], [[RES_]]{{.}}[[VAR_8_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_92_:%.+]] = vector.reduction <add>, [[VAR_11_]]#16 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_93_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_94_:%.+]] = arith.select [[VAR_93_]], [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_16_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_94_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_96_:%.+]] = arith.addf [[VAR_92_]], [[LOAD_PARAM_2_MEM_16_]] : f32
// CHECK:             krnl.store [[VAR_96_]], [[RES_]]{{.}}[[VAR_8_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_97_:%.+]] = vector.reduction <add>, [[VAR_11_]]#17 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_98_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_99_:%.+]] = arith.select [[VAR_98_]], [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_17_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_99_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_101_:%.+]] = arith.addf [[VAR_97_]], [[LOAD_PARAM_2_MEM_17_]] : f32
// CHECK:             krnl.store [[VAR_101_]], [[RES_]]{{.}}[[VAR_8_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_102_:%.+]] = vector.reduction <add>, [[VAR_11_]]#18 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_103_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_104_:%.+]] = arith.select [[VAR_103_]], [[VAR_9_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_18_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_104_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_106_:%.+]] = arith.addf [[VAR_102_]], [[LOAD_PARAM_2_MEM_18_]] : f32
// CHECK:             krnl.store [[VAR_106_]], [[RES_]]{{.}}[[VAR_9_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_107_:%.+]] = vector.reduction <add>, [[VAR_11_]]#19 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_108_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_109_:%.+]] = arith.select [[VAR_108_]], [[VAR_9_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_19_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_109_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_111_:%.+]] = arith.addf [[VAR_107_]], [[LOAD_PARAM_2_MEM_19_]] : f32
// CHECK:             krnl.store [[VAR_111_]], [[RES_]]{{.}}[[VAR_9_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[VAR_112_:%.+]] = vector.reduction <add>, [[VAR_11_]]#20 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_113_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_114_:%.+]] = arith.select [[VAR_113_]], [[VAR_9_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_20_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_114_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_116_:%.+]] = arith.addf [[VAR_112_]], [[LOAD_PARAM_2_MEM_20_]] : f32
// CHECK:             krnl.store [[VAR_116_]], [[RES_]]{{.}}[[VAR_9_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_1_:%.+]] = [[MAP_1_]]([[VAR_dim_]]) to [[MAP_2_]]([[VAR_dim_]])){
// CHECK-DAG:         [[VAR_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_1_:%.+]]:3 = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg4_1_:%.+]] = [[CST_0_]] to [[CST_2_]]) iter_args([[VAR_arg5_1_:%.+]] = [[VAR_cst_]][[VAR_arg6_1_:%.+]] = [[VAR_cst_]][[VAR_arg7_1_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_21_1_:%.+]] = arith.muli [[LOAD_PARAM_2_MEM_1_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_1_]], [[VAR_21_1_]]{{.}} : memref<?x8xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_21_1_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg5_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[VAR_21_1_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_4_]], [[VAR_arg6_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_2_]], [[VAR_21_1_]]{{.}} : memref<3x8xf32>, vector<4xf32>
// CHECK:               [[VAR_28_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_5_]], [[VAR_arg7_1_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_24_1_]], [[VAR_26_1_]], [[VAR_28_1_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_5_1_:%.+]] = vector.reduction <add>, [[VAR_4_1_]]#0 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_1_:%.+]] = arith.select [[VAR_6_1_]], [[VAR_2_1_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_21_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_7_1_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_9_1_:%.+]] = arith.addf [[VAR_5_1_]], [[LOAD_PARAM_2_MEM_21_]] : f32
// CHECK:             krnl.store [[VAR_9_1_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_0_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[LOOP_1_:%.+]] = vector.reduction <add>, [[VAR_4_1_]]#1 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_11_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_12_1_:%.+]] = arith.select [[VAR_11_1_]], [[VAR_2_1_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_22_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_12_1_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_14_1_:%.+]] = arith.addf [[LOOP_1_]], [[LOAD_PARAM_2_MEM_22_]] : f32
// CHECK:             krnl.store [[VAR_14_1_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_1_]]{{.}} : memref<?x3xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_23_:%.+]] = vector.reduction <add>, [[VAR_4_1_]]#2 : vector<4xf32> into f32
// CHECK-DAG:         [[VAR_16_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_17_1_:%.+]] = arith.select [[VAR_16_1_]], [[VAR_2_1_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_24_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_17_1_]], [[CST_0_]]{{.}} : memref<?x1xf32>
// CHECK:             [[VAR_19_1_:%.+]] = arith.addf [[LOAD_PARAM_2_MEM_23_]], [[LOAD_PARAM_2_MEM_24_]] : f32
// CHECK:             krnl.store [[VAR_19_1_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_2_]]{{.}} : memref<?x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x3xf32>
// CHECK:         }

}

// -----

// Not small-N: transA = 1 (rows of A are not contiguous along K). Tiled kernel.
func.func @test_gemm_small_n_reject_trans_a(%arg0: tensor<8x5xf32> {onnx.name = "a"}, %arg1: tensor<3x8xf32> {onnx.name = "b"}, %arg2: tensor<3xf32> {onnx.name = "c"}) -> (tensor<5x3xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<8x5xf32>, tensor<3x8xf32>, tensor<3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>

// CHECK-LABEL:  func.func @test_gemm_small_n_reject_trans_a
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x5xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<3x8xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<3xf32> {onnx.name = "c"}) -> (memref<5x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 128 : memref<5x3xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<5x3xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[BLOCK_IN__0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[BLOCK_IN__2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_0_]]#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__0_]], [[BLOCK_TILE__1_]], [[BLOCK_IN__1_]]) [0, 3, 5, 1, 6, 2, 4, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) with ([[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#0 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() alignment = 128 : memref<32x256xf32>
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() alignment = 128 : memref<256x64xf32>
// CHECK:             krnl.copy_to_tile_buffer [[RES_2_]], [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1], [[CST_0_dot_000000_]] {padToNext = [], tileSize = [], transpose = true} : memref<256x64xf32>, memref<3x8xf32>
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               krnl.copy_to_tile_buffer [[RES_1_]], [[PARAM_0_]]{{.}}[[VAR_2_]]#1, [[VAR_3_]]{{.}}, [[CST_0_dot_000000_]] {padToNext = [], tileSize = [], transpose = true} : memref<32x256xf32>, memref<8x5xf32>
// CHECK:               krnl.iterate([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) with (){
// CHECK:                 [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:                 krnl.matmul [[RES_1_]]{{.}}[[VAR_3_]], [[VAR_2_]]#1], [[RES_2_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0], [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__1_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_4_]]#1, [[VAR_4_]]#0, [[VAR_2_]]#1), ([[CST_5_]], [[CST_3_]], [[CST_8_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 16, 256], simdize = false} : memref<32x256xf32>, memref<256x64xf32>, memref<5x3xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<5x3xf32>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_1_]]#1] : memref<3xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_1_]], [[VAR_4_1_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<5x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5x3xf32>
// CHECK:         }

}

// -----

// Not small-N: N = 17 > 16. Tiled kernel.
func.func @test_gemm_small_n_reject_wide_n(%arg0: tensor<4x8xf32> {onnx.name = "a"}, %arg1: tensor<17x8xf32> {onnx.name = "b"}, %arg2: tensor<17xf32> {onnx.name = "c"}) -> (tensor<4x17xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<4x8xf32>, tensor<17x8xf32>, tensor<17xf32>) -> tensor<4x17xf32>
  return %0 : tensor<4x17xf32>
// CHECK-LABEL:  func.func @test_gemm_small_n_reject_wide_n
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<17x8xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<17xf32> {onnx.name = "c"}) -> (memref<4x17xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_17_:%.+]] = arith.constant 17 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 128 : memref<4x17xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<4x17xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[BLOCK_IN__0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[BLOCK_IN__2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_0_]]#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__0_]], [[BLOCK_TILE__1_]], [[BLOCK_IN__1_]], [[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]]) [0, 4, 5, 1, 3, 6, 2, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]], [[BLOCK_TILE__2_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 17, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__0_]], [[BLOCK_TILE__2_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() alignment = 128 : memref<32x256xf32>
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() alignment = 128 : memref<256x64xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() alignment = 128 : memref<32x256xf32>
// CHECK:             krnl.copy_to_tile_buffer [[RES_3_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1], [[CST_0_dot_000000_]] {padToNext = [], tileSize = []} : memref<32x256xf32>, memref<4x17xf32>
// CHECK:             krnl.iterate([[BLOCK_TILE__4_]]) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__4_]]) : (!krnl.loop) -> index
// CHECK:               krnl.copy_to_tile_buffer [[RES_1_]], [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]]{{.}}, [[CST_0_dot_000000_]] {padToNext = [], tileSize = []} : memref<32x256xf32>, memref<4x8xf32>
// CHECK:               krnl.copy_to_tile_buffer [[RES_2_]], [[PARAM_1_]]{{.}}[[VAR_2_]]#1, [[VAR_3_]]{{.}}, [[CST_0_dot_000000_]] {padToNext = [], tileSize = [], transpose = true} : memref<256x64xf32>, memref<17x8xf32>
// CHECK:               krnl.iterate([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) with (){
// CHECK:                 [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:                 krnl.matmul [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]]{{.}}, [[RES_2_]]{{.}}[[VAR_3_]], [[VAR_2_]]#1], [[RES_3_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1], ([[BLOCK_IN__1_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_4_]]#1, [[VAR_4_]]#0, [[VAR_3_]]), ([[CST_4_]], [[CST_17_]], [[CST_8_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 16, 256]} : memref<32x256xf32>, memref<256x64xf32>, memref<32x256xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:               }
// CHECK:             }
// CHECK:             krnl.copy_from_tile_buffer [[RES_3_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] {tileSize = []} : memref<32x256xf32>, memref<4x17xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 17){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<4x17xf32>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_1_]]#1] : memref<17xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_1_]], [[VAR_4_1_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<4x17xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x17xf32>
// CHECK:         }

}


// -----

// A rank-0 bias, broadcast to every output. N = 4, K = 4: one group of 4
// outputs and no K tail.
func.func @test_gemm_small_n_scalar_bias(%arg0: tensor<6x4xf32> {onnx.name = "a"}, %arg1: tensor<4x4xf32> {onnx.name = "b"}, %arg2: tensor<f32> {onnx.name = "c"}) -> (tensor<6x4xf32> {onnx.name = "y"}) {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<6x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
// CHECK-LABEL:  func.func @test_gemm_small_n_scalar_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6x4xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<4x4xf32> {onnx.name = "b"}, [[PARAM_2_:%.+]]: memref<f32> {onnx.name = "c"}) -> (memref<6x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 128 : memref<6x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]]:20 = krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[VAR_arg4_:%.+]] = [[CST_0_]] to [[CST_1_]]) iter_args([[VAR_arg5_:%.+]] = [[VAR_cst_]][[VAR_arg6_:%.+]] = [[VAR_cst_]][[VAR_arg7_:%.+]] = [[VAR_cst_]][[VAR_arg8_:%.+]] = [[VAR_cst_]][[VAR_arg9_:%.+]] = [[VAR_cst_]][[VAR_arg10_:%.+]] = [[VAR_cst_]][[VAR_arg11_:%.+]] = [[VAR_cst_]][[VAR_arg12_:%.+]] = [[VAR_cst_]][[VAR_arg13_:%.+]] = [[VAR_cst_]][[VAR_arg14_:%.+]] = [[VAR_cst_]][[VAR_arg15_:%.+]] = [[VAR_cst_]][[VAR_arg16_:%.+]] = [[VAR_cst_]][[VAR_arg17_:%.+]] = [[VAR_cst_]][[VAR_arg18_:%.+]] = [[VAR_cst_]][[VAR_arg19_:%.+]] = [[VAR_cst_]][[VAR_arg20_:%.+]] = [[VAR_cst_]][[VAR_arg21_:%.+]] = [[VAR_cst_]][[VAR_arg22_:%.+]] = [[VAR_cst_]][[VAR_arg23_:%.+]] = [[VAR_cst_]][[VAR_arg24_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[VAR_85_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_86_:%.+]] = arith.muli [[VAR_85_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_86_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_86_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_86_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_86_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_86_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_86_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_93_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg5_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_94_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg9_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_95_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg13_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_96_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg17_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_97_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_]], [[VAR_arg21_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[VAR_86_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_99_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg6_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_100_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg10_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_101_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg14_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_102_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg18_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_103_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_1_]], [[VAR_arg22_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_2_]], [[VAR_86_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_105_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg7_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_106_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_107_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg15_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_108_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg19_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_109_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_2_]], [[VAR_arg23_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_3_]], [[VAR_86_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_111_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_112_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg12_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_113_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg16_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_114_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_115_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_3_]], [[VAR_arg24_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_93_]], [[VAR_99_]], [[VAR_105_]], [[VAR_111_]], [[VAR_94_]], [[VAR_100_]], [[VAR_106_]], [[VAR_112_]], [[VAR_95_]], [[VAR_101_]], [[VAR_107_]], [[VAR_113_]], [[VAR_96_]], [[VAR_102_]], [[VAR_108_]], [[VAR_114_]], [[VAR_97_]], [[VAR_103_]], [[VAR_109_]], [[VAR_115_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_10_:%.+]] = vector.shuffle [[VAR_9_]]#0, [[VAR_9_]]#1 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = vector.shuffle [[VAR_9_]]#0, [[VAR_9_]]#1 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addf [[VAR_10_]], [[VAR_11_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = vector.shuffle [[VAR_9_]]#2, [[VAR_9_]]#3 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = vector.shuffle [[VAR_9_]]#2, [[VAR_9_]]#3 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[VAR_13_]], [[VAR_14_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.shuffle [[VAR_12_]], [[VAR_15_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = vector.shuffle [[VAR_12_]], [[VAR_15_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addf [[VAR_16_]], [[VAR_17_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_23_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_2_MEM_1_]], [[LOAD_PARAM_2_MEM_2_]], [[LOAD_PARAM_2_MEM_3_]] : vector<4xf32>
// CHECK:             [[VAR_24_:%.+]] = arith.addf [[VAR_18_]], [[VAR_23_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_24_]], [[RES_]]{{.}}[[VAR_3_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = vector.shuffle [[VAR_9_]]#4, [[VAR_9_]]#5 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = vector.shuffle [[VAR_9_]]#4, [[VAR_9_]]#5 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.addf [[VAR_25_]], [[VAR_26_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.shuffle [[VAR_9_]]#6, [[VAR_9_]]#7 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = vector.shuffle [[VAR_9_]]#6, [[VAR_9_]]#7 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_30_:%.+]] = arith.addf [[VAR_28_]], [[VAR_29_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = vector.shuffle [[VAR_27_]], [[VAR_30_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = vector.shuffle [[VAR_27_]], [[VAR_30_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.addf [[VAR_31_]], [[VAR_32_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_5_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_6_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_7_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_38_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_4_]], [[LOAD_PARAM_2_MEM_5_]], [[LOAD_PARAM_2_MEM_6_]], [[LOAD_PARAM_2_MEM_7_]] : vector<4xf32>
// CHECK:             [[VAR_39_:%.+]] = arith.addf [[VAR_33_]], [[VAR_38_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_39_]], [[RES_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = vector.shuffle [[VAR_9_]]#8, [[VAR_9_]]#9 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = vector.shuffle [[VAR_9_]]#8, [[VAR_9_]]#9 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.addf [[VAR_40_]], [[VAR_41_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = vector.shuffle [[VAR_9_]]#10, [[VAR_9_]]#11 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = vector.shuffle [[VAR_9_]]#10, [[VAR_9_]]#11 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_45_:%.+]] = arith.addf [[VAR_43_]], [[VAR_44_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = vector.shuffle [[VAR_42_]], [[VAR_45_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = vector.shuffle [[VAR_42_]], [[VAR_45_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.addf [[VAR_46_]], [[VAR_47_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_8_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_9_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_10_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_11_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_53_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_8_]], [[LOAD_PARAM_2_MEM_9_]], [[LOAD_PARAM_2_MEM_10_]], [[LOAD_PARAM_2_MEM_11_]] : vector<4xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.addf [[VAR_48_]], [[VAR_53_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_54_]], [[RES_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = vector.shuffle [[VAR_9_]]#12, [[VAR_9_]]#13 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_56_:%.+]] = vector.shuffle [[VAR_9_]]#12, [[VAR_9_]]#13 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.addf [[VAR_55_]], [[VAR_56_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_58_:%.+]] = vector.shuffle [[VAR_9_]]#14, [[VAR_9_]]#15 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_59_:%.+]] = vector.shuffle [[VAR_9_]]#14, [[VAR_9_]]#15 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_60_:%.+]] = arith.addf [[VAR_58_]], [[VAR_59_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_61_:%.+]] = vector.shuffle [[VAR_57_]], [[VAR_60_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_62_:%.+]] = vector.shuffle [[VAR_57_]], [[VAR_60_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.addf [[VAR_61_]], [[VAR_62_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_12_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_13_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_14_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_15_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_68_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_12_]], [[LOAD_PARAM_2_MEM_13_]], [[LOAD_PARAM_2_MEM_14_]], [[LOAD_PARAM_2_MEM_15_]] : vector<4xf32>
// CHECK:             [[VAR_69_:%.+]] = arith.addf [[VAR_63_]], [[VAR_68_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_69_]], [[RES_]]{{.}}[[VAR_6_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_70_:%.+]] = vector.shuffle [[VAR_9_]]#16, [[VAR_9_]]#17 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_71_:%.+]] = vector.shuffle [[VAR_9_]]#16, [[VAR_9_]]#17 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_72_:%.+]] = arith.addf [[VAR_70_]], [[VAR_71_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_73_:%.+]] = vector.shuffle [[VAR_9_]]#18, [[VAR_9_]]#19 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_74_:%.+]] = vector.shuffle [[VAR_9_]]#18, [[VAR_9_]]#19 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_75_:%.+]] = arith.addf [[VAR_73_]], [[VAR_74_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_76_:%.+]] = vector.shuffle [[VAR_72_]], [[VAR_75_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_77_:%.+]] = vector.shuffle [[VAR_72_]], [[VAR_75_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_78_:%.+]] = arith.addf [[VAR_76_]], [[VAR_77_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_16_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_17_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_18_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_19_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_83_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_16_]], [[LOAD_PARAM_2_MEM_17_]], [[LOAD_PARAM_2_MEM_18_]], [[LOAD_PARAM_2_MEM_19_]] : vector<4xf32>
// CHECK:             [[VAR_84_:%.+]] = arith.addf [[VAR_78_]], [[VAR_83_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_84_]], [[RES_]]{{.}}[[VAR_7_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_1_:%.+]] = 5 to 6){
// CHECK-DAG:         [[VAR_2_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_1_:%.+]]:4 = krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[VAR_arg4_1_:%.+]] = [[CST_0_]] to [[CST_1_]]) iter_args([[VAR_arg5_1_:%.+]] = [[VAR_cst_]][[VAR_arg6_1_:%.+]] = [[VAR_cst_]][[VAR_arg7_1_:%.+]] = [[VAR_cst_]][[VAR_arg8_1_:%.+]] = [[VAR_cst_]]) -> (vector<4xf32>vector<4xf32>vector<4xf32>vector<4xf32>){
// CHECK-DAG:           [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_2_MEM_2_:%.+]] = arith.muli [[LOAD_PARAM_2_MEM_1_]], [[CST_4_]] : index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_1_]], [[LOAD_PARAM_2_MEM_2_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[LOAD_PARAM_2_MEM_2_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_4_]], [[VAR_arg5_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_1_]], [[LOAD_PARAM_2_MEM_2_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_5_]], [[VAR_arg6_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_2_]], [[LOAD_PARAM_2_MEM_2_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_6_]], [[VAR_arg7_1_]] : vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_3_]], [[LOAD_PARAM_2_MEM_2_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               [[VAR_30_1_:%.+]] = vector.fma [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_7_]], [[VAR_arg8_1_]] : vector<4xf32>
// CHECK:               krnl.yield [[VAR_24_1_]], [[VAR_26_1_]], [[VAR_28_1_]], [[VAR_30_1_]] : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_5_1_:%.+]] = vector.shuffle [[VAR_4_1_]]#0, [[VAR_4_1_]]#1 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_6_1_:%.+]] = vector.shuffle [[VAR_4_1_]]#0, [[VAR_4_1_]]#1 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_1_:%.+]] = arith.addf [[VAR_5_1_]], [[VAR_6_1_]] : vector<4xf32>
// CHECK-DAG:         [[LOOP_1_:%.+]] = vector.shuffle [[VAR_4_1_]]#2, [[VAR_4_1_]]#3 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = vector.shuffle [[VAR_4_1_]]#2, [[VAR_4_1_]]#3 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_10_1_:%.+]] = arith.addf [[LOOP_1_]], [[VAR_9_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_11_1_:%.+]] = vector.shuffle [[VAR_7_1_]], [[VAR_10_1_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_12_1_:%.+]] = vector.shuffle [[VAR_7_1_]], [[VAR_10_1_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_1_:%.+]] = arith.addf [[VAR_11_1_]], [[VAR_12_1_]] : vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_20_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_21_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_22_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_23_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_18_1_:%.+]] = vector.from_elements [[LOAD_PARAM_2_MEM_20_]], [[LOAD_PARAM_2_MEM_21_]], [[LOAD_PARAM_2_MEM_22_]], [[LOAD_PARAM_2_MEM_23_]] : vector<4xf32>
// CHECK:             [[LOAD_PARAM_2_MEM_24_:%.+]] = arith.addf [[VAR_13_1_]], [[VAR_18_1_]] : vector<4xf32>
// CHECK:             vector.store [[LOAD_PARAM_2_MEM_24_]], [[RES_]]{{.}}[[VAR_2_1_]], [[CST_0_]]{{.}} : memref<6x4xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x4xf32>
// CHECK:         }

}

