// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @qlinearmatmul_i8_f32(%arg0: tensor<16x32xi8>, %arg1: tensor<1xf32>, %arg2: tensor<1xi8>, %arg3: tensor<32x64xi8>, %arg4: tensor<1xf32>, %arg5: tensor<1xi8>, %arg6: tensor<1xf32>, %arg7: tensor<1xi8>) -> (tensor<16x64xi8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xi8>, tensor<1xf32>, tensor<1xi8>, tensor<32x64xi8>, tensor<1xf32>, tensor<1xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<16x64xi8>
    return %0 : tensor<16x64xi8>

// CHECK-LABEL:  func.func @qlinearmatmul_i8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xi8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xi8>, [[PARAM_3_:%.+]]: memref<32x64xi8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xi8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xi8>) -> memref<16x64xi8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_20_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_20_]]#0, [[VAR_20_]]#1] : memref<16x32xi8>
// CHECK:             [[VAR_22_:%.+]] = arith.extsi [[LOAD_PARAM_0_MEM_]] : i8 to i32
// CHECK:             krnl.store [[VAR_22_]], [[RES_]]{{.}}[[VAR_20_]]#0, [[VAR_20_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_2_:%.+]] = arith.extsi [[LOAD_PARAM_2_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_2_]], [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 16, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_20_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_20_1_]]#0, [[VAR_20_1_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_22_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_23_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_22_1_]] : i32
// CHECK:             krnl.store [[VAR_23_]], [[RES_2_]]{{.}}[[VAR_20_1_]]#0, [[VAR_20_1_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 64){
// CHECK:             [[VAR_20_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_20_2_]]#0, [[VAR_20_2_]]#1] : memref<32x64xi8>
// CHECK:             [[VAR_22_2_:%.+]] = arith.extsi [[LOAD_PARAM_0_MEM_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_22_2_]], [[RES_3_]]{{.}}[[VAR_20_2_]]#0, [[VAR_20_2_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_6_:%.+]] = arith.extsi [[LOAD_PARAM_5_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_6_]], [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 64){
// CHECK:             [[VAR_20_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_20_3_]]#0, [[VAR_20_3_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_22_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_23_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_22_2_]] : i32
// CHECK:             krnl.store [[VAR_23_1_]], [[RES_5_]]{{.}}[[VAR_20_3_]]#0, [[VAR_20_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_9_:%.+]] = arith.extsi [[LOAD_PARAM_7_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_9_]], [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = 0 to 16, [[LOOP_4_]]#1 -> [[I_9_:%.+]] = 0 to 64, [[LOOP_4_]]#2 -> [[I_10_:%.+]] = 0 to 32){
// CHECK-DAG:         [[VAR_20_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.iterate([[LOOP_4_]]#2) with () iter_args([[VAR_arg11_:%.+]] = [[CST_0_]]) -> (i32){
// CHECK-DAG:           [[VAR_22_3_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]#2) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_20_4_]]#0, [[VAR_22_3_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_22_3_]], [[VAR_20_4_]]#1] : memref<32x64xi32>
// CHECK:               [[VAR_25_:%.+]] = arith.muli [[VAR_23_1_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_26_:%.+]] = arith.addi [[VAR_arg11_]], [[VAR_25_]] : i32
// CHECK:               krnl.yield [[VAR_26_]] : i32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_1_]], [[RES_7_]]{{.}}[[VAR_20_4_]]#0, [[VAR_20_4_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_11_:%.+]] = 0 to 16, [[LOOP_5_]]#1 -> [[I_12_:%.+]] = 0 to 64){
// CHECK:             [[VAR_20_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_20_5_]]#0, [[VAR_20_5_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_22_4_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_1_1_1_]] : i32 to f32
// CHECK:             krnl.store [[VAR_22_4_]], [[RES_8_]]{{.}}[[VAR_20_5_]]#0, [[VAR_20_5_]]#1] : memref<16x64xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_14_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_14_]], [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_9_MEM_:%.+]] = krnl.load [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_17_:%.+]] = arith.divf [[LOAD_RES_9_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_17_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_13_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_14_:%.+]] = 0 to 64){
// CHECK:             [[VAR_20_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[VAR_20_6_]]#0, [[VAR_20_6_]]#1] : memref<16x64xf32>
// CHECK-DAG:         [[VAR_22_4_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_23_2_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_22_4_]] : f32
// CHECK:             [[LOAD_RES_5_MEM_1_:%.+]] = math.floor [[VAR_23_2_]] : f32
// CHECK:             [[VAR_25_1_:%.+]] = arith.subf [[VAR_23_2_]], [[LOAD_RES_5_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_26_1_:%.+]] = arith.cmpf ogt, [[VAR_25_1_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.addf [[LOAD_RES_5_MEM_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.select [[VAR_26_1_]], [[VAR_27_]], [[LOAD_RES_5_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.mulf [[LOAD_RES_5_MEM_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_30_:%.+]] = math.floor [[VAR_29_]] : f32
// CHECK:             [[VAR_31_:%.+]] = arith.mulf [[VAR_30_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_32_:%.+]] = arith.subf [[LOAD_RES_5_MEM_1_]], [[VAR_31_]] : f32
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpf oeq, [[VAR_32_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.addf [[LOAD_RES_5_MEM_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.select [[VAR_33_]], [[VAR_34_]], [[LOAD_RES_5_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.cmpf oeq, [[VAR_25_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[VAR_35_]], [[VAR_28_]] : f32
// CHECK:             [[VAR_38_:%.+]] = arith.fptosi [[VAR_37_]] : f32 to i32
// CHECK:             krnl.store [[VAR_38_]], [[RES_11_]]{{.}}[[VAR_20_6_]]#0, [[VAR_20_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi8>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_15_:%.+]] = 0 to 16, [[LOOP_7_]]#1 -> [[I_16_:%.+]] = 0 to 64){
// CHECK:             [[VAR_20_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.load [[RES_11_]]{{.}}[[VAR_20_7_]]#0, [[VAR_20_7_]]#1] : memref<16x64xi32>
// CHECK-DAG:         [[VAR_22_4_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_23_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_1_1_1_1_]], [[VAR_22_4_1_]] : i32
// CHECK:             [[LOAD_RES_5_MEM_1_:%.+]] = arith.trunci [[VAR_23_3_]] : i32 to i8
// CHECK:             krnl.store [[LOAD_RES_5_MEM_1_]], [[RES_12_]]{{.}}[[VAR_20_7_]]#0, [[VAR_20_7_]]#1] : memref<16x64xi8>
// CHECK:           }
// CHECK:           return [[RES_12_]] : memref<16x64xi8>
// CHECK:         }
}

//-----

func.func @qlinearmatmul_ui8_f32(%arg0: tensor<16x32xui8>, %arg1: tensor<1xf32>, %arg2: tensor<1xui8>, %arg3: tensor<32x64xui8>, %arg4: tensor<1xf32>, %arg5: tensor<1xui8>, %arg6: tensor<1xf32>, %arg7: tensor<1xui8>) -> (tensor<16x64xui8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xui8>, tensor<1xf32>, tensor<1xui8>, tensor<32x64xui8>, tensor<1xf32>, tensor<1xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<16x64xui8>
    return %0 : tensor<16x64xui8>

// CHECK-LABEL:  func.func @qlinearmatmul_ui8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<32x64xui8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xui8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xui8>) -> memref<16x64xui8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_50_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_52_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.extui [[VAR_52_]] : i8 to i16
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<i16>
// CHECK:             [[VAR_55_:%.+]] = arith.subi [[VAR_53_]], [[LOAD_VAR_1_MEM_]] : i16
// CHECK:             [[VAR_56_:%.+]] = arith.trunci [[VAR_55_]] : i16 to i8
// CHECK:             [[VAR_57_:%.+]] = arith.extsi [[VAR_56_]] : i8 to i32
// CHECK:             krnl.store [[VAR_57_]], [[RES_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_6_:%.+]] = arith.extui [[VAR_5_]] : i8 to i16
// CHECK:           krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]][] : memref<i16>
// CHECK:           [[VAR_9_:%.+]] = arith.subi [[LOAD_RES_1_MEM_]], [[LOAD_VAR_3_MEM_]] : i16
// CHECK:           krnl.store [[VAR_9_]], [[RES_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_11_:%.+]] = arith.trunci [[LOAD_RES_2_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_11_]], [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_13_:%.+]] = arith.extsi [[LOAD_RES_3_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_13_]], [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 16, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_50_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_50_1_]]#0, [[VAR_50_1_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_52_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_53_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_52_1_]] : i32
// CHECK:             krnl.store [[VAR_53_1_]], [[RES_5_]]{{.}}[[VAR_50_1_]]#0, [[VAR_50_1_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 64){
// CHECK:             [[VAR_50_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_50_2_]]#0, [[VAR_50_2_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_52_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK-DAG:         [[VAR_53_2_:%.+]] = arith.extui [[VAR_52_2_]] : i8 to i16
// CHECK-DAG:         [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_15_]][] : memref<i16>
// CHECK:             [[VAR_55_1_:%.+]] = arith.subi [[VAR_53_2_]], [[LOAD_VAR_1_MEM_1_]] : i16
// CHECK:             [[VAR_56_1_:%.+]] = arith.trunci [[VAR_55_1_]] : i16 to i8
// CHECK:             [[VAR_57_1_:%.+]] = arith.extsi [[VAR_56_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_57_1_]], [[RES_6_]]{{.}}[[VAR_50_2_]]#0, [[VAR_50_2_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_17_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_5_MEM_]] : ui8 to i8
// CHECK:           [[VAR_20_:%.+]] = arith.extui [[VAR_19_]] : i8 to i16
// CHECK:           krnl.store [[VAR_20_]], [[RES_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]][] : memref<i16>
// CHECK:           [[VAR_23_:%.+]] = arith.subi [[LOAD_RES_7_MEM_]], [[LOAD_VAR_17_MEM_]] : i16
// CHECK:           krnl.store [[VAR_23_]], [[RES_8_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_25_:%.+]] = arith.trunci [[LOAD_RES_8_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_25_]], [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_9_MEM_:%.+]] = krnl.load [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_27_:%.+]] = arith.extsi [[LOAD_RES_9_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_27_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 64){
// CHECK:             [[VAR_50_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_52_2_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_53_3_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_52_2_]] : i32
// CHECK:             krnl.store [[VAR_53_3_]], [[RES_11_]]{{.}}[[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_29_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_7_MEM_]] : ui8 to i8
// CHECK:           [[VAR_32_:%.+]] = arith.extui [[VAR_31_]] : i8 to i16
// CHECK:           krnl.store [[VAR_32_]], [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_12_MEM_:%.+]] = krnl.load [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]][] : memref<i16>
// CHECK:           [[VAR_35_:%.+]] = arith.subi [[LOAD_RES_12_MEM_]], [[LOAD_VAR_29_MEM_]] : i16
// CHECK:           krnl.store [[VAR_35_]], [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_13_MEM_:%.+]] = krnl.load [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_37_:%.+]] = arith.trunci [[LOAD_RES_13_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_37_]], [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_14_MEM_:%.+]] = krnl.load [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_39_:%.+]] = arith.extsi [[LOAD_RES_14_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_39_]], [[RES_15_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = 0 to 16, [[LOOP_4_]]#1 -> [[I_9_:%.+]] = 0 to 64, [[LOOP_4_]]#2 -> [[I_10_:%.+]] = 0 to 32){
// CHECK-DAG:         [[VAR_50_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.iterate([[LOOP_4_]]#2) with () iter_args([[VAR_arg11_:%.+]] = [[CST_0_]]) -> (i32){
// CHECK-DAG:           [[VAR_52_3_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]#2) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_3_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_50_4_]]#0, [[VAR_52_3_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[RES_11_]]{{.}}[[VAR_52_3_]], [[VAR_50_4_]]#1] : memref<32x64xi32>
// CHECK:               [[VAR_55_2_:%.+]] = arith.muli [[VAR_53_3_]], [[LOAD_VAR_1_MEM_1_]] : i32
// CHECK:               [[VAR_56_2_:%.+]] = arith.addi [[VAR_arg11_]], [[VAR_55_2_]] : i32
// CHECK:               krnl.yield [[VAR_56_2_]] : i32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_1_]], [[RES_16_]]{{.}}[[VAR_50_4_]]#0, [[VAR_50_4_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_11_:%.+]] = 0 to 16, [[LOOP_5_]]#1 -> [[I_12_:%.+]] = 0 to 64){
// CHECK:             [[VAR_50_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_16_]]{{.}}[[VAR_50_5_]]#0, [[VAR_50_5_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_52_4_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_1_1_1_]] : i32 to f32
// CHECK:             krnl.store [[VAR_52_4_]], [[RES_17_]]{{.}}[[VAR_50_5_]]#0, [[VAR_50_5_]]#1] : memref<16x64xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_44_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_44_]], [[RES_18_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_18_MEM_:%.+]] = krnl.load [[RES_18_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_47_:%.+]] = arith.divf [[LOAD_RES_18_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_47_]], [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_13_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_14_:%.+]] = 0 to 64){
// CHECK:             [[VAR_50_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_17_]]{{.}}[[VAR_50_6_]]#0, [[VAR_50_6_]]#1] : memref<16x64xf32>
// CHECK-DAG:         [[VAR_52_4_:%.+]] = krnl.load [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_53_4_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_52_4_]] : f32
// CHECK:             [[LOAD_VAR_1_MEM_1_1_:%.+]] = math.floor [[VAR_53_4_]] : f32
// CHECK:             [[VAR_55_3_:%.+]] = arith.subf [[VAR_53_4_]], [[LOAD_VAR_1_MEM_1_1_]] : f32
// CHECK-DAG:         [[VAR_56_3_:%.+]] = arith.cmpf ogt, [[VAR_55_3_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_57_2_:%.+]] = arith.addf [[LOAD_VAR_1_MEM_1_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.select [[VAR_56_3_]], [[VAR_57_2_]], [[LOAD_VAR_1_MEM_1_1_]] : f32
// CHECK-DAG:         [[VAR_59_:%.+]] = arith.mulf [[LOAD_VAR_1_MEM_1_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_60_:%.+]] = math.floor [[VAR_59_]] : f32
// CHECK:             [[VAR_61_:%.+]] = arith.mulf [[VAR_60_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_62_:%.+]] = arith.subf [[LOAD_VAR_1_MEM_1_1_]], [[VAR_61_]] : f32
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.cmpf oeq, [[VAR_62_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_64_:%.+]] = arith.addf [[LOAD_VAR_1_MEM_1_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_65_:%.+]] = arith.select [[VAR_63_]], [[VAR_64_]], [[LOAD_VAR_1_MEM_1_1_]] : f32
// CHECK-DAG:         [[VAR_66_:%.+]] = arith.cmpf oeq, [[VAR_55_3_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[VAR_65_]], [[VAR_58_]] : f32
// CHECK:             [[VAR_68_:%.+]] = arith.fptosi [[VAR_67_]] : f32 to i32
// CHECK:             krnl.store [[VAR_68_]], [[RES_20_]]{{.}}[[VAR_50_6_]]#0, [[VAR_50_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<16x64xui8>
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_15_:%.+]] = 0 to 16, [[LOOP_7_]]#1 -> [[I_16_:%.+]] = 0 to 64){
// CHECK:             [[VAR_50_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.load [[RES_20_]]{{.}}[[VAR_50_7_]]#0, [[VAR_50_7_]]#1] : memref<16x64xi32>
// CHECK-DAG:         [[VAR_52_4_1_:%.+]] = krnl.load [[RES_15_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_53_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_1_1_1_1_]], [[VAR_52_4_1_]] : i32
// CHECK-DAG:         [[LOAD_VAR_1_MEM_1_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i32>
// CHECK:             [[VAR_55_4_:%.+]] = arith.addi [[VAR_53_5_]], [[LOAD_VAR_1_MEM_1_1_]] : i32
// CHECK:             [[VAR_56_4_:%.+]] = arith.trunci [[VAR_55_4_]] : i32 to i8
// CHECK:             [[VAR_57_3_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_4_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_57_3_]], [[RES_21_]]{{.}}[[VAR_50_7_]]#0, [[VAR_50_7_]]#1] : memref<16x64xui8>
// CHECK:           }
// CHECK:           return [[RES_21_]] : memref<16x64xui8>
// CHECK:         }
}
