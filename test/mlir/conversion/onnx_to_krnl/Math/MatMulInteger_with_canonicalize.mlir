// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_matmulinteger_per_tensor(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<1xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<1xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_matmulinteger_per_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_11_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_11_]]#0, [[VAR_11_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_14_:%.+]] = arith.extui [[VAR_13_]] : i8 to i32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_11_]]#0, [[VAR_11_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_3_:%.+]] = arith.extui [[VAR_2_]] : i8 to i32
// CHECK:           krnl.store [[VAR_3_]], [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 16, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_11_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_11_1_]]#0, [[VAR_11_1_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_13_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_14_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_13_1_]] : i32
// CHECK:             krnl.store [[VAR_14_1_]], [[RES_2_]]{{.}}[[VAR_11_1_]]#0, [[VAR_11_1_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 64){
// CHECK:             [[VAR_11_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_13_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_14_2_:%.+]] = arith.extui [[VAR_13_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_14_2_]], [[RES_3_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_7_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_8_:%.+]] = arith.extui [[VAR_7_]] : i8 to i32
// CHECK:           krnl.store [[VAR_8_]], [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 64){
// CHECK:             [[VAR_11_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_11_3_]]#0, [[VAR_11_3_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_13_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_14_3_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_13_2_]] : i32
// CHECK:             krnl.store [[VAR_14_3_]], [[RES_5_]]{{.}}[[VAR_11_3_]]#0, [[VAR_11_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = 0 to 16, [[LOOP_4_]]#1 -> [[I_9_:%.+]] = 0 to 64, [[LOOP_4_]]#2 -> [[I_10_:%.+]] = 0 to 32){
// CHECK-DAG:         [[VAR_11_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.iterate([[LOOP_4_]]#2) with () iter_args([[VAR_arg7_:%.+]] = [[CST_0_]]) -> (i32){
// CHECK-DAG:           [[VAR_13_3_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]#2) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_14_3_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_11_4_]]#0, [[VAR_13_3_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_13_3_]], [[VAR_11_4_]]#1] : memref<32x64xi32>
// CHECK:               [[VAR_16_:%.+]] = arith.muli [[VAR_14_3_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_17_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_16_]] : i32
// CHECK:               krnl.yield [[VAR_17_]] : i32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_1_]], [[RES_6_]]{{.}}[[VAR_11_4_]]#0, [[VAR_11_4_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
// CHECK:         }
}

// -----

func.func @test_matmulinteger_per_row_a(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<16xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<16xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_matmulinteger_per_row_a
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<16xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_12_:%.+]] = arith.extui [[VAR_11_]] : i8 to i32
// CHECK:             krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_9_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_9_1_]]{{.}} : memref<16xui8>
// CHECK:             [[VAR_11_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_12_1_:%.+]] = arith.extui [[VAR_11_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_12_1_]], [[RES_1_]]{{.}}[[VAR_9_1_]]{{.}} : memref<16xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [16, 1], strides: [1, 1] : memref<16xi32> to memref<16x1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_9_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_11_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_2_]]#0, [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:             [[VAR_12_2_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_11_1_]] : i32
// CHECK:             krnl.store [[VAR_12_2_]], [[RES_2_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_9_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_3_]]#0, [[VAR_9_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_11_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_12_3_:%.+]] = arith.extui [[VAR_11_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_12_3_]], [[RES_3_]]{{.}}[[VAR_9_3_]]#0, [[VAR_9_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_6_:%.+]] = arith.extui [[VAR_5_]] : i8 to i32
// CHECK:           krnl.store [[VAR_6_]], [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = 0 to 32, [[LOOP_4_]]#1 -> [[I_8_:%.+]] = 0 to 64){
// CHECK:             [[VAR_9_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_9_4_]]#0, [[VAR_9_4_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_11_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_12_4_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_11_2_]] : i32
// CHECK:             krnl.store [[VAR_12_4_]], [[RES_5_]]{{.}}[[VAR_9_4_]]#0, [[VAR_9_4_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_9_:%.+]] = 0 to 16, [[LOOP_5_]]#1 -> [[I_10_:%.+]] = 0 to 64, [[LOOP_5_]]#2 -> [[I_11_:%.+]] = 0 to 32){
// CHECK-DAG:         [[VAR_9_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.iterate([[LOOP_5_]]#2) with () iter_args([[VAR_arg7_:%.+]] = [[CST_0_]]) -> (i32){
// CHECK-DAG:           [[VAR_11_3_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]#2) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_12_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_9_5_]]#0, [[VAR_11_3_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_11_3_]], [[VAR_9_5_]]#1] : memref<32x64xi32>
// CHECK:               [[VAR_14_:%.+]] = arith.muli [[VAR_12_4_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_15_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_14_]] : i32
// CHECK:               krnl.yield [[VAR_15_]] : i32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_1_1_]], [[RES_6_]]{{.}}[[VAR_9_5_]]#0, [[VAR_9_5_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
// CHECK:         }
}

