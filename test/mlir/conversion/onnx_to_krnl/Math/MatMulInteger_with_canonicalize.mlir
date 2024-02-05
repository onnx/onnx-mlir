// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_matmulinteger_per_tensor(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<1xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<1xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// CHECK-LABEL:  func.func @test_matmulinteger_per_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_9_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_10_:%.+]] = arith.extui [[VAR_9_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_7_1_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_10_1_:%.+]] = arith.extui [[VAR_9_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[VAR_7_1_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_9_1_]] : i32
// CHECK:             krnl.store [[VAR_10_2_]], [[RES_2_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_9_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_3_:%.+]] = arith.extui [[VAR_9_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_3_]], [[RES_3_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_4_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_3_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_4_:%.+]] = arith.extui [[VAR_9_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_4_]], [[RES_4_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = 0 to 32, [[LOOP_5_]]#1 -> [[I_9_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_9_3_]] : i32
// CHECK:             krnl.store [[VAR_10_5_]], [[RES_5_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<i32>
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_11_:%.+]] = 0 to 64, [[LOOP_6_]]#2 -> [[I_12_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_7_]][] : memref<i32>
// CHECK:             krnl.iterate([[LOOP_6_]]#2) with (){
// CHECK:               [[VAR_9_4_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_10_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_7_6_]]#0, [[VAR_9_4_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_9_4_]], [[VAR_7_6_]]#1] : memref<32x64xi32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:               [[VAR_13_:%.+]] = arith.muli [[VAR_10_5_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_7_MEM_]], [[VAR_13_]] : i32
// CHECK:               krnl.store [[VAR_14_]], [[RES_7_]][] : memref<i32>
// CHECK:             }
// CHECK:             [[LOAD_RES_7_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:             krnl.store [[LOAD_RES_7_MEM_1_]], [[RES_6_]]{{.}}[[VAR_7_6_]]#0, [[VAR_7_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
// CHECK:         }
}

// -----

func.func @test_matmulinteger_per_row_a(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<16xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<16xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// CHECK-LABEL:  func.func @test_matmulinteger_per_row
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<16xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_9_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_10_:%.+]] = arith.extui [[VAR_9_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_7_1_]]{{.}} : memref<16xui8>
// CHECK:             [[VAR_9_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_10_1_:%.+]] = arith.extui [[VAR_9_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[VAR_7_1_]]{{.}} : memref<16xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [16, 1], strides: [1, 1] : memref<16xi32> to memref<16x1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_7_2_]]#0, [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_9_1_]] : i32
// CHECK:             krnl.store [[VAR_10_2_]], [[RES_2_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_9_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_3_:%.+]] = arith.extui [[VAR_9_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_3_]], [[RES_3_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_4_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_3_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_4_:%.+]] = arith.extui [[VAR_9_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_4_]], [[RES_4_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = 0 to 32, [[LOOP_5_]]#1 -> [[I_9_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_9_3_]] : i32
// CHECK:             krnl.store [[VAR_10_5_]], [[RES_5_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<i32>
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_11_:%.+]] = 0 to 64, [[LOOP_6_]]#2 -> [[I_12_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_7_]][] : memref<i32>
// CHECK:             krnl.iterate([[LOOP_6_]]#2) with (){
// CHECK:               [[VAR_9_4_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_10_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_7_6_]]#0, [[VAR_9_4_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_9_4_]], [[VAR_7_6_]]#1] : memref<32x64xi32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:               [[VAR_13_:%.+]] = arith.muli [[VAR_10_5_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_7_MEM_]], [[VAR_13_]] : i32
// CHECK:               krnl.store [[VAR_14_]], [[RES_7_]][] : memref<i32>
// CHECK:             }
// CHECK:             [[LOAD_RES_7_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:             krnl.store [[LOAD_RES_7_MEM_1_]], [[RES_6_]]{{.}}[[VAR_7_6_]]#0, [[VAR_7_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
// CHECK:         }
}
