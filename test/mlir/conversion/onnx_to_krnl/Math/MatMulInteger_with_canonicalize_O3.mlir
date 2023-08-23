// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --mcpu=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --mcpu=z16 to enable SIMD as we now need a machine
// can also use -march=x86-64 instead.

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_matmulinteger_per_tensor(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<1xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<1xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_matmulinteger_per_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
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
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_2) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_4_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_2_]]([[RES_4_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 512){
// CHECK:             [[VAR_7_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_2_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = vector.splat [[VAR_9_1_]] : vector<32xi32>
// CHECK:             [[VAR_11_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_10_2_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_11_]], [[VAR_reshape_4_]]{{.}}[[VAR_7_2_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_9_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_3_:%.+]] = arith.extui [[VAR_9_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_3_]], [[RES_5_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_6_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_4_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_3_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_4_:%.+]] = arith.extui [[VAR_9_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_4_]], [[RES_6_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_5_]]([[RES_8_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_5_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_5_]] -> [[I_7_:%.+]] = 0 to 2048){
// CHECK:             [[VAR_7_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[VAR_7_5_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = vector.splat [[VAR_9_3_]] : vector<32xi32>
// CHECK:             [[VAR_11_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_10_5_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_11_1_]], [[VAR_reshape_11_]]{{.}}[[VAR_7_5_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_10_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_6_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_6_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_6_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]]) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) with ([[LOOP_6_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_6_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_6_]]#2 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_7_6_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_7_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]4{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_7_6_]]#0, [[VAR_7_6_]]#1, [[VAR_7_6_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_10_]] : memref<16x64xi32>
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
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
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
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]]#1 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_7_2_]]#0, [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = vector.splat [[VAR_9_1_]] : vector<32xi32>
// CHECK:             [[VAR_11_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_10_2_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_11_]], [[RES_2_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>, vector<32xi32>
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
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_3_]]([[RES_6_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_7_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_5_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_5_]] -> [[I_8_:%.+]] = 0 to 2048){
// CHECK:             [[VAR_7_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_5_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = vector.splat [[VAR_9_3_]] : vector<32xi32>
// CHECK:             [[VAR_11_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_10_5_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_11_1_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_5_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_8_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_6_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_6_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_6_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]]) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) with ([[LOOP_6_]]#0 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_6_]]#1 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_6_]]#2 -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_7_6_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_5_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]0{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_7_6_]]#0, [[VAR_7_6_]]#1, [[VAR_7_6_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<16x64xi32>
// CHECK:         }
}
