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
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_2) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_4_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_2_]]([[RES_4_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 512){
// CHECK:             [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_11_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_13_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_14_1_:%.+]] = vector.splat [[VAR_13_1_]] : vector<32xi32>
// CHECK:             [[VAR_15_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_14_1_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_15_]], [[VAR_reshape_4_]]{{.}}[[VAR_11_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 64){
// CHECK:             [[VAR_11_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_13_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_14_2_:%.+]] = arith.extui [[VAR_13_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_14_2_]], [[RES_5_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_7_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_8_:%.+]] = arith.extui [[VAR_7_]] : i8 to i32
// CHECK:           krnl.store [[VAR_8_]], [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_5_]]([[RES_8_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:             [[VAR_11_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[VAR_11_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_13_2_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_14_3_:%.+]] = vector.splat [[VAR_13_2_]] : vector<32xi32>
// CHECK:             [[VAR_15_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_14_3_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_15_1_]], [[VAR_reshape_11_]]{{.}}[[VAR_11_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_10_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]]) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) with ([[LOOP_4_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_11_4_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_7_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]4{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_11_4_]]#0, [[VAR_11_4_]]#1, [[VAR_11_4_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
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
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]]#1 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_9_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<16x32xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_11_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_2_]]#0, [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:             [[VAR_12_2_:%.+]] = vector.splat [[VAR_11_1_]] : vector<32xi32>
// CHECK:             [[VAR_13_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_12_2_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_13_]], [[RES_2_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<16x32xi32>, vector<32xi32>
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
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_3_]]([[RES_6_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_7_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 2048){
// CHECK:             [[VAR_9_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_9_4_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:         [[VAR_11_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_12_4_:%.+]] = vector.splat [[VAR_11_2_]] : vector<32xi32>
// CHECK:             [[VAR_13_1_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_12_4_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_13_1_]], [[VAR_reshape_7_]]{{.}}[[VAR_9_4_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_8_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_5_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_5_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_5_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_5_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]]) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_5_]]#2 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_9_5_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_5_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]0{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_9_5_]]#0, [[VAR_9_5_]]#1, [[VAR_9_5_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<16x64xi32>
// CHECK:         }
}
