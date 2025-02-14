// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// use --mtriple=s390x-ibm-loz --march=z16 to enable SIMD as we now need a machine
// can also use --march=x86-64 instead.
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
// CHECK:             [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_12_:%.+]] = arith.extui [[VAR_11_]] : i8 to i32
// CHECK:             krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<16x32xi32>
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
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_2_]]([[RES_4_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 512){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_11_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_12_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_13_:%.+]] = vector.splat [[VAR_12_1_]] : vector<32xi32>
// CHECK:               [[VAR_14_:%.+]] = arith.subi [[VAR_11_1_]], [[VAR_13_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_14_]], [[VAR_reshape_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 64){
// CHECK:             [[VAR_9_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_1_]]#0, [[VAR_9_1_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_11_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_12_2_:%.+]] = arith.extui [[VAR_11_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_12_2_]], [[RES_5_]]{{.}}[[VAR_9_1_]]#0, [[VAR_9_1_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_6_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_7_:%.+]] = arith.extui [[VAR_6_]] : i8 to i32
// CHECK:           krnl.store [[VAR_7_]], [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_5_]]([[RES_8_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_11_2_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_12_2_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_13_1_:%.+]] = vector.splat [[VAR_12_2_]] : vector<32xi32>
// CHECK:               [[VAR_14_1_:%.+]] = arith.subi [[VAR_11_2_]], [[VAR_13_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_14_1_]], [[VAR_reshape_11_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_10_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__2_]]_13, [[BLOCK_IN__2_]]_14, [[BLOCK_TILE__2_]]_15, [[BLOCK_IN__2_]]_16) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_13, [[BLOCK_TILE__2_]]_15) with ([[LOOP_4_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_9_2_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_13, [[BLOCK_TILE__2_]]_15) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_7_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]2{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__2_]]_14, [[BLOCK_IN__2_]]_16), ([[VAR_9_2_]]#0, [[VAR_9_2_]]#1, [[VAR_9_2_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
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
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_10_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_11_:%.+]] = arith.extui [[VAR_10_]] : i8 to i32
// CHECK:             krnl.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_8_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_1_]]{{.}} : memref<16xui8>
// CHECK:             [[VAR_10_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_11_1_:%.+]] = arith.extui [[VAR_10_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_11_1_]], [[RES_1_]]{{.}}[[VAR_8_1_]]{{.}} : memref<16xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [16, 1], strides: [1, 1] : memref<16xi32> to memref<16x1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 16){
// CHECK-DAG:         [[VAR_8_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 32){
// CHECK:               [[VAR_10_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_11_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_8_2_]], [[VAR_10_2_]]{{.}} : memref<16x32xi32>, vector<32xi32>
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_2_]], [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:               [[VAR_13_:%.+]] = vector.splat [[LOAD_VAR_reinterpret_cast_MEM_]] : vector<32xi32>
// CHECK:               [[VAR_14_:%.+]] = arith.subi [[VAR_11_1_]], [[VAR_13_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_14_]], [[RES_2_]]{{.}}[[VAR_8_2_]], [[VAR_10_2_]]{{.}} : memref<16x32xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_4_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_8_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_8_3_]]#0, [[VAR_8_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_10_3_:%.+]] = builtin.unrealized_conversion_cast [[LOOP_3_]] : ui8 to i8
// CHECK:             [[VAR_11_2_:%.+]] = arith.extui [[VAR_10_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_11_2_]], [[RES_3_]]{{.}}[[VAR_8_3_]]#0, [[VAR_8_3_]]#1] : memref<32x64xi32>
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
// CHECK:           [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_5_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_5_]] -> [[I_7_:%.+]] = 0 to 2048){
// CHECK:               [[LOOP_3_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_10_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOOP_3_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_11_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.splat [[VAR_11_2_]] : vector<32xi32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subi [[VAR_10_3_]], [[LOAD_VAR_reinterpret_cast_MEM_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_13_1_]], [[VAR_reshape_7_]]{{.}}[[LOOP_3_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_8_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_6_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_6_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_6_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__2_]]_9, [[BLOCK_IN__2_]]_10, [[BLOCK_TILE__2_]]_11, [[BLOCK_IN__2_]]_12) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_9, [[BLOCK_TILE__2_]]_11) with ([[LOOP_6_]]#0 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_6_]]#1 -> [[I_9_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_6_]]#2 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_8_4_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_9, [[BLOCK_TILE__2_]]_11) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_5_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_8_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__2_]]_10, [[BLOCK_IN__2_]]_12), ([[VAR_8_4_]]#0, [[VAR_8_4_]]#1, [[VAR_8_4_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_8_]] : memref<16x64xi32>
// CHECK:         }
}

