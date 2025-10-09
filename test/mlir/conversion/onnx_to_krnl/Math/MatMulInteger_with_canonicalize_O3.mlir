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
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<16x32xui8>, memref<1xindex>) -> memref<512xui8>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 512){
// CHECK:               [[VAR_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_8_]]{{.}} : memref<512xui8>, vector<32xui8>
// CHECK:               [[VAR_10_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_]] : vector<32xui8> to vector<32xi8>
// CHECK:               [[VAR_11_:%.+]] = arith.extui [[VAR_10_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_11_]], [[VAR_reshape_2_]]{{.}}[[VAR_8_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_2_:%.+]] = arith.extui [[VAR_1_]] : i8 to i32
// CHECK:           krnl.store [[VAR_2_]], [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_5_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_6_:%.+]] = memref.reshape [[RES_]]([[RES_]]_5) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_6_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_8_:%.+]] = memref.reshape [[RES_4_]]([[RES_6_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 512){
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[VAR_8_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_10_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_11_1_:%.+]] = vector.broadcast [[VAR_10_1_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_12_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_]], [[VAR_11_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_12_]], [[VAR_reshape_8_]]{{.}}[[VAR_8_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_8_]]) : (memref<32x64xui8>, memref<1xindex>) -> memref<2048xui8>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_13_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_8_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_11_]]{{.}}[[VAR_8_2_]]{{.}} : memref<2048xui8>, vector<32xui8>
// CHECK:               [[VAR_10_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_1_]] : vector<32xui8> to vector<32xi8>
// CHECK:               [[VAR_11_2_:%.+]] = arith.extui [[VAR_10_2_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_11_2_]], [[VAR_reshape_13_]]{{.}}[[VAR_8_2_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_5_:%.+]] = arith.extui [[VAR_4_]] : i8 to i32
// CHECK:           krnl.store [[VAR_5_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_12_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_17_:%.+]] = memref.reshape [[RES_7_]]([[RES_12_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_13_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_19_:%.+]] = memref.reshape [[RES_11_]]([[RES_13_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_8_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__3_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_17_]]{{.}}[[VAR_8_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_10_2_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_11_3_:%.+]] = vector.broadcast [[VAR_10_2_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_12_1_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_]], [[VAR_11_3_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_12_1_]], [[VAR_reshape_19_]]{{.}}[[VAR_8_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_14_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__4_]]_21, [[BLOCK_IN__4_]]_22, [[BLOCK_TILE__4_]]_23, [[BLOCK_IN__4_]]_24) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__4_]], [[BLOCK_TILE__4_]]_21, [[BLOCK_TILE__4_]]_23) with ([[LOOP_4_]]#0 -> [[I_4_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_5_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[LOOP_3_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__4_]], [[BLOCK_TILE__4_]]_21, [[BLOCK_TILE__4_]]_23) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_4_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_11_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_14_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__4_]], [[BLOCK_IN__4_]]_22, [[BLOCK_IN__4_]]_24), ([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_14_]] : memref<16x64xi32>
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
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<16x32xui8>, memref<1xindex>) -> memref<512xui8>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 512){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_6_]]{{.}} : memref<512xui8>, vector<32xui8>
// CHECK:               [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_]] : vector<32xui8> to vector<32xi8>
// CHECK:               [[VAR_9_:%.+]] = arith.extui [[VAR_8_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_9_]], [[VAR_reshape_2_]]{{.}}[[VAR_6_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<16xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 16){
// CHECK:               [[VAR_6_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_6_1_]]{{.}} : memref<16xui8>, vector<16xui8>
// CHECK:               [[VAR_8_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_1_]] : vector<16xui8> to vector<16xi8>
// CHECK:               [[VAR_9_1_:%.+]] = arith.extui [[VAR_8_1_]] : vector<16xi8> to vector<16xi32>
// CHECK:               vector.store [[VAR_9_1_]], [[RES_3_]]{{.}}[[VAR_6_1_]]{{.}} : memref<16xi32>, vector<16xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_3_]] to offset: [0], sizes: [16, 1], strides: [1, 1] : memref<16xi32> to memref<16x1xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 32){
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_8_1_:%.+]] = vector.load [[RES_]]{{.}}[[LOOP_1_]], [[LOAD_VAR_reshape_MEM_1_]]{{.}} : memref<16x32xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[LOOP_1_]], [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:               [[VAR_10_:%.+]] = vector.broadcast [[VAR_9_1_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_11_:%.+]] = arith.subi [[VAR_8_1_]], [[VAR_10_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_11_]], [[RES_4_]]{{.}}[[LOOP_1_]], [[LOAD_VAR_reshape_MEM_1_]]{{.}} : memref<16x32xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_6_]]) : (memref<32x64xui8>, memref<1xindex>) -> memref<2048xui8>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_7_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__3_]]) with ([[LOOP_4_]] -> [[I_4_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_6_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_7_]]{{.}}[[VAR_6_2_]]{{.}} : memref<2048xui8>, vector<32xui8>
// CHECK:               [[VAR_8_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_1_1_]] : vector<32xui8> to vector<32xi8>
// CHECK:               [[VAR_9_2_:%.+]] = arith.extui [[VAR_8_2_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_9_2_]], [[VAR_reshape_9_]]{{.}}[[VAR_6_2_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_3_MEM_]] : ui8 to i8
// CHECK:           [[VAR_3_:%.+]] = arith.extui [[VAR_2_]] : i8 to i32
// CHECK:           krnl.store [[VAR_3_]], [[RES_8_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_10_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_13_:%.+]] = memref.reshape [[RES_5_]]([[RES_10_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_11_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_9_]]([[RES_11_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_5_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__4_]]) with ([[LOOP_5_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_6_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_13_]]{{.}}[[VAR_6_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_8_2_:%.+]] = krnl.load [[RES_8_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_9_3_:%.+]] = vector.broadcast [[VAR_8_2_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_10_1_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_]], [[VAR_9_3_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_10_1_]], [[VAR_reshape_15_]]{{.}}[[VAR_6_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_12_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_6_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_6_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_6_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__5_]], [[BLOCK_IN__5_]], [[BLOCK_TILE__5_]]_17, [[BLOCK_IN__5_]]_18, [[BLOCK_TILE__5_]]_19, [[BLOCK_IN__5_]]_20) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__5_]], [[BLOCK_TILE__5_]]_17, [[BLOCK_TILE__5_]]_19) with ([[LOOP_6_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_6_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_6_]]#2 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[LOOP_5_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__5_]], [[BLOCK_TILE__5_]]_17, [[BLOCK_TILE__5_]]_19) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_4_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_9_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_12_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__5_]], [[BLOCK_IN__5_]]_18, [[BLOCK_IN__5_]]_20), ([[LOOP_5_]]#0, [[LOOP_5_]]#1, [[LOOP_5_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_12_]] : memref<16x64xi32>
// CHECK:         }
}