// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --march=z16 to enable SIMD as we now need a machine
// can also use --march=x86-64 instead.

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.
func.func @test_mod_fp32(%arg0: tensor<60xf32>, %arg1: tensor<60xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 1 : si64} : (tensor<60xf32>, tensor<60xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["a","b"]'
// CHECK-LABEL:  func.func @test_mod_fp32
// CHECK-SAME:   ([[A_:%.+]]: memref<60xf32>, [[B_:%.+]]: memref<60xf32>) -> memref<60xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<256xi8> to memref<60xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 60){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = vector.load [[A_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = vector.load [[B_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.remf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_5_:%.+]] = math.copysign [[VAR_4_]], [[LOAD_A_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_5_]], [[VAR_view_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<60xf32>
// CHECK:         }
}

// -----

func.func @test_mean(%arg0: tensor<30xf32>, %arg1: tensor<30xf32>, %arg2: tensor<30xf32>) -> tensor<*xf32>  {
    %0 = "onnx.Mean"(%arg0, %arg1, %arg2) : (tensor<30xf32>, tensor<30xf32>, tensor<30xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["a","b","c"]'
// CHECK-LABEL:  func.func @test_mean
// CHECK-SAME:   ([[A_:%.+]]: memref<30xf32>, [[B_:%.+]]: memref<30xf32>, [[C_:%.+]]: memref<30xf32>) -> memref<30xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<3.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<128xi8> to memref<30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 30){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = vector.load [[A_]]{{.}}[[VAR_1_]]{{.}} : memref<30xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = vector.load [[B_]]{{.}}[[VAR_1_]]{{.}} : memref<30xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_C_MEM_:%.+]] = vector.load [[C_]]{{.}}[[VAR_1_]]{{.}} : memref<30xf32>, vector<32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_5_]], [[LOAD_C_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_7_]], [[VAR_view_]]{{.}}[[VAR_1_]]{{.}} : memref<30xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<30xf32>
// CHECK:         }
}

// -----


func.func @round(%arg0: tensor<15xf32>) -> tensor<*xf32> {
  %0 = "onnx.Round"(%arg0) : (tensor<15xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @round
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<15xf32>) -> memref<15xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<64xi8>
// CHECK:           [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<64xi8> to memref<15xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 15){
// CHECK:               [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<15xf32>, vector<16xf32>
// CHECK:               [[VAR_3_:%.+]] = vector.shape_cast [[LOAD_PARAM_0_MEM_]] : vector<16xf32> to vector<4x4xf32>
// CHECK:               [[VAR_4_:%.+]] = vector.extract [[VAR_3_]][0] : vector<4xf32> from vector<4x4xf32>
// CHECK:               [[VAR_5_:%.+]] = "krnl.round_even"([[VAR_4_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_6_:%.+]] = vector.insert [[VAR_5_]], [[VAR_3_]] [0] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = vector.extract [[VAR_3_]][1] : vector<4xf32> from vector<4x4xf32>
// CHECK:               [[VAR_8_:%.+]] = "krnl.round_even"([[VAR_7_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = vector.insert [[VAR_8_]], [[VAR_6_]] [1] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:           [[VAR_10_:%.+]] = vector.extract [[VAR_3_]][2] : vector<4xf32> from vector<4x4xf32>
// CHECK:               [[VAR_11_:%.+]] = "krnl.round_even"([[VAR_10_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_12_:%.+]] = vector.insert [[VAR_11_]], [[VAR_9_]] [2] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:           [[VAR_13_:%.+]] = vector.extract [[VAR_3_]][3] : vector<4xf32> from vector<4x4xf32>
// CHECK:               [[VAR_14_:%.+]] = "krnl.round_even"([[VAR_13_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:               [[VAR_15_:%.+]] = vector.insert [[VAR_14_]], [[VAR_12_]] [3] : vector<4xf32> into vector<4x4xf32>
// CHECK:               [[VAR_16_:%.+]] = vector.shape_cast [[VAR_15_]] : vector<4x4xf32> to vector<16xf32>
// CHECK:               vector.store [[VAR_16_]], [[VAR_view_]]{{.}}[[VAR_1_]]{{.}} : memref<15xf32>, vector<16xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<15xf32>
// CHECK:         }
}

// -----

// Should allow SIMD as there is broadcast but second arg is a scalar, so no need for complicated stuff
func.func private @test_elementwise_op_with_array_and_scalar_values(%arg0 : tensor<4x8x16xf32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<4x8x16xf32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_elementwise_op_with_array_and_scalar_values
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8x16xf32>, [[PARAM_1_:%.+]]: memref<f32>) -> memref<4x8x16xf32> {
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x8x16xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<4x8x16xf32>, memref<1xindex>) -> memref<512xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<4x8x16xf32>, memref<1xindex>) -> memref<512xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 512){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<512xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK:             [[VAR_4_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_4_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_5_]], [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<512xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x8x16xf32>
// CHECK:         }
}

// -----

// Should allow SIMD as there is broadcast but second arg is a scalar, so no need for complicated stuff
func.func private @test_elementwise_op_with_array_and_scalar_values_2(%arg0 : tensor<?x?x64xf32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x64xf32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_elementwise_op_with_array_and_scalar_values_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>, [[PARAM_1_:%.+]]: memref<f32>) -> memref<?x?x64xf32> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_64_]] : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x64xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_64_]] : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_3_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<?x?x64xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_3_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK:             [[VAR_8_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x64xf32>
// CHECK:         }
}

// -----

// SIMD for the lowest dim; possible broadcast for the top 2 dims.

func.func @roberta_partial_simd_1dim_v1(%arg0: tensor<?x?x768xf32>, %arg1: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<?x?x768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_0_]], [[VAR_1_]]{{.}}){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_13_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_7_]], [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_9_]], [[VAR_11_]], [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:               [[VAR_16_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// Same as above, but now the second arg is "constant" in the top 2 dims; and thus no need for broadcast select for the top 2 either.

func.func @roberta_partial_simd_1dim_v2(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<768xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// same as above, 2nd param has a useless 1 added in front.

func.func @roberta_partial_simd_1dim_v3(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1x768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1x768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1x768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_3_]]{{.}} : memref<1x768xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// same as above, 2nd param has 2 useless 1 added in front.

func.func @roberta_partial_simd_1dim_v4(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1x1x768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1x1x768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1x1x768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_3_]]{{.}} : memref<1x1x768xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// SIMD can compute over the entire data at once for an array of dynamic dimension.
func.func @roberta_partial_simd_1dim_tensor1(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_tensor1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_3_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_3_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_8_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// Same as above, now two 1 dim attribute for the scalar.
func.func @roberta_partial_simd_1dim_tensor2(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1x1xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1x1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_tensor2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1x1xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_3_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_3_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x1xf32>
// CHECK:             [[VAR_8_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// Same as above, now 3 ones above.
func.func @roberta_partial_simd_1dim_tensor3(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1x1x1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_tensor3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1x1x1xf32>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_3_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_3_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[CST_0_]]{{.}} : memref<1x1x1xf32>
// CHECK:             [[VAR_8_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// Same, now with a true scalar.
func.func @roberta_partial_simd_1dim_scalar(%arg0: tensor<?x?x768xf32>, %arg1: tensor<f32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<f32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_768_]] : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_3_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<?x?x768xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_3_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK:             [[VAR_8_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// has ?x? in the first 2 dims for both params; collapse of the lowest 2 dims.

func.func @roberta_partial_simd_2dim_v1(%arg0: tensor<?x?x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x?x96x8xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x96x8xf32>, memref<3xindex>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_6_]], [[RES_2_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_7_]], [[RES_2_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_2_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<?x?x96x8xf32>, memref<3xindex>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_0_]], [[RES_3_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_3_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_3_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_]]([[RES_]]_10) : (memref<?x?x96x8xf32>, memref<3xindex>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_0_]], [[VAR_1_]]{{.}}){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_13_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]], [[VAR_7_]], [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_9_MEM_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[VAR_9_]], [[VAR_11_]], [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:               [[VAR_16_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_9_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_16_]], [[VAR_reshape_11_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_13_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x96x8xf32>
// CHECK:         }
}

// -----

// has ?x2 and ?x? in the first 2 dims

func.func @roberta_partial_simd_2dim_v2(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x2x96x8xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_2_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x2x96x8xf32>, memref<3xindex>) -> memref<?x2x768xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_2_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_5_]], [[RES_2_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_2_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<?x?x96x8xf32>, memref<3xindex>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_0_]], [[RES_3_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_3_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_3_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_]]([[RES_]]_8) : (memref<?x2x96x8xf32>, memref<3xindex>) -> memref<?x2x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_2_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_10_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1, [[VAR_10_]]{{.}} : memref<?x2x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_7_MEM_:%.+]] = vector.load [[VAR_reshape_7_]]{{.}}[[VAR_6_]], [[VAR_8_]], [[VAR_10_]]{{.}} : memref<?x?x768xf32>, vector<32xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_7_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_13_]], [[VAR_reshape_9_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_10_]]{{.}} : memref<?x2x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x96x8xf32>
// CHECK:         }
}

// -----

// has ?x2 and ?x1 in the first 2 dims

func.func @roberta_partial_simd_2dim_v3(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x1x96x8xf32>) -> memref<?x2x96x8xf32> {
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1x96x8xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_1_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x2x96x8xf32>, memref<3xindex>) -> memref<?x2x768xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1x96x8xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_2_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_2_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_2_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<?x1x96x8xf32>, memref<3xindex>) -> memref<?x1x768xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_0_]], [[RES_3_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_3_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_3_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_]]([[RES_]]_6) : (memref<?x2x96x8xf32>, memref<3xindex>) -> memref<?x2x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1, [[VAR_8_]]{{.}} : memref<?x2x768xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_5_MEM_:%.+]] = vector.load [[VAR_reshape_5_]]{{.}}[[VAR_6_]], [[CST_0_]], [[VAR_8_]]{{.}} : memref<?x1x768xf32>, vector<32xf32>
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_5_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_11_]], [[VAR_reshape_7_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_8_]]{{.}} : memref<?x2x768xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x96x8xf32>
// CHECK:         }
}

// -----

// static size is not mod archVL, will do some SIMD some scalar.

func.func @roberta_partial_simd_2dim_not_0_mod_vl(%arg0: tensor<?x?x95x7xf32>, %arg1: tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x95x7xf32>, tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32>
    return %0 : tensor<?x?x95x7xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_not_0_mod_vl
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x95x7xf32>, [[PARAM_1_:%.+]]: memref<?x?x95x7xf32>) -> memref<?x?x95x7xf32> {
// CHECK-DAG:       [[CST_665_:%.+]] = arith.constant 665 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_2660_:%.+]] = arith.constant 2660 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_3_:%.+]] = arith.muli [[VAR_2_]], [[CST_2660_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_128_]] : index
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_4_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<?xi8> to memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_665_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x95x7xf32>, memref<3xindex>) -> memref<?x?x665xf32>
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_6_]], [[RES_2_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_7_]], [[RES_2_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_665_]], [[RES_2_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<?x?x95x7xf32>, memref<3xindex>) -> memref<?x?x665xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_0_]], [[RES_3_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_3_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_665_]], [[RES_3_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<?x?x95x7xf32>, memref<3xindex>) -> memref<?x?x665xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_0_]], [[VAR_1_]]{{.}}){
// CHECK-DAG:         [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_6_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_6_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_6_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_6_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 634){
// CHECK:               [[VAR_17_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]], [[VAR_17_]]{{.}} : memref<?x?x665xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_9_MEM_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[VAR_12_]], [[VAR_14_]], [[VAR_17_]]{{.}} : memref<?x?x665xf32>, vector<32xf32>
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_9_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_20_]], [[VAR_reshape_11_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_17_]]{{.}} : memref<?x?x665xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 640 to 665){
// CHECK:               [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_8_]], [[VAR_10_]], [[VAR_17_1_]]{{.}} : memref<?x?x665xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_9_MEM_1_:%.+]] = krnl.load [[VAR_reshape_9_]]{{.}}[[VAR_12_]], [[VAR_14_]], [[VAR_17_1_]]{{.}} : memref<?x?x665xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_1_]], [[LOAD_VAR_reshape_9_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_20_1_]], [[VAR_reshape_11_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_17_1_]]{{.}} : memref<?x?x665xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x?x95x7xf32>
// CHECK:         }
}

// -----

// Tests found in roberta when there are leading ones in tensors; they are ignored for the broadcasting as they contribute nothing.

func.func @add_from_test_with_1_1(%arg0 : tensor<1x1x128xf32>, %arg1 : tensor<128xf32>) -> tensor<*xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x1x128xf32>, tensor<128xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_from_test_with_1_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x128xf32>, [[PARAM_1_:%.+]]: memref<128xf32>) -> memref<1x1x128xf32> {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x128xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<1x1x128xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<1x1x128xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:               [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[VAR_4_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_4_]], [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x128xf32>
// CHECK:         }
}

// -----

// Same, with the 2 params swapped, to make sure order does not matter.

func.func @add_from_test_with_1_1_swapped(%arg0 : tensor<128xf32>, %arg1 : tensor<1x1x128xf32>) -> tensor<*xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<128xf32>, tensor<1x1x128xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_from_test_with_1_1_swapped
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128xf32>, [[PARAM_1_:%.+]]: memref<1x1x128xf32>) -> memref<1x1x128xf32> {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x128xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_1_]]) : (memref<1x1x128xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<1x1x128xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:               [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_4_]], [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x128xf32>
// CHECK:         }
}

// -----

func.func private @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

func.func private @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

func.func private @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_sum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----


func.func private @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----


func.func private @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_100_:%.+]] = arith.constant 100 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<512xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<512xi8> to memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_100_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x10xf32>, memref<1xindex>) -> memref<100xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 100){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.minnumf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_4_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<100xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// COM: Check float PRelu without broadcasting.
func.func @test_prelu_float(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_float
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_60_:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<256xi8> to memref<3x4x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<3x4x5xf32>, memref<1xindex>) -> memref<60xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<3x4x5xf32>, memref<1xindex>) -> memref<60xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<3x4x5xf32>, memref<1xindex>) -> memref<60xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 60){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xf32>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_VAR_reshape_2_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xi1>, vector<32xf32>
// CHECK:             vector.store [[VAR_6_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<60xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check int PRelu without broadcasting.
func.func @test_prelu_int(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x5xi32>) -> tensor<*xi32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xi32>, [[PARAM_1_:%.+]]: memref<3x4x5xi32>) -> memref<3x4x5xi32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0> : vector<32xi32>
// CHECK-DAG:       [[CST_60_:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<256xi8> to memref<3x4x5xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<3x4x5xi32>, memref<1xindex>) -> memref<60xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<3x4x5xi32>, memref<1xindex>) -> memref<60xi32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_60_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<3x4x5xi32>, memref<1xindex>) -> memref<60xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 60){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]{{.}} : memref<60xi32>, vector<32xi32>
// CHECK-DAG:         [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]{{.}} : memref<60xi32>, vector<32xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xi32>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.muli [[LOAD_VAR_reshape_2_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xi32>
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xi1>, vector<32xi32>
// CHECK:             vector.store [[VAR_6_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]{{.}} : memref<60xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<3x4x5xi32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
func.func @test_prelu_broadcast1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_broadcast1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#2] : memref<5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func.func @test_prelu_broadcast2(%arg0: tensor<3x4x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_broadcast2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<1x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2] : memref<1x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// case where only the lowest 2 dims are splatted.

func.func @add_partial_splat(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3x1x1xf32>) -> tensor<*xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x1x1xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_partial_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x1x1xf32>) -> memref<2x3x4x5xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_3_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_20_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<2x3x4x5xf32>, memref<3xindex>) -> memref<2x3x20xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_3_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_2_]]) : (memref<3x1x1xf32>, memref<2xindex>) -> memref<3x1xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_3_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_3_]], [[RES_3_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_20_]], [[RES_3_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_]]([[RES_]]_3) : (memref<2x3x4x5xf32>, memref<3xindex>) -> memref<2x3x20xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 20 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 20){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<2x3x20xf32>, vector<20xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_2_MEM_:%.+]] = krnl.load [[VAR_reshape_2_]]{{.}}[[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<3x1xf32>
// CHECK:               [[VAR_6_:%.+]] = vector.splat [[LOAD_VAR_reshape_2_MEM_]] : vector<20xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_6_]] : vector<20xf32>
// CHECK:               vector.store [[VAR_7_]], [[VAR_reshape_4_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<2x3x20xf32>, vector<20xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5xf32>
// CHECK:         }
}

// -----

// Test onnx.Erf lowering from onnx to kerneL

func.func @test_erf(%arg0: tensor<?x10xf32>) -> (tensor<*xf32>) {
    %0 = "onnx.Erf"(%arg0): (tensor<?x10xf32>) -> (tensor<*xf32>)
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func @test_erf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.erf [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.tanh [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_sinh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.subf [[VAR_cst_0_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = math.exp [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = math.exp [[VAR_6_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : vector<32xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_10_]], [[VAR_reshape_4_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_cosh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.subf [[VAR_cst_0_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = math.exp [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = math.exp [[VAR_6_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.addf [[VAR_7_]], [[VAR_8_]] : vector<32xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_10_]], [[VAR_reshape_4_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_cos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.cos [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sin(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.sin [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.log [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.subf [[VAR_cst_0_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = math.exp [[VAR_6_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[VAR_7_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_8_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_9_]], [[VAR_reshape_4_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.maxnumf [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_elu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_5_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_6_:%.+]] = math.exp [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.cmpf olt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.subf [[VAR_6_]], [[VAR_cst_0_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[VAR_8_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.select [[VAR_7_]], [[VAR_9_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xi1>, vector<32xf32>
// CHECK:               vector.store [[VAR_10_]], [[VAR_reshape_5_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_leakyrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_selu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_5_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_6_:%.+]] = math.exp [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.subf [[VAR_6_]], [[VAR_cst_0_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.select [[VAR_7_]], [[LOAD_VAR_reshape_MEM_]], [[VAR_8_]] : vector<32xi1>, vector<32xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_10_]], [[VAR_reshape_5_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_hardsigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_5_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.maxnumf [[VAR_6_]], [[VAR_cst_1_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.minnumf [[VAR_7_]], [[VAR_cst_0_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_8_]], [[VAR_reshape_5_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.divf [[VAR_cst_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_softplus
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.exp [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.addf [[VAR_6_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = math.log [[VAR_7_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_8_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softsign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_softsign
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.absf [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.addf [[VAR_6_]], [[VAR_cst_]] : vector<32xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_]], [[VAR_7_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_8_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.sqrt [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_sign_f(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_sign_f
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_5_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_cst_0_]], [[VAR_cst_]] : vector<32xi1>, vector<32xf32>
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.cmpf oeq, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_cst_1_]], [[VAR_7_]] : vector<32xi1>, vector<32xf32>
// CHECK:               vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_sign_i
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xi32>) -> memref<?x10xi32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-1> : vector<32xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<1> : vector<32xi32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0> : vector<32xi32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xi32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xi32>, memref<1xindex>) -> memref<?xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_5_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xi32>, memref<1xindex>) -> memref<?xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xi32>, vector<32xi32>
// CHECK:               [[VAR_6_:%.+]] = arith.cmpi sgt, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xi32>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_cst_0_]], [[VAR_cst_]] : vector<32xi1>, vector<32xi32>
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.cmpi eq, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_1_]] : vector<32xi32>
// CHECK:               [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_cst_1_]], [[VAR_7_]] : vector<32xi1>, vector<32xi32>
// CHECK:               vector.store [[VAR_9_]], [[VAR_reshape_5_]]{{.}}[[VAR_4_]]{{.}} : memref<?xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xi32>
// CHECK:         }
}

// -----


func.func private @test_abs_float(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_abs_float
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.absf [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_abs_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xi32>) -> memref<?x10xi32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xi32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xi32>, memref<1xindex>) -> memref<?xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xi32>, memref<1xindex>) -> memref<?xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xi32>, vector<32xi32>
// CHECK:               [[VAR_6_:%.+]] = math.absi [[LOAD_VAR_reshape_MEM_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xi32>
// CHECK:         }
}

// -----


func.func private @test_floor(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.floor [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func private @test_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]{{.}}){
// CHECK:               [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = math.ceil [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_clip(%arg0: tensor<128xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<128xf32> {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<128xf32>, tensor<f32>, tensor<f32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<f32>) -> memref<128xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = vector.splat [[LOAD_PARAM_1_MEM_]] : vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = vector.splat [[LOAD_PARAM_2_MEM_]] : vector<32xf32>
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.maxnumf [[VAR_4_]], [[LOAD_PARAM_0_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.minnumf [[VAR_6_]], [[VAR_7_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128xf32>
// CHECK:         }
}

// -----


func.func private @test_clip_default_min(%arg0: tensor<128xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<128xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<128xf32>, none, tensor<f32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_clip_default_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<f32>) -> memref<128xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_4_:%.+]] = vector.splat [[LOAD_PARAM_2_MEM_]] : vector<32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.minnumf [[VAR_4_]], [[LOAD_PARAM_0_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128xf32>
// CHECK:         }
}

// -----


func.func private @test_clip_default_min_int(%arg0: tensor<128xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<128xi32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<128xi32>, none, tensor<i32>) -> tensor<128xi32>
  return %0 : tensor<128xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_clip_default_min_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128xi32>, [[PARAM_1_:%.+]]: memref<i32>, [[PARAM_2_:%.+]]: memref<i32>) -> memref<128xi32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<128xi32>, vector<32xi32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i32>
// CHECK:             [[VAR_4_:%.+]] = vector.splat [[LOAD_PARAM_2_MEM_]] : vector<32xi32>
// CHECK:             [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[LOAD_PARAM_0_MEM_]] : vector<32xi32>
// CHECK:             vector.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<128xi32>, vector<32xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128xi32>
// CHECK:         }
}

