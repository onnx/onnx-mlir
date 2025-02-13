// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --march=z16 to enable SIMD as we now need a machine
// can also use --march=x86-64 instead.

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----

func.func @test_reduce_scalar_axes(%arg0: tensor<?x64x?xf32>) -> tensor<?x?xf32> {
  %axes= onnx.Constant dense<-2> : tensor<i64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x64x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-LABEL:  func.func @test_reduce_scalar_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x64x?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x64x?xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x64x?xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 64, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_1_]], [[VAR_dim_2_]]{{.}}){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x64x?xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<?x?xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----

// COM: Full reduction over all dimensions to a scalar value.
func.func @test_reduce_all_to_scalar(%arg0: tensor<?x64x?xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMax"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x64x?xf32>, none) -> tensor<*xf32>
  return %0: tensor<*xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-LABEL:  func.func @test_reduce_all_to_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x64x?xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x64x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[VAR_dim_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_]]) : (memref<?x64x?xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_8_]], [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_1_MEM_1_]] : vector<32xf32> into f32
// CHECK:           krnl.store [[VAR_4_]], [[RES_2_]][] : memref<f32>
// CHECK:           return [[RES_2_]] : memref<f32>
// CHECK:         }
}

// -----

func.func private @test_reducemax_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemin_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemin_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reduceprod_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProdV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reduceprod_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_1_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.mulf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %cst = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesumV11(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesumV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_reducesum1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_7_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<3x1x2xf32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_4_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.select [[VAR_4_1_]], [[CST_0_1_]], [[VAR_2_1_]]#0 : index
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi eq, [[VAR_6_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_1_]], [[CST_0_1_]], [[VAR_2_1_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[CST_0_1_]], [[VAR_2_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----


func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reducesum2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq, [[VAR_dim_]], [[CST_0_1_]] : index
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<3x1x2xf32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_5_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.select [[VAR_5_1_]], [[CST_0_1_]], [[VAR_3_1_]]#0 : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi eq, [[VAR_7_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_1_]], [[CST_0_1_]], [[VAR_3_1_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[VAR_3_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

// Original gpt2 reduction

func.func private @gpt2_original(%arg0 : tensor<?x?x768xf32>) -> tensor<?x?x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
  "func.return"(%0) : (tensor<?x?x1xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @gpt2_original
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>) -> memref<?x?x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x1xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK:           [[VAR_3_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_2_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_0_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_3) : (memref<?x?x1xf32>, memref<2xindex>) -> memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_9_]] {
// CHECK:               scf.for [[I_2_:%.+]] = [[VAR_7_]]#1 to [[VAR_dim_0_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 768){
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_14_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_17_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_12_:%.+]] = vector.reduction <add>, [[LOAD_RES_2_MEM_1_]] : vector<4xf32> into f32
// CHECK:                 [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[VAR_5_]] : f32
// CHECK:                 krnl.store [[VAR_13_]], [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]]#1)
// CHECK:               vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 768 step 4 {
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_2_MEM_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_12_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_4_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_2_MEM_5_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addf [[LOAD_RES_2_MEM_2_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.addf [[LOAD_RES_2_MEM_3_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.addf [[LOAD_RES_2_MEM_4_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_2_MEM_5_]], [[LOAD_PARAM_0_MEM_4_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_36_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_37_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_38_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_39_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_2_MEM_6_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_7_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_8_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_9_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_1_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_18_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addf [[VAR_18_]], [[VAR_17_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_20_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_8_]], [[LOAD_RES_2_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_8_]], [[LOAD_RES_2_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.addf [[VAR_24_]], [[VAR_23_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = vector.splat [[VAR_5_]] : vector<4xf32>
// CHECK:               [[VAR_27_:%.+]] = arith.divf [[VAR_25_]], [[VAR_26_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<?x?xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x1xf32>
// CHECK:         }
}

// -----

// reduction from GPT2 but with keepdims = 0

func.func private @gpt2_no_keepdims(%arg0 : tensor<?x?x768xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1], keepdims = 0 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<?x?x768xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @gpt2_no_keepdims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK:           [[VAR_3_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_2_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_9_]] {
// CHECK:               scf.for [[I_2_:%.+]] = [[VAR_7_]]#1 to [[VAR_dim_0_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 768){
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_14_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_17_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_12_:%.+]] = vector.reduction <add>, [[LOAD_RES_1_MEM_1_]] : vector<4xf32> into f32
// CHECK:                 [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[VAR_5_]] : f32
// CHECK:                 krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]]#1)
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 768 step 4 {
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_1_MEM_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_12_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_3_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_4_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_5_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addf [[LOAD_RES_1_MEM_2_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.addf [[LOAD_RES_1_MEM_3_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.addf [[LOAD_RES_1_MEM_4_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_1_MEM_5_]], [[LOAD_PARAM_0_MEM_4_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_36_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_37_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_38_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_39_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_1_MEM_6_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_7_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_8_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_9_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_1_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_18_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addf [[VAR_18_]], [[VAR_17_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_20_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_8_]], [[LOAD_RES_1_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_8_]], [[LOAD_RES_1_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.addf [[VAR_24_]], [[VAR_23_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = vector.splat [[VAR_5_]] : vector<4xf32>
// CHECK:               [[VAR_27_:%.+]] = arith.divf [[VAR_25_]], [[VAR_26_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<?x?xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----

// test flattening of 2 dims

func.func private @gpt2_reduce2(%arg0 : tensor<?x?x96x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1, -2], keepdims = 1 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<?x?x96x8xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @gpt2_reduce2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x?x1x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x1x1xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_768_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK:           [[VAR_3_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_2_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_768_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x96x8xf32>, memref<3xindex>) -> memref<?x?x768xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_0_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_]]([[RES_]]_6) : (memref<?x?x1x1xf32>, memref<2xindex>) -> memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_9_]] {
// CHECK:               scf.for [[I_2_:%.+]] = [[VAR_7_]]#1 to [[VAR_dim_0_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 768){
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_14_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_3_MEM_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_17_:%.+]] = arith.addf [[LOAD_RES_3_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_3_MEM_1_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_12_:%.+]] = vector.reduction <add>, [[LOAD_RES_3_MEM_1_]] : vector<4xf32> into f32
// CHECK:                 [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[VAR_5_]] : f32
// CHECK:                 krnl.store [[VAR_13_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOAD_RES_3_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]]#1)
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 768 step 4 {
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_3_MEM_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_12_1_]], [[I_4_]]{{.}} : memref<?x?x768xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_2_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_3_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_4_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_5_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addf [[LOAD_RES_3_MEM_2_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.addf [[LOAD_RES_3_MEM_3_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.addf [[LOAD_RES_3_MEM_4_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_3_MEM_5_]], [[LOAD_VAR_reshape_MEM_4_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_36_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_37_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_38_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_39_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_3_MEM_6_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_7_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_8_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_9_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_6_]], [[LOAD_RES_3_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_18_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_6_]], [[LOAD_RES_3_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addf [[VAR_18_]], [[VAR_17_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_20_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.addf [[VAR_24_]], [[VAR_23_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = vector.splat [[VAR_5_]] : vector<4xf32>
// CHECK:               [[VAR_27_:%.+]] = arith.divf [[VAR_25_]], [[VAR_26_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<?x?xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x1x1xf32>
// CHECK:         }
}

// -----

// test flattening of 2 dims, one that is not a multiple of VL

func.func private @gpt2_one_not_multiple(%arg0 : tensor<?x?x97x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1, -2], keepdims = 1 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<?x?x97x8xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @gpt2_one_not_multiple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x97x8xf32>) -> memref<?x?x1x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_773_:%.+]] = arith.constant 773 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_776_:%.+]] = arith.constant 776 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x1x1xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x8xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x8xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_776_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK:           [[VAR_3_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_2_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x8xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x8xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_776_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x97x8xf32>, memref<3xindex>) -> memref<?x?x776xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_0_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_]]([[RES_]]_6) : (memref<?x?x1x1xf32>, memref<2xindex>) -> memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_9_]] {
// CHECK:               scf.for [[I_2_:%.+]] = [[VAR_7_]]#1 to [[VAR_dim_0_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 773){
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_14_]]{{.}} : memref<?x?x776xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_3_MEM_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_17_:%.+]] = arith.addf [[LOAD_RES_3_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_17_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_3_MEM_1_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_12_:%.+]] = vector.reduction <add>, [[LOAD_RES_3_MEM_1_]] : vector<4xf32> into f32
// CHECK:                 [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[VAR_5_]] : f32
// CHECK:                 krnl.store [[VAR_13_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOAD_RES_3_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]]#1)
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               scf.for [[I_4_:%.+]] = [[CST_0_]] to [[CST_773_]] step [[CST_4_]] {
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_4_]]{{.}} : memref<?x?x776xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[I_4_]]{{.}} : memref<?x?x776xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_3_MEM_1_]], [[I_4_]]{{.}} : memref<?x?x776xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_12_1_]], [[I_4_]]{{.}} : memref<?x?x776xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_2_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_3_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_4_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_5_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addf [[LOAD_RES_3_MEM_2_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.addf [[LOAD_RES_3_MEM_3_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.addf [[LOAD_RES_3_MEM_4_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_3_MEM_5_]], [[LOAD_VAR_reshape_MEM_4_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_36_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_37_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_38_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_39_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_3_MEM_6_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_7_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_8_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_9_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_6_]], [[LOAD_RES_3_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_18_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_6_]], [[LOAD_RES_3_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addf [[VAR_18_]], [[VAR_17_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_20_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = vector.shuffle [[VAR_19_]], [[VAR_22_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.addf [[VAR_24_]], [[VAR_23_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = vector.splat [[VAR_5_]] : vector<4xf32>
// CHECK:               [[VAR_27_:%.+]] = arith.divf [[VAR_25_]], [[VAR_26_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<?x?xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x1x1xf32>
// CHECK:         }
}

// -----

// test flattening of 2 dims, , neither that are a multiple of VL, disabling SIMD

func.func private @gpt2_no_simd_as_not_mult_of_VL(%arg0 : tensor<?x?x97x9xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1, -2], keepdims = 1 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<?x?x97x9xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @gpt2_no_simd_as_not_mult_of_VL
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x97x9xf32>) -> memref<?x?x1x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_872_:%.+]] = arith.constant 872 : index
// CHECK-DAG:       [[CST_870_:%.+]] = arith.constant 870 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_873_:%.+]] = arith.constant 873 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x9xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x9xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x1x1xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x9xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x9xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_2_]] : index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_873_]] : index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_]]_0 : index
// CHECK:           [[VAR_3_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_2_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x97x9xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x97x9xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[VAR_dim_3_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[VAR_dim_4_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_873_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x?x97x9xf32>, memref<3xindex>) -> memref<?x?x873xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[VAR_dim_0_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_]]([[RES_]]_6) : (memref<?x?x1x1xf32>, memref<2xindex>) -> memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_9_]] {
// CHECK:               scf.for [[I_2_:%.+]] = [[VAR_7_]]#1 to [[VAR_dim_0_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 870){
// CHECK:                   [[VAR_15_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_15_]]{{.}} : memref<?x?x873xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_3_MEM_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_18_:%.+]] = arith.addf [[LOAD_RES_3_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_18_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 872 to 873){
// CHECK:                   [[VAR_15_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[I_2_]], [[VAR_15_1_]]{{.}} : memref<?x?x873xf32>
// CHECK-DAG:               [[LOAD_RES_3_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:                   [[VAR_18_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : f32
// CHECK:                   krnl.store [[VAR_18_1_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_3_MEM_2_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_13_:%.+]] = vector.reduction <add>, [[LOAD_RES_3_MEM_2_]] : vector<4xf32> into f32
// CHECK:                 [[VAR_14_:%.+]] = arith.divf [[VAR_13_]], [[VAR_5_]] : f32
// CHECK:                 krnl.store [[VAR_14_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1)
// CHECK-DAG:           [[LOAD_RES_3_MEM_2_:%.+]] = affine.apply [[MAP_4_]]([[VAR_7_]]#1)
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               scf.for [[I_5_:%.+]] = [[CST_0_]] to [[CST_870_]] step [[CST_4_]] {
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_5_]]{{.}} : memref<?x?x873xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[I_5_]]{{.}} : memref<?x?x873xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_2_]], [[I_5_]]{{.}} : memref<?x?x873xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_reshape_MEM_5_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_3_MEM_2_]], [[I_5_]]{{.}} : memref<?x?x873xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_3_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_4_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_5_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_3_MEM_6_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.addf [[LOAD_RES_3_MEM_3_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.addf [[LOAD_RES_3_MEM_4_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.addf [[LOAD_RES_3_MEM_5_]], [[LOAD_VAR_reshape_MEM_4_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.addf [[LOAD_RES_3_MEM_6_]], [[LOAD_VAR_reshape_MEM_5_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_48_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_49_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_50_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_51_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_6_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[CST_872_]]{{.}} : memref<?x?x873xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_7_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_1_]], [[CST_872_]]{{.}} : memref<?x?x873xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_8_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOOP_2_]], [[CST_872_]]{{.}} : memref<?x?x873xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_9_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[LOAD_RES_3_MEM_2_]], [[CST_872_]]{{.}} : memref<?x?x873xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_7_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_8_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_9_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_10_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.addf [[LOAD_RES_3_MEM_7_]], [[LOAD_VAR_reshape_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_3_MEM_8_]], [[LOAD_VAR_reshape_MEM_7_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.addf [[LOAD_RES_3_MEM_9_]], [[LOAD_VAR_reshape_MEM_8_]] : f32
// CHECK-DAG:           [[VAR_24_:%.+]] = arith.addf [[LOAD_RES_3_MEM_10_]], [[LOAD_VAR_reshape_MEM_9_]] : f32
// CHECK:               memref.store [[VAR_21_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:               memref.store [[VAR_22_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:               memref.store [[VAR_23_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:               memref.store [[VAR_24_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_11_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_12_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_13_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_14_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_11_]], [[LOAD_RES_3_MEM_12_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_30_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_11_]], [[LOAD_RES_3_MEM_12_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.addf [[VAR_30_]], [[VAR_29_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_13_]], [[LOAD_RES_3_MEM_14_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_13_]], [[LOAD_RES_3_MEM_14_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_34_:%.+]] = arith.addf [[VAR_33_]], [[VAR_32_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = vector.shuffle [[VAR_31_]], [[VAR_34_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_36_:%.+]] = vector.shuffle [[VAR_31_]], [[VAR_34_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.addf [[VAR_36_]], [[VAR_35_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_38_:%.+]] = vector.splat [[VAR_5_]] : vector<4xf32>
// CHECK:               [[VAR_39_:%.+]] = arith.divf [[VAR_37_]], [[VAR_38_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reshape_7_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<?x?xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x1x1xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemax_v13_bis(%arg0 : tensor<1028x256xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[-1], keepdims = 0 : si64} : (tensor<1028x256xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @test_reducemax_v13_bis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1028x256xf32>) -> memref<1028xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0xFF800000> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1028xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1028){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK:             vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 256 step 4 {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[I_1_]]{{.}} : memref<1028x256xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[I_1_]]{{.}} : memref<1028x256xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[I_1_]]{{.}} : memref<1028x256xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[I_1_]]{{.}} : memref<1028x256xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_3_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_2_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_3_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_26_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_28_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_29_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_4_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_5_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_6_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_7_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_5_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_4_]], [[LOAD_RES_1_MEM_5_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.maxnumf [[VAR_10_]], [[VAR_9_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.maxnumf [[VAR_13_]], [[VAR_12_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = vector.shuffle [[VAR_11_]], [[VAR_14_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.shuffle [[VAR_11_]], [[VAR_14_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_17_:%.+]] = arith.maxnumf [[VAR_16_]], [[VAR_15_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_17_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<1028xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1028xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemax_v13_small(%arg0 : tensor<7x8xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[-1], keepdims = 0 : si64} : (tensor<7x8xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @test_reducemax_v13_small
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x8xf32>) -> memref<7xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0xFF800000> : vector<4xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<7xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]])
// CHECK:             [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_2_]], [[CST_0_]] : index
// CHECK:             scf.if [[VAR_3_]] {
// CHECK:               scf.for [[I_1_:%.+]] = [[VAR_1_]] to [[CST_7_]] step [[CST_1_]] {
// CHECK:                 vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:                 krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 8){
// CHECK:                   [[VAR_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[I_1_]], [[VAR_7_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_10_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_10_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 [[VAR_6_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_1_MEM_1_]] : vector<4xf32> into f32
// CHECK:                 krnl.store [[VAR_6_]], [[RES_]]{{.}}[[I_1_]]{{.}} : memref<7xf32>
// CHECK:               }
// CHECK:             } else {
// CHECK-DAG:           [[LOOP_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]])
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]])
// CHECK-DAG:           [[VAR_6_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]])
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               affine.for [[I_3_:%.+]] = 0 to 8 step 4 {
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[I_3_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOOP_1_]], [[I_3_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOAD_RES_1_MEM_1_]], [[I_3_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_1_]], [[I_3_]]{{.}} : memref<7x8xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_3_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_4_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_5_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_2_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_29_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_3_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_4_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_5_]], [[LOAD_PARAM_0_MEM_4_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_28_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_29_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_30_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                 vector.store [[VAR_31_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_1_MEM_6_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_7_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_8_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_9_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_11_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_12_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_6_]], [[LOAD_RES_1_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.maxnumf [[VAR_12_]], [[VAR_11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_14_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_8_]], [[LOAD_RES_1_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_15_:%.+]] = vector.shuffle [[LOAD_RES_1_MEM_8_]], [[LOAD_RES_1_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_16_:%.+]] = arith.maxnumf [[VAR_15_]], [[VAR_14_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_17_:%.+]] = vector.shuffle [[VAR_13_]], [[VAR_16_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_18_:%.+]] = vector.shuffle [[VAR_13_]], [[VAR_16_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_19_:%.+]] = arith.maxnumf [[VAR_18_]], [[VAR_17_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<7xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<7xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemax_int_v13(%arg0 : tensor<128x256x768xi32>) -> tensor<*xi32> {
  %0 = "onnx.ReduceMaxV13"(%arg0) {axes = [-1], keepdims = 0 : si64, onnx_node_name = "ReduceMean_32"} : (tensor<128x256x768xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemax_int_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128x256x768xi32>) -> memref<128x256xi32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-2147483648> : vector<32xi32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128x256xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 128, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x32xi32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x32xi32>, vector<32xi32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 768){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_5_]]{{.}} : memref<128x256x768xi32>, vector<32xi32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x32xi32>, vector<32xi32>
// CHECK:               [[VAR_8_:%.+]] = arith.maxsi [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_8_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x32xi32>, vector<32xi32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x32xi32>, vector<32xi32>
// CHECK:             [[VAR_4_:%.+]] = vector.reduction <maxsi>, [[LOAD_RES_1_MEM_1_]] : vector<32xi32> into i32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<128x256xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128x256xi32>
// CHECK:         }
}

// -----

// model below created issue as the loop pattern was simplified and exposed an issue with KrnlToAffine lowering

func.func private @bertsquad10_same_pattern(%arg0 : tensor<?x256x768xf32>) -> tensor<?x256x1xf32> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2], keepdims = 1 : si64, onnx_node_name = "bert/embeddings/LayerNorm/moments/mean"} :
      (tensor<?x256x768xf32>) -> tensor<?x256x1xf32>
  "func.return"(%0) : (tensor<?x256x1xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 196608)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @bertsquad10_same_pattern
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x256x768xf32>) -> memref<?x256x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x768xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x256x1xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[VAR_2_:%.+]] = arith.floordivsi [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[VAR_dim_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<?x256x1xf32>, memref<2xindex>) -> memref<?x256xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_2_]]([[VAR_6_]]#1)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_6_]]#1)
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.apply [[MAP_4_]]([[VAR_6_]]#1)
// CHECK:             vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 768 step 4 {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[I_2_]]{{.}} : memref<?x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_7_]], [[I_2_]]{{.}} : memref<?x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_8_]], [[I_2_]]{{.}} : memref<?x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_9_]], [[I_2_]]{{.}} : memref<?x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addf [[LOAD_RES_2_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.addf [[LOAD_RES_2_MEM_2_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.addf [[LOAD_RES_2_MEM_3_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_33_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_34_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_35_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_36_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_4_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_5_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_6_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_7_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addf [[VAR_15_]], [[VAR_14_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_19_:%.+]] = arith.addf [[VAR_18_]], [[VAR_17_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = vector.shuffle [[VAR_16_]], [[VAR_19_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = vector.shuffle [[VAR_16_]], [[VAR_19_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = vector.splat [[VAR_4_]] : vector<4xf32>
// CHECK:             [[VAR_24_:%.+]] = arith.divf [[VAR_22_]], [[VAR_23_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_24_]], [[VAR_reshape_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<?x256xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x256x1xf32>
// CHECK:         }
}

// -----

// Same pattern as above, but with fully constant, just to add more checks.

func.func private @bertsquad10_const_pattern(%arg0 : tensor<1x256x768xf32>) -> tensor<1x256x1xf32> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2], keepdims = 1 : si64, onnx_node_name = "bert/embeddings/LayerNorm/moments/mean"} :
      (tensor<1x256x768xf32>) -> tensor<1x256x1xf32>
  "func.return"(%0) : (tensor<1x256x1xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @bertsquad10_const_pattern
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x256x768xf32>) -> memref<1x256x1xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<7.680000e+02> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x256x1xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<1x256x1xf32>, memref<2xindex>) -> memref<1x256xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1)
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 768 step 4 {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[I_2_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_2_]], [[I_2_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[I_2_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_4_]], [[I_2_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.addf [[LOAD_RES_2_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addf [[LOAD_RES_2_MEM_2_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.addf [[LOAD_RES_2_MEM_3_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_27_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_28_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_29_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_30_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_2_MEM_4_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_5_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_6_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_7_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[VAR_9_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[VAR_13_]], [[VAR_12_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = vector.shuffle [[VAR_11_]], [[VAR_14_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.shuffle [[VAR_11_]], [[VAR_14_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_17_:%.+]] = arith.addf [[VAR_16_]], [[VAR_15_]] : vector<4xf32>
// CHECK:             [[VAR_18_:%.+]] = arith.divf [[VAR_17_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_18_]], [[VAR_reshape_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<1x256xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x256x1xf32>
// CHECK:         }
}

