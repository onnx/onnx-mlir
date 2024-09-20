// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// -----

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized)
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.
// Test parallelization of Relu

func.func @test_relu_parallel(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1, s2] -> (s2)>
// CHECK-LABEL:  func.func @test_relu_parallel
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
// CHECK:             krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
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

