// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_lpNormalization(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.LpNormalization"(%arg0) {axis = 0: si64, p = 2 : si64} : (tensor<10x20xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_lpNormalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32>) -> memref<10x20xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_200_:%.+]] = arith.constant 200 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<896xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}[] : memref<896xi8> to memref<10x20xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_200_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<10x20xf32>, memref<1xindex>) -> memref<200xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_200_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_2_]]) : (memref<10x20xf32>, memref<1xindex>) -> memref<200xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_200_]], [[RES_3_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[VAR_view_]]([[RES_3_]]) : (memref<10x20xf32>, memref<1xindex>) -> memref<200xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 200){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_3_]]{{.}} : memref<200xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[VAR_3_]]{{.}} : memref<200xf32>, vector<32xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_6_]], [[VAR_reshape_4_]]{{.}}[[VAR_3_]]{{.}} : memref<200xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x20xf32>
// CHECK:           krnl.memset [[RES_4_]], [[CST_0_dot_000000_]] : memref<1x20xf32>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 20){
// CHECK:             [[LOOP_0_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.load [[VAR_view_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<10x20xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_]], [[LOOP_0_]]#1] : memref<1x20xf32>
// CHECK:             [[LOAD_VAR_reshape_2_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_1_]], [[VAR_3_1_]] : f32
// CHECK:             krnl.store [[LOAD_VAR_reshape_2_MEM_1_]], [[RES_4_]]{{.}}[[CST_0_]], [[LOOP_0_]]#1] : memref<1x20xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1x20xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_20_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_8_:%.+]] = memref.reshape [[RES_4_]]([[RES_6_]]) : (memref<1x20xf32>, memref<1xindex>) -> memref<20xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_20_]], [[RES_7_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_10_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<1x20xf32>, memref<1xindex>) -> memref<20xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_2_]] 20 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 20){
// CHECK:               [[VAR_3_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_8_]]{{.}}[[VAR_3_2_]]{{.}} : memref<20xf32>, vector<20xf32>
// CHECK:               [[LOAD_VAR_reshape_2_MEM_1_:%.+]] = math.sqrt [[LOAD_VAR_reshape_MEM_1_]] : vector<20xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_2_MEM_1_]], [[VAR_reshape_10_]]{{.}}[[VAR_3_2_]]{{.}} : memref<20xf32>, vector<20xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 10){
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_4_]] 20 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_4_]] -> [[I_5_:%.+]] = 0 to 20){
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_2_MEM_1_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_VAR_reshape_MEM_1_1_]]{{.}} : memref<10x20xf32>, vector<20xf32>
// CHECK-DAG:           [[VAR_6_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]], [[LOAD_VAR_reshape_MEM_1_1_]]{{.}} : memref<1x20xf32>, vector<20xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.divf [[LOAD_VAR_reshape_2_MEM_1_1_]], [[VAR_6_1_]] : vector<20xf32>
// CHECK:               vector.store [[VAR_7_]], [[RES_8_]]{{.}}[[LOOP_2_]], [[LOAD_VAR_reshape_MEM_1_1_]]{{.}} : memref<10x20xf32>, vector<20xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[PARAM_0_]] : memref<10x20xf32>
// CHECK:         }
}