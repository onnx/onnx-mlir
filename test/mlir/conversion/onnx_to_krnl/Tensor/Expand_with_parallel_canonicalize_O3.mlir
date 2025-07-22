// RUN: onnx-mlir-opt -O3 --march=z16 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized) 
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

// -----

func.func @expand_dyn(%arg0: tensor<1x8x1x?x64xf32>, %arg1: tensor<5xi64>) ->  tensor<1x8x4x?x64xf32> {
  %0 = "onnx.Expand"(%arg0, %arg1) : (tensor<1x8x1x?x64xf32>, tensor<5xi64>) -> tensor<1x8x4x?x64xf32>
  return %0 : tensor<1x8x4x?x64xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-LABEL:  func.func @expand_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x8x1x?x64xf32>, [[PARAM_1_:%.+]]: memref<5xi64>) -> memref<1x8x4x?x64xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x8x1x?x64xf32>
// CHECK:           [[VAR_2_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<1x8x4x?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[VAR_2_]], [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_4_:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index)
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_4_]]#3, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]], [[VAR_4_]]#1, [[CST_0_]], [[VAR_6_]], [[VAR_4_]]#4] : memref<1x8x1x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2, [[VAR_4_]]#3, [[VAR_4_]]#4] : memref<1x8x4x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x8x4x?x64xf32>
// CHECK:         }
}
