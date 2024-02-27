// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized) 
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

// Test parallelization of Transpose
func.func @test_transpose_block_1_last_dim_parallel(%arg0: tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3] } : (tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32>
    return %1 : tensor<?x12x256x64xf32>

    // mlir2FileCheck.py
    // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
    // CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 768 + d2 * 64)>
    // CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 64 + d2 * 16384)>
    // CHECK-LABEL:  func.func @test_transpose_block_1_last_dim_parallel
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x256x12x64xf32>) -> memref<?x12x256x64xf32> {
    // CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
    // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
    // CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
    // CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x12x256x64xf32>
    // CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
    // CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
    // CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
    // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12){
    // CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    // CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
    // CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
    // CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_3_]], [[VAR_2_]]) : (memref<?x12x256x64xf32>, memref<?x256x12x64xf32>, i64, index, index) -> ()
    // CHECK:           }
    // CHECK:           return [[RES_]] : memref<?x12x256x64xf32>
    // CHECK:         }
}
