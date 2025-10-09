// RUN: onnx-mlir-opt -O3 --march=z16 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

func.func @test_parallel_slice(%arg0 : tensor<1x32x?x64xf32>) -> tensor<1x32x?x32xf32> {
  %axes = onnx.Constant dense<[3]> : tensor<1xi64>
  %starts = onnx.Constant dense<[32]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9223372036854775807]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x32x?x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x32x?x32xf32>
  "func.return"(%1) : (tensor<1x32x?x32xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 32)>
// CHECK-LABEL:  func.func @test_parallel_slice
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x?x64xf32>) -> memref<1x32x?x32xf32> {
// CHECK:           [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x32x?x64xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<1x32x?x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#3)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<1x32x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x32x?x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x32x?x32xf32>
// CHECK:         }
}

