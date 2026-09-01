// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// Check the lowering of EyeLike with static shape and default k (main diagonal).
func.func private @test_eyelike_static(%arg0 : tensor<4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.EyeLike"(%arg0) : (tensor<4x4xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func private @test_eyelike_static
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x4xf32>) -> memref<4x4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_2_:%.+]] = arith.subi [[VAR_1_]]#1, [[VAR_1_]]#0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_dot_000000_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x4xf32>
// CHECK:         }
}

// -----

// Check the lowering of EyeLike with a non-zero k attribute and a dtype conversion.
func.func private @test_eyelike_k_dtype(%arg0 : tensor<3x4xi32>) -> tensor<*xf64> {
  %0 = "onnx.EyeLike"(%arg0) {k = 1 : si64, dtype = 11 : si64} : (tensor<3x4xi32>) -> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()

// CHECK-LABEL:  func.func private @test_eyelike_k_dtype
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xi32>) -> memref<3x4xf64> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4xf64>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_2_:%.+]] = arith.subi [[VAR_1_]]#1, [[VAR_1_]]#0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_dot_000000_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<3x4xf64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4xf64>
// CHECK:         }
}

// -----

// Check the lowering of EyeLike with dynamic shape.
func.func private @test_eyelike_dynamic(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.EyeLike"(%arg0) : (tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func private @test_eyelike_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_2_:%.+]] = arith.subi [[VAR_1_]]#1, [[VAR_1_]]#0 : index
// CHECK:             [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_0_1_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_dot_000000_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}
