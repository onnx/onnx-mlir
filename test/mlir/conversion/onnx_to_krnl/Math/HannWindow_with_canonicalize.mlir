// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_hannwindow(%arg0 : tensor<i32>) -> tensor<?xf32> {
  %0 = "onnx.HannWindow"(%arg0) {output_datatype = 1 : si64 , periodic = 0 : si64} : (tensor<i32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_hannwindow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i32>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_6_dot_28318548_:%.+]] = arith.constant 6.28318548 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i32 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_]] : i32 to f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.subf [[VAR_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_1_]])){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK:             [[VAR_7_:%.+]] = arith.sitofp [[VAR_6_]] : i64 to f32
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[VAR_7_]], [[CST_6_dot_28318548_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.divf [[VAR_8_]], [[VAR_3_]] : f32
// CHECK:             [[VAR_10_:%.+]] = math.cos [[VAR_9_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_12_:%.+]] = arith.mulf [[VAR_11_]], [[CST_5_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
}