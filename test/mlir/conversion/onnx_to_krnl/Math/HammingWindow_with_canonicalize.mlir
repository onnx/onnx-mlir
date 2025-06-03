// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s


func.func private @test_hammingwindow(%arg0 : tensor<i32>) -> tensor<?xf32> {
  %0 = "onnx.HammingWindow"(%arg0) {output_datatype = 1 : si64 , periodic = 1 : si64} : (tensor<i32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()
  
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_hammingwindow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i32>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_6_dot_283185_:%.+]] = arith.constant 6.283185 : f32
// CHECK-DAG:       [[CST_4_dot_565300_:%.+]] = arith.constant 4.565300e-01 : f32
// CHECK-DAG:       [[CST_5_dot_434700_:%.+]] = arith.constant 5.434700e-01 : f32
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i32 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_1_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK:             [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.mulf [[VAR_5_]], [[CST_6_dot_283185_]] : f32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_]] : i32 to f32
// CHECK:             [[VAR_8_:%.+]] = arith.divf [[VAR_6_]], [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = math.cos [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[CST_4_dot_565300_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[CST_5_dot_434700_]], [[VAR_10_]] : f32
// CHECK:             krnl.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
}
