// RUN: onnx-mlir-opt -O0 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck --check-prefix=CHECK-O0 %s

// -----
// COM: 2D matmul with no unrolling
func.func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py -a'["A", "B"]'
// CHECK-O0-LABEL:  func private @test_matmul1
// CHECK-O0-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-O0-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-O0-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK-O0-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-O0-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-O0:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 16){
// CHECK-O0:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-O0:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK-O0:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK-O0:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-O0-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_3_]]#0, [[VAR_5_]]{{.}} : memref<16x16xf32>
// CHECK-O0-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]], [[VAR_3_]]#1] : memref<16x16xf32>
// CHECK-O0-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK-O0:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK-O0:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK-O0:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK-O0:             }
// CHECK-O0:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK-O0:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x16xf32>
// CHECK-O0:           }
// CHECK-O0:           return [[RES_]] : memref<16x16xf32>
// CHECK-O0:         }
}
