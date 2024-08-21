// RUN: onnx-mlir-opt -O0 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: 2D matmul with no unrolling

func.func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func private @test_matmul1
// CHECK-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 16){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[LOOP_0_]]#2) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]]{{.}} : memref<16x16xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]], [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x16xf32>
// CHECK:         }
}

