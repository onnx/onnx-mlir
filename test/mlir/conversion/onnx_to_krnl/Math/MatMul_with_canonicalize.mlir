// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// COM: 2D matmul.
func.func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func private @test_matmul1
// CHECK-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]]{{.}} : memref<16x16xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]], [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x16xf32>
// CHECK:         }
}

// -----

// 2-D x N-D

func.func private @test_matmul2(%arg0 : tensor<10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"0":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul2
// CHECK-SAME:   ([[A_:%.+]]: memref<10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]#4) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// N-D x N-D

func.func private @test_matmul3(%arg0 : tensor<2x3x10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"0":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul3
// CHECK-SAME:   ([[A_:%.+]]: memref<2x3x10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]#4) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<2x3x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// 1-D x 2-D
func.func private @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"0": "RES"}'
// CHECK-LABEL:  func private @test_matmul4
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#0) : (!krnl.loop) -> index
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#1) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#1) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]], [[VAR_3_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10xf32>
// CHECK:         }
}

// -----

// 1-D x N-D
func.func private @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_matmul5
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<?x5x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[B_]], [[VAR_c0_]] : memref<?x5x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_4_]]#0, [[VAR_6_]], [[VAR_4_]]#1] : memref<?x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// N-D x 1-D
func.func private @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_matmul6
// CHECK-SAME:   ([[A_:%.+]]: memref<?x10x5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[A_]], [[VAR_c0_]] : memref<?x10x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_6_]]{{.}} : memref<?x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// 1-D x 1-D results in scalar
func.func private @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-LABEL:  func private @test_matmul7
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate() with ([[RES_1_]] -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[VAR_cst_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[RES_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
}
