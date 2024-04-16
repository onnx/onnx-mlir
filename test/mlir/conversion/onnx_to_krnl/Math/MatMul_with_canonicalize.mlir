// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----

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
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[RES_1_]]#4) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
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
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[RES_1_]]#4) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<2x3x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// 1-D x 2-D

func.func private @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"0":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul4
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
// CHECK-DAG:       [[RES_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[RES_1_]]#0) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#0) : (!krnl.loop) -> index
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[RES_1_]]#1) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#1) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_3_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]], [[VAR_1_]]{{.}} : memref<5x10xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10xf32>
// CHECK:         }
}

// -----

// 1-D x N-D

func.func private @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"1":"RES"}'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_matmul5
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<?x5x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[B_]], [[CST_0_]] : memref<?x5x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[RES_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[LOOP_0_]]#2) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_3_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[RES_1_]]#0, [[VAR_3_]], [[RES_1_]]#1] : memref<?x5x10xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[RES_1_]]#0, [[RES_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// N-D x 1-D

func.func private @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"1":"RES"}'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_matmul6
// CHECK-SAME:   ([[A_:%.+]]: memref<?x10x5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[A_]], [[CST_0_]] : memref<?x10x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[RES_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[LOOP_0_]]#2) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[RES_1_]]#0, [[RES_1_]]#1, [[VAR_3_]]{{.}} : memref<?x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[IterArg]], [[VAR_7_]] : f32
// CHECK:               krnl.yield [[VAR_8_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]]{{.}}[[RES_1_]]#0, [[RES_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// 1-D x 1-D results in scalar

func.func private @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"1":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul7
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<f32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate() with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 5){
// CHECK:             [[IterResult:%.+]] = krnl.iterate([[LOOP_0_]]) with () iter_args([[IterArg:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK:               [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_6_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_7_:%.+]] = arith.addf [[IterArg]], [[VAR_6_]] : f32
// CHECK:               krnl.yield [[VAR_7_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[IterResult]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
}

