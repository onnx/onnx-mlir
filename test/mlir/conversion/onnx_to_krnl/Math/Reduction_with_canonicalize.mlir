// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.
/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of i32.

func.func @test_reducemean_v13_i32_unknown_dims(%arg0 : tensor<3x?x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_reducemean_v13_i32_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?x2xi32>) -> memref<3x2xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xi32>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xi32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.trunci [[VAR_0_]] : i64 to i32
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xi32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_dim_0_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<3x?x2xi32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#2] : memref<3x2xi32>
// CHECK:             [[VAR_7_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#2] : memref<3x2xi32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<3x2xi32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divsi [[LOAD_RES_MEM_1_]], [[VAR_1_]] : i32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xi32>
// CHECK:         }
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of f32.

func.func @test_reducemean_v13_f32_unknown_dims(%arg0 : tensor<3x?x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_reducemean_v13_f32_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.sitofp [[VAR_0_]] : i64 to f32
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_dim_0_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<3x?x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<3x2xf32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[VAR_1_]] : f32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----

/// Check ReduceMeanV13 with f32.

func.func private @test_reducemean_v13_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xf32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----

/// Check ReduceMeanV13 with i32.

func.func private @test_reducemean_v13_i32(%arg0 : tensor<3x2x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xi32>) -> memref<3x2xi32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xi32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xi32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<3x2x2xi32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xi32>
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#2] : memref<3x2xi32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xi32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divsi [[LOAD_RES_MEM_1_]], [[CST_2_]] : i32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xi32>
// CHECK:         }
}

// -----


func.func private @test_reducemax_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducemin_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemin_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %cst = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesumV11(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesumV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_reducesum1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_7_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<3x1x2xf32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_4_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.select [[VAR_4_1_]], [[CST_0_1_]], [[VAR_2_1_]]#0 : index
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi eq, [[VAR_6_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_1_]], [[CST_0_1_]], [[VAR_2_1_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[CST_0_1_]], [[VAR_2_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----


func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reducesum2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq, [[VAR_dim_]], [[CST_0_1_]] : index
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<3x1x2xf32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_5_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.select [[VAR_5_1_]], [[CST_0_1_]], [[VAR_3_1_]]#0 : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi eq, [[VAR_7_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_1_]], [[CST_0_1_]], [[VAR_3_1_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[VAR_3_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reduceprod_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProdV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reduceprod_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_1_dot_000000_]] : memref<3x2xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.mulf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

