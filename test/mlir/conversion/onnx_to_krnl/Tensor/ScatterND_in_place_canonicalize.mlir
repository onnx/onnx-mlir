// RUN: onnx-mlir-opt --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Scattering into the buffer of `data` instead of copying it.

// `data` is a Relu result used only here: the updates are scattered into its
// buffer and there is no copy.
func.func @test_scatter_nd_in_place(%a: tensor<64x32xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}) {
  %d = "onnx.Relu"(%a) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>

// CHECK-LABEL:  func.func @test_scatter_nd_in_place
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x32xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x32xf32>
// CHECK:         }

}

// -----

// The same with a reduction: in place is independent of the reduction.
func.func @test_scatter_nd_in_place_add(%a: tensor<64x32xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}) {
  %d = "onnx.Relu"(%a) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) {reduction = "add"} : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>

// CHECK-LABEL:  func.func @test_scatter_nd_in_place_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x32xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<2x32xf32>
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x32xf32>
// CHECK:         }

}

// -----

// `data` is a function argument, owned by the caller: copy.
func.func @test_scatter_nd_no_in_place_arg(%a: tensor<64x32xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}) {
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%a, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>

// CHECK-LABEL:  func.func @test_scatter_nd_no_in_place_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x32xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_2048_]], [[CST_0_]], [[CST_0_]]) : (memref<64x32xf32>, memref<64x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x32xf32>
// CHECK:         }

}

// -----

// `data` is also returned, so its old values stay observable: copy.
func.func @test_scatter_nd_no_in_place_two_uses(%a: tensor<64x32xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}, tensor<64x32xf32> {onnx.name = "relu"}) {
  %d = "onnx.Relu"(%a) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0, %d : tensor<64x32xf32>, tensor<64x32xf32>

// CHECK-LABEL:  func.func @test_scatter_nd_no_in_place_two_uses
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x32xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}, memref<64x32xf32> {onnx.name = "relu"}) {
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK:           "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_2048_]], [[CST_0_]], [[CST_0_]]) : (memref<64x32xf32>, memref<64x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]], [[RES_]] : memref<64x32xf32>, memref<64x32xf32>
// CHECK:         }

}

// -----

// `data` is a constant, in read-only memory: copy.
func.func @test_scatter_nd_no_in_place_constant(%u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}) {
  %d = onnx.Constant dense<1.0> : tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>

// CHECK-LABEL:  func.func @test_scatter_nd_no_in_place_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [64, 32], value = dense<1.000000e+00> : tensor<64x32xf32>}> : () -> memref<64x32xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[VAR_0_]], [[CST_2048_]], [[CST_0_]], [[CST_0_]]) : (memref<64x32xf32>, memref<64x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_5_]], [[VAR_3_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x32xf32>
// CHECK:         }

}

// -----

// `data` is a view of another buffer (a Reshape of a Relu result): copy.
func.func @test_scatter_nd_no_in_place_view(%a: tensor<32x64xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}) {
  %s = onnx.Constant dense<[64, 32]> : tensor<2xi64>
  %r = "onnx.Relu"(%a) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  %d = "onnx.Reshape"(%r, %s) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>
// CHECK-LABEL:  func.func @test_scatter_nd_no_in_place_view
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<32x64xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<32x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 32, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 64){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<32x64xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<32x64xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<32x64xf32> to memref<64x32xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK:           "krnl.memcpy"([[RES_1_]], [[VAR_reinterpret_cast_]], [[CST_2048_]], [[CST_0_]], [[CST_0_]]) : (memref<64x32xf32>, memref<64x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<64x32xf32>
// CHECK:         }

}


// -----

// `data` is an Identity of a Relu result that Exp reads afterwards. Identity's
// lowering passes the Relu buffer through, so scattering into it would change
// what Exp reads: copy.
func.func @test_scatter_nd_no_in_place_pass_through(%a: tensor<64x32xf32> {onnx.name = "a"}, %u: tensor<2x32xf32> {onnx.name = "updates"}) -> (tensor<64x32xf32> {onnx.name = "output"}, tensor<64x32xf32> {onnx.name = "exp"}) {
  %x = "onnx.Relu"(%a) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  %d = "onnx.Identity"(%x) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  %i = onnx.Constant dense<[[3], [5]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%d, %i, %u) : (tensor<64x32xf32>, tensor<2x1xi64>, tensor<2x32xf32>) -> tensor<64x32xf32>
  %e = "onnx.Exp"(%x) : (tensor<64x32xf32>) -> tensor<64x32xf32>
  return %0, %e : tensor<64x32xf32>, tensor<64x32xf32>
// CHECK-LABEL:  func.func @test_scatter_nd_no_in_place_pass_through
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<64x32xf32> {onnx.name = "a"}, [[PARAM_1_:%.+]]: memref<2x32xf32> {onnx.name = "updates"}) -> (memref<64x32xf32> {onnx.name = "output"}, memref<64x32xf32> {onnx.name = "exp"}) {
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 64, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [5]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK:           "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_2048_]], [[CST_0_]], [[CST_0_]]) : (memref<64x32xf32>, memref<64x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_4_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<2x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_4_1_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<64x32xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 64, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_2_]]#0, [[VAR_4_2_]]#1] : memref<64x32xf32>
// CHECK:             [[VAR_6_2_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_6_2_]], [[RES_2_]]{{.}}[[VAR_4_2_]]#0, [[VAR_4_2_]]#1] : memref<64x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]], [[RES_2_]] : memref<64x32xf32>, memref<64x32xf32>
// CHECK:         }

}

