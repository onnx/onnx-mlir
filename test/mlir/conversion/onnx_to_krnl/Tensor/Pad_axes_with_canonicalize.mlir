// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Pad with an `axes` input. `pads` holds one begin then one end value per
// entry of `axes`, in `axes` order; the axes need not be sorted or contiguous.

// -----

// Scrambled axes: pads = [b2, b0, e2, e0] = [1, 2, 3, 4].
func.func @pad_axes_scrambled(%arg0: tensor<2x3x4xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %axes = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["data"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func.func @pad_axes_scrambled
// CHECK-SAME:   ([[DATA_:%.+]]: memref<2x3x4xf32>) -> memref<8x3x8xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<8x3x8xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<8x3x8xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#0)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_2_]], [[VAR_1_]]#1, [[VAR_3_]]{{.}} : memref<8x3x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<8x3x8xf32>
// CHECK:         }
}

// -----

// As in the ONNX Attention function body: pad the mask's last axis with
// runtime pads.
func.func @pad_axes_dynamic_pads(%arg0: tensor<4x?xi64>, %arg1: tensor<2xi64>, %arg2: tensor<1xi64>) -> tensor<*xi64> {
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %axes) {mode = "constant"} : (tensor<4x?xi64>, tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

// mlir2FileCheck.py -a '["data","pads","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func.func @pad_axes_dynamic_pads
// CHECK-SAME:   ([[DATA_:%.+]]: memref<4x?xi64>, [[PADS_:%.+]]: memref<2xi64>, [[CONSTANT_VALUE_:%.+]]: memref<1xi64>) -> memref<4x?xi64> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PADS_MEM_:%.+]] = krnl.load [[PADS_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PADS_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PADS_MEM_1_:%.+]] = krnl.load [[PADS_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PADS_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[DATA_]], [[CST_1_]] : memref<4x?xi64>
// CHECK:           [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]]) alignment = 16 : memref<4x?xi64>
// CHECK-DAG:       [[LOAD_CONSTANT_VALUE_MEM_:%.+]] = krnl.load [[CONSTANT_VALUE_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK:           krnl.memset [[RES_]], [[LOAD_CONSTANT_VALUE_MEM_]] : memref<4x?xi64>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[DATA_]], [[CST_1_]] : memref<4x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_0_]]){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}){
// CHECK:             [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<4x?xi64>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_8_]]{{.}} : memref<4x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x?xi64>
// CHECK:         }
}

// -----

func.func @pad_wrap_mode(%arg0: tensor<3x?xf32>) -> tensor<*xf32> {
  %pads = onnx.Constant dense<[0, 2, 0, 1]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "wrap"} : (tensor<3x?xf32>, tensor<4xi64>, none, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["data"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 3)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 - 2)>
// CHECK-LABEL:  func.func @pad_wrap_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<3x?xf32>) -> memref<3x?xf32> {
// CHECK:           [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[DATA_]], [[CST_1_]] : memref<3x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) alignment = 16 : memref<3x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_0_]])){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_2_]]#0)
// CHECK-DAG:         [[VAR_dim_0_:%.+]] = memref.dim [[DATA_]], [[CST_1_]] : memref<3x?xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]]#1)
// CHECK:             [[VAR_5_:%.+]] = arith.remsi [[VAR_4_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.remsi [[VAR_6_]], [[VAR_dim_0_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_3_]], [[VAR_7_]]{{.}} : memref<3x?xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x?xf32>
// CHECK:         }
}
