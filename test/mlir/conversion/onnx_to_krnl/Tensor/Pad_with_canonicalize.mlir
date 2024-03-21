// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @pad_constant_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func.func @pad_constant_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOAD_CONSTANT_VALUE_MEM_:%.+]] = krnl.load [[CONSTANT_VALUE_]][] : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_CONSTANT_VALUE_MEM_]] : memref<?x?x?x?xf32>
// CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK:             [[VAR_22_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_4_]]([[VAR_22_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_4_]]([[VAR_22_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:         [[VAR_25_:%.+]] = affine.apply [[MAP_4_]]([[VAR_22_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK-DAG:         [[VAR_26_:%.+]] = affine.apply [[MAP_4_]]([[VAR_22_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1, [[VAR_22_]]#2, [[VAR_22_]]#3] : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_23_]], [[VAR_24_]], [[VAR_25_]], [[VAR_26_]]{{.}} : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----


func.func @pad_edge_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "edge"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0)[s0] -> (d0 - s0)>
// CHECK-LABEL:  func.func @pad_edge_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_5_]]([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_6_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_7_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_21_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.cmpi sle, [[VAR_21_]]#0, [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK:             [[VAR_24_:%.+]] = arith.select [[VAR_22_]], [[CST_0_]], [[VAR_23_]] : index
// CHECK:             [[VAR_25_:%.+]] = arith.cmpi sge, [[VAR_24_]], [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[CST_0_]], [[VAR_24_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpi sle, [[VAR_21_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[CST_0_]], [[VAR_28_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.cmpi sge, [[VAR_29_]], [[CST_3_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_30_]], [[CST_2_]], [[VAR_29_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.cmpi sle, [[VAR_21_]]#2, [[VAR_11_]] : index
// CHECK-DAG:         [[VAR_33_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK:             [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[CST_0_]], [[VAR_33_]] : index
// CHECK:             [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_4_]] : index
// CHECK:             [[VAR_36_:%.+]] = arith.select [[VAR_35_]], [[CST_3_]], [[VAR_34_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.cmpi sle, [[VAR_21_]]#3, [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK:             [[VAR_39_:%.+]] = arith.select [[VAR_37_]], [[CST_0_]], [[VAR_38_]] : index
// CHECK:             [[VAR_40_:%.+]] = arith.cmpi sge, [[VAR_39_]], [[CST_5_]] : index
// CHECK:             [[VAR_41_:%.+]] = arith.select [[VAR_40_]], [[CST_4_]], [[VAR_39_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_26_]], [[VAR_31_]], [[VAR_36_]], [[VAR_41_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_21_]]#0, [[VAR_21_]]#1, [[VAR_21_]]#2, [[VAR_21_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_reflect_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "reflect"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0)[s0] -> (d0 - s0)>
// CHECK-LABEL:  func.func @pad_reflect_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_5_]]([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_6_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_7_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_21_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.cmpi slt, [[VAR_21_]]#0, [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_9_]]([[VAR_21_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK:             [[VAR_25_:%.+]] = arith.select [[VAR_22_]], [[VAR_23_]], [[VAR_24_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.subi [[CST_0_]], [[VAR_25_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.select [[VAR_26_]], [[VAR_27_]], [[VAR_25_]] : index
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_21_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:         [[VAR_31_:%.+]] = affine.apply [[MAP_9_]]([[VAR_21_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK:             [[VAR_32_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_31_]] : index
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpi sge, [[VAR_32_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.subi [[CST_4_]], [[VAR_32_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.select [[VAR_33_]], [[VAR_34_]], [[VAR_32_]] : index
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.cmpi slt, [[VAR_21_]]#2, [[VAR_11_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK-DAG:         [[VAR_38_:%.+]] = affine.apply [[MAP_9_]]([[VAR_21_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK:             [[VAR_39_:%.+]] = arith.select [[VAR_36_]], [[VAR_37_]], [[VAR_38_]] : index
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.cmpi sge, [[VAR_39_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.subi [[CST_6_]], [[VAR_39_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.select [[VAR_40_]], [[VAR_41_]], [[VAR_39_]] : index
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.cmpi slt, [[VAR_21_]]#3, [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_44_:%.+]] = affine.apply [[MAP_8_]]([[VAR_21_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK-DAG:         [[VAR_45_:%.+]] = affine.apply [[MAP_9_]]([[VAR_21_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK:             [[VAR_46_:%.+]] = arith.select [[VAR_43_]], [[VAR_44_]], [[VAR_45_]] : index
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.cmpi sge, [[VAR_46_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.subi [[CST_8_]], [[VAR_46_]] : index
// CHECK:             [[VAR_49_:%.+]] = arith.select [[VAR_47_]], [[VAR_48_]], [[VAR_46_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_28_]], [[VAR_35_]], [[VAR_42_]], [[VAR_49_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_21_]]#0, [[VAR_21_]]#1, [[VAR_21_]]#2, [[VAR_21_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_constant_mode_constant_pads(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0, 3, 2, 1]> : tensor<4xi64>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  %2 = "onnx.Pad"(%arg0, %0, %1, %cst) {mode = "constant"} : (tensor<16x16xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<18x20xf32>
  return %2 : tensor<18x20xf32>

// mlir2FileCheck.py -a'["data"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @pad_constant_mode_constant_pads
// CHECK-SAME:   ([[DATA_:%.+]]: memref<16x16xf32>) -> memref<18x20xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [1], value = dense<0.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<18x20xf32>
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_VAR_0_MEM_]] : memref<18x20xf32>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[PAD_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[CONSTANT_VALUE_:%.+]] = 0 to 16){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_]]#1)
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x16xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_4_]]{{.}} : memref<18x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<18x20xf32>
// CHECK:         }
}
