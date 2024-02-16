// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_expand_with_arith_constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[7, 1, 5]> : tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["input", "shape"]'
// CHECK-LABEL:  func @test_expand_with_arith_constant
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x1x6x1xf32>) -> memref<2x7x6x5xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x7x6x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[SHAPE_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 7, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 6, [[LOOP_0_]]#3 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<2x1x6x1xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<2x7x6x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x7x6x5xf32>
// CHECK:         }
}

// -----

func.func @expand_dyn(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32>  {
  %0 = "onnx.Expand"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
// mlir2FileCheck.py -a'["input", "shape"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-LABEL:  func @expand_dyn
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?xf32>, [[SHAPE_:%.+]]: memref<2xi64>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_SHAPE_MEM_:%.+]] = krnl.load [[SHAPE_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_SHAPE_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_SHAPE_MEM_1_:%.+]] = krnl.load [[SHAPE_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_SHAPE_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_4_]]{{.}}
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_7_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_6_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_7_]]){
// CHECK-DAG:         [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_4_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_10_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_5_]], [[CST_1_]] : index
// CHECK:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_10_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_12_]], [[VAR_14_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_10_]]#0, [[VAR_10_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}
