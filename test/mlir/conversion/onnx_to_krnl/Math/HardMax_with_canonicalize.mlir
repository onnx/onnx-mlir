// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_hardmax_axis_1(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["input"]'
// CHECK-LABEL:  func.func @test_hardmax_axis_1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5xindex>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<3x1x5xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<3x1x5xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_2_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<3x4x5xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_6_]] {
// CHECK:               krnl.store [[VAR_2_]]#1, [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<3x1x5xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[CST_0_]], [[VAR_2_1_]]#2] : memref<3x1x5xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_2_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x4x5xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_hardmax_unknown_dims(%arg0: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["input"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @test_hardmax_unknown_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[INPUT_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_2) {{.*}}: memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[INPUT_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_dim_3_]], [[VAR_dim_5_]]) {{.*}}: memref<?x1x?xindex>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<?x1x?xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_3_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_3_]], [[VAR_dim_4_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_3_]], [[VAR_dim_4_]], [[VAR_dim_5_]])){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<?x1x?xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_2_]]#2] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<?x?x?xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_6_]] {
// CHECK:               krnl.store [[VAR_2_]]#1, [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<?x1x?xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1), [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_2)){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[CST_0_]], [[VAR_2_1_]]#2] : memref<?x1x?xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_2_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<?x?x?xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<?x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?xf32>
// CHECK:         }
}
