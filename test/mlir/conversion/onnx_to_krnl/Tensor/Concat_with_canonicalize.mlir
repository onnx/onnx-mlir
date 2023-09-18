// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func private @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "func.return"(%1) : (tensor<5x5x9x32xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<5x5x9x32xf32>
  // CHECK: [[DEF_LOOPS0:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) with ([[DEF_LOOPS0]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS0]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS0]]#2 -> %arg5 = 0 to 1, [[DEF_LOOPS0]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[IV:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK: [[LOAD0:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x1x32xf32>
  // CHECK: krnl.store [[LOAD0]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS1]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS1]]#2 -> %arg5 = 0 to 3, [[DEF_LOOPS1]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[AFFINE_APPLY1:%.+]] = affine.apply #{{.*}}([[IV]]#2)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x3x32xf32>
  // CHECK: krnl.store [[LOAD1]], [[RES]][[[IV]]#0, [[IV]]#1, [[AFFINE_APPLY1]], [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2, [[DEF_LOOPS2]]#3) with ([[DEF_LOOPS2]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS2]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS2]]#2 -> %arg5 = 0 to 5, [[DEF_LOOPS2]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[AFFINE_APPLY2:%.+]] = affine.apply #{{.*}}([[IV]]#2)
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg2[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x5x32xf32>
  // CHECK: krnl.store [[LOAD2]], [[RES]][[[IV]]#0, [[IV]]#1, [[AFFINE_APPLY2]], [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: return [[RES]] :  memref<5x5x9x32xf32>
}

// -----

// Focus on accumulated offset for the store op in each loop
func.func @test_concat_5(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x3x32xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 : si64} : (tensor<?x?x?xf32>, tensor<?x3x32xf32>, tensor<?x?x?xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: [[MAP_5_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2)>
// CHECK-DAG: [[MAP_6_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-DAG: [[MAP_7_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 + 3)>
// CHECK-LABEL:  func.func @test_concat_5
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<?x3x32xf32>, [[PARAM_2_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x32xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_2_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_0_]]) {{.*}}: memref<?x?x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_4_1_]]#1){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1, [[VAR_4_1_]]#2] : memref<?x3x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_4_1_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_2_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_6_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_dim_]]_3), [[LOOP_2_]]#1 -> [[I_7_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_dim_]]_3, [[VAR_dim_]]_4), [[LOOP_2_]]#2 -> [[I_8_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_4_2_]]#1){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_2_]]#0, [[VAR_4_2_]]#1, [[VAR_4_2_]]#2] : memref<?x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_2_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_4_2_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x32xf32>
// CHECK:         }
}

// -----

// Please check the loop bounds for each input: should be same for dynamic
func.func @test_concat_4(%arg0 : tensor<?x1x?xf32>, %arg1 : tensor<?x3x32xf32>, %arg2 : tensor<?x5x?xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 : si64} : (tensor<?x1x?xf32>, tensor<?x3x32xf32>, tensor<?x5x?xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func @test_concat_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x?xf32>, [[PARAM_1_:%.+]]: memref<?x3x32xf32>, [[PARAM_2_:%.+]]: memref<?x5x?xf32>) -> memref<?x9x32xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x9x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_3_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<?x3x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_1_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_6_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_2_]]#1 -> [[I_7_:%.+]] = 0 to 5, [[LOOP_2_]]#2 -> [[I_8_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_3_2_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[VAR_3_2_]]#2] : memref<?x5x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_2_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x9x32xf32>
// CHECK:         }
}
