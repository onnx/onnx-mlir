// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// COM: test split with unknown dimensions and explicit split.
func.func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = onnx.Constant dense<[2, 30]> : tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<?x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:        [[MAP0:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:        [[MAP1:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL:  func @test_split_unknown_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x2x64xf32>, memref<?x30x64xf32>) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_1_]]) {{.*}} : memref<?x30x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_0_]]), [[LOOP_0]]#1 -> %arg2 = 0 to 2, [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x2x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_1_]]), [[LOOP_1]]#1 -> %arg2 = 0 to 30, [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP1]]{{.}}[[IV]]#1{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x30x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x2x64xf32>, memref<?x30x64xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and default split.
func.func @test_split_unknown_dimension_equal_split(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64 } : (tensor<?x?x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:       [[MAP0:#.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK:       [[MAP1:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:       [[MAP2:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       [[MAP3:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 ceildiv 2)>
// CHECK-LABEL: func @test_split_unknown_dimension_equal_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x?x64xf32>, memref<?x?x64xf32>) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_1_]], [[VAR_3_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_2_]], [[VAR_5_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_1_]]), [[LOOP_0]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_1_]], [[VAR_3_]]), [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_2_]]), [[LOOP_1]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_2_]], [[VAR_5_]]), [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP3]]([[IV]]#1){{.}}[[DIM_0_]]{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?x64xf32>, memref<?x?x64xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and explicit split.
func.func @test_splitv11_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:        [[MAP0:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:        [[MAP1:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL:  func @test_splitv11_unknown_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x2x64xf32>, memref<?x30x64xf32>) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_1_]]) {{.*}} : memref<?x30x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_0_]]), [[LOOP_0]]#1 -> %arg2 = 0 to 2, [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x2x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_1_]]), [[LOOP_1]]#1 -> %arg2 = 0 to 30, [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP1]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x30x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x2x64xf32>, memref<?x30x64xf32>
// CHECK:         }
}

// -----

// COM: test splitv11 with unknown dimensions and default split.
func.func @test_splitv11_unknown_dimension_equal_split(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64 } : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:       [[MAP0:#.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK:       [[MAP1:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:       [[MAP2:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       [[MAP3:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 ceildiv 2)>
// CHECK-LABEL: func @test_splitv11_unknown_dimension_equal_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x?x64xf32>, memref<?x?x64xf32>) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_1_]], [[VAR_3_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_2_]], [[VAR_5_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_1_]]), [[LOOP_0]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_1_]], [[VAR_3_]]), [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_2_]]), [[LOOP_1]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_2_]], [[VAR_5_]]), [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP3]]([[IV]]#1){{.}}[[DIM_0_]]{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?x64xf32>, memref<?x?x64xf32>
// CHECK:         }
}
