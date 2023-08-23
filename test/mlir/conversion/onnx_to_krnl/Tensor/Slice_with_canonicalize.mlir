// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Slice where all the parameters are constant.
func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[IV]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, -1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[5, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * -2 + 3)>
// CHECK-LABEL:  func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  // %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>
  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[-1, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[DIM_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#2)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_7_]], [[VAR_8_]]{{.}} : memref<?x4x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x3xf32>
// CHECK:         }
}

// -----

// Check where all is dynamic except input size and axis. The code was verified
// using a procedure simioar to mlir-run and by manually adding code to print the
// output as a vector

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<3x?x?xi64> {
   %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>

  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x?x?xi64>
  return %res : tensor<3x?x?xi64>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2xi64>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<3x?x?xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_minus_2147483648_:%.+]] = arith.constant -2147483648 : index
// CHECK-DAG:       [[CST_2147483647_:%.+]] = arith.constant 2147483647 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3, 4, 5], value = dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>} : () -> memref<3x4x5xi64>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_3_]]{{.}}
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_8_]], [[VAR_9_]], [[VAR_3_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_12_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_4_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.cmpi sgt, [[VAR_16_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_5_]], [[VAR_16_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[VAR_14_]], [[VAR_18_]] : index
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_22_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_5_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.select [[VAR_21_]], [[VAR_22_]], [[VAR_5_]] : index
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.cmpi sle, [[VAR_5_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.select [[VAR_24_]], [[CST_minus_1_]], [[VAR_23_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[CST_5_]], [[VAR_25_]] : index
// CHECK:           [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[CST_minus_1_]], [[VAR_27_]] : index
// CHECK:           [[VAR_30_:%.+]] = arith.cmpi sgt, [[VAR_29_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.select [[VAR_30_]], [[CST_5_]], [[VAR_29_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_0_]] : index
// CHECK:           [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[CST_0_]], [[VAR_27_]] : index
// CHECK:           [[VAR_34_:%.+]] = arith.cmpi sgt, [[VAR_33_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_5_]], [[VAR_33_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK:           [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[VAR_31_]], [[VAR_35_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.subi [[VAR_37_]], [[VAR_20_]] : index
// CHECK:           [[VAR_39_:%.+]] = arith.ceildivsi [[VAR_38_]], [[VAR_7_]] : index
// CHECK:           [[VAR_40_:%.+]] = arith.cmpi slt, [[VAR_39_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_41_:%.+]] = arith.select [[VAR_40_]], [[CST_0_]], [[VAR_39_]] : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_43_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_47_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_48_:%.+]] = arith.cmpi slt, [[VAR_43_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_49_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_43_]]{{.}}
// CHECK:           [[VAR_50_:%.+]] = arith.select [[VAR_48_]], [[VAR_49_]], [[VAR_43_]] : index
// CHECK:           [[VAR_51_:%.+]] = arith.cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_53_:%.+]] = arith.cmpi sgt, [[VAR_52_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[CST_3_]], [[VAR_52_]] : index
// CHECK-DAG:       [[VAR_55_:%.+]] = arith.cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_56_:%.+]] = arith.select [[VAR_55_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_57_:%.+]] = arith.cmpi sgt, [[VAR_56_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_58_:%.+]] = arith.select [[VAR_57_]], [[CST_4_]], [[VAR_56_]] : index
// CHECK-DAG:       [[VAR_59_:%.+]] = arith.cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_60_:%.+]] = arith.select [[VAR_59_]], [[VAR_54_]], [[VAR_58_]] : index
// CHECK-DAG:       [[VAR_61_:%.+]] = arith.cmpi slt, [[VAR_45_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_62_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_45_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_63_:%.+]] = arith.select [[VAR_61_]], [[VAR_62_]], [[VAR_45_]] : index
// CHECK-DAG:       [[VAR_64_:%.+]] = arith.cmpi sle, [[VAR_45_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_65_:%.+]] = arith.select [[VAR_64_]], [[CST_minus_1_]], [[VAR_63_]] : index
// CHECK-DAG:       [[VAR_66_:%.+]] = arith.cmpi sge, [[VAR_45_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[CST_4_]], [[VAR_65_]] : index
// CHECK:           [[VAR_68_:%.+]] = arith.cmpi slt, [[VAR_67_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[CST_minus_1_]], [[VAR_67_]] : index
// CHECK:           [[VAR_70_:%.+]] = arith.cmpi sgt, [[VAR_69_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_71_:%.+]] = arith.select [[VAR_70_]], [[CST_4_]], [[VAR_69_]] : index
// CHECK-DAG:       [[VAR_72_:%.+]] = arith.cmpi slt, [[VAR_67_]], [[CST_0_]] : index
// CHECK:           [[VAR_73_:%.+]] = arith.select [[VAR_72_]], [[CST_0_]], [[VAR_67_]] : index
// CHECK:           [[VAR_74_:%.+]] = arith.cmpi sgt, [[VAR_73_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_75_:%.+]] = arith.select [[VAR_74_]], [[CST_4_]], [[VAR_73_]] : index
// CHECK-DAG:       [[VAR_76_:%.+]] = arith.cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK:           [[VAR_77_:%.+]] = arith.select [[VAR_76_]], [[VAR_71_]], [[VAR_75_]] : index
// CHECK:           [[VAR_78_:%.+]] = arith.subi [[VAR_77_]], [[VAR_60_]] : index
// CHECK:           [[VAR_79_:%.+]] = arith.ceildivsi [[VAR_78_]], [[VAR_47_]] : index
// CHECK:           [[VAR_80_:%.+]] = arith.cmpi slt, [[VAR_79_]], [[CST_0_]] : index
// CHECK:           [[VAR_81_:%.+]] = arith.select [[VAR_80_]], [[CST_0_]], [[VAR_79_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_81_]], [[VAR_41_]]) {{.*}} : memref<3x?x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_81_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[VAR_41_]]){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_84_:%.+]] = arith.muli [[VAR_47_]], [[IV]]#1 : index
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.addi [[VAR_84_]], [[VAR_60_]] : index
// CHECK-DAG:         [[VAR_86_:%.+]] = arith.muli [[VAR_7_]], [[IV]]#2 : index
// CHECK:             [[VAR_87_:%.+]] = arith.addi [[VAR_86_]], [[VAR_20_]] : index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][[[IV]]#0, [[VAR_85_]], [[VAR_87_]]{{.}} : memref<3x4x5xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x?x?xi64>
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

