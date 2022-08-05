// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----

// Slice where all the parameters are constant.
func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_6_:%.+]] = affine.apply #map([[IV]]#0)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[IV]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  // %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  // slice * 1-3 1-4 with neg numbers
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map0([[DIM_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[IV]]#1)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply #map1([[IV]]#2)
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

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) {
   %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>

  // slice * 1-3 1-4 with neg numbers
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x?x?xi64>
  return

// CHECK-LABEL:  func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2xi64>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) {
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
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map0(){{.}}[[VAR_3_]]{{.}}
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
// CHECK-DAG:       [[VAR_22_:%.+]] = affine.apply #map0(){{.}}[[VAR_5_]]{{.}}
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
// CHECK-DAG:       [[VAR_49_:%.+]] = affine.apply #map1(){{.}}[[VAR_43_]]{{.}}
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
// CHECK-DAG:       [[VAR_62_:%.+]] = affine.apply #map1(){{.}}[[VAR_45_]]{{.}}
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

// -----

// GEMM with everything constant
func.func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
}

// -----

// Gemm with all dimensions dynamic
func.func @test_gemm_all_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_all_dyn
}

// -----

// A[10, *] * B[*, 10] result in constant size output but dyn reduction.
func.func @test_gemm_k_dyn(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x10xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_k_dyn
}

// -----

// Broadcast bias C is dym, so we don't know if its 1 -> broadcast or 10. Dyn test for that.
func.func @test_gemm_c_dyn(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_c_dyn
}

// -----

// Test tile with constant repeats
func.func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<4x8xf32>) -> memref<12x16xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<12x16xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 12, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 16){
// CHECK-NEXT:        [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3:%.+]] = affine.apply [[MAP0]]([[IV]]#0)
// CHECK-DAG:         [[VAR_4:%.+]] = affine.apply [[MAP1]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_3]], [[VAR_4]]{{.}} : memref<4x8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<12x16xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<12x16xf32>
// CHECK:         }
}

// -----

// Test tile without arith.constant repeats
func.func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile2
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<8xf32>, [[PARAM_1:%.+]]: memref<1xi64>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_1_MEM:%.+]] = krnl.load [[PARAM_1]]{{\[}}[[CST_0]]{{\]}} : memref<1xi64>
// CHECK:           [[VAR_1:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM]] : i64 to index
// CHECK:           [[VAR_2:%.+]] = affine.apply [[MAP0]](){{.}}[[VAR_1]]{{.}}
// CHECK-DAG:       [[RES:%.+]] = memref.alloc([[VAR_2]]) {{.*}} : memref<?xf32>
// CHECK-DAG:       [[LOOP_0:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0]]) with ([[LOOP_0]] -> [[I_0:%.+]] = 0 to [[MAP0]](){{.}}[[VAR_1]]{{.}}){
// CHECK-NEXT:        [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0]]) : (!krnl.loop) -> index  
// CHECK:             [[VAR_5:%.+]] = affine.apply [[MAP1]]([[IV]])
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_5]]{{.}} : memref<8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<?xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_4_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_:%.+]]#0, [[LOOP_0_:%.+]]#1, [[LOOP_0_:%.+]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_7_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 1, second example in ONNX for Gather.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [1, 2], value = dense<{{.}}[0, 2]{{.}}> : tensor<1x2xi64>} : () -> memref<1x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#1, [[IV]]#2] : memref<1x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_4_]]] : memref<3x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherElements along axis 0. Positive indices, so no select.
func.func @test_gather_elements_axis0(%arg0 : tensor<3x3xf32>) -> tensor<2x3xf32> {
  %indices = "onnx.Constant"() {value = dense<[[1, 2, 0], [2, 0, 0]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<2x3xi64>) -> tensor<2x3xf32>
  "func.return"(%0) : (tensor<2x3xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x3xf32>) -> memref<2x3xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 3], value = dense<{{.}}[1, 2, 0], [2, 0, 0]{{.}}> : tensor<2x3xi64>} : () -> memref<2x3xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x3xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[INDEX_CAST]], [[IV]]#1] : memref<3x3xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x3xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x3xf32>
// CHECK:         }
}

// -----

// Test GatherElements along axis 0. Negative indices.
func.func @test_gather_elements_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0:%.+]]#0, [[LOOP_0:%.+]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[INDEX_CAST]], [[CST_0]] : index
// CHECK-DAG:         [[VAR_1:%.+]] = arith.addi [[INDEX_CAST]], [[CST_3]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[VAR_1]], [[INDEX_CAST]] : index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[SEL]], [[IV]]#1] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherElements along axis 1. Positive indices, so no select.
func.func @test_gather_elements_axis1(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 0], [1, 0]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 0], [1, 0]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[IV]]#0, [[INDEX_CAST]]] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and explicit split.
func.func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
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
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
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
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
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

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of i32.
func.func @test_reducemean_i32_unknown_dims(%arg0 : tensor<3x?x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemean_i32_unknown_dims
  // CHECK: [[ONE:%.+]] = arith.constant 1 : index
  // CHECK: krnl.iterate
  // CHECK: krnl.iterate
  // CHECK: [[DIM:%.+]] = memref.dim %arg0, [[ONE]] : memref<3x?x2xi32>
  // CHECK: [[DIVISOR:%.+]] = arith.index_cast [[DIM]] : index to i32
  // CHECK: krnl.iterate
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of f32.
func.func @test_reducemean_f32_unknown_dims(%arg0 : tensor<3x?x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemean_f32_unknown_dims
  // CHECK: [[ONE:%.+]] = arith.constant 1 : index
  // CHECK: krnl.iterate
  // CHECK: krnl.iterate
  // CHECK: [[DIM:%.+]] = memref.dim %arg0, [[ONE]] : memref<3x?x2xf32>
  // CHECK: [[UNKNOWN_DIM_i64:%.+]] = arith.index_cast [[DIM]] : index to i64
  // CHECK: [[DIVISOR:%.+]] = arith.uitofp [[UNKNOWN_DIM_i64]] : i64 to f32
  // CHECK: krnl.iterate
}

// -----

// COM: Check the template for lowering binary operations whose output type can be different from its input type.
// With updated approach, no max is needed for the first dim as max(dim(arg0, 0), 1) is always dim(arg0, 0).
func.func @test_binary_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x5xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x4x5xf32>, tensor<1x?x1xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
// mlir2FileCheck.py
// CHECK-LABEL:  func @test_binary_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>, [[PARAM_1_:%.+]]: memref<1x?x1xf32>) -> memref<?x4x5xi1> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<1x?x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x4x5xf32>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[IV]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<1x?x1xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x4x5xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xi1>
// CHECK:         }
}

// -----

// COM: Check the template for lowering variadic operations and binary operations whose output type is the same as its input type: Min, Max, Add, Sub, etc. 
func.func @test_variadic_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x1xf32>, %arg1: tensor<?x?x5xf32>, %arg2: tensor<?x1x5xf32>) -> tensor<?x4x5xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<?x4x1xf32>, tensor<?x?x5xf32>, tensor<?x1x5xf32>) -> tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
// CHECK-DAG: #map0 = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func @test_variadic_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x1xf32>, [[PARAM_1_:%.+]]: memref<?x?x5xf32>, [[PARAM_2_:%.+]]: memref<?x1x5xf32>) -> memref<?x4x5xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x1x5xf32>
// CHECK:           [[VAR_4_:%.+]] = affine.max #map0(){{.}}[[VAR_0_]], [[VAR_1_]]{{.}}
// CHECK:           [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_3_]], [[VAR_4_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_3_]], [[VAR_4_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = memref.alloc([[VAR_6_]]) {{.*}} : memref<?x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map1([[VAR_6_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi sgt, [[VAR_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[IV]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_10_]], [[IV]]#1, [[CST_0_]]{{.}} : memref<?x4x1xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi sgt, [[VAR_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[IV]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.cmpi sgt, [[VAR_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[IV]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_13_]], [[VAR_15_]], [[IV]]#2{{.}} : memref<?x?x5xf32>
// CHECK:             [[VAR_17_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi sgt, [[VAR_3_]], [[CST_1_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[IV]]#0, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_20_]], [[CST_0_]], [[IV]]#2{{.}} : memref<?x1x5xf32>
// CHECK:             [[VAR_22_:%.+]] = arith.cmpf ogt, [[VAR_18_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_18_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_23_]], [[VAR_7_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x4x5xf32>
// CHECK:           }
// CHECK:           return [[VAR_7_]] : memref<?x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: Because of unidirectional broadcasting, always get constant dimensions from X even thought their values are 1.
func.func @test_prelu_broadcast_unknown_dims(%arg0: tensor<3x1x5xf32>, %arg1: tensor<3x?x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x1x5xf32>, tensor<3x?x1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: @test_prelu_broadcast_unknown_dims
  // CHECK-DAG: [[CST0_f32:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[CST0:%.+]] = arith.constant 0 : index 
  // CHECK:     [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x1x5xf32>
  // CHECK:     [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 1, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index) 
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x1x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1, [[CST0]]{{\]}} : memref<3x?x1xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST0_f32]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x1x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x1x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: If X's dimensions are unknown, get dimensions from slope whenever they are non-zero constants.
func.func @test_prelu_broadcast_unknown_dims1(%arg0: tensor<?x2x?xf32>, %arg1: tensor<?x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x2x?xf32>, tensor<?x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: @test_prelu_broadcast_unknown_dims1
  // CHECK-DAG: [[CST0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[CST0_f32:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:     [[DIM0_X:%.+]] = memref.dim %arg0, [[CST0]] : memref<?x2x?xf32>
  // CHECK:     [[DIM0_SLOPE:%.+]] = memref.dim %arg1, [[CST0]] : memref<?x5xf32>
  // CHECK:     [[RES:%.+]] = memref.alloc([[DIM0_X]]) {{.*}} : memref<?x2x5xf32>
  // CHECK:     [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to #map([[DIM0_X]]), [[MAIN_LOOP]]#1 -> %arg3 = 0 to 2, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index) 
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x?xf32>
  // CHECK:       [[GREATER_THAN_ONE:%.+]] = arith.cmpi sgt, [[DIM0_SLOPE]], [[CST1]] : index
  // CHECK:       [[SELECT1:%.+]] = arith.select [[GREATER_THAN_ONE]], [[IV]]#1, [[CST0]] : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[SELECT1]], [[IV]]#2] : memref<?x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST0_f32]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT2:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT2]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x5xf32>
  // CHECK:     }
  // CHECK:     return [[RES]] : memref<?x2x5xf32>
}

// -----

/// Check ReduceMean with f32.
func.func private @test_reducemean_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemean_f32
  // CHECK-DAG: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[DIVISOR:%.+]] = arith.constant 2.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK-DAG: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = arith.addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }

  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2){
  // CHECK:   [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:   [[LOAD3:%.+]] = krnl.load [[RES]][[[IV]]#0, [[IV]]#1] : memref<3x2xf32>
  // CHECK:   [[MEAN:%.+]] = arith.divf [[LOAD3]], [[DIVISOR]] : f32
  // CHECK:   krnl.store [[MEAN]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

/// Check ReduceMean with i32.
func.func private @test_reducemean_i32(%arg0 : tensor<3x2x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_reducemean_i32
  // CHECK-DAG: [[IDENTITY:%.+]] = arith.constant 0 : i32
  // CHECK-DAG: [[DIVISOR:%.+]] = arith.constant 2 : i32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xi32>
  // CHECK-DAG: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xi32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xi32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: [[REDUCE:%.+]] = arith.addi [[LOAD2]], [[LOAD1]] : i32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: }

  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2){
  // CHECK:   [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:   [[LOAD3:%.+]] = krnl.load [[RES]][[[IV]]#0, [[IV]]#1] : memref<3x2xi32>
  // CHECK:   [[MEAN:%.+]] = arith.divsi [[LOAD3]], [[DIVISOR]] : i32
  // CHECK:   krnl.store [[MEAN]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<3x2xi32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xi32>
}

// -----

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
// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func.func @test_prelu_broadcast3(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast3
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index) 
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[ZERO_INDEX]], [[IV]]#2] : memref<3x1x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----
// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func.func @test_prelu_broadcast4(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast4
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index) 
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[ZERO_INDEX]], [[IV]]#2] : memref<3x1x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----
// COM: 2D matmul.
func.func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py -a'["A", "B"]'
// CHECK-LABEL:  func private @test_matmul1
// CHECK-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-DAG:       [[VAR_c16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK:           krnl.memset [[RES_]], [[VAR_cst_]] : memref<16x16xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__:%.+]], [[BLOCK_IN__:%.+]] = krnl.block [[LOOP_0_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_0_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__]], [[BLOCK_IN__]], [[BLOCK_TILE__]]_0, [[BLOCK_IN__]]_1, [[BLOCK_TILE__]]_2, [[BLOCK_IN__]]_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__]], [[BLOCK_TILE__]]_0, [[BLOCK_TILE__]]_2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[VAR_c0_]] to [[VAR_c16_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[VAR_c0_]] to [[VAR_c16_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = [[VAR_c0_]] to [[VAR_c16_]]){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__]], [[BLOCK_TILE__]]_0, [[BLOCK_TILE__]]_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[A_]]{{.}}[[VAR_c0_]], [[VAR_c0_]]{{.}}, [[B_]]{{.}}[[VAR_c0_]], [[VAR_c0_]]{{.}}, [[RES_]]{{.}}[[VAR_c0_]], [[VAR_c0_]]{{.}}, ([[BLOCK_IN__]], [[BLOCK_IN__]]_1, [[BLOCK_IN__]]_3), ([[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2), ([[VAR_c16_]], [[VAR_c16_]], [[VAR_c16_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8], overcompute = false, simdize = true, unroll = true} : memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x16xf32>
// CHECK:         }
}

// -----
// 2-D x N-D
func.func private @test_matmul2(%arg0 : tensor<10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"0": "RES"}'
// CHECK-LABEL:  func private @test_matmul2
// CHECK-SAME:   ([[A_:%.+]]: memref<10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#4) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_3_]]#2, [[VAR_5_]]{{.}} : memref<10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_5_]], [[VAR_3_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// N-D x N-D
func.func private @test_matmul3(%arg0 : tensor<2x3x10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // mlir2FileCheck.py -a'["A", "B"]' -n'{"0": "RES"}'
// CHECK-LABEL:  func private @test_matmul3
// CHECK-SAME:   ([[A_:%.+]]: memref<2x3x10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#4) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_5_]]{{.}} : memref<2x3x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_5_]], [[VAR_3_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// 1-D x 2-D
func.func private @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"0": "RES"}'
// CHECK-LABEL:  func private @test_matmul4
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#0) : (!krnl.loop) -> index
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#1) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#1) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]], [[VAR_3_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10xf32>
// CHECK:         }
}

// -----

// 1-D x N-D
func.func private @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-LABEL:  func private @test_matmul5
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<?x5x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[B_]], [[VAR_c0_]] : memref<?x5x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_4_]]#0, [[VAR_6_]], [[VAR_4_]]#1] : memref<?x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// N-D x 1-D
func.func private @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-DAG: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_matmul6
// CHECK-SAME:   ([[A_:%.+]]: memref<?x10x5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[A_]], [[VAR_c0_]] : memref<?x10x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_6_]]{{.}} : memref<?x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// 1-D x 1-D results in scalar
func.func private @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-LABEL:  func private @test_matmul7
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate() with ([[RES_1_]] -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[VAR_cst_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[RES_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
}

// -----

func.func private @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x?x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 - 1)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #map3 = affine_map<(d0)[s0] -> (s0, d0 + 2)>
// CHECK-DAG: #map4 = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG: #map5 = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func private @test_pool_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x?x32xf32>) -> memref<1x3x?x31xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK:           [[VAR_1_:%.+]] = affine.apply #map0(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x3x?x31xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to #map1([[VAR_1_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_4_]][] : memref<f32>
// CHECK-DAG:         [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.max #map2([[IV]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.min #map3([[IV]]#2){{.}}[[VAR_5_]]{{.}}
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.max #map2([[IV]]#3)
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.min #map4([[IV]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.subi [[VAR_7_]], [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.subi [[VAR_9_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min #map5([[IV]]#2){{.}}[[VAR_5_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min #map5([[IV]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addi [[I_4_]], [[VAR_6_]] : index
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.addi [[I_5_]], [[VAR_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[VAR_19_]], [[VAR_20_]]{{.}} : memref<1x3x?x32xf32>
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]][] : memref<f32>
// CHECK:               [[VAR_23_:%.+]] = arith.addf [[LOAD_VAR_4_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_23_]], [[VAR_4_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_VAR_4_MEM_1_:%.+]] = krnl.load [[VAR_4_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_VAR_4_MEM_1_]], [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK-DAG:         [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.muli [[VAR_10_]], [[VAR_11_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i64
// CHECK:             [[VAR_17_:%.+]] = arith.sitofp [[VAR_16_]] : i64 to f32
// CHECK:             [[VAR_18_:%.+]] = arith.divf [[LOAD_VAR_2_MEM_]], [[VAR_17_]] : f32
// CHECK:             krnl.store [[VAR_18_]], [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<1x3x?x31xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_unknown_dimensions(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<?x?x?x?xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()



// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-LABEL:  func private @test_conv_unknown_dimensions
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<?x?x?x?xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>, [[BIAS_:%.+]]: memref<5xf32>) -> memref<?x5x?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[IMAGE_]], [[CST_0_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[IMAGE_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[IMAGE_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = affine.apply #map1(){{.}}[[VAR_3_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_2_]], [[VAR_4_]]) {{.*}}: memref<?x5x?x?xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map2([[VAR_1_]], [[VAR_3_]], [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_7_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply #map3([[VAR_7_]]#1, [[VAR_7_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to #map4([[VAR_7_]]#1, [[VAR_7_]]#2){{.}}[[VAR_2_]]{{.}}, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to #map5([[VAR_7_]]#1, [[VAR_7_]]#2){{.}}[[VAR_2_]], [[VAR_4_]]{{.}}){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_11_]][] : memref<f32>
// CHECK-DAG:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:           [[VAR_13_:%.+]] = memref.dim [[IMAGE_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-DAG:           [[VAR_14_:%.+]] = memref.dim [[IMAGE_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = max #map6([[VAR_10_]]#0) to min #map7([[VAR_10_]]#0){{.}}[[VAR_13_]]{{.}}, [[LOOP_2_]]#2 -> [[I_7_:%.+]] = max #map8([[VAR_10_]]#0, [[VAR_10_]]#1){{.}}[[VAR_13_]]{{.}} to min #map9([[VAR_10_]]#0, [[VAR_10_]]#1){{.}}[[VAR_13_]], [[VAR_14_]]{{.}}){
// CHECK:                 [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_19_:%.+]] = affine.apply #map10([[VAR_18_]]#0, [[VAR_7_]]#1)
// CHECK-DAG:             [[VAR_20_:%.+]] = affine.apply #map11([[VAR_18_]]#1, [[VAR_10_]]#0)
// CHECK-DAG:             [[VAR_21_:%.+]] = affine.apply #map11([[VAR_18_]]#2, [[VAR_10_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_7_]]#0, [[VAR_19_]], [[VAR_20_]], [[VAR_21_]]{{.}} : memref<?x?x?x?xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_8_]], [[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]][] : memref<f32>
// CHECK:                 [[VAR_25_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_26_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_]], [[VAR_25_]] : f32
// CHECK:                 krnl.store [[VAR_26_]], [[VAR_11_]][] : memref<f32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_8_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_17_]], [[VAR_5_]]{{.}}[[VAR_7_]]#0, [[VAR_8_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<?x5x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_5_]] : memref<?x5x?x?xf32>
// CHECK:         }
}

// -----

// CHECK-DAG: #map = affine_map<()[s0] -> (s0 * 10)>
func.func private @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func private @test_reshape
// CHECK:          ([[PARAM_0_:%.+]]: memref<?x10xf32>, [[PARAM_1_:%.+]]: memref<4xi64>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply #map(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.cmpi eq, [[VAR_3_]], [[CST_0_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_1_]], [[VAR_6_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi eq, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_10_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_1_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.muli [[VAR_8_]], [[VAR_14_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_2_]] : i64 to index
// CHECK:           [[VAR_18_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[CST_1_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.muli [[VAR_15_]], [[VAR_19_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_22_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_3_]] : i64 to index
// CHECK:           [[VAR_23_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[CST_1_]], [[VAR_22_]] : index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.muli [[VAR_20_]], [[VAR_24_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.select [[VAR_26_]], [[VAR_27_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[VAR_33_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.select [[VAR_35_]], [[VAR_36_]], [[VAR_22_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.muli [[VAR_37_]], [[VAR_34_]] : index
// CHECK:           [[VAR_39_:%.+]] = arith.muli [[VAR_38_]], [[VAR_31_]] : index
// CHECK:           [[VAR_40_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_28_]], [[VAR_31_]], [[VAR_34_]], [[VAR_37_]]{{.}}, strides: {{.}}[[VAR_39_]], [[VAR_38_]], [[VAR_37_]], 1] : memref<?x10xf32> to memref<?x?x?x?xf32>
// CHECK:           return [[VAR_40_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x2x32x64xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>) -> memref<1x5x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x27x58xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply #map0([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max #map1([[VAR_5_]]#0) to min #map2([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max #map3([[VAR_5_]]#0, [[VAR_5_]]#1) to min #map4([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply #map5([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply #map6([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply #map6([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x2x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-LABEL:  func private @test_conv_bias_no_pad
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x2x32x64xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>, [[BIAS_:%.+]]: memref<5xf32>) -> memref<1x5x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x27x58xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply #map0([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = max #map1([[VAR_5_]]#0) to min #map2([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_7_:%.+]] = max #map3([[VAR_5_]]#0, [[VAR_5_]]#1) to min #map4([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_11_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply #map5([[VAR_11_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_13_:%.+]] = affine.apply #map6([[VAR_11_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_14_:%.+]] = affine.apply #map6([[VAR_11_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_12_]], [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_11_]]#0, [[VAR_11_]]#1, [[VAR_11_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_18_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_19_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_18_]] : f32
// CHECK:                 krnl.store [[VAR_19_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_3_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_1_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad_w_group(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<6x3x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 3 : si64} : (tensor<1x9x32x64xf32>, tensor<6x3x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad_w_group
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x9x32x64xf32>, [[FILTER_:%.+]]: memref<6x3x6x7xf32>) -> memref<1x6x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x6x27x58xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply #map0([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max #map1([[VAR_5_]]#0) to min #map2([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max #map3([[VAR_5_]]#0, [[VAR_5_]]#1) to min #map4([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply #map5([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply #map6([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply #map6([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x9x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<6x3x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x6x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x6x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad_w_strides(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 2]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad_w_strides
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x9x32x64xf32>, [[FILTER_:%.+]]: memref<5x9x6x7xf32>) -> memref<1x5x14x29xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x14x29xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply #map0([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 14, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 29){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 9, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max #map1([[VAR_5_]]#0) to min #map2([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max #map3([[VAR_5_]]#0, [[VAR_5_]]#1) to min #map4([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply #map5([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply #map6([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply #map6([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x9x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<5x9x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x14x29xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x14x29xf32>
// CHECK:         }
}

// -----

// COM: if there is no opset information, we use opset 11.
func.func private @test_softmax(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 20, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 20, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 20, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 30){
// CHECK:               [[VAR_10_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 11.
func.func private @test_softmax_v11(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64, onnx_opset=11: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v11([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 20, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 20, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 20, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 30){
// CHECK:               [[VAR_10_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 13.

func.func private @test_softmax_v13(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64, onnx_opset=13: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v13([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 20){
// CHECK:               [[VAR_10_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

func.func @instance_norm(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3x4x5xf32> attributes {input_names = ["x", "s", "bias"], output_names = ["y"]} {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// mlir2FileCheck.py -a'["input", "scale", "bias"]'
// CHECK-LABEL:  func @instance_norm
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3x4x5xf32>, [[SCALE_:%.+]]: memref<3xf32>, [[BIAS_:%.+]]: memref<3xf32>) -> memref<2x3x4x5xf32> attributes {input_names = ["x", "s", "bias"], output_names = ["y"]} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 2.000000e+01 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_00999999977_:%.+]] = arith.constant 0.00999999977 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:               krnl.store [[VAR_20_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_8_:%.+]] = arith.divf [[LOAD_RES_1_MEM_1_]], [[CST_20_]] : f32
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_1_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_21_:%.+]] = arith.mulf [[VAR_20_1_]], [[VAR_20_1_]] : f32
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_1_MEM_2_]], [[VAR_21_]] : f32
// CHECK:               krnl.store [[VAR_22_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_10_:%.+]] = arith.divf [[LOAD_RES_1_MEM_3_]], [[CST_20_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[CST_0_dot_00999999977_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = math.sqrt [[VAR_11_]] : f32
// CHECK-DAG:         [[LOAD_SCALE_MEM_:%.+]] = krnl.load [[SCALE_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.divf [[LOAD_SCALE_MEM_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 4, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 5){
// CHECK:               [[VAR_17_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_INPUT_MEM_2_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_2_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_20_2_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_INPUT_MEM_1_]] : f32
// CHECK:               [[VAR_21_1_:%.+]] = arith.addf [[VAR_20_2_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_21_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_nonzero(%arg0: tensor<2x2xi1>) -> tensor<*xi64> attributes {input_names = ["condition"], output_names = ["result"]} {
    %0 = "onnx.NonZero"(%arg0) : (tensor<2x2xi1>) -> tensor<*xi64>
    return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_nonzero
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x2xi1>) -> memref<2x?xi64> attributes {input_names = ["condition"], output_names = ["result"]} {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:             [[VAR_9_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_]], [[RES_1_]]{{.}}[[VAR_9_]]{{.}} : memref<2xindex>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:             [[VAR_9_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_]], [[RES_2_]]{{.}}[[VAR_9_1_]]{{.}} : memref<2xindex>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_9_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<2x2xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_INPUT_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[CST_1_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_14_]], [[RES_]][] : memref<index>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_9_2_]]#0] : memref<2xindex>
// CHECK:             [[VAR_16_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_16_]], [[RES_1_]]{{.}}[[VAR_9_2_]]#0] : memref<2xindex>
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_9_2_]]#1] : memref<2xindex>
// CHECK:             [[VAR_18_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_18_]], [[RES_2_]]{{.}}[[VAR_9_2_]]#1] : memref<2xindex>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_RES_MEM_1_]]){
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_minus_1_]], [[RES_4_]][] : memref<index>
// CHECK:             krnl.store [[CST_0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_]]{{.}} : memref<2xindex>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_9_3_]], [[VAR_24_]] : index
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.cmpi eq, [[LOAD_RES_4_MEM_]], [[CST_minus_1_]] : index
// CHECK:               [[VAR_27_:%.+]] = arith.andi [[VAR_25_]], [[VAR_26_]] : i1
// CHECK:               [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_20_]], [[LOAD_RES_4_MEM_]] : index
// CHECK:               krnl.store [[VAR_28_]], [[RES_4_]][] : memref<index>
// CHECK:               krnl.store [[VAR_24_]], [[RES_5_]][] : memref<index>
// CHECK:             }
// CHECK:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[VAR_14_1_:%.+]] = arith.index_cast [[LOAD_RES_4_MEM_1_]] : index to i64
// CHECK:             krnl.store [[VAR_14_1_]], [[RES_3_]]{{.}}[[CST_0_]], [[VAR_9_3_]]{{.}} : memref<2x?xi64>
// CHECK:             krnl.store [[CST_minus_1_]], [[RES_4_]][] : memref<index>
// CHECK:             krnl.store [[CST_0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:               [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_20_1_]]{{.}} : memref<2xindex>
// CHECK-DAG:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:           [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:               [[VAR_24_1_:%.+]] = arith.addi [[LOAD_RES_5_MEM_1_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK-DAG:           [[VAR_25_1_:%.+]] = arith.cmpi slt, [[VAR_9_3_]], [[VAR_24_1_]] : index
// CHECK-DAG:           [[VAR_26_1_:%.+]] = arith.cmpi eq, [[LOAD_RES_4_MEM_2_]], [[CST_minus_1_]] : index
// CHECK:               [[VAR_27_1_:%.+]] = arith.andi [[VAR_25_1_]], [[VAR_26_1_]] : i1
// CHECK:               [[VAR_28_1_:%.+]] = arith.select [[VAR_27_1_]], [[VAR_20_1_]], [[LOAD_RES_4_MEM_2_]] : index
// CHECK:               krnl.store [[VAR_28_1_]], [[RES_4_]][] : memref<index>
// CHECK:               krnl.store [[VAR_24_1_]], [[RES_5_]][] : memref<index>
// CHECK:             }
// CHECK:             [[VAR_18_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             krnl.store [[VAR_19_]], [[RES_3_]]{{.}}[[CST_1_]], [[VAR_9_3_]]{{.}} : memref<2x?xi64>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<2x?xi64>
// CHECK:         }
}

// -----

func.func @test_mod_fp32(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 1 : si64} : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// mlir2FileCheck.py -a'["a", "b"]'
// CHECK-LABEL:  func @test_mod_fp32
// CHECK-SAME:   ([[A_:%.+]]: memref<6xf32>, [[B_:%.+]]: memref<6xf32>) -> memref<6xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value(%1) : (!krnl.loop) -> index 
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.remf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.copysign [[VAR_4_]], [[LOAD_A_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xf32>
// CHECK:         }
}

// -----

func.func @test_mean(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<*xf32>  {
    %0 = "onnx.Mean"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
// mlir2FileCheck.py -a'["a", "b", "c"]'
// CHECK-LABEL:  func @test_mean
// CHECK-SAME:   ([[A_:%.+]]: memref<3xf32>, [[B_:%.+]]: memref<3xf32>, [[C_:%.+]]: memref<3xf32>) -> memref<3xf32> {
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]][[[IV]]] : memref<3xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]][[[IV]]] : memref<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK-DAG:         [[LOAD_C_MEM_:%.+]] = krnl.load [[C_]][[[IV]]] : memref<3xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_4_]], [[LOAD_C_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_3_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]][[[IV]]] : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func.func @where(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// mlir2FileCheck.py -a'["condition", "x", "y"]'
// CHECK-LABEL:  func @where
// CHECK-SAME:   ([[CONDITION_:%.+]]: memref<2x2xi1>, [[X_:%.+]]: memref<2x2xf32>, [[Y_:%.+]]: memref<2x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi1>
// CHECK-DAG:         [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK-DAG:         [[LOAD_Y_MEM_:%.+]] = krnl.load [[Y_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.select [[LOAD_CONDITION_MEM_]], [[LOAD_X_MEM_]], [[LOAD_Y_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xf32>
// CHECK:         }
}

// -----

func.func @round(%arg0: tensor<15xf32>) -> tensor<*xf32> {
  %0 = "onnx.Round"(%arg0) : (tensor<15xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func @round
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<15xf32>) -> memref<15xf32> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<15xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 15){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]] : memref<15xf32>
// CHECK:             [[VAR_3_:%.+]] = math.floor [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpf ogt, [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.mulf [[VAR_3_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_9_:%.+]] = math.floor [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[VAR_3_]], [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpf oeq, [[VAR_11_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.cmpf oeq, [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_7_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]][[[IV]]] : memref<15xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<15xf32>
// CHECK:         }
}

// -----

func.func @pad_constant_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

//  use arg names: ['data', 'pad', 'constant_value']
// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG: #map0 = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG: #map2 = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG: #map3 = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG: #map4 = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func @pad_constant_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map1(){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply #map2(){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply #map3(){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOAD_CONSTANT_VALUE_MEM_:%.+]] = krnl.load [[CONSTANT_VALUE_]][] : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_CONSTANT_VALUE_MEM_]] : memref<?x?x?x?xf32>
// CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK:             [[VAR_23_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply #map4([[VAR_23_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:         [[VAR_25_:%.+]] = affine.apply #map4([[VAR_23_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:         [[VAR_26_:%.+]] = affine.apply #map4([[VAR_23_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK-DAG:         [[VAR_27_:%.+]] = affine.apply #map4([[VAR_23_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_23_]]#2, [[VAR_23_]]#3] : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_24_]], [[VAR_25_]], [[VAR_26_]], [[VAR_27_]]{{.}} : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_edge_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2) {mode = "edge"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG: #map0 = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG: #map2 = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG: #map3 = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG: #map4 = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG: #map5 = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG: #map6 = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG: #map7 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-DAG: #map8 = affine_map<(d0)[s0] -> (d0 - s0)>
// CHECK-LABEL:  func @pad_edge_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map1(){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply #map2(){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply #map3(){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map4([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map5([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to #map6([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to #map7([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_22_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi sle, [[VAR_22_]]#0, [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply #map8([[VAR_22_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK:             [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[CST_0_]], [[VAR_24_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[CST_0_]], [[VAR_25_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.cmpi sle, [[VAR_22_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_29_:%.+]] = affine.apply #map8([[VAR_22_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK:             [[VAR_30_:%.+]] = arith.select [[VAR_28_]], [[CST_0_]], [[VAR_29_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.cmpi sge, [[VAR_30_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[CST_2_]], [[VAR_30_]] : index
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpi sle, [[VAR_22_]]#2, [[VAR_11_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = affine.apply #map8([[VAR_22_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK:             [[VAR_35_:%.+]] = arith.select [[VAR_33_]], [[CST_0_]], [[VAR_34_]] : index
// CHECK:             [[VAR_36_:%.+]] = arith.cmpi sge, [[VAR_35_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[CST_3_]], [[VAR_35_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpi sle, [[VAR_22_]]#3, [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_39_:%.+]] = affine.apply #map8([[VAR_22_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK:             [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[CST_0_]], [[VAR_39_]] : index
// CHECK:             [[VAR_41_:%.+]] = arith.cmpi sge, [[VAR_40_]], [[CST_5_]] : index
// CHECK:             [[VAR_42_:%.+]] = arith.select [[VAR_41_]], [[CST_4_]], [[VAR_40_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_27_]], [[VAR_32_]], [[VAR_37_]], [[VAR_42_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1, [[VAR_22_]]#2, [[VAR_22_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_reflect_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2) {mode = "reflect"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG: #map0 = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG: #map2 = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG: #map3 = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG: #map4 = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG: #map5 = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG: #map6 = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG: #map7 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-LABEL:  func @pad_reflect_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map1(){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply #map2(){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply #map3(){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map4([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map5([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to #map6([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to #map7([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_22_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_22_]]#0, [[VAR_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.select [[VAR_23_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpi sge, [[VAR_26_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.subi [[CST_0_]], [[VAR_26_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_28_]], [[VAR_26_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_22_]]#1, [[VAR_6_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_30_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.cmpi sge, [[VAR_33_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.subi [[CST_4_]], [[VAR_33_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.select [[VAR_34_]], [[VAR_35_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.cmpi slt, [[VAR_22_]]#2, [[VAR_11_]] : index
// CHECK:             [[VAR_40_:%.+]] = arith.select [[VAR_37_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.cmpi sge, [[VAR_40_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.subi [[CST_6_]], [[VAR_40_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.select [[VAR_41_]], [[VAR_42_]], [[VAR_40_]] : index
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.cmpi slt, [[VAR_22_]]#3, [[VAR_16_]] : index
// CHECK:             [[VAR_47_:%.+]] = arith.select [[VAR_44_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.subi [[CST_8_]], [[VAR_47_]] : index
// CHECK:             [[VAR_50_:%.+]] = arith.select [[VAR_48_]], [[VAR_49_]], [[VAR_47_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_29_]], [[VAR_36_]], [[VAR_43_]], [[VAR_50_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1, [[VAR_22_]]#2, [[VAR_22_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_constant_mode_constant_pads(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3, 2, 1]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32> } : () -> tensor<1xf32>
  %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<16x16xf32>, tensor<4xi64>, tensor<1xf32>) -> tensor<18x20xf32>
  return %2 : tensor<18x20xf32>

// mlir2FileCheck.py -a'["data"]'
// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func @pad_constant_mode_constant_pads
// CHECK-SAME:   ([[DATA_:%.+]]: memref<16x16xf32>) -> memref<18x20xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<0.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<18x20xf32>
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<1xf32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_VAR_0_MEM_]] : memref<18x20xf32>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply #map([[VAR_4_]]#1)
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<16x16xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<18x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<18x20xf32>
// CHECK:         }
}

// -----

func.func @test_expand_with_arith.constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["input", "shape"]'
// CHECK-LABEL:  func @test_expand_with_arith.constant
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
// CHECK-DAG: #map = affine_map<()[s0, s1] -> (s1, s0)>
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
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.max #map(){{.}}[[VAR_1_]], [[VAR_4_]]{{.}}
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.max #map(){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}
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

// -----

func.func @test_cumsum_constant_axis(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis ="onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> 
  %0 = "onnx.CumSum"(%arg0, %axis) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]6] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis ="onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> 
  %0 = "onnx.CumSum"(%arg0, %axis) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]6] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----


func.func @test_cumsum_constant_axis_exclusive_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis ="onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> 
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.subi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]6] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----


func.func @test_cumsum_constant_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis ="onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> 
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]6] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map1(){{.}}[[VAR_1_]], [[VAR_1_]]5]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.subi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi sge, [[VAR_29_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map1(){{.}}[[VAR_1_]], [[VAR_1_]]5]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_29_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.subi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi sge, [[VAR_20_]], [[CST_0_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.subi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_0_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map1(){{.}}[[VAR_1_]], [[VAR_1_]]5]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.subi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi sge, [[VAR_29_1_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.addi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_20_]], [[CST_2_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_3_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map1(){{.}}[[VAR_1_]], [[VAR_1_]]5]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.addi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_dims(%arg0: tensor<?x?xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<?x?xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #map1 = affine_map<(d0)[s0, s1] -> (d0)>
// CHECK-DAG: #map2 = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG: #map3 = affine_map<(d0, d1)[s0, s1] -> (s1 + 1)>
// CHECK-DAG: #map4 = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-LABEL:  func @test_cumsum_dynamic_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<?x?xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_5_]], [[VAR_6_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_8_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_9_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_8_]], [[VAR_9_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.select [[VAR_11_]], [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK:           [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[VAR_15_]], [[VAR_13_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[VAR_16_]] : index to i64
// CHECK:           [[VAR_18_:%.+]] = arith.sitofp [[VAR_17_]] : i64 to f32
// CHECK:           [[VAR_19_:%.+]] = math.log2 [[VAR_18_]] : f32
// CHECK:           [[VAR_20_:%.+]] = arith.fptosi [[VAR_19_]] : f32 to i64
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.index_cast [[VAR_20_]] : i64 to index
// CHECK-DAG:       [[VAR_22_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_23_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map1([[VAR_22_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map2([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}){
// CHECK:             [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map3([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}){
// CHECK:             [[VAR_26_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_26_1_]] : index to i64
// CHECK:             [[VAR_28_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_29_:%.+]] = math.exp2 [[VAR_28_]] : f32
// CHECK:             [[VAR_30_:%.+]] = arith.fptosi [[VAR_29_]] : f32 to i64
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.index_cast [[VAR_30_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to #map4([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to #map2([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}){
// CHECK:               [[VAR_34_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<?x?xf64>
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.cmpi eq, [[CST_0_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.subi [[VAR_34_]]#0, [[VAR_31_]] : index
// CHECK:               [[VAR_38_:%.+]] = arith.cmpi sge, [[VAR_37_]], [[CST_0_]] : index
// CHECK:               [[VAR_39_:%.+]] = arith.andi [[VAR_36_]], [[VAR_38_]] : i1
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.select [[VAR_39_]], [[VAR_37_]], [[VAR_34_]]#0 : index
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.cmpi eq, [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.subi [[VAR_34_]]#1, [[VAR_31_]] : index
// CHECK:               [[VAR_43_:%.+]] = arith.cmpi sge, [[VAR_42_]], [[CST_0_]] : index
// CHECK:               [[VAR_44_:%.+]] = arith.andi [[VAR_41_]], [[VAR_43_]] : i1
// CHECK-DAG:           [[VAR_45_:%.+]] = arith.ori [[VAR_44_]], [[VAR_39_]] : i1
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.select [[VAR_44_]], [[VAR_42_]], [[VAR_34_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_40_]], [[VAR_46_]]{{.}} : memref<?x?xf64>
// CHECK:               [[VAR_48_:%.+]] = arith.select [[VAR_45_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_48_]] : f64
// CHECK:               krnl.store [[VAR_49_]], [[RES_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to #map4([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to #map2([[VAR_22_]], [[VAR_23_]]){{.}}[[VAR_1_]], [[VAR_21_]]{{.}}){
// CHECK:               [[VAR_34_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<?x?xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf64>
// CHECK:         }
}

// -----
// Compress on axis 0, with enough conditions, test elided

func.func @compress_axis0(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
//  use arg names: ['input', 'condition']
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_11_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_11_]]{{.}} : memref<?x2xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----
// Compress on axis 0, with not enough conditions, test not elided

func.func @compress_axis0_not_enough(%arg0: tensor<3x2xf32>, %arg1: tensor<2xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0_not_enough
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<2xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<2xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_5_1_]], [[VAR_c2_]] : index
// CHECK:             scf.if [[LOAD_CONDITION_MEM_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<2xi1>
// CHECK:               [[VAR_8_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[VAR_8_1_]] {
// CHECK-DAG:             [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_12_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_12_]]{{.}} : memref<3x2xf32>
// CHECK:                   krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_12_]]{{.}} : memref<?x2xf32>
// CHECK:                 }
// CHECK:                 [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----
// Compress on axis 1, with enough conditions, test elided

func.func @compress_axis1(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<3x?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<3x?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<3x?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 3){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_11_]], [[VAR_5_1_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_11_]], [[LOAD_RES_MEM_2_]]{{.}} : memref<3x?xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x?xf32>
// CHECK:         }
}

// -----
// Compress witn no axis , with not enough conditions, test not elided

func.func @compress_no_axis_not_elided(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

// CHECK-LABEL:  func @compress_no_axis_not_elided
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi slt, [[LOAD_CONDITION_MEM_1_]], [[VAR_c3_]] : index
// CHECK:             scf.if [[VAR_8_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<3xi1>
// CHECK:               [[LOAD_RES_MEM_2_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[LOAD_RES_MEM_2_]] {
// CHECK-DAG:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:             [[LOAD_RES_MEM_3_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_3_]]{{.}} : memref<?xf32>
// CHECK:                 [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_3_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_14_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:               [[VAR_11_1_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}

// -----
// Compress witn no axis , with enough conditions, test elided

func.func @compress_no_axis_enough_cond(%arg0: tensor<3x2xf32>, %arg1: tensor<6xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<6xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
// CHECK-LABEL:  func @compress_no_axis_enough_cond
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<6xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_9_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_9_1_]] {
// CHECK-DAG:           [[VAR_11_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]]{{.}} : memref<?xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_13_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_3_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:             krnl.store [[LOAD_RES_MEM_3_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}

// -----

func.func @test_hardmax_axis_1(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_hardmax_axis_1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5xindex>
// CHECK:           krnl.memset [[RES_1_]], [[VAR_c0_]] : memref<3x1x5xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_c0_]], [[VAR_4_]]#2] : memref<3x1x5xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_4_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<3x4x5xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_8_]] {
// CHECK:               krnl.store [[VAR_4_]]#1, [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_c0_]], [[VAR_4_]]#2] : memref<3x1x5xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_c0_]], [[VAR_4_1_]]#2] : memref<3x1x5xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_4_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[VAR_cst_0_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1, [[VAR_4_1_]]#2] : memref<3x4x5xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[VAR_cst_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1, [[VAR_4_1_]]#2] : memref<3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_hardmax_unknown_dims(%arg0: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["input"]'
// CHECK-DAG: #map0 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #map2 = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func @test_hardmax_unknown_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[INPUT_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[INPUT_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[INPUT_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {{.*}}: memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[INPUT_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[INPUT_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.dim [[INPUT_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_6_]]) {{.*}}: memref<?x1x?xindex>
// CHECK:           krnl.memset [[RES_1_]], [[VAR_c0_]] : memref<?x1x?xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map0([[VAR_4_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map1([[VAR_4_]], [[VAR_5_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to #map2([[VAR_4_]], [[VAR_5_]], [[VAR_6_]])){
// CHECK:             [[VAR_10_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_10_]]#0, [[VAR_c0_]], [[VAR_10_]]#2] : memref<?x1x?xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_10_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_10_]]#2] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_10_]]#0, [[VAR_10_]]#1, [[VAR_10_]]#2] : memref<?x?x?xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_14_]] {
// CHECK:               krnl.store [[VAR_10_]]#1, [[RES_1_]]{{.}}[[VAR_10_]]#0, [[VAR_c0_]], [[VAR_10_]]#2] : memref<?x1x?xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to #map0([[VAR_0_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to #map1([[VAR_0_]], [[VAR_1_]]), [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to #map2([[VAR_0_]], [[VAR_1_]], [[VAR_2_]])){
// CHECK:             [[VAR_10_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_10_1_]]#0, [[VAR_c0_]], [[VAR_10_1_]]#2] : memref<?x1x?xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_10_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[VAR_cst_0_]], [[RES_]]{{.}}[[VAR_10_1_]]#0, [[VAR_10_1_]]#1, [[VAR_10_1_]]#2] : memref<?x?x?xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[VAR_cst_]], [[RES_]]{{.}}[[VAR_10_1_]]#0, [[VAR_10_1_]]#1, [[VAR_10_1_]]#2] : memref<?x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?xf32>
// CHECK:         }
}

// -----

func.func @top_k(%arg0: tensor<3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @top_k
// CHECK-SAME:   ([[X_:%.+]]: memref<3x4xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<3x?xf32>, memref<3x?xi64>) {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x4xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_8_]]#1, [[RES_2_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<3x4xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_8_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = #map([[VAR_8_1_]]#1) to 4){
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_8_1_]]#1] : memref<3x4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_10_]]{{.}} : memref<3x4xindex>
// CHECK-DAG:           [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_1_]]#0, [[LOAD_RES_2_MEM_]]{{.}} : memref<3x4xf32>
// CHECK:               [[LOAD_X_MEM_1_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_1_]]#0, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<3x4xf32>
// CHECK:               [[VAR_15_:%.+]] = arith.cmpf olt, [[LOAD_X_MEM_]], [[LOAD_X_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_15_]] {
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_8_1_]]#1] : memref<3x4xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_]], [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_10_]]{{.}} : memref<3x4xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_8_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x4xindex>
// CHECK:             [[LOAD_X_MEM_2_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_2_]]#0, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_2_]], [[RES_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xf32>
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_1_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x?xf32>, memref<3x?xi64>
// CHECK:         }
}

// -----

func.func @top_k_smallest(%arg0: tensor<3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 0 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-DAG: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @top_k_smallest
// CHECK-SAME:   ([[X_:%.+]]: memref<3x4xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<3x?xf32>, memref<3x?xi64>) {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x4xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_8_]]#1, [[RES_2_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<3x4xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_8_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = #map([[VAR_8_1_]]#1) to 4){
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_8_1_]]#1] : memref<3x4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_10_]]{{.}} : memref<3x4xindex>
// CHECK-DAG:           [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_1_]]#0, [[LOAD_RES_2_MEM_]]{{.}} : memref<3x4xf32>
// CHECK:               [[LOAD_X_MEM_1_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_1_]]#0, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<3x4xf32>
// CHECK:               [[VAR_15_:%.+]] = arith.cmpf ogt, [[LOAD_X_MEM_]], [[LOAD_X_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_15_]] {
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_8_1_]]#1] : memref<3x4xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_]], [[RES_2_]]{{.}}[[VAR_8_1_]]#0, [[VAR_10_]]{{.}} : memref<3x4xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_8_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x4xindex>
// CHECK:             [[LOAD_X_MEM_2_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_2_]]#0, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_2_]], [[RES_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xf32>
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_1_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x?xf32>, memref<3x?xi64>
// CHECK:         }
}

// -----

func.func @top_k_unknown_dims(%arg0: tensor<?x?xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<?x?xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-DAG: #map0 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #map3 = affine_map<(d0, d1) -> (d1 - 1)>
// CHECK-DAG: #map4 = affine_map<(d0, d1, d2) -> (d2 + 1)>
// CHECK-DAG: #map5 = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #map6 = affine_map<(d0)[s0] -> (s0)>
// CHECK-LABEL:  func @top_k_unknown_dims
// CHECK-SAME:   ([[X_:%.+]]: memref<?x?xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<?x?xf32>, memref<?x?xi64>) {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[X_]], [[VAR_c0_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_2_]], [[VAR_1_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_2_]], [[VAR_1_]]) {{.*}}: memref<?x?xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[X_]], [[VAR_c0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.dim [[X_]], [[VAR_c1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_5_]], [[VAR_6_]]) {{.*}}: memref<?x?xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map0([[VAR_5_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to #map1([[VAR_5_]], [[VAR_6_]])){
// CHECK:             [[VAR_11_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_11_]]#1, [[RES_2_]]{{.}}[[VAR_11_]]#0, [[VAR_11_]]#1] : memref<?x?xindex>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to #map2([[VAR_5_]], [[VAR_6_]]), [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to #map3([[VAR_5_]], [[VAR_6_]])){
// CHECK-DAG:         [[VAR_11_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = #map4([[VAR_5_]], [[VAR_6_]], [[VAR_11_1_]]#1) to #map5([[VAR_5_]], [[VAR_6_]], [[VAR_11_1_]]#1)){
// CHECK-DAG:           [[VAR_13_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_11_1_]]#0, [[VAR_11_1_]]#1] : memref<?x?xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_11_1_]]#0, [[VAR_13_]]{{.}} : memref<?x?xindex>
// CHECK-DAG:           [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_11_1_]]#0, [[LOAD_RES_2_MEM_]]{{.}} : memref<?x?xf32>
// CHECK:               [[LOAD_X_MEM_1_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_11_1_]]#0, [[LOAD_RES_2_MEM_1_]]{{.}} : memref<?x?xf32>
// CHECK:               [[VAR_18_:%.+]] = arith.cmpf olt, [[LOAD_X_MEM_]], [[LOAD_X_MEM_1_]] : f32
// CHECK:               scf.if [[VAR_18_]] {
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_2_]]{{.}}[[VAR_11_1_]]#0, [[VAR_11_1_]]#1] : memref<?x?xindex>
// CHECK:                 krnl.store [[LOAD_RES_2_MEM_]], [[RES_2_]]{{.}}[[VAR_11_1_]]#0, [[VAR_13_]]{{.}} : memref<?x?xindex>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to #map0([[VAR_2_]]), [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to #map6([[VAR_2_]]){{.}}[[VAR_1_]]{{.}}){
// CHECK:             [[VAR_11_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<?x?xindex>
// CHECK:             [[LOAD_X_MEM_2_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_11_2_]]#0, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_2_]], [[RES_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<?x?xf32>
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_1_]]{{.}}[[VAR_11_2_]]#0, [[VAR_11_2_]]#1] : memref<?x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?xf32>, memref<?x?xi64>
// CHECK:         }
}

// -----

func.func @test_loop_tiny_yolo() -> tensor<?xi32> {
    %0 = "onnx.Constant"() {value = dense<7> : tensor<i64>} : () -> tensor<i64>
    %1 = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %3:2 = "onnx.Loop"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<i32>):  // no predecessors
      %4 = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %5 = "onnx.Add"(%arg2, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      onnx.Return %arg1, %5, %arg2 : tensor<i1>, tensor<i32>, tensor<i32>
    }) {input_names = ["i", "cond", "prev"], output_names = ["cond_out", "current", "range"]} : (tensor<i64>, tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<?xi32>)
    return %3#1 : tensor<?xi32>

// CHECK-LABEL:  func @test_loop_tiny_yolo
// CHECK-SAME:   () -> memref<?xi32> {
// CHECK-DAG:       [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<7> : tensor<i64>} : () -> memref<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<true> : tensor<i1>} : () -> memref<i1>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<0> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i32>
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi32>
// CHECK-DAG:       [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]][] : memref<i32>
// CHECK:           krnl.store [[LOAD_VAR_2_MEM_]], [[RES_]][] : memref<i32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK:           krnl.store [[LOAD_VAR_1_MEM_]], [[RES_2_]][] : memref<i1>
// CHECK:           [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_12_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> %arg0 = [[ZERO]] to [[VAR_12_]]){
// CHECK:             [[I_0_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<i1>
// CHECK:             scf.if [[LOAD_RES_2_MEM_]] {
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.index_cast [[I_0_]] : index to i64
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloc() : memref<i64>
// CHECK:               krnl.store [[VAR_14_]], [[RES_3_]][] : memref<i64>
// CHECK-DAG:           [[VAR_16_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloc() : memref<i32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]][] : memref<i32>
// CHECK:               [[VAR_20_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_VAR_16_MEM_]] : i32
// CHECK:               krnl.store [[VAR_20_]], [[RES_4_]][] : memref<i32>
// CHECK:               [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK:               krnl.store [[LOAD_VAR_1_MEM_1_]], [[RES_2_]][] : memref<i1>
// CHECK:               [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[I_0_]]{{.}} : memref<?xi32>
// CHECK:               [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_4_MEM_]], [[RES_]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xi32>
// CHECK:         }
}

// -----

// Check lowering transpose to a view op when the order of the dimensions whose
// value is not 1 is unchanged.
func.func @test_transpose_lowered_to_a_view_op(%arg0: tensor<?x1x1x384xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<?x1x1x384xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL:  func @test_transpose_lowered_to_a_view_op
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x384xf32>) -> memref<?x384x1x1xf32> {
  // CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
  // CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x1x384xf32>
  // CHECK:           [[VAR_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_0_]], 384, 1, 1], strides: [384, 1, 1, 1] : memref<?x1x1x384xf32> to memref<?x384x1x1xf32>
  // CHECK:           return [[VAR_1_]] : memref<?x384x1x1xf32>
  // CHECK:         }
}

// -----

// Check lowering transpose to a view op when the order of the dimensions whose
// value is not 1 is unchanged.
// The order of the dimension whose value is not 1 is changed by transpose.
func.func @test_transpose_lowered_to_a_view_op_inv(%arg0: tensor<?x1x1x384xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [3, 0, 1, 2]} : (tensor<?x1x1x384xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL:  func @test_transpose_lowered_to_a_view_op_inv
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x384xf32>) -> memref<384x?x1x1xf32> {
  // CHECK-NOT:       memref.reinterpret_cast

}
