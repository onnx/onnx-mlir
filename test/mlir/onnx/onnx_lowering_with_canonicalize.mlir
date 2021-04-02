// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map, which are otherwise
// before the function, and thus are hard to test.

// -----

// Slice where all the parameters are constant.
func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = constant unit
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2) {
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = constant unit
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3) {
// CHECK:             [[VAR_6_:%.+]] = affine.apply #map([[I_0_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3xf32>
// CHECK:         }
}

// -----

func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2) {
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, -1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, -1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2) {
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2) {
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, -2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2) {
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  // %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  // slice * 1-3 1-4 with neg numbers
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%res) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[2, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<-1> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = alloc([[DIM_0_]]) : memref<?x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[DIM_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3) {
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply #map([[I_1_]])
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply #map([[I_2_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_7_]], [[VAR_8_]]{{.}} : memref<?x4x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x2x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x3xf32>
// CHECK:         }
}

// -----

// Check where all is dynamic except input size and axis. The code was verified
// using a procedure simioar to mlir-run and by manually adding code to print the
// output as a vector

func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) {
   %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>

  // slice * 1-3 1-4 with neg numbers
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x?x?xi64>
  return

// CHECK-LABEL:  func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2xi64>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_5_:%.+]] = constant 5 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = constant -1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_minus_2147483648_:%.+]] = constant -2147483648 : index
// CHECK-DAG:       [[CST_2147483647_:%.+]] = constant 2147483647 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_0", shape = [3, 4, 5], value = dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>} : () -> memref<3x4x5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[2, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_8_:%.+]] = cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map0(){{.}}[[VAR_3_]]{{.}}
// CHECK:           [[VAR_10_:%.+]] = select [[VAR_8_]], [[VAR_9_]], [[VAR_3_]] : index
// CHECK:           [[VAR_11_:%.+]] = cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = select [[VAR_11_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = cmpi sgt, [[VAR_12_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = select [[VAR_13_]], [[CST_4_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_16_:%.+]] = select [[VAR_15_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_17_:%.+]] = cmpi sgt, [[VAR_16_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = select [[VAR_17_]], [[CST_5_]], [[VAR_16_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = select [[VAR_19_]], [[VAR_14_]], [[VAR_18_]] : index
// CHECK-DAG:       [[VAR_21_:%.+]] = cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_22_:%.+]] = affine.apply #map0(){{.}}[[VAR_5_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = select [[VAR_21_]], [[VAR_22_]], [[VAR_5_]] : index
// CHECK-DAG:       [[VAR_24_:%.+]] = cmpi sle, [[VAR_5_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = select [[VAR_24_]], [[CST_minus_1_]], [[VAR_23_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = cmpi sge, [[VAR_5_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_27_:%.+]] = select [[VAR_26_]], [[CST_5_]], [[VAR_25_]] : index
// CHECK:           [[VAR_28_:%.+]] = cmpi slt, [[VAR_27_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_29_:%.+]] = select [[VAR_28_]], [[CST_minus_1_]], [[VAR_27_]] : index
// CHECK:           [[VAR_30_:%.+]] = cmpi sgt, [[VAR_29_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = select [[VAR_30_]], [[CST_5_]], [[VAR_29_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = cmpi slt, [[VAR_27_]], [[CST_0_]] : index
// CHECK:           [[VAR_33_:%.+]] = select [[VAR_32_]], [[CST_0_]], [[VAR_27_]] : index
// CHECK:           [[VAR_34_:%.+]] = cmpi sgt, [[VAR_33_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = select [[VAR_34_]], [[CST_5_]], [[VAR_33_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK:           [[VAR_37_:%.+]] = select [[VAR_36_]], [[VAR_31_]], [[VAR_35_]] : index
// CHECK:           [[VAR_38_:%.+]] = subi [[VAR_37_]], [[VAR_20_]] : index
// CHECK:           [[VAR_39_:%.+]] = ceildivi_signed [[VAR_38_]], [[VAR_7_]] : index
// CHECK:           [[VAR_40_:%.+]] = cmpi slt, [[VAR_39_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_41_:%.+]] = select [[VAR_40_]], [[CST_0_]], [[VAR_39_]] : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_43_:%.+]] = index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_47_:%.+]] = index_cast [[LOAD_PARAM_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_48_:%.+]] = cmpi slt, [[VAR_43_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_49_:%.+]] = affine.apply #map1(){{.}}[[VAR_43_]]{{.}}
// CHECK:           [[VAR_50_:%.+]] = select [[VAR_48_]], [[VAR_49_]], [[VAR_43_]] : index
// CHECK:           [[VAR_51_:%.+]] = cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_52_:%.+]] = select [[VAR_51_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_53_:%.+]] = cmpi sgt, [[VAR_52_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_54_:%.+]] = select [[VAR_53_]], [[CST_3_]], [[VAR_52_]] : index
// CHECK-DAG:       [[VAR_55_:%.+]] = cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_56_:%.+]] = select [[VAR_55_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_57_:%.+]] = cmpi sgt, [[VAR_56_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_58_:%.+]] = select [[VAR_57_]], [[CST_4_]], [[VAR_56_]] : index
// CHECK-DAG:       [[VAR_59_:%.+]] = cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_60_:%.+]] = select [[VAR_59_]], [[VAR_54_]], [[VAR_58_]] : index
// CHECK-DAG:       [[VAR_61_:%.+]] = cmpi slt, [[VAR_45_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_62_:%.+]] = affine.apply #map1(){{.}}[[VAR_45_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_63_:%.+]] = select [[VAR_61_]], [[VAR_62_]], [[VAR_45_]] : index
// CHECK-DAG:       [[VAR_64_:%.+]] = cmpi sle, [[VAR_45_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_65_:%.+]] = select [[VAR_64_]], [[CST_minus_1_]], [[VAR_63_]] : index
// CHECK-DAG:       [[VAR_66_:%.+]] = cmpi sge, [[VAR_45_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_67_:%.+]] = select [[VAR_66_]], [[CST_4_]], [[VAR_65_]] : index
// CHECK:           [[VAR_68_:%.+]] = cmpi slt, [[VAR_67_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_69_:%.+]] = select [[VAR_68_]], [[CST_minus_1_]], [[VAR_67_]] : index
// CHECK:           [[VAR_70_:%.+]] = cmpi sgt, [[VAR_69_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_71_:%.+]] = select [[VAR_70_]], [[CST_4_]], [[VAR_69_]] : index
// CHECK-DAG:       [[VAR_72_:%.+]] = cmpi slt, [[VAR_67_]], [[CST_0_]] : index
// CHECK:           [[VAR_73_:%.+]] = select [[VAR_72_]], [[CST_0_]], [[VAR_67_]] : index
// CHECK:           [[VAR_74_:%.+]] = cmpi sgt, [[VAR_73_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_75_:%.+]] = select [[VAR_74_]], [[CST_4_]], [[VAR_73_]] : index
// CHECK-DAG:       [[VAR_76_:%.+]] = cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK:           [[VAR_77_:%.+]] = select [[VAR_76_]], [[VAR_71_]], [[VAR_75_]] : index
// CHECK:           [[VAR_78_:%.+]] = subi [[VAR_77_]], [[VAR_60_]] : index
// CHECK:           [[VAR_79_:%.+]] = ceildivi_signed [[VAR_78_]], [[VAR_47_]] : index
// CHECK:           [[VAR_80_:%.+]] = cmpi slt, [[VAR_79_]], [[CST_0_]] : index
// CHECK:           [[VAR_81_:%.+]] = select [[VAR_80_]], [[CST_0_]], [[VAR_79_]] : index
// CHECK-DAG:       [[RES_:%.+]] = alloc([[VAR_81_]], [[VAR_41_]]) : memref<3x?x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_81_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[VAR_41_]]) {
// CHECK:             [[VAR_84_:%.+]] = muli [[VAR_47_]], [[I_1_]] : index
// CHECK-DAG:         [[VAR_85_:%.+]] = addi [[VAR_84_]], [[VAR_60_]] : index
// CHECK-DAG:         [[VAR_86_:%.+]] = muli [[VAR_7_]], [[I_2_]] : index
// CHECK:             [[VAR_87_:%.+]] = addi [[VAR_86_]], [[VAR_20_]] : index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[I_0_]], [[VAR_85_]], [[VAR_87_]]{{.}} : memref<3x4x5xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x?x?xi64>
// CHECK:           }
// CHECK:           dealloc [[RES_]] : memref<3x?x?xi64>
// CHECK:           return
// CHECK:         }
}

// -----

// GEMM with everything constant
func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 5) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_0_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:               [[VAR_11_:%.+]] = mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_12_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_11_]] : f32
// CHECK:               krnl.store [[VAR_12_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK-DAG:         [[VAR_4_:%.+]] = mulf [[CST_1_dot_000000_]], [[LOAD_RES_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[I_1_]]{{.}} : memref<10xf32>
// CHECK:             [[VAR_6_:%.+]] = mulf [[CST_5_dot_000000_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = addf [[VAR_4_]], [[VAR_6_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Gemm with all dimensions dynamic
func @test_gemm_all_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>, [[PARAM_1_:%.+]]: memref<?x?xf32>, [[PARAM_2_:%.+]]: memref<?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = dim [[PARAM_1_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[DIM_3_:%.+]] = dim [[PARAM_2_]], [[CST_0_]] : memref<?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = alloc([[DIM_0_]], [[DIM_2_]]) : memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[DIM_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[DIM_2_]]) {
// CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[DIM_1_]]) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_0_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:               [[VAR_17_:%.+]] = mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_18_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_17_]] : f32
// CHECK:               krnl.store [[VAR_18_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[VAR_7_:%.+]] = cmpi sgt, [[DIM_3_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = select [[VAR_7_]], [[I_1_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = mulf [[CST_1_dot_000000_]], [[LOAD_RES_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_8_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_12_:%.+]] = mulf [[CST_5_dot_000000_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_13_:%.+]] = addf [[VAR_10_]], [[VAR_12_]] : f32
// CHECK:             krnl.store [[VAR_13_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----

// A[10, *] * B[*, 10] result in constant size output but dyn reduction.
func @test_gemm_k_dyn(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x10xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_k_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>, [[PARAM_1_:%.+]]: memref<?x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<10x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[DIM_0_]]) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_0_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:               [[VAR_12_:%.+]] = mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_13_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_12_]] : f32
// CHECK:               krnl.store [[VAR_13_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK-DAG:         [[VAR_5_:%.+]] = mulf [[CST_1_dot_000000_]], [[LOAD_RES_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[I_1_]]{{.}} : memref<10xf32>
// CHECK:             [[VAR_7_:%.+]] = mulf [[CST_5_dot_000000_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = addf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Broadcast bias C is dym, so we don't know if its 1 -> broadcast or 10. Dyn test for that.
func @test_gemm_c_dyn(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_c_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<?xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<10x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_2_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 5) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_0_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:               [[VAR_14_:%.+]] = mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_15_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_14_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
// CHECK:             [[VAR_4_:%.+]] = cmpi sgt, [[DIM_0_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = select [[VAR_4_]], [[I_1_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = mulf [[CST_1_dot_000000_]], [[LOAD_RES_MEM_1_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_9_:%.+]] = mulf [[CST_5_dot_000000_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_10_:%.+]] = addf [[VAR_7_]], [[VAR_9_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Test tile with constant repeats
func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @test_tile1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8xf32>) -> memref<12x16xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<12x16xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[3, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 12, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16) {
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply #map0([[I_0_]])
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply #map1([[I_1_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]], [[VAR_4_]]{{.}} : memref<4x8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<12x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<12x16xf32>
// CHECK:         }
}

// -----

// Test tile without constant repeats
func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @test_tile2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8xf32>, [[PARAM_1_:%.+]]: memref<1xi64>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_0_]]{{\]}} : memref<1xi64>
// CHECK:           [[VAR_1_:%.+]] = index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK:           [[VAR_2_:%.+]] = affine.apply #map0(){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = alloc([[VAR_2_]]) : memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_2_]]) {
// CHECK:             [[VAR_5_:%.+]] = affine.apply #map1([[I_0_]])
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]{{.}} : memref<8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "std.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2, 2], value = dense<{{.}}[0, 1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2) {
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[I_2_]]{{.}} : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func @test_gather_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "std.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2) {
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = addi [[VAR_4_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[I_2_]]{{.}} : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 1, second example in ONNX for Gather.
func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "std.return"(%0) : (tensor<3x1x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<3x1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", shape = [1, 2], value = dense<{{.}}[0, 2]{{.}}> : tensor<1x2xi64>} : () -> memref<1x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2) {
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[I_1_]], [[I_2_]]{{.}} : memref<1x2xi64>
// CHECK:             [[VAR_4_:%.+]] = index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_4_]]{{.}} : memref<3x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and explicit split.
func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_split_unknown_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x2x64xf32>, memref<?x30x64xf32>) {
// CHECK:           [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = alloc([[DIM_0_]]) : memref<?x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = alloc([[DIM_1_]]) : memref<?x30x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[DIM_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 64) {
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x2x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[DIM_1_]], [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 30, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 64) {
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply #map([[I_4_]])
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_3_]], [[LOAD_PARAM_0_MEM_1_]], [[I_5_]]{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[I_3_]], [[I_4_]], [[I_5_]]{{.}} : memref<?x30x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x2x64xf32>, memref<?x30x64xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and default split.
func @test_split_unknown_dimension_equal_split(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 : si64 } : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_split_unknown_dimension_equal_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x?x64xf32>, memref<?x?x64xf32>) {
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply #map0(){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = alloc([[DIM_1_]], [[VAR_3_]]) : memref<?x?x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = alloc([[DIM_2_]], [[VAR_5_]]) : memref<?x?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[DIM_1_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_3_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 64) {
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[DIM_2_]], [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[VAR_5_]], [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 64) {
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply #map1([[I_4_]]){{.}}[[DIM_0_]]{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_3_]], [[LOAD_PARAM_0_MEM_1_]], [[I_5_]]{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[I_3_]], [[I_4_]], [[I_5_]]{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?x64xf32>, memref<?x?x64xf32>
// CHECK:         }
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of i32.
func @test_reducemean_i32_unknown_dims(%arg0 : tensor<3x?x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xi32>)-> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemean_i32_unknown_dims
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: krnl.iterate
  // CHECK: krnl.iterate
  // CHECK: [[DIM:%.+]] = dim %arg0, [[ONE]] : memref<3x?x2xi32>
  // CHECK: [[DIVISOR:%.+]] = index_cast [[DIM]] : index to i32
  // CHECK: krnl.iterate
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of f32.
func @test_reducemean_f32_unknown_dims(%arg0 : tensor<3x?x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemean_f32_unknown_dims
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: krnl.iterate
  // CHECK: krnl.iterate
  // CHECK: [[DIM:%.+]] = dim %arg0, [[ONE]] : memref<3x?x2xf32>
  // CHECK: [[UNKNOWN_DIM_i64:%.+]] = index_cast [[DIM]] : index to i64
  // CHECK: [[DIVISOR:%.+]] = uitofp [[UNKNOWN_DIM_i64]] : i64 to f32
  // CHECK: krnl.iterate
}

// -----

// COM: Check the template for lowering binary operations whose output type can be different from its input type.
func @test_binary_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x5xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x4x5xf32>, tensor<1x?x1xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
// CHECK-LABEL:  func @test_binary_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>, [[PARAM_1_:%.+]]: memref<1x?x1xf32>) -> memref<?x4x5xi1> {
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_1_]], [[CST_1_]] : memref<1x?x1xf32>
// CHECK:           [[VAR_2_:%.+]] = affine.max #map(){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = alloc([[VAR_2_]]) : memref<?x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_2_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[I_0_]], [[I_1_]], [[I_2_]]] : memref<?x4x5xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = cmpi sgt, [[DIM_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = select [[VAR_6_]], [[I_1_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_7_]], [[CST_0_]]{{.}} : memref<1x?x1xf32>
// CHECK:             [[VAR_9_:%.+]] = cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_9_]], [[RES_]]{{\[}}[[I_0_]], [[I_1_]], [[I_2_]]{{\]}} : memref<?x4x5xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xi1>
// CHECK:         }
}

// -----

// COM: Check the template for lowering variadic operations and binary operations whose output type is the same as its input type: Min, Max, Add, Sub, etc. 
func @test_variadic_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x1xf32>, %arg1: tensor<?x?x5xf32>, %arg2: tensor<?x1x5xf32>) -> tensor<?x4x5xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<?x4x1xf32>, tensor<?x?x5xf32>, tensor<?x1x5xf32>) -> tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
// CHECK-LABEL:  func @test_variadic_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x1xf32>, [[PARAM_1_:%.+]]: memref<?x?x5xf32>, [[PARAM_2_:%.+]]: memref<?x1x5xf32>) -> memref<?x4x5xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x1xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[DIM_3_:%.+]] = dim [[PARAM_2_]], [[CST_0_]] : memref<?x1x5xf32>
// CHECK:           [[VAR_4_:%.+]] = affine.max #map(){{.}}[[DIM_0_]], [[DIM_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = alloc([[VAR_4_]]) : memref<?x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_4_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5) {
// CHECK:             [[VAR_7_:%.+]] = cmpi sgt, [[DIM_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_8_:%.+]] = select [[VAR_7_]], [[I_0_]], [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]], [[I_1_]], [[CST_0_]]{{.}} : memref<?x4x1xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = cmpi sgt, [[DIM_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = select [[VAR_10_]], [[I_0_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = cmpi sgt, [[DIM_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_13_:%.+]] = select [[VAR_12_]], [[I_1_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_11_]], [[VAR_13_]], [[I_2_]]{{.}} : memref<?x?x5xf32>
// CHECK:             [[VAR_15_:%.+]] = cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_16_:%.+]] = select [[VAR_15_]], [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_17_:%.+]] = cmpi sgt, [[DIM_3_]], [[CST_1_]] : index
// CHECK:             [[VAR_18_:%.+]] = select [[VAR_17_]], [[I_0_]], [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_18_]], [[CST_0_]], [[I_2_]]{{.}} : memref<?x1x5xf32>
// CHECK:             [[VAR_20_:%.+]] = cmpf ogt, [[VAR_16_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_21_:%.+]] = select [[VAR_20_]], [[VAR_16_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_21_]], [[RES_]]{{\[}}[[I_0_]], [[I_1_]], [[I_2_]]{{\]}} : memref<?x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: Because of unidirectional broadcasting, always get constant dimensions from X even thought their values are 1.
func @test_prelu_broadcast_unknown_dims(%arg0: tensor<3x1x5xf32>, %arg1: tensor<3x?x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x1x5xf32>, tensor<3x?x1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: @test_prelu_broadcast_unknown_dims
  // CHECK-DAG: [[CST0_f32:%.+]] = constant 0.000000e+00 : f32
  // CHECK-DAG: [[CST0:%.+]] = constant 0 : index 
  // CHECK:     [[RES:%.+]] = alloc() : memref<3x1x5xf32>
  // CHECK:     [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 1, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x1x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, %arg3, [[CST0]]{{\]}} : memref<3x?x1xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST0_f32]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x1x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x1x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: If X's dimensions are unknown, get dimensions from slope whenever they are non-zero constants.
func @test_prelu_broadcast_unknown_dims1(%arg0: tensor<?x2x?xf32>, %arg1: tensor<?x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x2x?xf32>, tensor<?x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: @test_prelu_broadcast_unknown_dims1
  // CHECK-DAG: [[CST0:%.+]] = constant 0 : index
  // CHECK-DAG: [[CST1:%.+]] = constant 1 : index
  // CHECK-DAG: [[CST0_f32:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[DIM0_X:%.+]] = dim %arg0, [[CST0]] : memref<?x2x?xf32>
  // CHECK:     [[DIM0_SLOPE:%.+]] = dim %arg1, [[CST0]] : memref<?x5xf32>
  // CHECK:     [[RES:%.+]] = alloc([[DIM0_X]]) : memref<?x2x5xf32>
  // CHECK:     [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to [[DIM0_X]], [[MAIN_LOOP]]#1 -> %arg3 = 0 to 2, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<?x2x?xf32>
  // CHECK:       [[GREATER_THAN_ONE:%.+]] = cmpi sgt, [[DIM0_SLOPE]], [[CST1]] : index
  // CHECK:       [[SELECT1:%.+]] = select [[GREATER_THAN_ONE]], %arg3, [[CST0]] : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[SELECT1]], %arg4] : memref<?x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST0_f32]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT2:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT2]], [[RES]][%arg2, %arg3, %arg4] : memref<?x2x5xf32>
  // CHECK:     }
  // CHECK:     return [[RES]] : memref<?x2x5xf32>
}

// -----

/// Check ReduceMean with f32.
func private @test_reducemean_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemean_f32
  // CHECK-DAG: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK-DAG: [[DIVISOR_I64:%.+]] = constant 2 : i64 
  // CHECK-DAG: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK-DAG: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }

  // CHECK: [[DIVISOR:%.+]] = uitofp [[DIVISOR_I64]] : i64 to f32
  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2) {
  // CHECK:   [[LOAD3:%.+]] = krnl.load [[RES]][%arg1, %arg2] : memref<3x2xf32>
  // CHECK:   [[MEAN:%.+]] = divf [[LOAD3]], [[DIVISOR]] : f32
  // CHECK:   krnl.store [[MEAN]], [[RES]][%arg1, %arg2] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

/// Check ReduceMean with i32.
func private @test_reducemean_i32(%arg0 : tensor<3x2x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi32>)-> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_reducemean_i32
  // CHECK: [[IDENTITY:%.+]] = constant 0 : i32
  // CHECK: [[DIVISOR:%.+]] = constant 2 : i32
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xi32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xi32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xi32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: [[REDUCE:%.+]] = addi [[LOAD2]], [[LOAD1]] : i32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: }

  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2) {
  // CHECK:   [[LOAD3:%.+]] = krnl.load [[RES]][%arg1, %arg2] : memref<3x2xi32>
  // CHECK:   [[MEAN:%.+]] = divi_signed [[LOAD3]], [[DIVISOR]] : i32
  // CHECK:   krnl.store [[MEAN]], [[RES]][%arg1, %arg2] : memref<3x2xi32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xi32>
}

// -----

func private @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "std.return"(%1) : (tensor<5x5x9x32xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = alloc() : memref<5x5x9x32xf32>
  // CHECK: [[DEF_LOOPS0:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) with ([[DEF_LOOPS0]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS0]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS0]]#2 -> %arg5 = 0 to 1, [[DEF_LOOPS0]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[LOAD0:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x1x32xf32>
  // CHECK: krnl.store [[LOAD0]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS1]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS1]]#2 -> %arg5 = 0 to 3, [[DEF_LOOPS1]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[AFFINE_APPLY1:%.+]] = affine.apply #{{.*}}(%arg5)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg1[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x3x32xf32>
  // CHECK: krnl.store [[LOAD1]], [[RES]][%arg3, %arg4, [[AFFINE_APPLY1]], %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2, [[DEF_LOOPS2]]#3) with ([[DEF_LOOPS2]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS2]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS2]]#2 -> %arg5 = 0 to 5, [[DEF_LOOPS2]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[AFFINE_APPLY2:%.+]] = affine.apply #{{.*}}(%arg5)
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg2[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x5x32xf32>
  // CHECK: krnl.store [[LOAD2]], [[RES]][%arg3, %arg4, [[AFFINE_APPLY2]], %arg6] : memref<5x5x9x32xf32>

  // CHECK: return [[RES]] :  memref<5x5x9x32xf32>
}

// -----
// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func @test_prelu_broadcast3(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast3
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, [[ZERO_INDEX]], %arg4] : memref<3x1x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----
// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func @test_prelu_broadcast3(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast3
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, [[ZERO_INDEX]], %arg4] : memref<3x1x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----
// COM: 2D matmul.
func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]'
// CHECK-LABEL:  func private @test_matmul1
// CHECK-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<16x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_16_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_16_]]) {
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x16xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[loop_block_:%.+]], [[VAR_loop_local_:%.+]] = krnl.block [[LOOP_1_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[loop_block_0_:%.+]], [[VAR_loop_local_1_:%.+]] = krnl.block [[LOOP_1_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[loop_block_2_:%.+]], [[VAR_loop_local_3_:%.+]] = krnl.block [[LOOP_1_]]#2 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[loop_block_]], [[VAR_loop_local_]], [[loop_block_]]_0, [[VAR_loop_local_]]_1, [[loop_block_]]_2, [[VAR_loop_local_]]_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[loop_block_]], [[loop_block_]]_0, [[loop_block_]]_2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = [[CST_0_]] to [[CST_16_]], [[LOOP_1_]]#1 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_16_]], [[LOOP_1_]]#2 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_16_]]) {
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[loop_block_]], [[loop_block_]]_0, [[loop_block_]]_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[A_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[B_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[VAR_loop_local_]], [[VAR_loop_local_]]_1, [[VAR_loop_local_]]_3), ([[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2), ([[CST_16_]], [[CST_16_]], [[CST_16_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 4], overcompute = false, simdize = true, unroll = true} : memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x16xf32>
// CHECK:         }
}