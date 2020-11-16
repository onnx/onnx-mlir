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

// CHECK-LABEL:       func @test_slice_constant_default_axes
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK:           [[ALLOC:%.+]] = alloc() : memref<1x2xf32>
// CHECK:           [[START:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[END:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[STEPS:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[DIM0:%.+]] = 0 to 1, [[ITERS]]#1 -> [[DIM1:%.+]] = 0 to 2) {
// CHECK:             [[VAL:%.+]] = affine.load [[VAR_arg0]][symbol([[DIM0]]) + 1, symbol([[DIM1]]) * 2] : memref<2x4xf32>
// CHECK:             affine.store [[VAL]], [[ALLOC]][symbol([[DIM0]]), symbol([[DIM1]])] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[ALLOC]] : memref<1x2xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = constant unit
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_slice_constant_default_steps
// CHECK-SAME:     ([[INPUT:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK:           [[ALLOC:%.+]] = alloc() : memref<1x3xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_3:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[I:%.+]] = 0 to 1, [[ITERS]]#1 -> [[J:%.+]] = 0 to 3) {
// CHECK:             [[VAR_5:%.+]] = affine.load [[INPUT]][symbol([[I]]) + 1, symbol([[J]])] : memref<2x4xf32>
// CHECK:             affine.store [[VAR_5]], [[ALLOC]][symbol([[I]]), symbol([[J]])] : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[ALLOC]] : memref<1x3xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_slice_all_constant
// CHECK-SAME:     ([[INPUT:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK:           [[ALLOC:%.+]] = alloc() : memref<1x2xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_3:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_4:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[I:%.+]] = 0 to 1, [[ITERS]]#1 -> [[J:%.+]] = 0 to 2) {
// CHECK:             [[VAR_6:%.+]] = affine.load [[INPUT]][symbol([[I]]) + 1, symbol([[J]]) * 2] : memref<2x4xf32>
// CHECK:             affine.store [[VAR_6]], [[ALLOC]][symbol([[I]]), symbol([[J]])] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[ALLOC]] : memref<1x2xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_slice_all_constant_negative
// CHECK-SAME:     ([[INPUT:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK:           [[DATA:%.+]] = alloc() : memref<1x2xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, -1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_3:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, -1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_4:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[I:%.+]] = 0 to 1, [[ITERS]]#1 -> [[J:%.+]] = 0 to 2) {
// CHECK:             [[VAR_6:%.+]] = affine.load [[INPUT]][symbol([[I]]) + 1, symbol([[J]]) * 2] : memref<2x4xf32>
// CHECK:             affine.store [[VAR_6]], [[DATA]][symbol([[I]]), symbol([[J]])] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[DATA]] : memref<1x2xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_slice_all_constant_end_outofbound
// CHECK-SAME:     ([[INPUT:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK:           [[DATA:%.+]] = alloc() : memref<1x2xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_3:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_4:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[I:%.+]] = 0 to 1, [[ITERS]]#1 -> [[J:%.+]] = 0 to 2) {
// CHECK:             [[VAR_6:%.+]] = affine.load [[INPUT]][symbol([[I]]) + 1, symbol([[J]]) * 2] : memref<2x4xf32>
// CHECK:             affine.store [[VAR_6]], [[DATA]][symbol([[I]]), symbol([[J]])] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[DATA]] : memref<1x2xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_slice_all_constant_negative_steps
// CHECK-SAME:     ([[DATA:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK:           [[INPUT:%.+]] = alloc() : memref<1x2xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[1, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_3:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<[2, 0]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_4:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<[1, -2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[ITERS:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[ITERS]]#0, [[ITERS]]#1) with ([[ITERS]]#0 -> [[I:%.+]] = 0 to 1, [[ITERS]]#1 -> [[J:%.+]] = 0 to 2) {
// CHECK:             [[VAR_6:%.+]] = affine.load [[DATA]][symbol([[I]]) + 1, symbol([[J]]) * -2 + 3] : memref<2x4xf32>
// CHECK:             affine.store [[VAR_6]], [[INPUT]][symbol([[I]]), symbol([[J]])] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[INPUT]] : memref<1x2xf32>
// CHECK:         }
// CHECK:       }
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

//CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
//CHECK-SAME:   ([[DATA_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
//CHECK:           [[CST_0_:%.+]] = constant 0 : index
//CHECK:           [[AXIS_:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[2, 1]> : tensor<2xi64>} : () -> memref<2xi64>
//CHECK:           [[START_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
//CHECK:           [[END_:%.+]] = "krnl.global"() {name = "constant_2", shape = [2], value = dense<-1> : tensor<2xi64>} : () -> memref<2xi64>
//CHECK:           [[STEP_:%.+]] = "krnl.global"() {name = "constant_3", shape = [2], value = dense<1> : tensor<2xi64>} : () -> memref<2xi64>
//CHECK:           [[VAR_4_:%.+]] = dim [[DATA_]], [[CST_0_]] : memref<?x4x5xf32>
//CHECK:           [[RES_:%.+]] = alloc([[VAR_4_]]) : memref<?x2x3xf32>
//CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_4_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3) {
//CHECK:             [[LOAD_DATA_MEM_:%.+]] = affine.load [[DATA_]][symbol([[I_0_]]), symbol([[I_1_]]) + 1, symbol([[I_2_]]) + 1] : memref<?x4x5xf32>
//CHECK:             affine.store [[LOAD_DATA_MEM_]], [[RES_]][symbol([[I_0_]]), symbol([[I_1_]]), symbol([[I_2_]])] : memref<?x2x3xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<?x2x3xf32>
//CHECK:         }
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
//CHECK-LABEL:  func @compute_slice_all_dyn
//CHECK-SAME:   ([[START_:%.+]]: memref<2xi64>, [[END_:%.+]]: memref<2xi64>, [[STEP_:%.+]]: memref<2xi64>) {
//CHECK:           [[CST_5_:%.+]] = constant 5 : index
//CHECK:           [[CST_3_:%.+]] = constant 3 : index
//CHECK:           [[CST_minus_2147483648_:%.+]] = constant -2147483648 : index
//CHECK:           [[CST_2147483647_:%.+]] = constant 2147483647 : index
//CHECK:           [[CST_minus_1_:%.+]] = constant -1 : index
//CHECK:           [[CST_4_:%.+]] = constant 4 : index
//CHECK:           [[CST_0_:%.+]] = constant 0 : index
//CHECK:           [[DATA_:%.+]] = "krnl.global"() {name = "constant_0", shape = [3, 4, 5], value = dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>} : () -> memref<3x4x5xi64>
//CHECK:           [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_1", shape = [2], value = dense<[2, 1]> : tensor<2xi64>} : () -> memref<2xi64>
//CHECK:           [[LOAD_START_MEM_:%.+]] = affine.load [[START_]][0] : memref<2xi64>
//CHECK:           [[VAR_3_:%.+]] = index_cast [[LOAD_START_MEM_]] : i64 to index
//CHECK:           [[LOAD_END_MEM_:%.+]] = affine.load [[END_]][0] : memref<2xi64>
//CHECK:           [[VAR_5_:%.+]] = index_cast [[LOAD_END_MEM_]] : i64 to index
//CHECK:           [[LOAD_STEP_MEM_:%.+]] = affine.load [[STEP_]][0] : memref<2xi64>
//CHECK:           [[VAR_7_:%.+]] = index_cast [[LOAD_STEP_MEM_]] : i64 to index
//CHECK:           [[VAR_8_:%.+]] = cmpi "slt", [[VAR_3_]], [[CST_0_]] : index
//CHECK:           [[VAR_9_:%.+]] = affine.apply #map1(){{.}}[[VAR_3_]]{{.}}
//CHECK:           [[VAR_10_:%.+]] = select [[VAR_8_]], [[VAR_9_]], [[VAR_3_]] : index
//CHECK:           [[VAR_11_:%.+]] = cmpi "slt", [[VAR_10_]], [[CST_0_]] : index
//CHECK:           [[VAR_12_:%.+]] = select [[VAR_11_]], [[CST_0_]], [[VAR_10_]] : index
//CHECK:           [[VAR_13_:%.+]] = cmpi "sgt", [[VAR_12_]], [[CST_4_]] : index
//CHECK:           [[VAR_14_:%.+]] = select [[VAR_13_]], [[CST_4_]], [[VAR_12_]] : index
//CHECK:           [[VAR_15_:%.+]] = cmpi "slt", [[VAR_10_]], [[CST_0_]] : index
//CHECK:           [[VAR_16_:%.+]] = select [[VAR_15_]], [[CST_0_]], [[VAR_10_]] : index
//CHECK:           [[VAR_17_:%.+]] = cmpi "sgt", [[VAR_16_]], [[CST_5_]] : index
//CHECK:           [[VAR_18_:%.+]] = select [[VAR_17_]], [[CST_5_]], [[VAR_16_]] : index
//CHECK:           [[VAR_19_:%.+]] = cmpi "slt", [[VAR_7_]], [[CST_0_]] : index
//CHECK:           [[VAR_20_:%.+]] = select [[VAR_19_]], [[VAR_14_]], [[VAR_18_]] : index
//CHECK:           [[VAR_21_:%.+]] = cmpi "slt", [[VAR_5_]], [[CST_0_]] : index
//CHECK:           [[VAR_22_:%.+]] = affine.apply #map1(){{.}}[[VAR_5_]]{{.}}
//CHECK:           [[VAR_23_:%.+]] = select [[VAR_21_]], [[VAR_22_]], [[VAR_5_]] : index
//CHECK:           [[VAR_24_:%.+]] = cmpi "slt", [[VAR_5_]], [[CST_minus_2147483648_]] : index
//CHECK:           [[VAR_25_:%.+]] = select [[VAR_24_]], [[CST_minus_1_]], [[VAR_23_]] : index
//CHECK:           [[VAR_26_:%.+]] = cmpi "sge", [[VAR_5_]], [[CST_2147483647_]] : index
//CHECK:           [[VAR_27_:%.+]] = select [[VAR_26_]], [[CST_5_]], [[VAR_25_]] : index
//CHECK:           [[VAR_28_:%.+]] = cmpi "slt", [[VAR_27_]], [[CST_minus_1_]] : index
//CHECK:           [[VAR_29_:%.+]] = select [[VAR_28_]], [[CST_minus_1_]], [[VAR_27_]] : index
//CHECK:           [[VAR_30_:%.+]] = cmpi "sgt", [[VAR_29_]], [[CST_5_]] : index
//CHECK:           [[VAR_31_:%.+]] = select [[VAR_30_]], [[CST_5_]], [[VAR_29_]] : index
//CHECK:           [[VAR_32_:%.+]] = cmpi "slt", [[VAR_27_]], [[CST_0_]] : index
//CHECK:           [[VAR_33_:%.+]] = select [[VAR_32_]], [[CST_0_]], [[VAR_27_]] : index
//CHECK:           [[VAR_34_:%.+]] = cmpi "sgt", [[VAR_33_]], [[CST_5_]] : index
//CHECK:           [[VAR_35_:%.+]] = select [[VAR_34_]], [[CST_5_]], [[VAR_33_]] : index
//CHECK:           [[VAR_36_:%.+]] = cmpi "slt", [[VAR_7_]], [[CST_0_]] : index
//CHECK:           [[VAR_37_:%.+]] = select [[VAR_36_]], [[VAR_31_]], [[VAR_35_]] : index
//CHECK:           [[VAR_38_:%.+]] = subi [[VAR_37_]], [[VAR_20_]] : index
//CHECK:           [[VAR_39_:%.+]] = ceildivi_signed [[VAR_38_]], [[VAR_7_]] : index
//CHECK:           [[VAR_40_:%.+]] = cmpi "slt", [[VAR_39_]], [[CST_0_]] : index
//CHECK:           [[VAR_41_:%.+]] = select [[VAR_40_]], [[CST_0_]], [[VAR_39_]] : index
//CHECK:           [[LOAD_START_MEM_1_:%.+]] = affine.load [[START_]][1] : memref<2xi64>
//CHECK:           [[VAR_43_:%.+]] = index_cast [[LOAD_START_MEM_1_]] : i64 to index
//CHECK:           [[LOAD_END_MEM_1_:%.+]] = affine.load [[END_]][1] : memref<2xi64>
//CHECK:           [[VAR_45_:%.+]] = index_cast [[LOAD_END_MEM_1_]] : i64 to index
//CHECK:           [[LOAD_STEP_MEM_1_:%.+]] = affine.load [[STEP_]][1] : memref<2xi64>
//CHECK:           [[VAR_47_:%.+]] = index_cast [[LOAD_STEP_MEM_1_]] : i64 to index
//CHECK:           [[VAR_48_:%.+]] = cmpi "slt", [[VAR_43_]], [[CST_0_]] : index
//CHECK:           [[VAR_49_:%.+]] = affine.apply #map3(){{.}}[[VAR_43_]]{{.}}
//CHECK:           [[VAR_50_:%.+]] = select [[VAR_48_]], [[VAR_49_]], [[VAR_43_]] : index
//CHECK:           [[VAR_51_:%.+]] = cmpi "slt", [[VAR_50_]], [[CST_0_]] : index
//CHECK:           [[VAR_52_:%.+]] = select [[VAR_51_]], [[CST_0_]], [[VAR_50_]] : index
//CHECK:           [[VAR_53_:%.+]] = cmpi "sgt", [[VAR_52_]], [[CST_3_]] : index
//CHECK:           [[VAR_54_:%.+]] = select [[VAR_53_]], [[CST_3_]], [[VAR_52_]] : index
//CHECK:           [[VAR_55_:%.+]] = cmpi "slt", [[VAR_50_]], [[CST_0_]] : index
//CHECK:           [[VAR_56_:%.+]] = select [[VAR_55_]], [[CST_0_]], [[VAR_50_]] : index
//CHECK:           [[VAR_57_:%.+]] = cmpi "sgt", [[VAR_56_]], [[CST_4_]] : index
//CHECK:           [[VAR_58_:%.+]] = select [[VAR_57_]], [[CST_4_]], [[VAR_56_]] : index
//CHECK:           [[VAR_59_:%.+]] = cmpi "slt", [[VAR_47_]], [[CST_0_]] : index
//CHECK:           [[VAR_60_:%.+]] = select [[VAR_59_]], [[VAR_54_]], [[VAR_58_]] : index
//CHECK:           [[VAR_61_:%.+]] = cmpi "slt", [[VAR_45_]], [[CST_0_]] : index
//CHECK:           [[VAR_62_:%.+]] = affine.apply #map3(){{.}}[[VAR_45_]]{{.}}
//CHECK:           [[VAR_63_:%.+]] = select [[VAR_61_]], [[VAR_62_]], [[VAR_45_]] : index
//CHECK:           [[VAR_64_:%.+]] = cmpi "slt", [[VAR_45_]], [[CST_minus_2147483648_]] : index
//CHECK:           [[VAR_65_:%.+]] = select [[VAR_64_]], [[CST_minus_1_]], [[VAR_63_]] : index
//CHECK:           [[VAR_66_:%.+]] = cmpi "sge", [[VAR_45_]], [[CST_2147483647_]] : index
//CHECK:           [[VAR_67_:%.+]] = select [[VAR_66_]], [[CST_4_]], [[VAR_65_]] : index
//CHECK:           [[VAR_68_:%.+]] = cmpi "slt", [[VAR_67_]], [[CST_minus_1_]] : index
//CHECK:           [[VAR_69_:%.+]] = select [[VAR_68_]], [[CST_minus_1_]], [[VAR_67_]] : index
//CHECK:           [[VAR_70_:%.+]] = cmpi "sgt", [[VAR_69_]], [[CST_4_]] : index
//CHECK:           [[VAR_71_:%.+]] = select [[VAR_70_]], [[CST_4_]], [[VAR_69_]] : index
//CHECK:           [[VAR_72_:%.+]] = cmpi "slt", [[VAR_67_]], [[CST_0_]] : index
//CHECK:           [[VAR_73_:%.+]] = select [[VAR_72_]], [[CST_0_]], [[VAR_67_]] : index
//CHECK:           [[VAR_74_:%.+]] = cmpi "sgt", [[VAR_73_]], [[CST_4_]] : index
//CHECK:           [[VAR_75_:%.+]] = select [[VAR_74_]], [[CST_4_]], [[VAR_73_]] : index
//CHECK:           [[VAR_76_:%.+]] = cmpi "slt", [[VAR_47_]], [[CST_0_]] : index
//CHECK:           [[VAR_77_:%.+]] = select [[VAR_76_]], [[VAR_71_]], [[VAR_75_]] : index
//CHECK:           [[VAR_78_:%.+]] = subi [[VAR_77_]], [[VAR_60_]] : index
//CHECK:           [[VAR_79_:%.+]] = ceildivi_signed [[VAR_78_]], [[VAR_47_]] : index
//CHECK:           [[VAR_80_:%.+]] = cmpi "slt", [[VAR_79_]], [[CST_0_]] : index
//CHECK:           [[VAR_81_:%.+]] = select [[VAR_80_]], [[CST_0_]], [[VAR_79_]] : index
//CHECK:           [[RES_:%.+]] = alloc([[VAR_81_]], [[VAR_41_]]) : memref<3x?x?xi64>
//CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_81_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[VAR_41_]]) {
//CHECK:             [[VAR_84_:%.+]] = muli [[VAR_47_]], [[I_1_]] : index
//CHECK:             [[VAR_85_:%.+]] = addi [[VAR_84_]], [[VAR_60_]] : index
//CHECK:             [[VAR_86_:%.+]] = muli [[VAR_7_]], [[I_2_]] : index
//CHECK:             [[VAR_87_:%.+]] = addi [[VAR_86_]], [[VAR_20_]] : index
//CHECK:             [[VAR_88_:%.+]] = load [[DATA_]]{{.}}[[I_0_]], [[VAR_85_]], [[VAR_87_]]{{.}} : memref<3x4x5xi64>
//CHECK:             affine.store [[VAR_88_]], [[RES_]][symbol([[I_0_]]), symbol([[I_1_]]), symbol([[I_2_]])] : memref<3x?x?xi64>
//CHECK:           }
//CHECK:           dealloc [[RES_]] : memref<3x?x?xi64>
//CHECK:           return
//CHECK:         }
}

// -----

// GEMM with everything constant
func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_gemm
// CHECK-SAME:     ([[A:%.+]]: memref<5x10xf32>, [[B:%.+]]: memref<5x10xf32>, [[C:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
// CHECK:           [[ALPHA:%.+]] = constant 1.000000e+00 : f32
// CHECK:           [[BETA:%.+]] = constant 5.000000e+00 : f32
// CHECK:           [[ZERO:%.+]] = constant 0.000000e+00 : f32
// CHECK:           [[RES:%.+]] = alloc() : memref<10x10xf32>
// CHECK:           [[VAR_1:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[VAR_1]]#0, [[VAR_1]]#1) with ([[VAR_1]]#0 -> [[VAR_arg3:%.+]] = 0 to 10, [[VAR_1]]#1 -> [[VAR_arg4:%.+]] = 0 to 10) {
// CHECK:             affine.store [[ZERO]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_2:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[VAR_2]]) with ([[VAR_2]] -> [[VAR_arg5:%.+]] = 0 to 5) {
// CHECK:               [[AA:%.+]] = affine.load [[A]][symbol([[VAR_arg5]]), symbol([[VAR_arg3]])] : memref<5x10xf32>
// CHECK:               [[BB:%.+]] = affine.load [[B]][symbol([[VAR_arg5]]), symbol([[VAR_arg4]])] : memref<5x10xf32>
// CHECK:               [[RR:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:               [[VAR_11:%.+]] = mulf [[AA]], [[BB]] : f32
// CHECK:               [[VAR_12:%.+]] = addf [[RR]], [[VAR_11]] : f32
// CHECK:               affine.store [[VAR_12]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             }
// CHECK:             [[RRR:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_4:%.+]] = mulf [[ALPHA]], [[RRR]] : f32
// CHECK:             [[CC:%.+]] = affine.load [[C]][symbol([[VAR_arg4]])] : memref<10xf32>
// CHECK:             [[VAR_6:%.+]] = mulf [[BETA]], [[CC]] : f32
// CHECK:             [[VAR_7:%.+]] = addf [[VAR_4]], [[VAR_6]] : f32
// CHECK:             affine.store [[VAR_7]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Gemm with all dimensions dynamic
func @test_gemm_all_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_gemm_all_dyn
// CHECK-SAME:     ([[A:%.+]]: memref<?x?xf32>, [[B:%.+]]: memref<?x?xf32>, [[C:%.+]]: memref<?xf32>) -> memref<?x?xf32> {
// CHECK:           [[VAR_cst_:%.+]] = constant 1.000000e+00 : f32
// CHECK:           [[VAR_cst_0_:%.+]] = constant 5.000000e+00 : f32
// CHECK:           [[FZERO:%.+]] = constant 0.000000e+00 : f32
// CHECK:           [[ONE:%.+]] = constant 1 : index
// CHECK:           [[ZERO:%.+]] = constant 0 : index
// CHECK:           [[DIM_A0:%.+]] = dim [[A]], [[ONE]] : memref<?x?xf32>
// CHECK:           [[DIM_A1:%.+]] = dim [[A]], [[ZERO]] : memref<?x?xf32>
// CHECK:           [[DIM_B1:%.+]] = dim [[B]], [[ONE]] : memref<?x?xf32>
// CHECK:           [[DIM_C:%.+]] = dim [[C]], [[ZERO]] : memref<?xf32>
// CHECK:           [[RES:%.+]] = alloc([[DIM_A0]], [[DIM_B1]]) : memref<?x?xf32>
// CHECK:           [[VAR_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[VAR_5_]]#0, [[VAR_5_]]#1) with ([[VAR_5_]]#0 -> [[VAR_arg3_:%.+]] = 0 to [[DIM_A0]], [[VAR_5_]]#1 -> [[VAR_arg4_:%.+]] = 0 to [[DIM_B1]]) {
// CHECK:             affine.store [[FZERO]], [[RES]][symbol([[VAR_arg3_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:             [[VAR_6_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[VAR_6_]]) with ([[VAR_6_]] -> [[VAR_arg5_:%.+]] = 0 to [[DIM_A1]]) {
// CHECK:               [[VAR_14_:%.+]] = affine.load [[A]][symbol([[VAR_arg5_]]), symbol([[VAR_arg3_]])] : memref<?x?xf32>
// CHECK:               [[VAR_15_:%.+]] = affine.load [[B]][symbol([[VAR_arg5_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:               [[VAR_16_:%.+]] = affine.load [[RES]][symbol([[VAR_arg3_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:               [[VAR_17_:%.+]] = mulf [[VAR_14_]], [[VAR_15_]] : f32
// CHECK:               [[VAR_18_:%.+]] = addf [[VAR_16_]], [[VAR_17_]] : f32
// CHECK:               affine.store [[VAR_18_]], [[RES]][symbol([[VAR_arg3_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_7_:%.+]] = cmpi "sgt", [[DIM_C]], [[ONE]] : index
// CHECK:             [[VAR_8_:%.+]] = select [[VAR_7_]], [[VAR_arg4_]], [[ZERO]] : index
// CHECK:             [[VAR_9_:%.+]] = affine.load [[RES]][symbol([[VAR_arg3_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:             [[VAR_10_:%.+]] = mulf [[VAR_cst_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_11_:%.+]] = load [[C]]{{.}}[[VAR_8_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_12_:%.+]] = mulf [[VAR_cst_0_]], [[VAR_11_]] : f32
// CHECK:             [[VAR_13_:%.+]] = addf [[VAR_10_]], [[VAR_12_]] : f32
// CHECK:             affine.store [[VAR_13_]], [[RES]][symbol([[VAR_arg3_]]), symbol([[VAR_arg4_]])] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<?x?xf32>
// CHECK:         }
}

// -----

// A[10, *] * B[*, 10] result in constant size output but dyn reduction.
func @test_gemm_k_dyn(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x10xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_gemm_k_dyn
// CHECK-SAME:     ([[A:%.+]]: memref<?x10xf32>, [[B:%.+]]: memref<?x10xf32>, [[C:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
// CHECK:           [[VAR_c0:%.+]] = constant 0 : index
// CHECK:           [[VAR_cst:%.+]] = constant 1.000000e+00 : f32
// CHECK:           [[VAR_cst_0:%.+]] = constant 5.000000e+00 : f32
// CHECK:           [[ZERO:%.+]] = constant 0.000000e+00 : f32
// CHECK:           [[RES:%.+]] = alloc() : memref<10x10xf32>
// CHECK:           [[DIM_K:%.+]] = dim [[A]], [[VAR_c0]] : memref<?x10xf32>
// CHECK:           [[VAR_2:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[VAR_2]]#0, [[VAR_2]]#1) with ([[VAR_2]]#0 -> [[VAR_arg3:%.+]] = 0 to 10, [[VAR_2]]#1 -> [[VAR_arg4:%.+]] = 0 to 10) {
// CHECK:             affine.store [[ZERO]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_3:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[VAR_3]]) with ([[VAR_3]] -> [[VAR_arg5:%.+]] = 0 to [[DIM_K]]) {
// CHECK:               [[VAR_9:%.+]] = affine.load [[A]][symbol([[VAR_arg5]]), symbol([[VAR_arg3]])] : memref<?x10xf32>
// CHECK:               [[VAR_10:%.+]] = affine.load [[B]][symbol([[VAR_arg5]]), symbol([[VAR_arg4]])] : memref<?x10xf32>
// CHECK:               [[VAR_11:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:               [[VAR_12:%.+]] = mulf [[VAR_9]], [[VAR_10]] : f32
// CHECK:               [[VAR_13:%.+]] = addf [[VAR_11]], [[VAR_12]] : f32
// CHECK:               affine.store [[VAR_13]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             }
// CHECK:             [[VAR_4:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_5:%.+]] = mulf [[VAR_cst]], [[VAR_4]] : f32
// CHECK:             [[VAR_6:%.+]] = affine.load [[C]][symbol([[VAR_arg4]])] : memref<10xf32>
// CHECK:             [[VAR_7:%.+]] = mulf [[VAR_cst_0]], [[VAR_6]] : f32
// CHECK:             [[VAR_8:%.+]] = addf [[VAR_5]], [[VAR_7]] : f32
// CHECK:             affine.store [[VAR_8]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Broadcast bias C is dym, so we don't know if its 1 -> broadcast or 10. Dyn test for that.
func @test_gemm_c_dyn(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:       func @test_gemm_c_dyn
// CHECK-SAME:     ([[A:%.+]]: memref<5x10xf32>, [[B:%.+]]: memref<5x10xf32>, [[C:%.+]]: memref<?xf32>) -> memref<10x10xf32> {
// CHECK:           [[VAR_cst:%.+]] = constant 1.000000e+00 : f32
// CHECK:           [[VAR_cst_0:%.+]] = constant 5.000000e+00 : f32
// CHECK:           [[ZERO:%.+]] = constant 0.000000e+00 : f32
// CHECK:           [[VAR_c1:%.+]] = constant 1 : index
// CHECK:           [[VAR_c0:%.+]] = constant 0 : index
// CHECK:           [[RES:%.+]] = alloc() : memref<10x10xf32>
// CHECK:           [[VAR_1:%.+]] = dim [[C]], [[VAR_c0]] : memref<?xf32>
// CHECK:           [[VAR_2:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[VAR_2]]#0, [[VAR_2]]#1) with ([[VAR_2]]#0 -> [[VAR_arg3:%.+]] = 0 to 10, [[VAR_2]]#1 -> [[VAR_arg4:%.+]] = 0 to 10) {
// CHECK:             affine.store [[ZERO]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_3:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[VAR_3]]) with ([[VAR_3]] -> [[VAR_arg5:%.+]] = 0 to 5) {
// CHECK:               [[VAR_11:%.+]] = affine.load [[A]][symbol([[VAR_arg5]]), symbol([[VAR_arg3]])] : memref<5x10xf32>
// CHECK:               [[VAR_12:%.+]] = affine.load [[B]][symbol([[VAR_arg5]]), symbol([[VAR_arg4]])] : memref<5x10xf32>
// CHECK:               [[VAR_13:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:               [[VAR_14:%.+]] = mulf [[VAR_11]], [[VAR_12]] : f32
// CHECK:               [[VAR_15:%.+]] = addf [[VAR_13]], [[VAR_14]] : f32
// CHECK:               affine.store [[VAR_15]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             }
// CHECK:             [[NO_BROADCAST:%.+]] = cmpi "sgt", [[VAR_1]], [[VAR_c1]] : index
// CHECK:             [[C_INDEX:%.+]] = select [[NO_BROADCAST]], [[VAR_arg4]], [[VAR_c0]] : index
// CHECK:             [[RRR:%.+]] = affine.load [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:             [[VAR_7:%.+]] = mulf [[VAR_cst]], [[RRR]] : f32
// CHECK:             [[CC:%.+]] = load [[C]]{{.}}[[C_INDEX]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_9:%.+]] = mulf [[VAR_cst_0]], [[CC]] : f32
// CHECK:             [[VAR_10:%.+]] = addf [[VAR_7]], [[VAR_9]] : f32
// CHECK:             affine.store [[VAR_10]], [[RES]][symbol([[VAR_arg3]]), symbol([[VAR_arg4]])] : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Test tile with constant repeats
func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:       func @test_tile1
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<4x8xf32>) -> memref<12x16xf32> {
// CHECK:           [[VAR_0:%.+]] = alloc() : memref<12x16xf32>
// CHECK:           [[VAR_1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[3, 2]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_2:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[VAR_2]]#0, [[VAR_2]]#1) with ([[VAR_2]]#0 -> [[VAR_arg1:%.+]] = 0 to 12, [[VAR_2]]#1 -> [[VAR_arg2:%.+]] = 0 to 16) {
// CHECK:             [[VAR_3:%.+]] = affine.load [[VAR_arg0]][symbol([[VAR_arg1]]) mod 4, symbol([[VAR_arg2]]) mod 8] : memref<4x8xf32>
// CHECK:             affine.store [[VAR_3]], [[VAR_0]][symbol([[VAR_arg1]]), symbol([[VAR_arg2]])] : memref<12x16xf32>
// CHECK:           }
// CHECK:           return [[VAR_0]] : memref<12x16xf32>
// CHECK:         }
// CHECK:       }
}

// -----

// Test tile without constant repeats
func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:       func @test_tile2
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<8xf32>, [[VAR_arg1:%.+]]: memref<1xi64>) -> memref<?xf32> {
// CHECK:           [[VAR_0:%.+]] = affine.load [[VAR_arg1]][0] : memref<1xi64>
// CHECK:           [[VAR_1:%.+]] = index_cast [[VAR_0]] : i64 to index
// CHECK:           [[VAR_2:%.+]] = affine.apply #map1(){{.}}[[VAR_1]]{{.}}
// CHECK:           [[VAR_3:%.+]] = alloc([[VAR_2]]) : memref<?xf32>
// CHECK:           [[VAR_4:%.+]] = krnl.define_loops 1
// CHECK:           [[VAR_5:%.+]] = affine.apply #map1(){{.}}[[VAR_1]]{{.}}
// CHECK:           krnl.iterate([[VAR_4]]) with ([[VAR_4]] -> [[VAR_arg2:%.+]] = 0 to [[VAR_5]]) {
// CHECK:             [[VAR_6:%.+]] = affine.load [[VAR_arg0]][symbol([[VAR_arg2]]) mod 8] : memref<8xf32>
// CHECK:             affine.store [[VAR_6]], [[VAR_3]][symbol([[VAR_arg2]])] : memref<?xf32>
// CHECK:           }
// CHECK:           return [[VAR_3]] : memref<?xf32>
// CHECK:         }
}


