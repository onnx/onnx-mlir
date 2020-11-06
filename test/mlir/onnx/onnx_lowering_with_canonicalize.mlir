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
