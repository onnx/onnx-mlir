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

