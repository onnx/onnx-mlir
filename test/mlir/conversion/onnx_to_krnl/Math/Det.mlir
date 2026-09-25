// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func private @test_det_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<f32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_3_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.minsi [[CST_3_2_]], [[CST_3_3_]] : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_0_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_1_]] {
// CHECK:             [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             scf.for [[I_1_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_2_]] {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = scf.for [[I_0_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_]] iter_args([[I_1_:%.+]] = [[CST_1_dot_000000_]]) -> (f32) {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_0_]], [[I_0_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]]:2 = scf.for [[VAR_arg3_:%.+]] = [[VAR_4_]] to [[VAR_0_]] step [[CST_1_]] iter_args([[VAR_arg4_:%.+]] = [[VAR_3_1_:%.+]], [[VAR_arg5_:%.+]] = [[I_0_]]) -> (f32, index) {
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_arg3_]], [[I_0_]]{{.}} : memref<?x?xf32>
// CHECK:               [[VAR_12_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[VAR_12_]], [[VAR_arg4_]] : f32
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_12_]], [[VAR_arg4_]] : f32
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.select [[VAR_13_]], [[VAR_arg3_]], [[VAR_arg5_]] : index
// CHECK:               scf.yield [[VAR_14_]], [[VAR_15_]] : f32, index
// CHECK:             }
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi ne, [[VAR_5_]]#1, [[I_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = scf.if [[VAR_6_]] -> (f32) {
// CHECK-DAG:           [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:               scf.for [[I_2_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_3_]] {
// CHECK-DAG:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:             [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:                 krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:               [[LOAD_RES_1_MEM_3_:%.+]] = arith.mulf [[I_1_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK:               scf.yield [[LOAD_RES_1_MEM_3_]] : f32
// CHECK:             } else {
// CHECK:               scf.yield [[I_1_]] : f32
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_0_]], [[I_0_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_4_]], [[CST_0_dot_000000_]] : f32
// CHECK:             scf.if [[VAR_10_]] {
// CHECK-DAG:           [[LOAD_RES_1_MEM_3_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK-DAG:           [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:               scf.for [[I_3_:%.+]] = [[LOAD_RES_1_MEM_3_]] to [[VAR_0_]] step [[CST_1_4_]] {
// CHECK:                 [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_3_]], [[I_0_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:             [[VAR_13_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_5_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:             [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:                 scf.for [[I_4_:%.+]] = [[I_0_]] to [[VAR_0_]] step [[CST_1_5_]] {
// CHECK-DAG:               [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_0_]], [[I_4_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_7_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<?x?xf32>
// CHECK:                   [[VAR_16_:%.+]] = arith.mulf [[VAR_13_1_]], [[LOAD_RES_1_MEM_6_]] : f32
// CHECK:                   [[VAR_17_:%.+]] = arith.subf [[LOAD_RES_1_MEM_7_]], [[VAR_16_]] : f32
// CHECK:                   krnl.store [[VAR_17_]], [[RES_1_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<?x?xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield [[VAR_9_]] : f32
// CHECK:           }
// CHECK:           krnl.store [[VAR_1_]], [[RES_]][] : memref<f32>
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
func.func private @test_det_2d(%arg0 : tensor<3x3xf32>) -> tensor<f32> {
  %0 = "onnx.Det"(%arg0) : (tensor<3x3xf32>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()
}

// -----

// CHECK-LABEL:  func.func private @test_det_nd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x4xf32>) -> memref<2xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2xf32>
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_3_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.minsi [[CST_4_2_]], [[CST_4_3_]] : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_1_]] {
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_0_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             scf.for [[I_1_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_2_]] {
// CHECK:               [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:               scf.for [[I_2_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_3_]] {
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x4x4xf32>
// CHECK:                 krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[I_1_]], [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[VAR_1_:%.+]] = scf.for [[I_1_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_]] iter_args([[I_2_:%.+]] = [[CST_1_dot_000000_]]) -> (f32) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_3_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_4_:%.+]] = arith.addi [[I_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = scf.for [[VAR_arg4_:%.+]] = [[VAR_4_]] to [[VAR_0_]] step [[CST_1_]] iter_args([[VAR_arg5_:%.+]] = [[VAR_3_1_:%.+]], [[VAR_arg6_:%.+]] = [[I_1_]]) -> (f32, index) {
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_arg4_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_12_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:                 [[VAR_13_:%.+]] = arith.cmpf ogt, [[VAR_12_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_12_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_15_:%.+]] = arith.select [[VAR_13_]], [[VAR_arg4_]], [[VAR_arg6_]] : index
// CHECK:                 scf.yield [[VAR_14_]], [[VAR_15_]] : f32, index
// CHECK:               }
// CHECK:               [[VAR_6_:%.+]] = arith.cmpi ne, [[VAR_5_]]#1, [[I_1_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = scf.if [[VAR_6_]] -> (f32) {
// CHECK-DAG:             [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:                 scf.for [[I_3_:%.+]] = [[CST_0_]] to [[VAR_0_]] step [[CST_1_4_]] {
// CHECK-DAG:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                   krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[I_1_]], [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                   krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = arith.mulf [[I_2_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK:                 scf.yield [[LOAD_RES_1_MEM_3_]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield [[I_2_]] : f32
// CHECK:               }
// CHECK:               [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_4_]], [[CST_0_dot_000000_]] : f32
// CHECK:               scf.if [[VAR_10_]] {
// CHECK-DAG:             [[LOAD_RES_1_MEM_3_:%.+]] = arith.addi [[I_1_]], [[CST_1_]] : index
// CHECK-DAG:             [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:                 scf.for [[I_4_:%.+]] = [[LOAD_RES_1_MEM_3_]] to [[VAR_0_]] step [[CST_1_5_]] {
// CHECK:                   [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_4_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[VAR_13_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_5_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:               [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:                   scf.for [[I_5_:%.+]] = [[I_1_]] to [[VAR_0_]] step [[CST_1_6_]] {
// CHECK-DAG:                 [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_7_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_4_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK:                     [[VAR_16_:%.+]] = arith.mulf [[VAR_13_1_]], [[LOAD_RES_1_MEM_6_]] : f32
// CHECK:                     [[VAR_17_:%.+]] = arith.subf [[LOAD_RES_1_MEM_7_]], [[VAR_16_]] : f32
// CHECK:                     krnl.store [[VAR_17_]], [[RES_1_]]{{.}}[[I_4_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               scf.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[VAR_1_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2xf32>
// CHECK:         }
func.func private @test_det_nd(%arg0 : tensor<2x4x4xf32>) -> tensor<2xf32> {
  %0 = "onnx.Det"(%arg0) : (tensor<2x4x4xf32>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
}

// -----

// CHECK-LABEL:  func.func private @test_det_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>) -> memref<?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK:           [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.minsi [[VAR_dim_3_]], [[VAR_dim_5_]] : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_2_]] to [[VAR_dim_]] step [[CST_1_3_]] {
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_0_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             scf.for [[I_1_:%.+]] = [[CST_0_1_]] to [[VAR_0_]] step [[CST_1_4_]] {
// CHECK:               [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK:               scf.for [[I_2_:%.+]] = [[CST_0_1_]] to [[VAR_0_]] step [[CST_1_5_]] {
// CHECK:                 [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x?x?xf32>
// CHECK:                 krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_1_]]{{.}}[[I_1_]], [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[VAR_1_:%.+]] = scf.for [[I_1_:%.+]] = [[CST_0_1_]] to [[VAR_0_]] step [[CST_1_2_]] iter_args([[I_2_:%.+]] = [[CST_1_dot_000000_]]) -> (f32) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_3_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_4_:%.+]] = arith.addi [[I_1_]], [[CST_1_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = scf.for [[VAR_arg4_:%.+]] = [[VAR_4_]] to [[VAR_0_]] step [[CST_1_2_]] iter_args([[VAR_arg5_:%.+]] = [[VAR_3_1_:%.+]], [[VAR_arg6_:%.+]] = [[I_1_]]) -> (f32, index) {
// CHECK-DAG:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_arg4_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK:                 [[VAR_12_:%.+]] = math.absf [[LOAD_RES_1_MEM_]] : f32
// CHECK:                 [[VAR_13_:%.+]] = arith.cmpf ogt, [[VAR_12_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_12_]], [[VAR_arg5_]] : f32
// CHECK-DAG:             [[VAR_15_:%.+]] = arith.select [[VAR_13_]], [[VAR_arg4_]], [[VAR_arg6_]] : index
// CHECK:                 scf.yield [[VAR_14_]], [[VAR_15_]] : f32, index
// CHECK:               }
// CHECK:               [[VAR_6_:%.+]] = arith.cmpi ne, [[VAR_5_]]#1, [[I_1_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = scf.if [[VAR_6_]] -> (f32) {
// CHECK-DAG:             [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:                 scf.for [[I_3_:%.+]] = [[CST_0_1_]] to [[VAR_0_]] step [[CST_1_6_]] {
// CHECK-DAG:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                   krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[I_1_]], [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                   krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_5_]]#1, [[I_3_]]{{.}} : memref<?x?xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_1_MEM_3_:%.+]] = arith.mulf [[I_2_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK:                 scf.yield [[LOAD_RES_1_MEM_3_]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield [[I_2_]] : f32
// CHECK:               }
// CHECK:               [[LOAD_RES_1_MEM_4_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.cmpf one, [[LOAD_RES_1_MEM_4_]], [[CST_0_dot_000000_]] : f32
// CHECK:               scf.if [[VAR_10_]] {
// CHECK-DAG:             [[LOAD_RES_1_MEM_3_:%.+]] = arith.addi [[I_1_]], [[CST_1_2_]] : index
// CHECK-DAG:             [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK:                 scf.for [[I_4_:%.+]] = [[LOAD_RES_1_MEM_3_]] to [[VAR_0_]] step [[CST_1_7_]] {
// CHECK:                   [[LOAD_RES_1_MEM_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_4_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:               [[VAR_13_1_:%.+]] = arith.divf [[LOAD_RES_1_MEM_5_]], [[LOAD_RES_1_MEM_4_]] : f32
// CHECK-DAG:               [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK:                   scf.for [[I_5_:%.+]] = [[I_1_]] to [[VAR_0_]] step [[CST_1_8_]] {
// CHECK-DAG:                 [[LOAD_RES_1_MEM_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_1_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_7_:%.+]] = krnl.load [[RES_1_]]{{.}}[[I_4_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK:                     [[VAR_16_:%.+]] = arith.mulf [[VAR_13_1_]], [[LOAD_RES_1_MEM_6_]] : f32
// CHECK:                     [[VAR_17_:%.+]] = arith.subf [[LOAD_RES_1_MEM_7_]], [[VAR_16_]] : f32
// CHECK:                     krnl.store [[VAR_17_]], [[RES_1_]]{{.}}[[I_4_]], [[I_5_]]{{.}} : memref<?x?xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               scf.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[VAR_1_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
func.func private @test_det_dynamic(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32> {
  %0 = "onnx.Det"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()
}
