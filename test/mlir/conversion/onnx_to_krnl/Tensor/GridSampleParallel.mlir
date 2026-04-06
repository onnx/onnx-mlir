// RUN: onnx-mlir-opt -O3 --march=z17 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Note here: we added the march = z17 because not all machines have support for round even operation.
// Recent Z machines have it, same with recent macs.

// -----

// COM: Test 2D GridSample with small batch (BS=1) and many channels (C=20) for parallelization
func.func @test_gridsample_2d_parallel_small_batch(%arg0: tensor<1x20x4x4xf32>, %arg1: tensor<1x2x2x2xf32>) -> tensor<1x20x2x2xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x20x4x4xf32>, tensor<1x2x2x2xf32>) -> tensor<1x20x2x2xf32>
  return %0 : tensor<1x20x2x2xf32>

// CHECK-LABEL:  func.func @test_gridsample_2d_parallel_small_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x20x4x4xf32>, [[PARAM_1_:%.+]]: memref<1x2x2x2xf32>) -> memref<1x20x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x20x2x2xf32>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2x4xindex>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2x4xf32>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2xi8>
// CHECK:           [[LOOP_N_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_N_]]) with ([[LOOP_N_]] -> [[I_N_:%.+]] = 0 to 1){
// CHECK:             [[LOOP_PLAN_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_PLAN_]]#0, [[LOOP_PLAN_]]#1)
// CHECK:             [[LOOP_APPLY_:%.+]]:3 = krnl.define_loops 3
// CHECK:             krnl.parallel([[LOOP_APPLY_]]#0)
// CHECK:             krnl.iterate([[LOOP_APPLY_]]#0, [[LOOP_APPLY_]]#1, [[LOOP_APPLY_]]#2)
// CHECK:               krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x20x2x2xf32>
}

// -----

// COM: Test 2D GridSample with large batch (BS=16) and few channels (C=3) for parallelization
func.func @test_gridsample_2d_parallel_large_batch(%arg0: tensor<16x3x4x4xf32>, %arg1: tensor<16x2x2x2xf32>) -> tensor<16x3x2x2xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<16x3x4x4xf32>, tensor<16x2x2x2xf32>) -> tensor<16x3x2x2xf32>
  return %0 : tensor<16x3x2x2xf32>

// CHECK-LABEL:  func.func @test_gridsample_2d_parallel_large_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x3x4x4xf32>, [[PARAM_1_:%.+]]: memref<16x2x2x2xf32>) -> memref<16x3x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x3x2x2xf32>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2x4xindex>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2x4xf32>
// CHECK:           memref.alloc({{.*}}) {{.*}}: memref<2x2xi8>
// CHECK:           [[LOOP_N_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_N_]]) with ([[LOOP_N_]] -> [[I_N_:%.+]] = 0 to 16){
// CHECK:             [[LOOP_PLAN_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_PLAN_]]#0, [[LOOP_PLAN_]]#1)
// CHECK:             [[LOOP_APPLY_:%.+]]:3 = krnl.define_loops 3
// CHECK:             krnl.iterate([[LOOP_APPLY_]]#0, [[LOOP_APPLY_]]#1, [[LOOP_APPLY_]]#2)
// CHECK:               krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x3x2x2xf32>
}

// -----

// COM: Test 3D GridSample with small batch (BS=1) and many channels (C=20) for parallelization
func.func @test_gridsample_3d_parallel_small_batch(%arg0: tensor<1x20x2x2x2xf32>, %arg1: tensor<1x2x2x2x3xf32>) -> tensor<1x20x2x2x2xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x20x2x2x2xf32>, tensor<1x2x2x2x3xf32>) -> tensor<1x20x2x2x2xf32>
  return %0 : tensor<1x20x2x2x2xf32>

// CHECK-LABEL:  func.func @test_gridsample_3d_parallel_small_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x20x2x2x2xf32>, [[PARAM_1_:%.+]]: memref<1x2x2x2x3xf32>) -> memref<1x20x2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x20x2x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.parallel([[LOOP_0_]]#1)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x20x2x2x2xf32>
}

// -----

// COM: Test 3D GridSample with large batch (BS=16) and few channels (C=3) for parallelization
func.func @test_gridsample_3d_parallel_large_batch(%arg0: tensor<16x3x2x2x2xf32>, %arg1: tensor<16x2x2x2x3xf32>) -> tensor<16x3x2x2x2xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<16x3x2x2x2xf32>, tensor<16x2x2x2x3xf32>) -> tensor<16x3x2x2x2xf32>
  return %0 : tensor<16x3x2x2x2xf32>

// CHECK-LABEL:  func.func @test_gridsample_3d_parallel_large_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x3x2x2x2xf32>, [[PARAM_1_:%.+]]: memref<16x2x2x2x3xf32>) -> memref<16x3x2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x3x2x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.parallel([[LOOP_0_]]#0)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x3x2x2x2xf32>
}