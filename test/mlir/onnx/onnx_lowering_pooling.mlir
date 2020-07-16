// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s | FileCheck %s

// CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> ((s2 ceildiv s4) * s4 - s2, d0 * s3 - s2)>
// CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0, d0 * s3 + (s1 - 1) * s4 - s2 + 1)>
// CHECK-DAG: #{{.*}} = affine_map<() -> (0)>
// CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

// CHECK-DAG: #[[AFFINE_MAP1:.+]] = affine_map<(d0)[s0, s1, s2, s3] -> ((d0 + s1 - (s0 - 1) * s3 - 1) floordiv s2 + 1)>

func @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_pool_general_computation

  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {
  // CHECK:     {{.*}} = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

func @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x?x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_pool_unknown_dimensions
  // CHECK: [[C0:%.+]] = constant 2 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[C0]] : memref<1x3x?x32xf32>
  // CHECK: [[KERNEL:%.+]] = constant 2 : index
  // CHECK: [[PAD:%.+]] = constant 0 : index
  // CHECK: [[STRIDE:%.+]] = constant 1 : index
  // CHECK: [[DILATION:%.+]] = constant 1 : index
  // CHECK: [[AFFINE_APPLY:%.+]] = affine.apply #[[AFFINE_MAP1]]([[DIM]]){{.*}}[[KERNEL]], [[PAD]], [[STRIDE]], [[DILATION]]{{.*}}
  // CHECK: [[RES:%.+]] = alloc([[AFFINE_APPLY]]) : memref<1x3x?x31xf32>
}

func @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

func @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

func @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[SUM:%.+]] = addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     affine.store [[SUM]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   [[NUMERATOR:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   affine.store [[AVERAGE]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[GREATER:%.+]] = cmpf "ogt", [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     [[SELECT:%.+]] = select [[GREATER]], [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     affine.store [[SELECT]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK-NOT:   {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK-NOT:   affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}
