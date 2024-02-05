// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_batchnorm_testmode_Nd(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x2x1x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x1x3xf32>
  return %0 : tensor<1x2x1x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_Nd
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x2x1x3xf32>
  // CHECK-DAG: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 2){
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<2xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<2xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<2xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<2xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 1, [[DEF_LOOPS]]#2 -> %arg7 = 0 to 1, [[DEF_LOOPS]]#3 -> %arg8 = 0 to 3){
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x2x1x3xf32>
}

// -----

func.func private @test_batchnorm_testmode_1d(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

  // CHECK-LABEL: test_batchnorm_testmode_1d
  // CHECK: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  // CHECK: [[SCALE:%.+]] = krnl.load %arg1[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[BIAS:%.+]] = krnl.load %arg2[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[MEAN:%.+]] = krnl.load %arg3[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[VARIANCE:%.+]] = krnl.load %arg4[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg5 = 0 to 10){
  // CHECK:   [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg5] : memref<10xf32>
  // CHECK:   [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:   [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:   [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:   [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:   [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:   [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:   krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg5] : memref<10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func.func private @test_batchnorm_testmode_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<10x3xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<10x3xf32>
  return %0 : tensor<10x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_2d
  // CHECK: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x3xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 3){
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<3xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<3xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<3xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<3xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 10){
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5] : memref<10x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5] : memref<10x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x3xf32>
}

