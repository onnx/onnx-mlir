// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

func @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemax
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducemin(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemin
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0x7F800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reduceprod(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProd"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduceprod
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 1.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = mulf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSum"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesum
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}
