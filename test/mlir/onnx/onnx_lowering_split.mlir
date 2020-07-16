// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s | FileCheck %s

// CHECK: [[INDEX_MAP1:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK: [[INDEX_MAP2:#.+]] = affine_map<(d0) -> (d0 + 2)>

func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 0} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_split_equal

  // CHECK: [[RES_1:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP1]](%arg1)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%[[INDEX]], %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_1:%.+]] = alloc() : memref<16x30x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<16x2x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP2]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1, split = [2, 30]} : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_split_unknown_dimension

  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x?x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc([[DIM_0]]) : memref<?x2x64xf32>
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim %arg0, [[C0_0]] : memref<?x?x64xf32>
  // CHECK: [[RES_1:%.+]] = alloc([[DIM_1]]) : memref<?x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES_0]], [[C0_2]] : memref<?x2x64xf32>
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to [[DIM_0]], [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<?x?x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<?x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[C0_3:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim [[RES_1]], [[C0_3]] : memref<?x30x64xf32>
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to [[DIM_1]], [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP2]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<?x?x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<?x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<?x2x64xf32>, memref<?x30x64xf32>
}