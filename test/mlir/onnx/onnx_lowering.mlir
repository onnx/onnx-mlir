// RUN: onnf-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

module {
  func @test_sigmoid(%a1 : tensor<?x10xf32>, %a2 : tensor<?x10xf32>) -> tensor<*xf32> {
    %0 = "onnx.Add"(%a1, %a2) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
  }
}

// CHECK: func @test_sigmoid([[ARG0:%.+]]: memref<?x10xf32>, [[ARG1:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK: [[DIM_0:%.+]] = dim [[ARG0]], 0 : memref<?x10xf32>
// CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
// CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
// CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
// CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
// CHECK: } : () -> (!krnl.loop, !krnl.loop)
// CHECK: [[DIM_2:%.+]] = dim [[ARG0]], 0 : memref<?x10xf32>
// CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
// CHECK: [[LOAD1:%.+]] = load [[ARG0]][%arg2, %arg3] : memref<?x10xf32>
// CHECK: [[LOAD2:%.+]] = load [[ARG1]][%arg2, %arg3] : memref<?x10xf32>
// CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
// CHECK: store [[ADDF]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
// CHECK: return [[RES]] : memref<?x10xf32>
