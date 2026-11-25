// RUN: onnx-mlir-opt -O3 --march=z17 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt -O3 --march=z17 --parallel-must-repay-fork=false --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s --check-prefix=OFF
// GROUND-ALL: -c="-O3 --parallel" -r="--parallel-must-repay-fork=false"

// The per-entry fork-cost test at a site that opts in (mustRepayFork):
// a full SIMD reduction to one element, entered once. On z17 (forkCost 2,700,
// 4 threads, bodyCost 2 for ReduceSum: a load and an add per element) a static
// size needs about 5,400 elements to repay one fork. With the test off, the
// trip-count floor alone decides, and every case below is parallel.

// -----

// 1,024 elements: below the fork cost, so the reduction stays sequential.
// GROUND-THIS:
func.func @test_fork_cost_small_static(%arg0 : tensor<16x64xf32>) -> tensor<f32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 0 : si64} : (tensor<16x64xf32>, none) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:  func.func @test_fork_cost_small_static
// CHECK-NOT:      krnl.parallel
// CHECK:          return
// OFF-LABEL:    func.func @test_fork_cost_small_static
// OFF:            krnl.parallel
}

// -----

// 16,384 elements: repays the fork, so the reduction is parallel.
// GROUND-THIS:
func.func @test_fork_cost_large_static(%arg0 : tensor<256x64xf32>) -> tensor<f32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 0 : si64} : (tensor<256x64xf32>, none) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:  func.func @test_fork_cost_large_static
// CHECK:          krnl.parallel
// OFF-LABEL:    func.func @test_fork_cost_large_static
// OFF:            krnl.parallel
}

// -----

// A dynamic size gets the benefit of the doubt: it may be large, so the
// reduction stays parallel (a dynamic size would otherwise count as 1).
// GROUND-THIS: --shape-info=0:16x64
func.func @test_fork_cost_dynamic(%arg0 : tensor<?x64xf32>) -> tensor<f32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 0 : si64} : (tensor<?x64xf32>, none) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:  func.func @test_fork_cost_dynamic
// CHECK:          krnl.parallel
// OFF-LABEL:    func.func @test_fork_cost_dynamic
// OFF:            krnl.parallel
}
