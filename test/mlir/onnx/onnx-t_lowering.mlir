// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

func private @test_geluf32(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Gelu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_geluf32
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[MINUSBETA:%.+]] = constant -1.702000e+00 : f32
  // CHECK: [[EXP_INPUT:%.+]] = mulf [[LOAD]], [[MINUSBETA]] : f32
  // CHECK: [[EXP_RES:%.+]] = exp [[EXP_INPUT]] : f32
  // CHECK: [[DENOMINATOR:%.+]] = addf [[EXP_RES]], [[ONE]] : f32
  // CHECK: [[GELU_RES:%.+]] = divf [[LOAD]], [[DENOMINATOR]] : f32
  // CHECK: krnl.store [[GELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_gelubf16(%arg0 : tensor<?x10xbf16>) -> tensor<*xbf16> {
  %0 = "onnx.Gelu"(%arg0) : (tensor<?x10xbf16>) -> tensor<*xbf16>
  "std.return"(%0) : (tensor<*xbf16>) -> ()

  // CHECK-LABEL: test_gelubf16
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xbf16>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xbf16>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xbf16>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xbf16>
  // CHECK: [[ONE:%.+]] = constant 1.000000e+00 : bf16
  // CHECK: [[MINUSBETA:%.+]] = constant -1.703130e+00 : bf16
  // CHECK: [[EXP_INPUT:%.+]] = mulf [[LOAD]], [[MINUSBETA]] : bf16
  // CHECK: [[EXP_RES:%.+]] = exp [[EXP_INPUT]] : bf16
  // CHECK: [[DENOMINATOR:%.+]] = addf [[EXP_RES]], [[ONE]] : bf16
  // CHECK: [[GELU_RES:%.+]] = divf [[LOAD]], [[DENOMINATOR]] : bf16
  // CHECK: krnl.store [[GELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xbf16>
  // CHECK: return [[RES]] : memref<?x10xbf16>
}
