// RUN: onnx-mlir-opt --march=arm64 --shape-inference --fusion-op-transform --convert-onnx-to-krnl="enable-simd=false" %s -split-input-file | FileCheck %s

// enable-simd=false (independent of -march) must degrade the fused
// lowering to a plain scalar loop, not skip it or crash: VL=1 makes
// simdIterateIE fall straight through to krnl.load/krnl.store on scalar
// f32 (no krnl.block, no vector types) -- see the "SIMD-disabled behavior"
// note in FusedSplitOpGather.cpp. Still only one memref.alloc: the
// single-buffer win over the unfused Slice/Concat baseline holds even
// with SIMD off.

func.func @rotate_half_no_simd(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
  return %3 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @rotate_half_no_simd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4x8x128xf32>) -> memref<1x4x8x128xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<1x4x8x128xf32>
// CHECK:           krnl.iterate
// Low half: scalar krnl.load/krnl.store copy -- no krnl.block, no vectors.
// CHECK-NOT:       krnl.block
// CHECK:               [[LOW_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>
// CHECK:               krnl.store [[LOW_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>
// High half: scalar load, negate, store.
// CHECK:               [[HIGH_VAL_:%.+]] = krnl.load [[PARAM_0_]][{{.*}}] : memref<1x4x8x128xf32>
// CHECK:               [[NEG_VAL_:%.+]] = arith.negf [[HIGH_VAL_]] : f32
// CHECK:               krnl.store [[NEG_VAL_]], [[ALLOC_]][{{.*}}] : memref<1x4x8x128xf32>
// CHECK:           return [[ALLOC_]]
}
