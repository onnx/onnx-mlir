// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

func private @memset(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca() : memref<8x4x20x30xf32>
  %f0 = arith.constant 0.0 : f32

  krnl.memset %A, %f0 : memref<8x4x20x30xf32>
  return
// CHECK-LABEL:  func private @memset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: index, [[PARAM_1_:%.+]]: index) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<8x4x20x30xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 8 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 4 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 20 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 30 {
// CHECK:                   affine.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<8x4x20x30xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @memset_dyn(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca(%p0) : memref<?x4x20x30xf32>
  %f0 = arith.constant 0.0 : f32

  krnl.memset %A, %f0 : memref<?x4x20x30xf32>
  return
// CHECK-LABEL:  func private @memset_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: index, [[PARAM_1_:%.+]]: index) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca([[PARAM_0_]]) : memref<?x4x20x30xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[PARAM_0_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 4 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 20 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 30 {
// CHECK:                   affine.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<?x4x20x30xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

