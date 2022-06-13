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

// -----

#map = affine_map<(d0) -> (d0 floordiv 64, d0 mod 64)>
func private @memref_with_affine(%arg0: memref<3xf32, #map>) -> memref<3xf32, #map> {
  %0 = memref.alloc() : memref<3xf32, #map>
  %1 = krnl.define_loops 1
  krnl.iterate(%1) with (%1 -> %arg1 = 0 to 3){
    %2 = krnl.get_induction_var_value(%1) : (!krnl.loop) -> index
    %3 = krnl.load %arg0[%2] : memref<3xf32, #map>
    krnl.store %3, %0[%2] : memref<3xf32, #map>
  }
  return %0 : memref<3xf32, #map>
// CHECK-DAG: #map = affine_map<(d0) -> (d0 floordiv 64, d0 mod 64)>
// CHECK-LABEL:  func private @memref_with_affine
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3xf32, #map>) -> memref<3xf32, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() : memref<3xf32, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]]{{.}} : memref<3xf32, #map>
// CHECK:             affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<3xf32, #map>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32, #map>
// CHECK:         }
}
