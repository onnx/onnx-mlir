// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

func.func private @memset(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca() : memref<8x4x20x30xf32>
  %f0 = arith.constant 0.0 : f32

  krnl.memset %A, %f0 : memref<8x4x20x30xf32>
  return
// CHECK-LABEL:  func private @memset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: index, [[PARAM_1_:%.+]]: index) attributes {llvm.emit_c_interface} {
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

func.func private @memset_dyn(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca(%p0) : memref<?x4x20x30xf32>
  %f0 = arith.constant 0.0 : f32

  krnl.memset %A, %f0 : memref<?x4x20x30xf32>
  return
// CHECK-LABEL:  func private @memset_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: index, [[PARAM_1_:%.+]]: index) attributes {llvm.emit_c_interface} {
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
func.func private @memref_with_affine(%arg0: memref<3xf32, #map>) -> memref<3xf32, #map> {
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
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3xf32, #map>) -> memref<3xf32, #map> attributes {llvm.emit_c_interface} {
// CHECK:           [[RES_:%.+]] = memref.alloc() : memref<3xf32, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]]{{.}} : memref<3xf32, #map>
// CHECK:             affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<3xf32, #map>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32, #map>
// CHECK:         }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d2 floordiv 32, d2 mod 32, d1 mod 64)>
func.func @krnl_get_linear_offset_index_1(%arg0: memref<?x128x256xf32, #map>, %arg1: index, %arg2: index) -> index {
  %c5 = arith.constant 5: index
  %c10 = arith.constant 10: index
  %0 = memref.alloc(%arg1) : memref<?x128x256xf32, #map>
  %1 = krnl.get_linear_offset_index %arg0 at [%arg2, %c5, %c10] : memref<?x128x256xf32, #map>
  return %1: index

// CHECK-LABEL:  func.func @krnl_get_linear_offset_index
// CHECK:           [[VAR_0_:%.+]] = krnl.get_linear_offset_index {{.*}} at {{.*}} : memref<?x128x256xf32, #map>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0)>
func.func @krnl_get_linear_offset_index_2(%arg0: memref<?x2x8x32x64xf32>, %arg1: index, %arg2: index) -> index {
  %0 = krnl.get_linear_offset_index %arg0 at [%arg2, 0, 0, 10, 5] : memref<?x2x8x32x64xf32>
  return %0 : index

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 32768 + 645)>
// CHECK-LABEL:  func.func @krnl_get_linear_offset_index
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x8x32x64xf32>, [[PARAM_1_:%.+]]: index, [[PARAM_2_:%.+]]: index) -> index attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[PARAM_2_]]{{.}}
// CHECK:           return [[VAR_0_]] : index
// CHECK:         }
}

// -----

#map = affine_map<(d0) -> (d0 + 64)>
func.func @prefetch(%arg0: memref<256x512xf32>) -> memref<256x512xf32> attributes {input_names = ["x"], output_names = ["output"]} {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<256x512xf32>
  %0:2 = krnl.define_loops 2
  krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg1 = 0 to 256, %0#1 -> %arg2 = 0 to 512){
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %2 = krnl.load %arg0[%1#0, %1#1] : memref<256x512xf32>
    %3 = affine.apply #map(%1#1)
    krnl.prefetch %arg0[%1#0, %3], read, locality<3>, data : memref<256x512xf32>
    %4 = krnl.load %arg0[%1#0, %1#1] : memref<256x512xf32>
    %5 = arith.addf %2, %4 : f32
    krnl.store %5, %alloc[%1#0, %1#1] : memref<256x512xf32>
  }
  return %alloc : memref<256x512xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @prefetch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x512xf32>) -> memref<256x512xf32> attributes {input_names = ["x"], llvm.emit_c_interface, output_names = ["output"]} {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x512xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 256 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 512 {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<256x512xf32>
// CHECK:               affine.prefetch [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]] + 64], read, locality<3>, data : memref<256x512xf32>
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<256x512xf32>
// CHECK:               [[VAR_2_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               affine.store [[VAR_2_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<256x512xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<256x512xf32>
// CHECK:         }
}
