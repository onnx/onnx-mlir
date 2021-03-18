// RUN: onnx-mlir-opt --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

///////////////////////////////////////////////////////////////////////////////
// COPY TO

// -----

// fully enclosed
func private @copy_to(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  %i10 = constant 10 : index
  %i12 = constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i10, %i12], %f0 : memref<4x6xf32>, memref<40x60xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 10, [[I_1_]] + 12] : memref<40x60xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_nopad(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  %i10 = constant 10 : index
  %i12 = constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i10, %i12], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 10, [[I_1_]] + 12] : memref<40x60xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_nopad_last_fully_in(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  // last fully enclosed
  %i36 = constant 36 : index
  %i54 = constant 54 : index
  krnl.copy_to_tile_buffer %B, %A [%i36, %i54], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_last_fully_in
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<40x60xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_nopad_runtime_param(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  // runtime start indices (no problem as we have multiples)
  krnl.copy_to_tile_buffer %B, %A [%p0, %p1], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_runtime_param
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<40x60xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_nopad_partial(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // both dim partial, no padding
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_pad_partial(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // same, padding to full
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[4,6]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_pad_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ZERO_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = 2 to 6 {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_3_:%.+]] = 3 to 4 {
// CHECK:             affine.for [[I_4_:%.+]] = 0 to 6 {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_pad_partial_mod3(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // same, padding to mod 3
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[3,3]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_pad_partial_mod3
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ZERO_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = 2 to 3 {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_runtime_start_indices(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // runtime start indices
  krnl.copy_to_tile_buffer %B, %AA [%p0, %p1], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// ignore-dag: #map0 = affine_map<()[s0] -> (-s0 + 39, 4)>
// ignore-dag: #map1 = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_to_runtime_start_indices
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min #map0(){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min #map1(){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_to_runtime_start_indices_pad3(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // runtime start indices with padding to mod 3
  krnl.copy_to_tile_buffer %B, %AA [%p0, %p1], %f0 {padToNext=[3,3]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// ignore-dag: #map0 = affine_map<()[s0] -> (-s0 + 39, 4)>
// ignore-dag: #map1 = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_to_runtime_start_indices_pad3
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ZERO_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.min #map0(){{.}}[[START0_]]{{.}}
// CHECK:           [[VAR_3_:%.+]] = ceildivi_signed [[VAR_2_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = muli [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.min #map1(){{.}}[[START1_]]{{.}}
// CHECK:           [[VAR_6_:%.+]] = ceildivi_signed [[VAR_5_]], [[CST_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = muli [[VAR_6_]], [[CST_3_]] : index
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min #map0(){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min #map1(){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = max #map1(){{.}}[[START1_]]{{.}} to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_3_:%.+]] = max #map0(){{.}}[[START0_]]{{.}} to [[VAR_4_]] {
// CHECK:             affine.for [[I_4_:%.+]] = 0 to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func @copy_to_nested(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<10x60xf32>
  %f0 = constant 0.0 : f32
  %c0 = constant 0 : index

  affine.for %i = 0 to 40 step 10 {
      krnl.copy_to_tile_buffer %B, %A [%i, %c0], %f0 : memref<10x60xf32>, memref<40x60xf32>
  }
  return 

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func @copy_to_nested
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<10x60xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 40 step 10 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 10 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 60 {
// CHECK:                 [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.*}} + {{.*}}, [[I_2_]]{{.}} : memref<40x60xf32>
// CHECK:                 affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_1_]], [[I_2_]]{{.}} : memref<10x60xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func @copy_to_nested_partial(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<45x60xf32>
  %B = alloca() : memref<10x60xf32>
  %f0 = constant 0.0 : f32
  %c0 = constant 0 : index

  affine.for %i = 0 to 45 step 10 {
      krnl.copy_to_tile_buffer %B, %A [%i, %c0], %f0 : memref<10x60xf32>, memref<45x60xf32>
  }
  return

// ignore-dag: #map = affine_map<(d0) -> (-d0 + 45, 10)>
// CHECK-LABEL:  func @copy_to_nested_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<45x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<10x60xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 45 step 10 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min #map([[I_0_]]) {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 60 {
// CHECK:                 [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.*}} + {{.*}}, [[I_2_]]{{.}} : memref<45x60xf32>
// CHECK:                 affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_1_]], [[I_2_]]{{.}} : memref<10x60xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

///////////////////////////////////////////////////////////////////////////////
// COPY FROM

// -----

func private @copy_from_simple(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  // fully enclosed
  %i10 = constant 10 : index
  %i12 = constant 12 : index
  krnl.copy_from_tile_buffer %B, %A [%i10, %i12]: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + 10, [[I_1_]] + 12] : memref<40x60xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_from_simple_last(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  // last fully enclosed
  %i36 = constant 36 : index
  %i54 = constant 54 : index
  krnl.copy_from_tile_buffer %B, %A [%i36, %i54] : memref<4x6xf32>, memref<40x60xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple_last
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<40x60xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_from_simple_runtime(%p0 : index, %p1 : index) -> () {
  %A = alloca() : memref<40x60xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32

  // runtime start indices (no problem as we have multiples)
  krnl.copy_from_tile_buffer %B, %A [%p0, %p1] : memref<4x6xf32>, memref<40x60xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple_runtime
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<40x60xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_from_partial(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // both dim partial
  krnl.copy_from_tile_buffer %B, %AA [%i36, %i54] : memref<4x6xf32>, memref<39x56xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2 {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + 36, [[I_1_]] + 54] : memref<39x56xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func private @copy_from_partial_runtime(%p0 : index, %p1 : index) -> () {
  %AA = alloca() : memref<39x56xf32>
  %B = alloca() : memref<4x6xf32>
  %f0 = constant 0.0 : f32
  %i36 = constant 36 : index
  %i54 = constant 54 : index

  // runtime start indices
  krnl.copy_from_tile_buffer %B, %AA [%p0, %p1]: memref<4x6xf32>, memref<39x56xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// ignore-dag: #map0 = affine_map<()[s0] -> (-s0 + 39, 4)>
// ignore-dag: #map1 = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_from_partial_runtime
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) {
// CHECK-DAG:       [[ORGINAL_:%.+]] = alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min #map0(){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min #map1(){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}
