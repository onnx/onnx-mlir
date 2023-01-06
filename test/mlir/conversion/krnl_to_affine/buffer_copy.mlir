// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

///////////////////////////////////////////////////////////////////////////////
// COPY TO

// -----

// fully enclosed
func.func private @copy_to(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i10, %i12], %f0 : memref<4x6xf32>, memref<40x60xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

// Fully enclosed, with source memref of higher rank than buffer.
func.func private @copy_to_larger_source(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca() : memref<5x10x40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index
  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i2, %i3, %i10, %i12], %f0 : memref<4x6xf32>, memref<5x10x40x60xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_larger_source
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<5x10x40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]][2, 3, [[I_0_]] + 10, [[I_1_]] + 12] : memref<5x10x40x60xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

// Same with transpose
func.func private @copy_to_larger_transposed_source(%p0 : index, %p1 : index) -> () {
  //A source, B buffer
  %A = memref.alloca() : memref<5x10x60x40xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index
  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i2, %i3, %i12, %i10], %f0 {transpose=true}: 
    memref<4x6xf32>, memref<5x10x60x40xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_larger_transposed_source
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<5x10x60x40xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]][2, 3, [[I_1_]] + 12, [[I_0_]] + 10] : memref<5x10x60x40xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func private @copy_to_nopad(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_to_tile_buffer %B, %A [%i10, %i12], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_to_nopad_last_fully_in(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // last fully enclosed
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index
  krnl.copy_to_tile_buffer %B, %A [%i36, %i54], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_last_fully_in
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_to_nopad_runtime_param(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // runtime start indices (no problem as we have multiples)
  krnl.copy_to_tile_buffer %B, %A [%p0, %p1], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_runtime_param
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_to_nopad_partial(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // both dim partial, no padding
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_nopad_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_to_pad_partial(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // same, padding to full
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[4,6]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_pad_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ZERO_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

// Same, transposed
func.func private @copy_to_pad_partial_transposed(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<56x39xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // same, padding to full
  krnl.copy_to_tile_buffer %B, %AA [%i54, %i36], %f0 {padToNext=[4,6], transpose=true}: 
    memref<4x6xf32>, memref<56x39xf32>
  return

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_pad_partial_transposed
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<56x39xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2 {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_1_]] + 54, [[I_0_]] + 36] : memref<56x39xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = 2 to 6 {
// CHECK:               affine.store [[CST_0_dot_000000_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_3_:%.+]] = 3 to 4 {
// CHECK:             affine.for [[I_4_:%.+]] = 0 to 6 {
// CHECK:               affine.store [[CST_0_dot_000000_]], [[BUFFER_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func private @copy_to_pad_partial_mod3(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // same, padding to mod 3
  krnl.copy_to_tile_buffer %B, %AA [%i36, %i54], %f0 {padToNext=[3,3]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_to_pad_partial_mod3
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ZERO_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_to_runtime_start_indices(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // runtime start indices
  krnl.copy_to_tile_buffer %B, %AA [%p0, %p1], %f0 {padToNext=[1,1]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (-s0 + 39, 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func.func private @copy_to_runtime_start_indices
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min [[MAP_0_]](){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min [[MAP_1_]](){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func private @copy_to_runtime_start_indices_pad3(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // runtime start indices with padding to mod 3
  krnl.copy_to_tile_buffer %B, %AA [%p0, %p1], %f0 {padToNext=[3,3]}: memref<4x6xf32>, memref<39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (-s0 + 39, 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_to_runtime_start_indices_pad3
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ZERO_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.min [[MAP_1_]](){{.}}[[START0_]]{{.}}
// CHECK:           [[VAR_3_:%.+]] = arith.ceildivsi [[VAR_2_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.min [[MAP_2_]](){{.}}[[START1_]]{{.}}
// CHECK:           [[VAR_6_:%.+]] = arith.ceildivsi [[VAR_5_]], [[CST_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_3_]] : index
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min [[MAP_1_]](){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min [[MAP_2_]](){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = max [[MAP_2_]](){{.}}[[START1_]]{{.}} to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_3_:%.+]] = max [[MAP_1_]](){{.}}[[START0_]]{{.}} to [[VAR_4_]] {
// CHECK:             affine.for [[I_4_:%.+]] = 0 to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

// same, with source mem of larger rank.
func.func private @copy_to_runtime_start_indices_larger_source(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<5x10x39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i2 = arith.constant 2 : index
  %i5 = arith.constant 5 : index
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // runtime start indices
  krnl.copy_to_tile_buffer %B, %AA [%i2, %i5, %p0, %p1], %f0 {padToNext=[3,3]}: memref<4x6xf32>, memref<5x10x39x56xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (-s0 + 39, 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_to_runtime_start_indices_larger_source
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ZERO_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<5x10x39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.min [[MAP_0_]](){{.}}[[START0_]]{{.}}
// CHECK:           [[VAR_3_:%.+]] = arith.ceildivsi [[VAR_2_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.min [[MAP_1_]](){{.}}[[START1_]]{{.}}
// CHECK:           [[VAR_6_:%.+]] = arith.ceildivsi [[VAR_5_]], [[CST_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_3_]] : index
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min [[MAP_0_]](){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min [[MAP_1_]](){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_ORGINAL_MEM_:%.+]] = affine.load [[ORGINAL_]][2, 5, [[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<5x10x39x56xf32>
// CHECK:               affine.store [[LOAD_ORGINAL_MEM_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:             affine.for [[I_2_:%.+]] = max [[MAP_1_]](){{.}}[[START1_]]{{.}} to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_3_:%.+]] = max [[MAP_0_]](){{.}}[[START0_]]{{.}} to [[VAR_4_]] {
// CHECK:             affine.for [[I_4_:%.+]] = 0 to [[VAR_7_]] {
// CHECK:               affine.store [[ZERO_]], [[BUFFER_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<4x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func @copy_to_nested(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<10x60xf32>
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  affine.for %i = 0 to 40 step 10 {
      krnl.copy_to_tile_buffer %B, %A [%i, %c0], %f0 : memref<10x60xf32>, memref<40x60xf32>
  }
  return 

// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func @copy_to_nested
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<10x60xf32>
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

func.func @copy_to_nested_partial(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<45x60xf32>
  %B = memref.alloca() : memref<10x60xf32>
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  affine.for %i = 0 to 45 step 10 {
      krnl.copy_to_tile_buffer %B, %A [%i, %c0], %f0 : memref<10x60xf32>, memref<45x60xf32>
  }
  return

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 45, 10)>
// CHECK-LABEL:  func @copy_to_nested_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<45x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<10x60xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 45 step 10 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min [[MAP_0_]]([[I_0_]]) {
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

func.func private @copy_from_simple(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // fully enclosed
  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_from_tile_buffer %B, %A [%i10, %i12]: memref<4x6xf32>, memref<40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_from_simple_from_larger(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<5x10x40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // fully enclosed
  %i2 = arith.constant 2 : index
  %i5 = arith.constant 5 : index
  %i10 = arith.constant 10 : index
  %i12 = arith.constant 12 : index
  krnl.copy_from_tile_buffer %B, %A [%i2, %i5, %i10, %i12]: memref<4x6xf32>, memref<5x10x40x60xf32>
  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple_from_larger
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<5x10x40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]][2, 5, [[I_0_]] + 10, [[I_1_]] + 12] : memref<5x10x40x60xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func private @copy_from_simple_last(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // last fully enclosed
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index
  krnl.copy_from_tile_buffer %B, %A [%i36, %i54] : memref<4x6xf32>, memref<40x60xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple_last
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_from_simple_runtime(%p0 : index, %p1 : index) -> () {
  %A = memref.alloca() : memref<40x60xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32

  // runtime start indices (no problem as we have multiples)
  krnl.copy_from_tile_buffer %B, %A [%p0, %p1] : memref<4x6xf32>, memref<40x60xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_simple_runtime
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<40x60xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_from_partial(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // both dim partial
  krnl.copy_from_tile_buffer %B, %AA [%i36, %i54] : memref<4x6xf32>, memref<39x56xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-LABEL:  func private @copy_from_partial
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
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

func.func private @copy_from_partial_runtime(%p0 : index, %p1 : index) -> () {
  %AA = memref.alloca() : memref<39x56xf32>
  %B = memref.alloca() : memref<4x6xf32>
  %f0 = arith.constant 0.0 : f32
  %i36 = arith.constant 36 : index
  %i54 = arith.constant 54 : index

  // runtime start indices
  krnl.copy_from_tile_buffer %B, %AA [%p0, %p1]: memref<4x6xf32>, memref<39x56xf32>

  return
// mlir2FileCheck.py -n'{"0": "ORGINAL", "1": "BUFFER", "cst": "ZERO"}' -a'["start0", "start1"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (-s0 + 39, 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (-s0 + 56, 6)>
// CHECK-LABEL:  func private @copy_from_partial_runtime
// CHECK-SAME:   ([[START0_:%.+]]: index, [[START1_:%.+]]: index) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[ORGINAL_:%.+]] = memref.alloca() : memref<39x56xf32>
// CHECK-DAG:       [[BUFFER_:%.+]] = memref.alloca() : memref<4x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to min [[MAP_0_]](){{.}}[[START0_]]{{.}} {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to min [[MAP_1_]](){{.}}[[START1_]]{{.}} {
// CHECK:               [[LOAD_BUFFER_MEM_:%.+]] = affine.load [[BUFFER_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK:               affine.store [[LOAD_BUFFER_MEM_]], [[ORGINAL_]]{{.}}[[I_0_]] + symbol([[START0_]]), [[I_1_]] + symbol([[START1_]])] : memref<39x56xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}
