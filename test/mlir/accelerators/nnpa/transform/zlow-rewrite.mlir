// RUN: onnx-mlir-opt --maccel=NNPA --zlow-rewrite --canonicalize %s -split-input-file | FileCheck %s

func.func @test_remove_unstick_view_stick(%arg0: memref<7x4x1x8x32x64xf16>) -> (memref<7x4x1x8x32x64xf16>){
    %0 = memref.alloc() {alignment = 16 : i64} : memref<7x1x256x200xf32>
    "zlow.unstick"(%arg0, %0) {layout = "4DS"} : (memref<7x4x1x8x32x64xf16>, memref<7x1x256x200xf32>) -> ()
    %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [7, 256, 200], strides: [51200, 200, 1] : memref<7x1x256x200xf32> to memref<7x256x200xf32>
    %2 = memref.alloc() {alignment = 4096 : i64} : memref<7x4x1x8x32x64xf16>
    "zlow.stick"(%1, %2) {layout = "3DS"} : (memref<7x256x200xf32>, memref<7x4x1x8x32x64xf16>) -> ()
    "func.return"(%2) : (memref<7x4x1x8x32x64xf16>) -> ()

    // CHECK-LABEL: test_remove_unstick_view_stick
    // CHECK-NEXT: return %arg0 : memref<7x4x1x8x32x64xf16>
    // CHECK-NOT: "zlow.unstick"
    // CHECK-NOT: "zlow.stick"
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> transpose -> stick.
// Test a simple transpose.

#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @unstick_transpose_stick(%arg0: memref<5x10xf16, #map>) -> memref<10x5xf16, #map> {
  // Unstick
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  // Transpose
  %1 = memref.alloc() {alignment = 16 : i64} : memref<10x5xf32>
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %3 = affine.load %0[%arg1, %arg2] : memref<5x10xf32>
      affine.store %3, %1[%arg2, %arg1] : memref<10x5xf32>
    }
  }
  // Stick
  %4 = memref.alloc() {alignment = 4096 : i64} : memref<10x5xf16, #map>
  "zlow.stick"(%1, %4) {layout = "2D"} : (memref<10x5xf32>, memref<10x5xf16, #map>) -> ()
  return %4 : memref<10x5xf16, #map>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @unstick_transpose_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf16, #map>) -> memref<10x5xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x5xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 5 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 10 {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<5x10xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_1_]], [[I_0_]]{{.}} : memref<10x5xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x5xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> transpose -> stick.
// Test how rewriting works with unknown dimensions.
// In this case, arguments of AllocOp need to be moved also.
#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @unstick_transpose_stick_unknown(%arg0: memref<?x?xf16, #map>) -> memref<?x?xf16, #map> {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %dim0 = memref.dim %arg0, %cst0:  memref<?x?xf16, #map>
  %dim1 = memref.dim %arg0, %cst1:  memref<?x?xf16, #map>
  // Unstick
  %0 = memref.alloc(%dim0, %dim1) {alignment = 4096 : i64} : memref<?x?xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<?x?xf16, #map>, memref<?x?xf32>) -> ()
  // Transpose
  %1 = memref.alloc(%dim1, %dim0) {alignment = 16 : i64} : memref<?x?xf32>
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %3 = affine.load %0[%arg1, %arg2] : memref<?x?xf32>
      affine.store %3, %1[%arg2, %arg1] : memref<?x?xf32>
    }
  }
  // Stick
  %dim0_ = memref.dim %1, %cst0:  memref<?x?xf32>
  %dim1_ = memref.dim %1, %cst1:  memref<?x?xf32>
  %4 = memref.alloc(%dim1_, %dim0_) {alignment = 4096 : i64} : memref<?x?xf16, #map>
  "zlow.stick"(%1, %4) {layout = "2D"} : (memref<?x?xf32>, memref<?x?xf16, #map>) -> ()
  return %4 : memref<?x?xf16, #map>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @unstick_transpose_stick_unknown
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf16, #map>) -> memref<?x?xf16, #map> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf16, #map>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 5 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 10 {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x?xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_1_]], [[I_0_]]{{.}} : memref<?x?xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> affine-for -> stick
// where stick and unstick have different layouts.

#map_nchw = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
#map_2d = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
#map = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
func.func @unstick_affinefor_stick_diff_layout(%arg0: memref<1x1x1x2048xf16, #map_nchw>) -> memref<1x2048xf16, #map_2d> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<1x2048x1x1xf32>
  "zlow.unstick"(%arg0, %alloc) {layout = "NCHW"} : (memref<1x1x1x2048xf16, #map_nchw>, memref<1x2048x1x1xf32>) -> ()
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x2048xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 2048 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 1 {
          %0 = affine.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x1x1xf32>
          %1 = affine.apply #map(%arg2, %arg3, %arg4)
          affine.store %0, %alloc_0[%arg1, %1] : memref<1x2048xf32>
        }
      }
    }
  }
  %alloc_1 = memref.alloc() {alignment = 4096 : i64} : memref<1x2048xf16, #map_2d>
  "zlow.stick"(%alloc_0, %alloc_1) {layout = "2D"} : (memref<1x2048xf32>, memref<1x2048xf16, #map_2d>) -> ()
  return %alloc_1 : memref<1x2048xf16, #map_2d>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @unstick_affinefor_stick_diff_layout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x1x2048xf16, #map>) -> memref<1x2048xf16, #map1> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2048xf16, #map1>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 1 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 2048 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 1 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 1 {
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<1x1x1x2048xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]] + [[I_2_]] + [[I_3_]]{{.}} : memref<1x2048xf16, #map1>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2048xf16, #map1>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> affine-for -> stick
// where layout is NCHW. This is to check whether access indices are rearranged
// correctly or not.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
func.func @unstick_affinefor_stick_nchw(%arg0: memref<?x56x56x128xf16, #map>) -> memref<?x58x58x128xf16, #map> {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x56x56x128xf16, #map>
  %alloc = memref.alloc(%dim) {alignment = 4096 : i64} : memref<?x128x56x56xf32>
  "zlow.unstick"(%arg0, %alloc) {layout = "NCHW"} : (memref<?x56x56x128xf16, #map>, memref<?x128x56x56xf32>) -> ()
  %alloc_0 = memref.alloc(%dim) {alignment = 16 : i64} : memref<?x128x58x58xf32>
  affine.for %arg1 = 0 to %dim {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 56 {
        affine.for %arg4 = 0 to 56 {
          %0 = affine.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<?x128x56x56xf32>
          affine.store %0, %alloc_0[%arg1, %arg2, %arg3 + 1, %arg4 + 1] : memref<?x128x58x58xf32>
        }
      }
    }
  }
  %alloc_1 = memref.alloc(%dim) {alignment = 4096 : i64} : memref<?x58x58x128xf16, #map>
  "zlow.stick"(%alloc_0, %alloc_1) {layout = "NCHW"} : (memref<?x128x58x58xf32>, memref<?x58x58x128xf16, #map>) -> ()
  return %alloc_1 : memref<?x58x58x128xf16, #map>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func.func @unstick_affinefor_stick_nchw
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x56x56x128xf16, #map>) -> memref<?x58x58x128xf16, #map> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x56x56x128xf16, #map>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x58x58x128xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 128 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 56 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 56 {
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<?x56x56x128xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_2_]] + 1, [[I_3_]] + 1, [[I_1_]]{{.}} : memref<?x58x58x128xf16, #map>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x58x58x128xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> concat -> stick.
#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @unstick_concat_stick(%arg0: memref<5x10xf16, #map>, %arg1: memref<5x10xf16, #map>) -> memref<10x10xf16, #map> attributes {llvm.emit_c_interface} {
  // Unstick
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %alloc) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  %alloc_0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg1, %alloc_0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  // Concat
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<10x10xf32>
  affine.for %arg2 = 0 to 5 {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %alloc[%arg2, %arg3] : memref<5x10xf32>
      affine.store %0, %alloc_1[%arg2, %arg3] : memref<10x10xf32>
    }
  }
  affine.for %arg2 = 0 to 5 {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %alloc_0[%arg2, %arg3] : memref<5x10xf32>
      affine.store %0, %alloc_1[%arg2 + 5, %arg3] : memref<10x10xf32>
    }
  }
  // Stick
  %alloc_2 = memref.alloc() {alignment = 4096 : i64} : memref<10x10xf16, #map>
  "zlow.stick"(%alloc_1, %alloc_2) {layout = "2D"} : (memref<10x10xf32>, memref<10x10xf16, #map>) -> ()
  return %alloc_2 : memref<10x10xf16, #map>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @unstick_concat_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf16, #map>, [[PARAM_1_:%.+]]: memref<5x10xf16, #map>) -> memref<10x10xf16, #map> attributes {llvm.emit_c_interface} {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x10xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 5 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 10 {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<5x10xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 5 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 10 {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<5x10xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[I_2_]] + 5, [[I_3_]]{{.}} : memref<10x10xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x10xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> split -> stick.
#map2D = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
#map = affine_map<(d0) -> (d0 + 3)>
func.func @unstick_split_stick(%arg0: memref<2x6xf16, #map2D>) -> (memref<2x3xf16, #map2D>, memref<2x3xf16, #map2D>){
  // Unstick
  %unstick = memref.alloc() {alignment = 4096 : i64} : memref<2x6xf32>
  "zlow.unstick"(%arg0, %unstick) {layout = "2D"} : (memref<2x6xf16, #map2D>, memref<2x6xf32>) -> ()
  // Split
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x3xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<2x3xf32>
  affine.for %arg1 = 0 to 2 {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %unstick[%arg1, %arg2] : memref<2x6xf32>
      affine.store %0, %alloc[%arg1, %arg2] : memref<2x3xf32>
    }
  }
  affine.for %arg1 = 0 to 2 {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.apply #map(%arg2)
      %1 = affine.load %unstick[%arg1, %0] : memref<2x6xf32>
      affine.store %1, %alloc_0[%arg1, %arg2] : memref<2x3xf32>
    }
  }
  // Stick
  %alloc_stick = memref.alloc() {alignment = 4096 : i64} : memref<2x3xf16, #map2D>
  "zlow.stick"(%alloc, %alloc_stick) {layout = "2D"} : (memref<2x3xf32>, memref<2x3xf16, #map2D>) -> ()
  %alloc_0_stick = memref.alloc() {alignment = 4096 : i64} : memref<2x3xf16, #map2D>
  "zlow.stick"(%alloc_0, %alloc_0_stick) {layout = "2D"} : (memref<2x3xf32>, memref<2x3xf16, #map2D>) -> ()
  return %alloc_stick, %alloc_0_stick : memref<2x3xf16, #map2D>, memref<2x3xf16, #map2D>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func.func @unstick_split_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x6xf16, #map>) -> (memref<2x3xf16, #map>, memref<2x3xf16, #map>) {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 2 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 3 {
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x6xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x3xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 2 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 3 {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]] + 3] : memref<2x6xf16, #map>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<2x3xf16, #map>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_0 : memref<2x3xf16, #map>, memref<2x3xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> concat -> stick.
// Test NCHW layout with static dimensions.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
func.func @unstick_concat_stick_nchw(%arg0: memref<1x5x7x3xf16, #map>, %arg1: memref<1x5x7x3xf16, #map>) -> memref<2x5x7x3xf16, #map> {
  %0 = memref.alloc() {alignment = 16: i64} : memref<1x3x5x7xf32>
  "zlow.unstick"(%arg0, %0) {layout = "NCHW"} : (memref<1x5x7x3xf16, #map>, memref<1x3x5x7xf32>) -> ()
  %1 = memref.alloc() {alignment = 16: i64} : memref<1x3x5x7xf32>
  "zlow.unstick"(%arg1, %1) {layout = "NCHW"} : (memref<1x5x7x3xf16, #map>, memref<1x3x5x7xf32>) -> ()
  // Concat
  %2 = memref.alloc() {alignment = 16 : i64} : memref<2x3x5x7xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 7 {
          %4 = affine.load %0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x5x7xf32>
          affine.store %4, %2[%arg2, %arg3, %arg4, %arg5] : memref<2x3x5x7xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 7 {
          %4 = affine.load %1[%arg2, %arg3, %arg4, %arg5] : memref<1x3x5x7xf32>
          affine.store %4, %2[%arg2 + 1, %arg3, %arg4, %arg5] : memref<2x3x5x7xf32>
        }
      }
    }
  }
  %3 = memref.alloc() {alignment = 4096 : i64} : memref<2x5x7x3xf16, #map>
  "zlow.stick"(%2, %3) {layout = "NCHW"} : (memref<2x3x5x7xf32>, memref<2x5x7x3xf16, #map>) -> ()
  return %3 : memref<2x5x7x3xf16, #map>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func.func @unstick_concat_stick_nchw
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x5x7x3xf16, #map>, [[PARAM_1_:%.+]]: memref<1x5x7x3xf16, #map>) -> memref<2x5x7x3xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x5x7x3xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 1 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 3 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 5 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 7 {
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<1x5x7x3xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<2x5x7x3xf16, #map>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_4_:%.+]] = 0 to 1 {
// CHECK:             affine.for [[I_5_:%.+]] = 0 to 3 {
// CHECK:               affine.for [[I_6_:%.+]] = 0 to 5 {
// CHECK:                 affine.for [[I_7_:%.+]] = 0 to 7 {
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[PARAM_1_]]{{.}}[[I_4_]], [[I_6_]], [[I_7_]], [[I_5_]]{{.}} : memref<1x5x7x3xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[I_4_]] + 1, [[I_6_]], [[I_7_]], [[I_5_]]{{.}} : memref<2x5x7x3xf16, #map>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x5x7x3xf16, #map>
// CHECK:         }
}

// -----

// Remove zlow.stick and zlow.unstick in pattern: unstick -> concat -> stick.
// Test NCHW layout with unknown dimensions.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
func.func @unstick_concat_stick_nchw_unknown_dims(%arg0: memref<?x?x?x?xf16, #map>, %arg1: memref<?x?x?x?xf16, #map>) -> memref<?x?x?x?xf16, #map> {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %cst2 = arith.constant 2 : index
  %cst3 = arith.constant 3 : index

  %dim0_0 = memref.dim %arg0, %cst0:  memref<?x?x?x?xf16, #map>
  %dim1_0 = memref.dim %arg0, %cst1:  memref<?x?x?x?xf16, #map>
  %dim2_0 = memref.dim %arg0, %cst2:  memref<?x?x?x?xf16, #map>
  %dim3_0 = memref.dim %arg0, %cst3:  memref<?x?x?x?xf16, #map>

  %dim0_1 = memref.dim %arg1, %cst0:  memref<?x?x?x?xf16, #map>
  %dim1_1 = memref.dim %arg1, %cst1:  memref<?x?x?x?xf16, #map>
  %dim2_1 = memref.dim %arg1, %cst2:  memref<?x?x?x?xf16, #map>
  %dim3_1 = memref.dim %arg1, %cst3:  memref<?x?x?x?xf16, #map>

  // Unstick
  %0 = memref.alloc(%dim0_0, %dim1_0, %dim2_0, %dim3_0) {alignment = 16: i64} : memref<?x?x?x?xf32>
  "zlow.unstick"(%arg0, %0) {layout = "NCHW"} : (memref<?x?x?x?xf16, #map>, memref<?x?x?x?xf32>) -> ()
  %1 = memref.alloc(%dim0_1, %dim1_1, %dim2_1, %dim3_1) {alignment = 16: i64} : memref<?x?x?x?xf32>
  "zlow.unstick"(%arg1, %1) {layout = "NCHW"} : (memref<?x?x?x?xf16, #map>, memref<?x?x?x?xf32>) -> ()
  // Concat
  %concat_dim = affine.apply #map1(%dim0_0, %dim0_1)
  %2 = memref.alloc(%concat_dim, %dim1_0, %dim2_0, %dim3_0) {alignment = 16 : i64} : memref<?x?x?x?xf32>
  affine.for %arg2 = 0 to %dim0_0 {
    affine.for %arg3 = 0 to %dim1_0 {
      affine.for %arg4 = 0 to %dim2_0 {
        affine.for %arg5 = 0 to %dim3_0 {
          %4 = affine.load %0[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
          affine.store %4, %2[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to %dim0_1 {
    affine.for %arg3 = 0 to %dim1_1 {
      affine.for %arg4 = 0 to %dim2_1 {
        affine.for %arg5 = 0 to %dim3_1 {
          %concat_idx = affine.apply #map1(%arg2, %dim0_0)
          %4 = affine.load %1[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
          affine.store %4, %2[%concat_idx, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  // Stick
  %dim0_2 = memref.dim %2, %cst0:  memref<?x?x?x?xf32>
  %dim1_2 = memref.dim %2, %cst1:  memref<?x?x?x?xf32>
  %dim2_2 = memref.dim %2, %cst2:  memref<?x?x?x?xf32>
  %dim3_2 = memref.dim %2, %cst3:  memref<?x?x?x?xf32>
  %3 = memref.alloc(%dim0_2, %dim1_2, %dim2_2, %dim3_2) {alignment = 4096 : i64} : memref<?x?x?x?xf16, #map>
  "zlow.stick"(%2, %3) {layout = "NCHW"} : (memref<?x?x?x?xf32>, memref<?x?x?x?xf16, #map>) -> ()
  return %3 : memref<?x?x?x?xf16, #map>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL:  func.func @unstick_concat_stick_nchw_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?x?xf16, #map>, [[PARAM_1_:%.+]]: memref<?x?x?x?xf16, #map>) -> memref<?x?x?x?xf16, #map> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<?x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_1_]], [[CST_3_]] : memref<?x?x?x?xf16, #map>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_3]
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?x?x?xf16, #map>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[VAR_dim_0_]] {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to [[VAR_dim_1_]] {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to [[VAR_dim_2_]] {
// CHECK:                   [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<?x?x?x?xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<?x?x?x?xf16, #map>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_4_:%.+]] = 0 to [[VAR_dim_3_]] {
// CHECK:             affine.for [[I_5_:%.+]] = 0 to [[VAR_dim_4_]] {
// CHECK:               affine.for [[I_6_:%.+]] = 0 to [[VAR_dim_5_]] {
// CHECK:                 affine.for [[I_7_:%.+]] = 0 to [[VAR_dim_6_]] {
// CHECK:                   [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[PARAM_1_]]{{.}}[[I_4_]], [[I_6_]], [[I_7_]], [[I_5_]]{{.}} : memref<?x?x?x?xf16, #map>
// CHECK:                   affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[I_4_]] + symbol([[VAR_dim_]]), [[I_6_]], [[I_7_]], [[I_5_]]{{.}} : memref<?x?x?x?xf16, #map>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf16, #map>
// CHECK:         }
}

// -----

// Do not support layout 1D and 2DS since their access index functions are
// incorrect: https://github.com/onnx/onnx-mlir/issues/1940
#map = affine_map<(d0, d1) -> (d0, d1 floordiv 64, 0, 0, 31, d1 mod 64)>
func.func @should_not_rewrite_unstick_transpose_stick_1(%arg0: memref<5x10xf16, #map>) -> memref<10x5xf16, #map> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2DS"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  %1 = memref.alloc() {alignment = 16 : i64} : memref<10x5xf32>
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %3 = affine.load %0[%arg1, %arg2] : memref<5x10xf32>
      affine.store %3, %1[%arg2, %arg1] : memref<10x5xf32>
    }
  }
  %4 = memref.alloc() {alignment = 4096 : i64} : memref<10x5xf16, #map>
  "zlow.stick"(%1, %4) {layout = "2DS"} : (memref<10x5xf32>, memref<10x5xf16, #map>) -> ()
  return %4 : memref<10x5xf16, #map>

// CHECK-LABEL:  func.func @should_not_rewrite_unstick_transpose_stick_1
// CHECK: zlow.unstick
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load {{.*}} memref<5x10xf32>
// CHECK: affine.store {{.*}} memref<10x5xf32>
// CHECK: zlow.stick
}

// -----

// Do not rewrite because other ops rather than zlow.unstick are consumming a CPU MemRef.
#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @should_not_rewrite_unstick_transpose_stick_2(%arg0: memref<5x10xf16, #map>) -> memref<10x5xf16, #map> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()

  // There are two consumers of %0. We cannot totally remove %0 and zlow.unstick.

  // First consumer of %0 that is followed by zlow.stick
  %1 = memref.alloc() {alignment = 16 : i64} : memref<10x5xf32>
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %3 = affine.load %0[%arg1, %arg2] : memref<5x10xf32>
      affine.store %3, %1[%arg2, %arg1] : memref<10x5xf32>
    }
  }

  // Second consumer of %0 that is NOT followed by zlow.stick
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %6 = affine.load %0[%arg1, %arg2] : memref<5x10xf32>
      affine.store %6, %0[%arg1, %arg2] : memref<5x10xf32>
    }
  }

  %4 = memref.alloc() {alignment = 4096 : i64} : memref<10x5xf16, #map>
  "zlow.stick"(%1, %4) {layout = "2D"} : (memref<10x5xf32>, memref<10x5xf16, #map>) -> ()
  return %4 : memref<10x5xf16, #map>

// CHECK-LABEL:  func.func @should_not_rewrite_unstick_transpose_stick_2
// CHECK: zlow.unstick
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load {{.*}} memref<5x10xf32>
// CHECK: affine.store {{.*}} memref<10x5xf32>
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load {{.*}} memref<5x10xf32>
// CHECK: affine.store {{.*}} memref<5x10xf32>
// CHECK: zlow.stick
}

// -----

// Do not rewrite because other ops rather than zlow.stick are consumming a CPU MemRef.
#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @should_not_rewrite_unstick_transpose_stick_3(%arg0: memref<5x10xf16, #map>) -> memref<10x5xf16, #map> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  %1 = memref.alloc() {alignment = 16 : i64} : memref<10x5xf32>
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %3 = affine.load %0[%arg1, %arg2] : memref<5x10xf32>
      affine.store %3, %1[%arg2, %arg1] : memref<10x5xf32>
    }
  }
  // There are two consumers of %1. We cannot totally remove %1 and zlow.stick.
  // First consumer of %1 that is zlow.stick
  %4 = memref.alloc() {alignment = 4096 : i64} : memref<10x5xf16, #map>
  "zlow.stick"(%1, %4) {layout = "2D"} : (memref<10x5xf32>, memref<10x5xf16, #map>) -> ()

  // Second consumer of %1 that is NOT zlow.stick
  affine.for %arg1 = 0 to 5 {
    affine.for %arg2 = 0 to 10 {
      %6 = affine.load %1[%arg2, %arg1] : memref<10x5xf32>
      affine.store %6, %1[%arg2, %arg1] : memref<10x5xf32>
    }
  }
  return %4 : memref<10x5xf16, #map>

// CHECK-LABEL:  func.func @should_not_rewrite_unstick_transpose_stick_3
// CHECK: zlow.unstick
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load {{.*}} memref<5x10xf32>
// CHECK: affine.store {{.*}} memref<10x5xf32>
// CHECK: zlow.stick
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.load {{.*}} memref<10x5xf32>
// CHECK: affine.store {{.*}} memref<10x5xf32>
}

// -----

// Do not rewrite because there is a AffineStoreOp without AffineLoadOp in pattern: unstick -> pad -> stick
// TODO: support this pattern.

// COM: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// COM: func.func @should_not_rewrite_unstick_pad_stick_nchw(%arg0: memref<?x56x56x128xf16, #map>) -> memref<?x58x58x128xf16, #map> {
// COM:   %c0 = arith.constant 0 : index
// COM:   %dim = memref.dim %arg0, %c0 : memref<?x56x56x128xf16, #map>
// COM:   %alloc = memref.alloc(%dim) {alignment = 4096 : i64} : memref<?x128x56x56xf32>
// COM:   "zlow.unstick"(%arg0, %alloc) {layout = "NCHW"} : (memref<?x56x56x128xf16, #map>, memref<?x128x56x56xf32>) -> ()
// COM:   %alloc_0 = memref.alloc(%dim) {alignment = 16 : i64} : memref<?x128x58x58xf32>
// COM:   %cst = arith.constant 0.000000e+00 : f32
// COM:   affine.for %arg1 = 0 to %dim {
// COM:     affine.for %arg2 = 0 to 128 {
// COM:       affine.for %arg3 = 0 to 58 {
// COM:         affine.for %arg4 = 0 to 58 {
// COM:           affine.store %cst, %alloc_0[%arg1, %arg2, %arg3, %arg4] : memref<?x128x58x58xf32>
// COM:         }
// COM:       }
// COM:     }
// COM:   }
// COM:   affine.for %arg1 = 0 to %dim {
// COM:     affine.for %arg2 = 0 to 128 {
// COM:       affine.for %arg3 = 0 to 56 {
// COM:         affine.for %arg4 = 0 to 56 {
// COM:           %0 = affine.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<?x128x56x56xf32>
// COM:           affine.store %0, %alloc_0[%arg1, %arg2, %arg3 + 1, %arg4 + 1] : memref<?x128x58x58xf32>
// COM:         }
// COM:       }
// COM:     }
// COM:   }
// COM:   %alloc_1 = memref.alloc(%dim) {alignment = 4096 : i64} : memref<?x58x58x128xf16, #map>
// COM:   "zlow.stick"(%alloc_0, %alloc_1) {layout = "NCHW"} : (memref<?x128x58x58xf32>, memref<?x58x58x128xf16, #map>) -> ()
// COM:   return %alloc_1 : memref<?x58x58x128xf16, #map>
// COM: 
// COM: // CHECK-LABEL:  func.func @should_not_rewrite_unstick_pad_stick_nchw
// COM: // CHECK: zlow.unstick
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.store
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.for
// COM: // CHECK: affine.load
// COM: // CHECK: affine.store
// COM: // CHECK: zlow.stick
// COM: }

