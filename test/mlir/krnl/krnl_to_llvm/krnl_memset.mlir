// RUN: onnx-mlir-opt --convert-krnl-to-affine --normalize-memrefs --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s
#map_2ds = affine_map<(d0, d1) -> (d0, d1 floordiv 64, 0, 0, 31, d1 mod 64)>
func.func @lowering_krnl_memset(%arg0: memref<1xi64>, %arg1: memref<1xi64>) -> (memref<?x?xf16, #map_2ds>) {
  %cst_0 = arith.constant 0 : index
  %cst_f0 = arith.constant 0.0 : f16
  %cst_f1 = arith.constant 1.0 : f16
  %0 = affine.load %arg0[%cst_0] : memref<1xi64>
  %i0 = arith.index_cast %0 : i64 to index
  %1 = affine.load %arg1[%cst_0] : memref<1xi64>
  %i1 = arith.index_cast %1 : i64 to index
  %2 = memref.alloc(%i0, %i1) {alignment = 4096 : i64} : memref<?x?xf16, #map_2ds>
  // Set all elements including padding elements to 0.0.
  krnl.memset %2, %cst_f0 {delayed = true} : memref<?x?xf16, #map_2ds>
  // Set visible elements (non-padding elements) to 1.0
  krnl.memset %2, %cst_f1 {delayed = false} : memref<?x?xf16, #map_2ds>
  return %2 : memref<?x?xf16, #map_2ds>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
// CHECK-LABEL:  func.func @lowering_krnl_memset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1xi64>, [[PARAM_1_:%.+]]: memref<1xi64>) -> memref<?x?x1x1x32x?xf16> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f16
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]][0] : memref<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = affine.load [[PARAM_1_]][0] : memref<1xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK:           [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_3_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_4_]]) {{.*}}: memref<?x?x1x1x32x64xf16>
// CHECK:           [[VAR_6_:%.+]] = memref.cast [[RES_]] : memref<?x?x1x1x32x64xf16> to memref<?x?x1x1x32x?xf16>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_1_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_3_]]{{.}} {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 1 {
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to 1 {
// CHECK:                   affine.for [[I_4_:%.+]] = 0 to 32 {
// CHECK:                     affine.for [[I_5_:%.+]] = 0 to 64 {
// CHECK:                       affine.store [[VAR_cst_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]], [[I_4_]], [[I_5_]]{{.}} : memref<?x?x1x1x32x64xf16>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_6_:%.+]] = 0 to [[VAR_1_]] {
// CHECK:             affine.for [[I_7_:%.+]] = 0 to [[VAR_3_]] {
// CHECK:               affine.store [[VAR_cst_0_]], [[RES_]]{{.}}[[I_6_]], [[I_7_]] floordiv 64, 0, 0, 31, [[I_7_]] mod 64] : memref<?x?x1x1x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_6_]] : memref<?x?x1x1x32x?xf16>
// CHECK:         }
}

