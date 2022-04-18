// RUN: (onnx-mlir-opt --normalize-memrefs %s 2>&1 || true) | FileCheck --check-prefix=FAILED %s

// COM: Current MLIR normalize-memres does not support multiple dereferencing uses
// in a single op, check expected failure emitted by MLIR. 

// FAILED-LABEL: onnx-mlir-opt 
// FAILED: "multiple dereferencing uses in a single op not supported"

// RUN: onnx-mlir-opt --zlow-dummyop-for-multideref --normalize-memrefs --canonicalize %s | FileCheck --check-prefix=PASSED %s

// COM: Check normalize memrefs when there are multiple dereferencing uses in a single op.
// COM: Check that --zlow-dummyop-for-multideref can help to bypass the issue.
#map = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
module {
  func @should_normalize(%arg0: memref<3x4x5xf16, #map>, %arg1: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
    %c2 = arith.constant 2 : index
    %c5_i64 = arith.constant 5 : i64
    %c1 = arith.constant 1 : index
    %c4_i64 = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c3_i64 = arith.constant 3 : i64
    %0 = memref.alloc() {alignment = 4096 : i64} : memref<3x4x5xf16, #map>
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3xi64>
    krnl.store %c3_i64, %1[%c0] : memref<3xi64>
    krnl.store %c4_i64, %1[%c1] : memref<3xi64>
    krnl.store %c5_i64, %1[%c2] : memref<3xi64>
    "zlow.add"(%arg0, %arg0, %1, %0) {layout = "3D"} : (memref<3x4x5xf16, #map>, memref<3x4x5xf16, #map>, memref<3xi64>, memref<3x4x5xf16, #map>) -> ()
    return %0 : memref<3x4x5xf16, #map>
  }
  // PASSED-LABEL: @should_normalize
  // PASSED-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x3x1x32x64xf16>, [[PARAM_1_:%.+]]: memref<1x1x3x1x32x64xf16>) -> memref<1x1x3x1x32x64xf16> {
  // PASSED-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
  // PASSED-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
  // PASSED-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
  // PASSED-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
  // PASSED-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
  // PASSED-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
  // PASSED-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x3x1x32x64xf16>
  // PASSED-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
  // PASSED:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
  // PASSED:           krnl.store [[VAR_c4_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
  // PASSED:           krnl.store [[VAR_c5_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
  // PASSED:           "zlow.add"([[PARAM_0_]], [[PARAM_0_]], [[RES_1_]], [[RES_]]) {layout = "3D"} : (memref<1x1x3x1x32x64xf16>, memref<1x1x3x1x32x64xf16>, memref<3xi64>, memref<1x1x3x1x32x64xf16>) -> ()
  // PASSED:           return [[RES_]] : memref<1x1x3x1x32x64xf16>
  // PASSED:         }
}

