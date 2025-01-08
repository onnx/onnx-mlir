// RUN: (onnx-mlir-opt --march=z16 --maccel=NNPA --normalize-memrefs %s 2>&1 || true) | FileCheck --check-prefix=FAILED %s

// COM: Current MLIR normalize-memres does not support multiple dereferencing uses
// in a single op, check expected failure emitted by MLIR. 

// FAILED: "multiple dereferencing uses in a single op not supported"

// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zlow-dummyop-for-multideref --normalize-memrefs --canonicalize %s | FileCheck --check-prefix=PASSED %s

// COM: Check normalize memrefs when there are multiple dereferencing uses in a single op.
// COM: Check that --zlow-dummyop-for-multideref can help to bypass the issue.
#map = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
module {
  func.func @should_normalize(%arg0: memref<3x4x5xf16, #map>, %arg1: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
    %0 = memref.alloc() {alignment = 4096 : i64} : memref<3x4x5xf16, #map>
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3xi64>
    "zlow.add"(%arg0, %arg0, %1, %0) {layout = "3D"} : (memref<3x4x5xf16, #map>, memref<3x4x5xf16, #map>, memref<3xi64>, memref<3x4x5xf16, #map>) -> ()
    return %0 : memref<3x4x5xf16, #map>
  }
  // PASSED-LABEL: @should_normalize
  // PASSED-SAME:   ([[PARAM_0_:%.+]]: memref<1x1x3x1x32x64xf16>, [[PARAM_1_:%.+]]: memref<1x1x3x1x32x64xf16>) -> memref<1x1x3x1x32x64xf16> {
  // PASSED-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x3x1x32x64xf16>
  // PASSED-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
  // PASSED:           "zlow.add"([[PARAM_0_]], [[PARAM_0_]], [[RES_1_]], [[RES_]]) {layout = "3D"} : (memref<1x1x3x1x32x64xf16>, memref<1x1x3x1x32x64xf16>, memref<3xi64>, memref<1x1x3x1x32x64xf16>) -> ()
  // PASSED:           return [[RES_]] : memref<1x1x3x1x32x64xf16>
  // PASSED:         }
}

