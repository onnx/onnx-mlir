// RUN: onnx-mlir --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --EmitMLIR --printIR -tag="test" %s | FileCheck %s

// -----

// Transpose will be done directly on stickified data, so no need to unstickify.

func.func @transpose_on_ztensor(%arg0: tensor<3x5xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = "onnx.Relu" (%0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %2 = "onnx.Transpose"(%1) {perm = [1,0]} : (tensor<3x5xf32>) -> tensor<5x3xf32>
  %3 = "onnx.Relu" (%2) : (tensor<5x3xf32>) -> tensor<5x3xf32>
  %4 = "onnx.Relu" (%3) : (tensor<5x3xf32>) -> tensor<5x3xf32>
  onnx.Return %4 : tensor<5x3xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @transpose_on_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5xf32>) -> memref<5x3xf32> attributes {llvm.emit_c_interface} {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "2D"} : (memref<3x5xf32>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_3", shape = [2], value = dense<[3, 5]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_]], [[VAR_0_]], [[RES_]]_0) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_2", shape = [2], value = dense<[3, 5]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_1_]], [[VAR_1_]], [[RES_2_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK:               [[LOAD_RES_2_MEM_:%.+]] = affine.load [[RES_2_]][0, 0, 0, 0, [[I_0_]], [[I_1_]]{{.}} : memref<1x1x1x1x32x64xf16>
// CHECK:               affine.store [[LOAD_RES_2_MEM_]], [[RES_3_]][0, 0, 0, 0, [[I_1_]], [[I_0_]]{{.}} : memref<1x1x1x1x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_1", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_3_]], [[VAR_2_]], [[RES_4_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_4_]], [[VAR_3_]], [[RES_5_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<5x3xf32>
// CHECK:           "zlow.unstick"([[RES_5_]], [[RES_6_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<5x3xf32>) -> ()
// CHECK:           return [[RES_6_]] : memref<5x3xf32>
// CHECK:         }
}
