// RUN: onnx-mlir --maccel=NNPA --EmitMLIR --printIR %s | FileCheck %s

// -----

// Transpose will be done directly on stickified data, so no need to unstickify.

func.func @transpose_on_ztensor(%arg0: tensor<3x5xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = "onnx.Transpose"(%0) {perm = [1,0]} : (tensor<3x5xf32>) -> tensor<5x3xf32>
  %2 = "onnx.Relu" (%1) : (tensor<5x3xf32>) -> tensor<5x3xf32>
  onnx.Return %2 : tensor<5x3xf32>

// CHECK-LABEL:  func.func @transpose_on_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5xf32>) -> memref<5x3xf32> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [1], value = dense<-8.57315738E+9> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x5xf32>
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = affine.load [[VAR_0_]][0] : memref<1xf32>
// CHECK:               [[VAR_5_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:               [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:               affine.store [[VAR_6_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           "zlow.stick"([[RES_]], [[RES_]]_0) {layout = "2D"} : (memref<3x5xf32>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_1", shape = [2], value = dense<[3, 5]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_1_]], [[VAR_1_]], [[RES_2_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 5 {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[RES_2_]][0, 0, 0, 0, [[I_2_]], [[I_3_]]{{.}} : memref<1x1x1x1x32x64xf16>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_3_]][0, 0, 0, 0, [[I_3_]], [[I_2_]]{{.}} : memref<1x1x1x1x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_3_]], [[VAR_2_]], [[RES_4_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<5x3xf32>
// CHECK:           "zlow.unstick"([[RES_4_]], [[RES_5_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<5x3xf32>) -> ()
// CHECK:           return [[RES_5_]] : memref<5x3xf32>
// CHECK:         }
}

