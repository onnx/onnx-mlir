// RUN: onnx-mlir --maccel=NNPA --EmitMLIR --printIR %s | FileCheck %s

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>

// Transpose will be done directly on stickified data, so no need to unstickify.
func.func @transpose_on_ztensor(%arg0: tensor<3x5xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = "onnx.Transpose"(%0) {perm = [1,0]} : (tensor<3x5xf32>) -> tensor<5x3xf32>
  %2 = "onnx.Relu" (%1) : (tensor<5x3xf32>) -> tensor<5x3xf32>
  return %2 : tensor<5x3xf32>

// CHECK-LABEL:  func.func @transpose_on_ztensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5xf32>) -> memref<5x3xf32> attributes {llvm.emit_c_interface} {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "2D"} : (memref<3x5xf32>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_1", shape = [2], value = dense<[3, 5]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_]], [[VAR_0_]], [[RES_]]_0) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 3 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = affine.load [[RES_1_]][0, 0, 0, 0, [[I_0_]], [[I_1_]]] : memref<1x1x1x1x32x64xf16>
// CHECK:               affine.store [[LOAD_RES_1_MEM_]], [[RES_2_]][0, 0, 0, 0, [[I_1_]], [[I_0_]]] : memref<1x1x1x1x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1x1x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [2], value = dense<[5, 3]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           "zlow.relu"([[RES_2_]], [[VAR_1_]], [[RES_3_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<2xi64>, memref<1x1x1x1x32x64xf16>) -> ()
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<5x3xf32>
// CHECK:           "zlow.unstick"([[RES_3_]], [[RES_4_]]) {layout = "2D"} : (memref<1x1x1x1x32x64xf16>, memref<5x3xf32>) -> ()
// CHECK:           return [[RES_4_]] : memref<5x3xf32>
// CHECK:         }
}

// -----

// Transpose will be done directly on stickified data, so no need to unstickify.
func.func @transpose_on_ztensor_unknown_dims(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.Transpose"(%0) {perm = [1,0]} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.Relu" (%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>

// CHECK-LABEL:  func.func @transpose_on_ztensor_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>) -> memref<?x?xf32> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[RES_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[VAR_cast_]]) {layout = "2D"} : (memref<?x?xf32>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_2_]], [[VAR_3_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_2_:%.+]] = memref.cast [[RES_1_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_4_]], [[RES_2_]][0] : memref<2xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_5_]], [[RES_2_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_]], [[RES_2_]], [[VAR_cast_]]_2) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_7_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK:           [[VAR_cast_5_:%.+]] = memref.cast [[RES_3_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[VAR_dim_0_]] {
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = affine.load [[RES_1_]][0, [[I_1_]] floordiv 64, 0, [[I_0_]] floordiv 32, [[I_0_]] mod 32, [[I_1_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:               affine.store [[LOAD_RES_1_MEM_]], [[RES_3_]][0, [[I_0_]] floordiv 64, 0, [[I_1_]] floordiv 32, [[I_1_]] mod 32, [[I_0_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[VAR_8_]], [[VAR_9_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_7_:%.+]] = memref.cast [[RES_4_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_10_]], [[RES_5_]][0] : memref<2xi64>
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_11_]], [[RES_5_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_5_]], [[RES_5_]], [[VAR_cast_7_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK:           [[RES_6_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) {{.*}}: memref<?x?xf32>
// CHECK:           "zlow.unstick"([[VAR_cast_7_]], [[RES_6_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<?x?xf32>) -> ()
// CHECK:           return [[RES_6_]] : memref<?x?xf32>
// CHECK:         }
}
