// RUN: onnx-mlir --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --EmitMLIR --printIR -tag="test" %s | FileCheck %s

// -----

// Transpose will be done directly on stickified data, so no need to unstickify.
func.func @transpose_on_ztensor_unknown_dims(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.Relu" (%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.Transpose"(%1) {perm = [1,0]} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "onnx.Relu" (%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "onnx.Relu" (%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %4 : tensor<?x?xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
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
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_7_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_5_:%.+]] = memref.cast [[RES_3_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_8_]], [[RES_4_]][0] : memref<2xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_9_]], [[RES_4_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_2_]], [[RES_4_]], [[VAR_cast_5_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_10_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_11_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_10_]], [[VAR_11_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK:           [[VAR_cast_8_:%.+]] = memref.cast [[RES_5_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[VAR_dim_0_]] {
// CHECK:               [[LOAD_RES_3_MEM_:%.+]] = affine.load [[RES_3_]][0, [[I_1_]] floordiv 64, 0, [[I_0_]] floordiv 32, [[I_0_]] mod 32, [[I_1_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:               affine.store [[LOAD_RES_3_MEM_]], [[RES_5_]][0, [[I_0_]] floordiv 64, 0, [[I_1_]] floordiv 32, [[I_1_]] mod 32, [[I_0_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_12_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_13_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_6_:%.+]] = memref.alloc([[VAR_12_]], [[VAR_13_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_10_:%.+]] = memref.cast [[RES_6_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_14_]], [[RES_7_]][0] : memref<2xi64>
// CHECK:           [[VAR_15_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_15_]], [[RES_7_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_8_]], [[RES_7_]], [[VAR_cast_10_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_16_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_17_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_8_:%.+]] = memref.alloc([[VAR_16_]], [[VAR_17_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_13_:%.+]] = memref.cast [[RES_8_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_18_]], [[RES_9_]][0] : memref<2xi64>
// CHECK:           [[VAR_19_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_19_]], [[RES_9_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_10_]], [[RES_9_]], [[VAR_cast_13_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK:           [[RES_10_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) {{.*}}: memref<?x?xf32>
// CHECK:           "zlow.unstick"([[VAR_cast_13_]], [[RES_10_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<?x?xf32>) -> ()
// CHECK:           return [[RES_10_]] : memref<?x?xf32>
// CHECK:         }
}
