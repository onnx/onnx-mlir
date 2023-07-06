// RUN: onnx-mlir --maccel=NNPA --EmitMLIR --printIR %s | FileCheck %s

// -----

// Transpose will be done directly on stickified data, so no need to unstickify.
func.func @transpose_on_ztensor_unknown_dims(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.Relu" (%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.Transpose"(%0) {perm = [1,0]} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.Relu" (%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %2 : tensor<?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>

// CHECK-LABEL:  func.func @transpose_on_ztensor_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>) -> memref<?x?xf32> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [1], value = dense<-8.57315738E+9> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[VAR_dim_0_]] {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = affine.load [[VAR_0_]][0] : memref<1xf32>
// CHECK:               [[VAR_15_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:               affine.store [[VAR_16_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_2_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[RES_1_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK:           "zlow.stick"([[RES_]], [[VAR_cast_]]) {layout = "2D"} : (memref<?x?xf32>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_3_]], [[VAR_4_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_3_:%.+]] = memref.cast [[RES_2_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_5_]], [[RES_3_]][0] : memref<2xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_6_]], [[RES_3_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_]], [[RES_3_]], [[VAR_cast_]]_3) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[VAR_7_]], [[VAR_8_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK:           [[VAR_cast_6_:%.+]] = memref.cast [[RES_4_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK:           affine.for [[I_2_:%.+]] = 0 to [[VAR_dim_]] {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to [[VAR_dim_0_]] {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.load [[RES_2_]][0, [[I_3_]] floordiv 64, 0, [[I_2_]] floordiv 32, [[I_2_]] mod 32, [[I_3_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:               affine.store [[LOAD_PARAM_0_MEM_1_]], [[RES_4_]][0, [[I_2_]] floordiv 64, 0, [[I_3_]] floordiv 32, [[I_3_]] mod 32, [[I_2_]] mod 64] : memref<1x?x1x?x32x64xf16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_10_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_9_]], [[VAR_10_]]) {{.*}}: memref<1x?x1x?x32x64xf16>
// CHECK-DAG:       [[VAR_cast_8_:%.+]] = memref.cast [[RES_5_]] : memref<1x?x1x?x32x64xf16> to memref<1x?x1x?x?x?xf16>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           affine.store [[VAR_11_]], [[RES_6_]][0] : memref<2xi64>
// CHECK:           [[VAR_12_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           affine.store [[VAR_12_]], [[RES_6_]][1] : memref<2xi64>
// CHECK:           "zlow.relu"([[VAR_cast_6_]], [[RES_6_]], [[VAR_cast_8_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<2xi64>, memref<1x?x1x?x?x?xf16>) -> ()
// CHECK:           [[RES_7_:%.+]] = memref.alloc([[VAR_dim_0_]], [[VAR_dim_]]) {{.*}}: memref<?x?xf32>
// CHECK:           "zlow.unstick"([[VAR_cast_8_]], [[RES_7_]]) {layout = "2D"} : (memref<1x?x1x?x?x?xf16>, memref<?x?xf32>) -> ()
// CHECK:           return [[RES_7_]] : memref<?x?xf32>
// CHECK:         }
}
