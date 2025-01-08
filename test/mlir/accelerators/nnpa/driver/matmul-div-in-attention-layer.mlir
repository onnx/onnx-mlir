// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitMLIR --nnpa-enable-scalar-bcast-binary --printIR %s | FileCheck %s

// Check whether the compiler can remove unstick/stick so that the output of zdnn matmul is passed directly to zdnn div.
func.func @matmul_div(%arg0: tensor<?x12x?x64xf32>) -> tensor<?x?x?x?xf32> {
  %scalar = onnx.Constant dense<8.000000e+00> : tensor<f32>
  %b = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2]} : (tensor<?x12x?x64xf32>) -> tensor<?x12x64x?xf32>
  %m = "onnx.MatMul"(%arg0, %b) : (tensor<?x12x?x64xf32>, tensor<?x12x64x?xf32>) -> tensor<?x12x?x?xf32>
  %r = "onnx.Div"(%m, %scalar) : (tensor<?x12x?x?xf32>, tensor<f32>) -> tensor<?x12x?x?xf32>
  "onnx.Return"(%r) : (tensor<?x12x?x?xf32>) -> ()

// CHECK-LABEL: func.func @matmul_div
// CHECK:           memref.alloc
// CHECK:           memref.alloc
// CHECK:           [[ALLOC:%.+]] = memref.alloc({{.*}}) {{.*}}: memref<?x?x1x?x32x64xf16>
// CHECK-DAG:       [[MATMUL_RES:%.+]] = memref.cast [[ALLOC]] : memref<?x?x1x?x32x64xf16> to memref<?x?x1x?x?x?xf16>
// CHECK:           "zlow.matmul"({{.*}}, {{.*}}, {{.*}}, {{.*}}, [[MATMUL_RES]]) {is_bcast1 = 0 : si64, is_bcast23 = 0 : si64, is_stacked = -1 : si64, transposeA = 0 : si64, transposeB = 0 : si64} : (memref<?x1x1x?x?x64xf16>, memref<?x?x1x2x32x?xf16>, memref<?x?x1x1x32x?xf16>, memref<4xi64>, memref<?x?x1x?x?x?xf16>) -> ()
// CHECK-NOT:        "zlow.stick"
// CHECK-NOT:        "zlow.unstick"
// CHECK:           "zlow.div"([[MATMUL_RES]], {{.*}}, {{.*}}, {{.*}}) {layout = "3DS"} : (memref<?x?x1x?x?x?xf16>, memref<?x?x1x?x?x?xf16>, memref<3xi64>, memref<?x?x1x?x?x?xf16>) -> ()
}
