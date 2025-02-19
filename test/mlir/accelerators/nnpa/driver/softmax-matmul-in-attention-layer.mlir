// RUN: onnx-mlir --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --EmitMLIR --printIR %s | FileCheck %s

// Check whether the compiler can remove unstick/stick so that the output of zdnn softmax is passed directly to zdnn matmul.
func.func @softmax_matmul(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = "onnx.Softmax"(%arg0) {axis = 3 : si64} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2 = "onnx.MatMul"(%0, %1) {axis = 3 : si64} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  "onnx.Return"(%2) : (tensor<?x?x?x?xf32>) -> ()

// CHECK-LABEL: @softmax_matmul
// CHECK:        "zlow.stick"
// CHECK:        [[ALLOC:%.+]] = memref.alloc({{.*}}, {{.*}}, {{.*}}) {alignment = 4096 : i64} : memref<?x?x1x?x32x64xf16>
// CHECK:        [[SOFTMAX_OUT:%.+]] = memref.cast [[ALLOC]] : memref<?x?x1x?x32x64xf16> to memref<?x?x1x?x?x?xf16>
// CHECK:        "zlow.softmax"({{.*}}, {{.*}}, {{.*}}, [[SOFTMAX_OUT]]) {act_func = "ACT_NONE"} {{.*}}
// CHECK:        "zlow.matmul"({{.*}}, [[SOFTMAX_OUT]], {{.*}}, {{.*}}, {{.*}}) {{.*}} 
}
