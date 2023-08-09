
// RUN: onnx-mlir --maccel=NNPA -v %s -o %t 2>&1 | FileCheck %s

// -----
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-NEXT {{clang|c|g}}++{{.*}} compile-config.o -o compile-config.so -shared -fPIC -L{{.*}}/lib -lRuntimeNNPA -lzdnn -lcruntime 

