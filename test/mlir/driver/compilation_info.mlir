// RUN: onnx-mlir --EmitLLVMIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK: llvm.mlir.global internal constant @om_compilation_info_json_compilation_info("{\0A\22compiler_version\22: \22{{.*}}\22,\0A\22compile_options\22: \22--EmitLLVMIR --printIR compilation_info.mlir\22,\0A\22op_stats\22: {\0A  \22func.return.3D\22 : 1,\0A  \22onnx.Relu.3D\22 : 1\0A}\0A}\00") {addr_space = 0 : i32}

// CHECK: llvm.func @omCompilationInfo{{.*}}() -> !llvm.ptr
// CHECK:   {{.*}} = llvm.mlir.addressof @om_compilation_info_json
// CHECK:   llvm.return {{.*}} : !llvm.ptr
