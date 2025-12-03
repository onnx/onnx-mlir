// RUN: onnx-mlir --EmitLLVMIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?xf32> {onnx.dim_params = "0:a", onnx.name = "x"}) -> (tensor<?xf32> {onnx.dim_params = "0:a", onnx.name = "exp"}) {
    %0 = "onnx.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
    onnx.Return %0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()

// CHECK-NOT: onnx.dim_params
// CHECK-NOT: onnx.name
// CHECK-NOT: Unhandled parameter attribute

// CHECK:     llvm.func @main_graph_remove_onnx_arg_attrs([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr, [[arg2_:%.+]]: i64, [[arg3_:%.+]]: i64, [[arg4_:%.+]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface}
// CHECK:     llvm.func @_mlir_ciface_main_graph_remove_onnx_arg_attrs([[arg0_:%.+]]: !llvm.ptr, [[arg1_:%.+]]: !llvm.ptr) attributes {llvm.emit_c_interface}
}
