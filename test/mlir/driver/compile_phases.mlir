// Disabled as cruntime does not exist in test env (AMD): onnx-mlir %s -o %t 2>&1 | FileCheck --check-prefix=EMIT-LIB %s && rm %t.so
// RUN: onnx-mlir %s --EmitObj -o %t 2>&1 | FileCheck --check-prefix=EMIT-OBJ %s && rm %t.o
// Disabled as jni libs do not exist in test env (AMD): onnx-mlir %s --EmitJNI -o %t 2>&1 | FileCheck --check-prefix=EMIT-JNI %s && rm %t.jar
// RUN: onnx-mlir %s --EmitLLVMIR -o %t 2>&1 | FileCheck --check-prefix=EMIT-LLVMIR %s && rm %t.onnx.mlir

// EMIT-OBJ: [1/5] {{.*}} Importing ONNX Model to MLIR Module from
// EMIT-OBJ: [2/5] {{.*}} Compiling and Optimizing MLIR Module
// EMIT-OBJ: [3/5] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// EMIT-OBJ: [4/5] {{.*}} Generating Object from LLVM Bitcode
// EMIT-OBJ: [5/5] {{.*}} Compilation completed

// EMIT-JNI: [1/8] {{.*}} Importing ONNX Model to MLIR Module from
// EMIT-JNI: [2/8] {{.*}} Compiling and Optimizing MLIR Module
// EMIT-JNI: [3/8] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// EMIT-JNI: [4/8] {{.*}} Generating Object from LLVM Bitcode
// EMIT-JNI: [5/8] {{.*}} Generating JNI Object
// EMIT-JNI: [6/8] {{.*}} Linking and Generating the Output Shared Library
// EMIT-JNI: [7/8] {{.*}} Creating JNI Jar
// EMIT-JNI: [8/8] {{.*}} Compilation completed

// EMIT-LLVMIR: [1/3] {{.*}} Importing ONNX Model to MLIR Module from
// EMIT-LLVMIR: [2/3] {{.*}} Compiling and Optimizing MLIR Module
// EMIT-LLVMIR: [3/3] {{.*}} Compilation completed
module {
  func.func @main_graph(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    onnx.Return %arg0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}


