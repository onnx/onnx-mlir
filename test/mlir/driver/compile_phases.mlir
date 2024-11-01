// RUN: onnx-mlir %s -o %t| FileCheck --check-prefix=LIB %s && rm %t.so
// RUN: onnx-mlir %s --EmitObj -o %t| FileCheck --check-prefix=OBJ %s && rm %t.o
// RUN: onnx-mlir %s --EmitJNI -o %t| FileCheck --check-prefix=JNI %s && rm %t.jar

// LIB: [1/6] {{.*}} Importing ONNX Model to MLIR Module from
// LIB: [2/6] {{.*}} Compiling and Optimizing MLIR Module
// LIB: [3/6] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// LIB: [4/6] {{.*}} Generating Object from LLVM Bitcode
// LIB: [5/6] {{.*}} Linking and Generating the Output Shared Library
// LIB: [6/6] {{.*}} Compilation completed

// OBJ: [1/5] {{.*}} Importing ONNX Model to MLIR Module from
// OBJ: [2/5] {{.*}} Compiling and Optimizing MLIR Module
// OBJ: [3/5] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// OBJ: [4/5] {{.*}} Generating Object from LLVM Bitcode
// OBJ: [5/5] {{.*}} Compilation completed

// JNI: [1/8] {{.*}} Importing ONNX Model to MLIR Module from
// JNI: [2/8] {{.*}} Compiling and Optimizing MLIR Module
// JNI: [3/8] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// JNI: [4/8] {{.*}} Generating Object from LLVM Bitcode
// JNI: [5/8] {{.*}} Generating JNI Object
// JNI: [6/8] {{.*}} Linking and Generating the Output Shared Library
// JNI: [7/8] {{.*}} Creating JNI Jar
// JNI: [8/8] {{.*}} Compilation completed
module {
  func.func @main_graph(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    onnx.Return %arg0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}


