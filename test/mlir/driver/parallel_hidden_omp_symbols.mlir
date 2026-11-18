// Test that a model compiled with --parallel keeps the OpenMP runtime it
// embeds (libompruntime.a) private: the runtime is linked into the model, but
// none of its symbols are exported. Otherwise the dynamic loader may bind the
// model's OpenMP calls to another OpenMP runtime loaded in the same process
// (for example PyTorch's libomp), which crashes at the model's first parallel
// region with more than one thread.

// REQUIRES: openmp
// REQUIRES: system-linux || system-darwin

// RUN: onnx-mlir -O3 --parallel %s -o %t | FileCheck --check-prefix=COMPILE %s

// The OpenMP runtime is linked into the model (as local symbols).
// RUN: llvm-nm --defined-only %t.so | FileCheck --check-prefix=LINKED %s

// The model's entry points are exported.
// RUN: llvm-nm %if system-linux %{ -D %} --extern-only --defined-only %t.so | FileCheck --check-prefix=EXPORTED %s

// No OpenMP runtime symbol is exported.
// RUN: llvm-nm %if system-linux %{ -D %} --extern-only --defined-only %t.so | FileCheck --check-prefix=NOT-EXPORTED %s

// RUN: rm %t.so

// COMPILE: Compilation completed

// LINKED: __kmpc_fork_call

// EXPORTED-DAG: omQueryEntryPoints
// EXPORTED-DAG: run_main_graph

// Match at the start of the symbol name only (the model's own symbols carry a
// tag derived from the file name), plus libomp's mangled C++ symbols.
// NOT-EXPORTED-NOT: {{ [_.]*(kmp|omp_|ompc_|ompt_|GOMP_|gomp_)}}
// NOT-EXPORTED-NOT: {{ _+Z.*(kmp|distributedBarrier)}}

module {
  func.func @main_graph(%arg0: tensor<64x1024xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<64x1024xf32>, tensor<64x1024xf32>) -> tensor<64x1024xf32>
    onnx.Return %0 : tensor<64x1024xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
