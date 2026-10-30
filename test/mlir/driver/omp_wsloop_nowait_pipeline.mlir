// RUN: onnx-mlir -O3 --parallel --EmitLLVMIR --printIR %s | FileCheck %s
// RUN: onnx-mlir -O3 --parallel --disable-omp-nowait --EmitLLVMIR --printIR %s | FileCheck --check-prefix=BARRIER %s

// Pins the pipeline order: omp-wsloop-nowait runs after the OpenMP regions
// are created and no later pass adds code between the wsloop and the end of
// its parallel region.

module {
  func.func @main_graph(%arg0: tensor<64x1024xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<64x1024xf32>, tensor<64x1024xf32>) -> tensor<64x1024xf32>
    onnx.Return %0 : tensor<64x1024xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()

// CHECK:         omp.parallel
// CHECK-NEXT:      omp.wsloop nowait {
// CHECK:           omp.terminator

// BARRIER:       omp.parallel
// BARRIER-NEXT:    omp.wsloop {
// BARRIER:         omp.terminator
}
