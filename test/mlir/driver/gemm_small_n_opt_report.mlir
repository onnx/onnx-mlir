// RUN: onnx-mlir -O3 --march=z16 --opt-report=Simd --EmitMLIR %s -o %t 2>&1 | FileCheck %s --check-prefix=SIMD
// RUN: onnx-mlir -O3 --march=z16 -parallel --opt-report=Parallel --EmitMLIR %s -o %t 2>&1 | FileCheck %s --check-prefix=PAR

// The small-N Gemm kernel reports its SIMD and parallel decisions. Report
// comments must not contain commas (onnxToKrnlSimdReport asserts on them), so
// compiling with --opt-report is a test of its own.

module {
  func.func @main_graph(%arg0: tensor<?x96xf32> {onnx.name = "a"}, %arg1: tensor<12x96xf32> {onnx.name = "b"}, %arg2: tensor<12xf32> {onnx.name = "c"}) -> (tensor<?x12xf32> {onnx.name = "y"}) {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64, onnx_node_name = "gemm_small_n"} : (tensor<?x96xf32>, tensor<12x96xf32>, tensor<12xf32>) -> tensor<?x12xf32>
    onnx.Return %0 : tensor<?x12xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// SIMD: ==SIMD-REPORT==, onnx.Gemm-simd, gemm_small_n, small-N gemm with N=12 and simd along K, 4, 96
// PAR: ==PAR-REPORT==, onnx.Gemm-par, gemm_small_n, GEMM small N row blocks, 0, -1
