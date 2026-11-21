// RUN: onnx-mlir --mtriple=x86_64-unknown-linux-gnu --march=znver3 --EmitObj -v %s -o %t 2>&1 | FileCheck --check-prefix=CPU %s
// RUN: onnx-mlir --mtriple=x86_64-unknown-linux-gnu --march=znver3 --mcpu=skylake --EmitObj -v %s -o %t 2>&1 | FileCheck --check-prefix=MCPU %s
// RUN: onnx-mlir --mtriple=x86_64-unknown-linux-gnu --march=x86-64 --EmitObj -v %s -o %t 2>&1 | FileCheck --check-prefix=GENERIC %s
// RUN: onnx-mlir -O3 --mtriple=x86_64-unknown-linux-gnu --march=znver3 --EmitMLIR --printIR %s | FileCheck --check-prefix=SIMD %s

// REQUIRES: x86

// An x86 CPU name given with --march, as --march=native returns on x86 hosts,
// is not a valid llc --march target: it is passed as --mcpu with
// --march=x86-64, and enables the same SIMD support as --march=x86-64.

// CPU:          opt{{.*}} --march=x86-64 --mcpu=znver3
// CPU:          llc{{.*}} --march=x86-64 --mcpu=znver3

// MCPU:         llc{{.*}} --march=x86-64 --mcpu=skylake

// GENERIC:      llc{{.*}} --march=x86-64
// GENERIC-NOT:  --mcpu
// GENERIC-SAME: -filetype=obj

// SIMD:         vector<{{[0-9]+}}xf32>
module {
  func.func @main_graph(%arg0: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<16x64xf32>) -> tensor<16x64xf32>
    onnx.Return %0 : tensor<16x64xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
