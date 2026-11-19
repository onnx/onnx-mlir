// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --zlow-stick-expansion %s | FileCheck --check-prefix=MAX %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=max --zlow-stick-expansion %s | FileCheck --check-prefix=MAX %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit --zlow-stick-expansion %s | FileCheck --check-prefix=LIMIT %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit3e9 --zlow-stick-expansion %s | FileCheck --check-prefix=LIMIT3E9 %s

// COM: Saturation bounds of a compiler-generated stick (zlow.stick expanded
// COM: into a loop): +/-DLF16_MAX by default and with
// COM: --nnpa-saturation=max/mask; +/-DLF16_MAX/4 with limit; with
// COM: limit<value>, the value rounded down to dlfloat16 (3e9 ->
// COM: 2.99892736e9).

#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_stick_saturation_limit(%arg0: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
  %alloc = memref.alloc() alignment = 4096 : memref<16x8x128xf16, #map>
  "zlow.stick"(%arg0, %alloc) {layout = "3DS"} : (memref<16x8x128xf32>, memref<16x8x128xf16, #map>) -> ()
  return %alloc : memref<16x8x128xf16, #map>

// MAX-LABEL:  func.func @test_stick_saturation_limit
// MAX-DAG:      arith.constant dense<-8.57315738E+9> : vector<4xf32>
// MAX-DAG:      arith.constant dense<8.57315738E+9> : vector<4xf32>

// LIMIT-LABEL:  func.func @test_stick_saturation_limit
// LIMIT-DAG:      arith.constant dense<-2.14328934E+9> : vector<4xf32>
// LIMIT-DAG:      arith.constant dense<2.14328934E+9> : vector<4xf32>
// LIMIT-NOT:      8.57315738E+9

// LIMIT3E9-LABEL:  func.func @test_stick_saturation_limit
// LIMIT3E9-DAG:      arith.constant dense<-2.99892736E+9> : vector<4xf32>
// LIMIT3E9-DAG:      arith.constant dense<2.99892736E+9> : vector<4xf32>
}
