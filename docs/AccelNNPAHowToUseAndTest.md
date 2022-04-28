<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build and test for Accelerator NNPA

## Build

The following CMake variable is required to build onnx-mlir for NNPA.

- `-DONNX_MLIR_ACCELERATORS=NNPA`

## Test

Lit tests and numerical tests are provided for NNPA.

- Lit tests

When building onnx-mlir for NNPA, lit tests for NNPA run with the same command with CPU. The lit tests for NNPA are included in `test/mlir/accelerators/nnpa`.

```
cmake --build . --target check-onnx-lit
```

- Numerical tests

Numerical tests for NNPA are provided in `test/accelerator/NNPA/numrical`. Currently tests for MatMul2D and Gemm are the same with CPU. For LSTM and GRU, different parameter configurations are provided for NNPA-specific tests. In these tests, appropriate ATOL and RTOL are used to pass the tests.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```