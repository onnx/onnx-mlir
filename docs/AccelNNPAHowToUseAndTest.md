<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build and test for Accelerator NNPA

## Build

The following CMake variable is required to build onnx-mlir for NNPA.

- `-DONNX_MLIR_ACCELERATORS=NNPA`

## Test

Lit tests and numerical tests are provided for NNPA.

- Lit tests

When building onnx-mlir for NNPA, lit tests for NNPA also run with the command for CPU. The lit tests for NNPA are included in `test/mlir/accelerators/nnpa`.

```
cmake --build . --target check-onnx-lit
```

- Numerical tests

Numerical tests for NNPA are provided in `test/accelerator/NNPA/numrical`. Currently tests for MatMul2D, Gemm, LSTM, and GRU are provided and run by using following command. For MatMul2D and Gemm, the same test code with CPU is used. For LSTM and GRU, different parameter configurations are provided for NNPA-specific tests. In these tests, appropriate ATOL and RTOL are used to pass the tests.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```