<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build and test for Accelerator NNPA

Neural Network Processing Assist Facility (NNPA) is implemented on processor units of IBM z16. Onnx-mlir uses [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN) to use it. Building and lit tests runs on other IBM Z systems(eg. z15), but numerical tests need to run on z16.

## Build

Add following CMake option to build onnx-mlir for NNPA. Regarding build command for Linux OS, see [here](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md/#build)

- `-DONNX_MLIR_ACCELERATORS=NNPA`

## Test

- Lit tests

The lit tests for NNPA are included in `test/mlir/accelerators/nnpa`. When building onnx-mlir for NNPA, these lit tests also run with the following same command with CPU.

```
cmake --build . --target check-onnx-lit
```

- Numerical tests

Numerical tests for NNPA are provided in `test/accelerator/NNPA/numerical`. Currently tests for MatMul2D, Gemm, LSTM, and GRU are provided and run by using following command. For MatMul2D and Gemm, the same test code with CPU is used. For LSTM and GRU, different parameter configurations are provided for NNPA-specific tests. In these tests, appropriate ATOL and RTOL are used to pass the tests.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```