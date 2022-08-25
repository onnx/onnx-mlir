<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build and test for Accelerator NNPA

Neural Network Processing Assist Facility (NNPA) is implemented on processor units of IBM z16. Onnx-mlir can use it via  [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN). Building and lit tests runs on other IBM Z systems(eg. z15), but numerical tests need to run on z16.

## Build

Add following CMake option to build onnx-mlir for NNPA. Regarding build command for Linux OS, see [here](BuildOnLinuxOSX.md/#build)

- `-DONNX_MLIR_ACCELERATORS=NNPA`

## Test

### Lit tests

The lit tests for NNPA are included in `test/mlir/accelerators/nnpa`. When building onnx-mlir for NNPA, these lit tests also run with the following same command with CPU.

```
cmake --build . --target check-onnx-lit
```

### Numerical tests

Numerical tests for NNPA are provided in `test/accelerators/NNPA/numerical`. Currently tests for Conv2D, MatMul2D, Gemm, LSTM, and GRU are provided and run using following command. These tests can check if a zDNN instruction is included in the generated shared library using an environment variable `TEST_INSTRUCTION`. Also, to check the accuracy of the results, ATOL and RTOL can be set by using environment `TEST_ATOL` and `TEST_RTOL`. An environment variable `TEST_DATARANGE` are provided to set lower and upper bound of data range. They can be set "<lower bound>,<upper bound>" such as "-0.1,0.1". To configure the test cases, an environment variable `TEST_CONFIG` are provided. Current configurations are written in section of each test below.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```

These tests uses the same test code with numerical tests for CPU (`test/modellib` and `test/numerial`), but uses different cmake file(`test/accelerator/NNPA/numerical/CMakeLists.txt`).

##### Conv2D
Since Conv2D in zDNN library only supports the case where dilation equals to one, dilation is always set to one in the test. Also, padding types are set as VALID and SAME_UPPER since they are only suppored. All dimensions are static since dynamic height and weight dimension are currently not supported. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-dim=static -dilation=1 -padding=valid_upper".

##### Gemm
`alpha` and `beta` in Gemm are always one, which are supported case by zDNN library. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-alpha=1 -beta=1".

##### LSTM
Peephole tensor is not tested since LSTM in zDNN library does not support it. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-peephole=0".

##### GRU
GRU of zDNN library supports only the case where the linear transformation is applied before multiplying by the output of the reset gata. It is configured automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-linearBeforeReset=1".

### Backend tests

Backend tests for NNPA are provided in `test/accelerators/NNPA/backend`. It can be run with following command. Only test cases supported by zDNN runs as listed in `test/accelerators/NNPA/backend/CMakeLists.txt`.

```
cmake --build . --config Release --target check-onnx-backend-nnpa
```

ATOL and RTOL for NNPA are set using environment variables `TEST_ATOL` and `TEST_RTOL` in the `CMakeLists.txt`.
Also, the environment variables `TEST_INSTRUCTION_CHECK` and `TEST_CASE_BY_USER` allow you to check if the NNPA instruction is generated in the shared library. In `CMakeLists.txt`, `TEST_INSTRUCTION_CHECK` is set to true and `TEST_CASE_BY_USER` contains the test case and instruction name. If the instruction name is not found in the shared library, the test will fail.
