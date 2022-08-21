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

Numerical tests for NNPA are provided in `test/accelerators/NNPA/numerical`. Currently tests for Conv2D, MatMul2D, Gemm, LSTM, and GRU are provided and run by using following command. These tests check if a zDNN instruction is generated in shared library in adition to accurary check.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```

These tests uses the same test code with numerical tests for CPU (`test/modellib` and `test/numerial`), but uses different cmake file(`test/accelerator/NNPA/numerical/CMakeLists.txt`).

##### Conv2D
Since Conv2D of zDNN library does not support the case where dilations equal to one, `-dilation=1` option are added in `test/numerical/TestConv.cpp`. Also, since only VALID and SAME_UPPER as pading type are supported, `-padding=valid_upper is prepared to use the pading type. Currently dynamic height and weight dimension are not supported. So `-dim=static` are provided.
To set data range for input data and weightsi n Conv2D, an environment variable `TestConvNNPA_DATARANGE` are used. Currently the value is 0.1 written in cmake file to pass the test.

##### Gemm
Since `alpha` and `beta` should be one for Matmul of zDNN library, `-alpha=1` and `-beta=1` options are added in `test/numerical/TestGemm.cpp` and set in the CMakeLists.txt (`test/accelerator/NNPA/numerical/CMakeLists.txt`)

##### LSTM
Since LSTM of zDNN library does not support peephole tensor, `-peephole=0` are added in `test/numerial/TestLSTM.cpp` and set in the CMakeLists.

##### GRU
Since GRU of zDNN library support only LinearBeforeReset=1, `-linerBeforeReset=1` option is added in `test/numerial/TestGRU.cpp` and set in the CMakeLists.

### Backend tests

Backend tests for NNPA are provided in `test/accelerators/NNPA/backend`. It can be run with following command. Only test cases supported by zDNN runs as listed in `test/accelerators/NNPA/backend/CMakeLists.txt`.

```
cmake --build . --config Release --target check-onnx-backend-nnpa
```

ATOL and RTOL for NNPA are set using environment variables `TEST_ATOL` and `TEST_RTOL` in the `CMakeLists.txt`.
Also, the environment variables `TEST_INSTRUCTION_CHECK` and `TEST_CASE_BY_USER` allow you to check if the NNPA instruction is generated in the shared library. In `CMakeLists.txt`, `TEST_INSTRUCTION_CHECK` is set to true and `TEST_CASE_BY_USER` contains the test case and instruction name. If the instruction name is not found in the shared library, the test will fail.
