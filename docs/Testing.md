# Testing

In onnx-mlir, there are three types of tests to ensure correctness of implementation:

## ONNX Backend Tests

Backend tests are end-to-end tests for onnx-mlir based on onnx node tests.
To invoke the test, use the following command:

```
cmake --build . --config Release --target check-onnx-backend
``` 
Packages, such as third_party/onnx and ssl, needs to be installed to run the backend test.

The node tests in onnx that will be run by check-onnx-backend is defined by variable test_to_enable in test/backend/test.py. User can test one test case by environment variable BACKEND_TEST. For example,
```
BACKEND_TEST=selected_test_name cmake --build . --config Release --target check-onnx-backend
```
With BACKEND_TEST specified, the intermedia result, the .onnx file and .so file, are kept in build/test/backend for debugging.

When the ONNX-to-Krnl conversion of an operator is added, the corresponding backend tests for this operator should be added to test.py. The available test cases can be found in third_part/onnx/onnx/backend/test/case/node. Please note to add suffix `_cpu` to the onnx test name. 

The onnx node tests usually have known dimension size for input tensors. To test tensor with unknown dimension, the model importer (Build/FrontendONNXTransformer.cpp) provides a functionality to generate such cases. When the environment variable, `IMPORTER_FORCE_DYNAMIC`, is set, the frontend import will turn the first dimension (by default) of some input tensor of the model into -1. 
```
IMPORTER_FORCE_DYNAMIC=-1 all the inputs will be changed
IMPORTER_FORCE_DYNAMIC=0 the first input will be changed
IMPORTER_FORCE_DYNAMIC=n input[n] will be changed
```
For example, the default model for test_add_cpu is:

`func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>`

with IMPORTER_FORCE_DYNAMIC=-1, the result is:

`func @main_graph(%arg0: tensor<?x4x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<?x4x5xf32>`

with IMPORTER_FORCE_DYNAMIC=0, the result is:

  `func @main_graph(%arg0: tensor<?x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>`.

Which dimension to be changed can be specified with env variable `IMPORTER_FORCE_DYNAMIC`.
```
IMPORTER_FORCE_DYNAMIC_DIM=-1 all the dimensions to be changed
IMPORTER_FORCE_DYNAMIC_DIM=0 the first dimension
IMPORTER_FORCE_DYNAMIC_DIM=n the n+1 th dimension
```
For example, with `IMPORTER_FORCE_DYNAMIC=0 IMPORTER_FORCE_DYNAMIC_DIM=1`, the result is:

`func @main_graph(%arg0: tensor<3x?x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> `

This is a way to use existing node test for dynamic tensors. Since not all test case can pass with dynamic tensor, there is a list in test/backend/test.py, test_not_for_dynamic, to specify which test can not pass with IMPORTER_FORCE_DYNAMIC is defined.

## LLVM FileCheck Tests

TODO.

## Numerical Tests

Numerical tests are used to test for numerical correctness in addition to the tests provided by the ONNX package.
The goal is to provide extensive numerical value based unit tests; this is very important for ensuring that
optimization transformations are valid and correct: more corner cases will arise as we specialize for specific 
architecture parameters (like vector width). Numerical tests generates extensive amount of numerical value-based 
unit tests based on simple, naive (and extremely slow) implementation of operations being tested, used to verify 
the correctness of our operation lowering and optimization.

Numerical tests should be structured such that the following two components are independent and separate:
- Generation of test case parameters (for instance, the dimensions of convolutions N, C, H, W, kH, kW ...).
- Checking that the values produced by onnx-mlir is consistent with those produced by naive implementation.

The motivation is that there are two ways we want to generate test case parameters:
- Exhaustive generation of test case parameters. Where we want to exhaustively test the correctness of a small range
of parameters (for instance, if we would like to test and verify that 3x3 convolution is correctly implmented for
all valid padding configurations.)
- When the possible parameter space is extremely large, we can rely on RapidCheck to randomly generate test cases
that becomes increasingly large as smaller test cases succeed. And it also automatically shrinks the test cases
in the event that an error occurs. For example, the following RapidCheck test case automatically generates test
case parameters (N from between 1 and 10, C from within 1 and 20 etc...). By default rc::check will draw 100 sets of
test case parameters and invoke the value checking function `isOMConvTheSameAsNaiveImplFor`.

```cpp
  // RapidCheck test case generation.
  rc::check("convolution implementation correctness", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 20);
    const auto H = *rc::gen::inRange(5, 20);
    const auto W = *rc::gen::inRange(5, 20);

    const auto kH = *rc::gen::inRange(1, 15);
    const auto kW = *rc::gen::inRange(1, 15);

    // We don't want an entire window of padding.
    const auto pHBegin = *rc::gen::inRange(0, kH - 1);
    const auto pHEnd = *rc::gen::inRange(0, kH - 1);
    const auto pWBegin = *rc::gen::inRange(0, kW - 1);
    const auto pWEnd = *rc::gen::inRange(0, kW - 1);

    // Make sure we have at least 1 output per dimension.
    RC_PRE((H >= kH) && (W > kW));

    RC_ASSERT(isOMConvTheSameAsNaiveImplFor(
        N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd));
  });
```
  
