# Testing

In onnx-mlir, there are three types of tests to ensure correctness of implementation:

## ONNX Backend Tests

TODO.

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
  