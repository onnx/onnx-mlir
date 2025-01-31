<!--- SPDX-License-Identifier: Apache-2.0 -->

# Overview 
 
NNPA in IBM Telum II supports 8-bit signed-integer quantized matrix multiplications. This document shows how to compile an ONNX model for 8-bit quantization on NNPA. When not following these steps, models will still be accelerated when targeting Telum systems using a mixture of 16-bit floating-point numbers for computations mapped to the Telum's Integrated AI accelerator and 32-bit floating-point numbers for computations mapped to the Telum CPUs.

There are two approaches to using quantization in the onnx-mlir compiler, depending on the input ONNX model to the compile:
- The input model is a quantized model that was quantized by other frameworks such as ONNX Runtime. In this case, the input ONNX model contains 8-bit operations, and the onnx-mlir compiler selects suitable 8-bit operations to run on NNPA. There is no special compile flags needed to enable quantization when compiling this quantized model. Hence, we do not discuss this case in this document.
  - In this approach, the compiler supports both static and dynamic quantized models.
- The input model is a non-quantized model, e.g. operations operate on float32 data types. In this case, the onnx-mlir compiler provides several quantization options in order to quantize the model during compilation, then run the compiled model on NNPA. The remaining of this document describes this approach.
  - In this approach, the compiler only supports dynamic quantization.

In both approaches, the following constraints are applied:
- Only per-tensor quantization is supported, meaning `scale` and `zero_point` are computed per-tensor and are scalar values.
- Target quantization data type is 8-bit signed-integer.
 
Quantization requires NNPA in IBM Telum II, meaning that the following compile flags must be specified to enable quantization: `-maccel=NNPA -march=arch15`.

# Dynamic quantization by the compiler

Again, it is important to note that the onnx-mlir compiler currently:
- supports per-tensor dynamic quantization, and
- quantizes data tensors from float32 to 8-bit signed integer. If a data tensor in the input model is already in 8-bit singed integer, the compiler will not quantize it again.

The compiler provides two compile flags for dynamically quantizing a model at compile time:
- `--nnpa-quant-dynamic` to enable dynamic quantization.
- `--nnpa-quant-op-types` to specify the types of ONNX operations to quantize manually, e.g. `MatMul,Conv`.

Users can specify whether or not to symmetrize data for activations and weights by using options `symActivation, asymActivation, symWeight, asymWeight` as values for `--nnpa-quant-dynamic`.
For examples, to asymmetrize data for activations and to symmetrize data for weights, one can use `--nnpa-quant-dynamic=asymActivation,symWeight`.

By specifying `--nnpa-quant-dynamic` only, the compiler will decide quantization options and operation types by itself.

## Computing `scale` and `zero_point` 
The compiler uses the following equations to compute `scale` and `zero_point` for 8-bit signed integer quantization.

Asymmetric quantization
```
scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
zero_point = cast(round(saturate(qmin - min(x)/scale)))
```
where
- `x` is the input tensor to quantize,
- data range is adjusted to include 0,
- `qmax=127` and `qmin=-128` are the max and min values for quantization range.
- `saturate` is to saturate to `[-128, 127]`.

Symmetric quantization
```
scale = max(abs(x)) / 127
zero_point = 0
```

Given `scale` and `zero_point`, the input `x` is quantized to
```
quantized_x = x/scale + zero_point
```

# Performance notes

It is often the case that symmetric quantization leads to better inference performance but poorer accuracy than asymmetric quantization.
Users may want to experiment with different quantization schemes to find the best combination for their own model.

# Resources
- [A visual guide to quantization](https://www.maartengrootendorst.com/blog/quantization/)
