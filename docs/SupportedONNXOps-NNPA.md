<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops_NNPA' -->

# Supported ONNX Operation for Target *NNPA*.

Onnx-mlir currently supports ONNX operations targeting up to opset 23. Limitations are listed when applicable. This documentation highlights the minimum and maximum opset versions that are fully supported by onnx-mlir and not the version changes.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* **Supported Opsets** indicates the lowest and highest opset a model may have for onnx-mlir to support compiling a model with the operator.
   * A * indicates onnx-mlir is compatible with the latest version of that operator available as of opset 23.
   * A ^ indicates onnx-mlir is compatible with the latest level of the NNPA Architecture which is z17.


NNPA for z16 and z17 have hardware limitations in dimension index size and tensor size, which are described in [NNPALimit.cpp](../src/Accelerators/NNPA/Support/NNPALimit.cpp). They are large enough for normal use cases, but if your model exceeds the limitations, CPU is used instead of NNPA. NNPA currently only support DLFLOAT16 as its data type. Common data formats like FP32, FP16, BFLOAT need to undergo data conversions to the NNPA internal format DLFLOAT16. Hence ONNX ops which updated their tensors to BFLOAT16 will not be natively supported on NNPA.  Onnx-mlir with NNPA utilizes hardware when possible. To accomplish this, the compiler converts ONNX ops to [ZHigh](Dialects/zhigh.md) ops, [ZLow](Dialects/zlow.md) ops, and are processed by the [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN).


Refer to the [Qunatization-NNPA.md](https://github.com/onnx/onnx-mlir/blob/main/docs/Quantization-NNPA.md#limiations) page for limitations pertaining to quantization support on z17.




| Op |Supported Opsets (inclusive) |Minimum NNPA Level(Inclusive) |Limitations |Notes |
| --- |--- |--- |--- |--- |
| **Add** |6 - * |z16 - ^ |Shape of input tensors must be the same since broadcasting is not supported. | |
| **AveragePool** |6 - * |z16 - ^ |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors (N x C x H x W).<br>- `kernel_shape` must be static.<br>- `count_include_pad` must be default value(0).<br>- `ceil_mode` must be default value(0). | |
| **BatchNormalization** |6 - * |z16 - ^ |Input and output tensor must be 4D(N x C x H x W). | |
| **Conv** |6 - * |z16 - ^ |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- Dimension in Height and weight must be static.<br>- `group` must be default value(1).<br>- `dilations` must be default value(1).<br>- Input and output tensors must have 4D (N x C x H x W).<br>- `kernel_shape` must be static. | |
| **ConvTranspose** |6 - * |z16 - ^ |- 1D and 3D not supported because Conv1D and Conv3D not supported in zDNN. non-default `dilations` not supported because dilated convolution not supported in zDNN. | |
| **Div** |6 - * |z16 - ^ |Shape of input tensors must be the same since broadcasting is not supported. | |
| **Exp** |6 - * |z16 - ^ |Input tensor must have 4 dimensions. | |
| **GRU** |7 - * |z16 - ^ |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- If `B` and `initial_h` are given, they must have static dimensions.<br>- `sequence_lens` is not supported for bidirectional GRU.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `linear_before_reset` must be 1.<br>- `layout` is not supported. | |
| **Gelu** |20 - * |z17 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
| **Gemm** |6 - * |z16 - ^ |- `alpha` and `beta` must be default value(1).<br>- Rank of `C` must be 1 or 2. If the rank is 1, the dimension of `C` must be the same with the seconde dimension of `B`.<br>. | |
| **GlobalAveragePool** |6 - * |z16 - ^ |- Input shape must be 4D tensor(NCHW).<br>- Dimensions in `H` and `W` must be static. | |
| **LSTM** |7 - * |z16 - ^ |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- `B` and `initial_h` have static dimensions if given. `B`'s direction dim must be 1 or 2.<br>- `P`(peepholes), `activation_alpha`, and `activation_beta` are not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `input_forget` must be default value(0).<br>- `layout` is not supported. | |
| **LeakyRelu** |6 - * |z17 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
| **Log** |6 - * |z16 |Input tensor must have 4 dimensions. | |
| **LogSoftmax** |6 - * |z16 - ^ | | |
| **MatMul** |6 - * |z16 - ^ |Ranks of input tensors must be (Rank of A, Rank of B) = (M, N), where M >= 2 and N >= 2. | |
| **MatMulInteger** |10 - * |z17 - ^ | | |
| **Max** |6 - * |z16 - ^ |Shape of input tensors must be the same since broadcasting is not supported. | |
| **MaxPool** |6 - * |z16 - ^ |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors(N x C x H x W).<br>- `kernel_shape` must be static.<br>- `ceil_mode` must be default value(0).<br>- `dilations` must be default value(1). | |
| **Min** |6 - * |z16 - ^ |Shape of input tensors must be the same since broadcasting is not supported. | |
| **Mul** |6 - * |z16 - ^ |Shape of input tensors should be the same since broadcasting is not supported. | |
| **Pow** |7 - * |z16 - ^ |- Exponent should be a scalar integer and less or equal to 64. | |
| **QLinearMatMul** |10 - * |z17 - ^ |Only support i8 and ui8 for zeropoint, and f32 for scale. | |
| **ReduceMax** |6 - * |z17 - ^ |- `keepdims` must be 1.<br>- `noop_with_empty_axes` must be 0.<br>- Does not support reduction over multiple axes.<br>- We do not support `do_not_keepdims` backend tests.<br>- Only support reduction over the innermost dimension. | |
| **ReduceMean** |6 - * |z16 - ^ |- `keepdims` must be 1.<br>- Input tensor must be 4D tensors and `axis` must be [2, 3]. | |
| **ReduceMin** |6 - * |z17 - ^ |- `keepdims` must be 1.<br>- `noop_with_empty_axes` must be 0.<br>- Does not support reduction over multiple axes.<br>- We do not support `do_not_keepdims` backend tests.<br>- Only support reduction over the innermost dimension. | |
| **Relu** |6 - * |z16 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
| **Sigmoid** |6 - * |z16 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
| **Softmax** |6 - * |z16 - ^ |- `axis` must be the last dimension, i.e. `rank - 1` or -1. | |
| **Softplus** |6 - * |z16 - ^ |The operations immediately before and after the Softplus operation must be executed on the NNPA. Otherwise, Softplus is executed on the CPU. This limitation is set to avoid performance degradation. | |
| **Sqrt** |6 - * |z17 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
| **Sub** |6 - * |z16 - ^ |Shape of input tensors should be the same since broadcasting is not supported. | |
| **Sum** |6 - * |z16 - ^ |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Single input not supported. | |
| **Tanh** |6 - * |z16 - ^ |Input tensor must be less than or equal to 4 dimensions. | |
