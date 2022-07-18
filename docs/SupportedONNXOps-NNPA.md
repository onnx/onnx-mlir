<!--- Automatically generated, do not edit. -->
<!--- python documentOps.py --arch NNPA --input test/accelerators/NNPA/backend/CMakeLists.txt --path utils --notes --unsupported -->

# Supported ONNX Operation for Target *NNPA*.

Onnx-mlir currently supports ONNX operations targeting up to opset 14. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


NNPA has hardware limitations in dimension index size and tensor size, which are described in [NNPALimit.h](../src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h). They are enough large for normal use cases, but if your model exceeds the limitations, CPU is used instead of NNPA.


| Op |Up to Opset |Limitations |Notes |
| --- |--- |--- |--- |
| **Add** |14 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **AveragePool** |11 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors (N x C x H x W).<br>- `kernel_shape` must be static.<br>- `count_include_pad` must be default value(0).<br>- `ceil_mode` must be default value(0). | |
| **BatchNormalization** |14 |Input and output tensor must be 4D(N x C x H x W). | |
| **Conv** |11 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- Dimension in Height and weight must be static.<br>- `group` must be default value(1).<br>- `dilations` must be default value(1).<br>- Input and output tensors must have 4D (N x C x H x W).<br>- `kernel_shape` must be static. | |
| **Div** |14 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Exp** |13 |Input tensor must have 4 dimensions. | |
| **GRU** |14 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- If `B` and `initial_h` are given, they must have static dimensions.<br>- `sequence_lens` is not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `linear_before_reset` must be 1.<br>- `layout` is not supported. | |
| **Gemm** |13 |- `alpha` and `beta` must be default value(1).<br>- Rank of `C` must be 1 or 2. If the rank is 1, the dimension of `C` must be the same with the seconde dimension of `B`. | |
| **GlobalAveragePool** |1 |- Input shape must be 4D tensor(NCHW).<br>- Dimensions in `H` and `W` must be static. | |
| **LSTM** |14 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- `B` and `initial_h` have static dimensions if given. `B`'s direction dim must be 1 or 2.<br>- `P`(peepholes), `activation_alpha`, and `activation_beta` are not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `input_forget` must be default value(0).<br>- `layout` is not supported. | |
| **Log** |13 |Input tensor must have 4 dimensions. | |
| **LogSoftmax** |13 | | |
| **MatMul** |13 |Ranks of input tensors must be (Rank of A, Rank of B) = (2, 2), (3, 3), and (3, 2). | |
| **Max** |13 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **MaxPool** |12 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors(N x C x H x W).<br>- `kernel_shape` must be static.<br>- `ceil_mode` must be default value(0).<br>- `dilations` must be default value(1). | |
| **Min** |13 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Mul** |14 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **ReduceMean** |13 |- `keepdims` must be 1.<br>- Input tensor must be 4D tensors and `axis` must be [2, 3]. | |
| **Relu** |14 |Input tensor must be less than or equal to 4 dimensions. | |
| **Sigmoid** |13 |Input tensor must be less than or equal to 4 dimensions. | |
| **Softmax** |13 |- Rank of input tensor must be 2.<br>- `axis` must be 1 or -1. | |
| **Sub** |14 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Sum** |13 |- All inputs must have the same static shape (Broadcasting not supported.)<br>- Single input not supported. | |
| **Tanh** |13 |Input tensor must be less than or equal to 4 dimensions. | |
