<!--- Automatically generated, do not edit. -->
<!--- python documentOps.py --arch NNPA --input /home/imaihal/work/onnx-mlir/test/accelerators/NNPA/backend/CMakeLists.txt --path /home/imaihal/work/onnx-mlir/utils --notes --unsupported -->

# Supported ONNX Operation for Target *NNPA*.

Onnx-mlir currently support ONNX operations targeting up to opset 14. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodifiedto Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


| Op |Up to Opset |Limitations |Notes |
| --- |--- |--- |--- |
| **Add** |14 |<ul><li>Shape of input tensors should be the same since broadcasting is not supported</li><li>Unknown dimensions in input tensor is not supported</li></ul>. | |
| **AveragePool** |11 |Support padding type of VALID and SAME UPPER. ceil_mode not supported. 4D tensors(N x C x H x W) are supported as input and output. static kernel shape is supported. count_include_pad must be default value. ceil_mode not supported. | |
| **BatchNormalization** |9 |4D tensors(N x C x H x W) are supported as input and output. | |
| **Conv** |11 |Support padding type of VALID and SAME UPPER. Not supported if height and weight dims are unknown. Defult group(=1) and default dilations(=1) are supported. 4D tensors(N x C x H x W) are supported as input and output. static kernel shape is supported. | |
| **Div** |14 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **Exp** |13 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **GRU** |7 |direction and hidden_size in W must have static dimensions. R must have static dimensions. B and initial_h have static dimensions if given. B's direction dim must be 1 or 2. sequence_lens, activation_alpha, and activation_beta not supported. The default activations (["Sigmoid", "Tanh", "Tanh"]) are supported. clip(Cell clip threshold) not supported. hidden_size should be equal to the hidden size in other inputs. linear_before_reset must be 1. | |
| **Gemm** |13 |Alpha and beta must be 1. Rank of input tensor A and B must be 2, and rank of C must be 1 or 2. If rank of C is 1, second dim of B must be the same with dim of C. | |
| **GlobalAveragePool** |1 |Input shape must be HCHW. Unknown dim in height and width not supported. | |
| **LSTM** |7 |direction and hidden_size in W must have static dimensions. R must have static dimensions. B and initial_h have static dimensions if given. B's direction dim must be 1 or 2. | |
| **Log** |13 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **LogSoftmax** |13 | | |
| **MatMul** |13 |(Rank of A, Rank of B) must be (2, 2), (3, 3), and (3, 2). | |
| **Max** |13 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **MaxPool** |12 | | |
| **Min** |13 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **Mul** |14 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **Softmax** |13 |Rank of input tensor must be 2. axis must be 1 or -1. | |
| **Sub** |14 |Shape of input tensors should be the same since broadcasting is not supported. Unknown dimensions in input tensor is not supported. | |
| **Sum** |13 |All inputs must have the same static shape (Broadcasting not supported.) Single input not supported. | |
