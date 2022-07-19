<!--- Automatically generated, do not edit. -->
<!--- python documentOps.py --arch cpu --input test/backend/inference_backend.py --path utils --notes --unsupported -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 16. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


| Op |Up to Opset |Limitations |Notes |
| --- |--- |--- |--- |
| **Abs** |13 | | |
| **Acos** |7 | | |
| **Acosh** |9 | | |
| **Adagrad** |1 | | |
| **Adam** |1 | | |
| **Add** |14 |No support for short integers. | |
| **And** |7 | | |
| **ArgMax** |13 | | |
| **ArgMin** |13 | | |
| **Asin** |7 | | |
| **Asinh** |9 | | |
| **Atan** |7 | | |
| **Atanh** |9 | | |
| **AveragePool** |11 | | |
| **BatchNormalization** |9 |Training not supported. | |
| **Bernoulli** |unsupported | | |
| **Bitshift** |unsupported | | |
| **Cast** |13 |Cast only between float and double types. | |
| **CastLike** |unsupported | | |
| **Ceil** |13 | | |
| **Celu** |12 | | |
| **Clip** |13, 12, 11, 6 |No support for short integers. | |
| **Compress** |11 | | |
| **Concat** |13 | | |
| **Constant** |13 | | |
| **ConstantOfShape** |9 | | |
| **Conv** |11 | | |
| **ConvInteger** |10 | | |
| **ConvTranspose** |11 | | |
| **Cos** |7 | | |
| **Cosh** |9 | | |
| **CumSum** |14 | | |
| **DepthToSpace** |13 | | |
| **DequatizeLinear** |unsupported | | |
| **Det** |11 | | |
| **Div** |14 |No support for short integers. | |
| **Dropout** |13 |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 | | |
| **Einsum** |12 |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. | |
| **Elu** |6 | | |
| **Equal** |13 | | |
| **Erf** |13 | | |
| **Exp** |13 | | |
| **Expand** |13 | | |
| **Eyelike** |unsupported | | |
| **Flatten** |13 | | |
| **Floor** |13 | | |
| **GRU** |7 |Batchwise test is not supported. | |
| **Gather** |13 | | |
| **GatherElements** |13 | | |
| **GatherND** |13 | | |
| **Gemm** |13 | | |
| **GlobalAveragePool** |1 | | |
| **GlobalMaxPool** |1 | | |
| **Greater** |13 | | |
| **GreaterOrEqual** |16 | | |
| **GridSample** |unsupported | | |
| **HardSigmoid** |6 | | |
| **HardSwish** |unsupported | | |
| **Hardmax** |13 | | |
| **Identity** |16 |Sequence identity not supported. | |
| **If** |16 | | |
| **InstanceNormalization** |6 | | |
| **IsInf** |10 | | |
| **IsNan** |unsupported | | |
| **LRN** |13 | | |
| **LSTM** |7 |No support for batchwise examples. | |
| **LeakyRelu** |16 | | |
| **Less** |13 | | |
| **LessOrEqual** |16 | | |
| **Log** |13 | | |
| **LogSoftmax** |13 |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |16 |No support for opset 13 and 16 at this time. | |
| **MatMul** |13 | | |
| **MatMulInteger** |10 | | |
| **Max** |13 |No support for short floats and unsigned int. | |
| **MaxPool** |12 |Does not support argmax and short ints. Support single output only. | |
| **MaxUnpool** |11 | | |
| **Mean** |13 | | |
| **MeanVarianceNormalization** |13 | | |
| **Min** |13 |Does not support short floats and unsigned numbers. | |
| **Mod** |13 |Support float and double only. | |
| **Momentum** |1 | | |
| **Mul** |14 |Does not support short integers. | |
| **Neg** |13 | | |
| **NegativeLogLikelihoodLoss** |13 | | |
| **NonMaxSuppression** |11 | | |
| **NonZero** |13 | | |
| **Not** |1 | | |
| **OneHot** |11 | | |
| **OptionalGetElement** |unsupported | | |
| **OptionalHasElement** |unsupported | | |
| **Or** |7 | | |
| **PRelu** |16 | | |
| **Pad** |13, 11, 2 | | |
| **Pow** |15 |No support for power with integer types. | |
| **QLinearConv** |10 | | |
| **QLinearMatmul** |unsupported | | |
| **QuantizeLinear** |13 | | |
| **RNN** |14 |Batchwise not supported. | |
| **Range** |11 | | |
| **Reciprocal** |13 | | |
| **ReduceL1** |13 | | |
| **ReduceL2** |13 | | |
| **ReduceLogSum** |13 | | |
| **ReduceLogSumExp** |13 | | |
| **ReduceMax** |13 | | |
| **ReduceMean** |13 | | |
| **ReduceMin** |13 | | |
| **ReduceProd** |13 | | |
| **ReduceSum** |13, 11 |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |13 | | |
| **Relu** |14 | | |
| **Reshape** |13 | | |
| **Resize** |13, 11, 10 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. | |
| **ReverseSequence** |10 | | |
| **RoiAlign** |10 | | |
| **Round** |11 | | |
| **Scan** |16 |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **ScatterElements** |13 |Does not support duplicate indices. | |
| **ScatterND** |16 |Does not support scatternd add/multiply. | |
| **Selu** |6 | | |
| **SequenceInsert** |11 | | |
| **Shape** |13 | | |
| **Shrink** |9 | | |
| **Sigmoid** |13 | | |
| **Sign** |13 | | |
| **Sin** |7 | | |
| **Sinh** |9 | | |
| **Size** |13 | | |
| **Slice** |13 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |13 | | |
| **SoftmaxCrossEntropyLoss** |13 | | |
| **Softplus** |1 | | |
| **Softsign** |1 | | |
| **SpaceToDepth** |13 | |Example works, the other is imprecise. To investigate. |
| **Split** |13, 11 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **Sqrt** |13 | | |
| **Squeeze** |13, 11 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StrNormalizer** |unsupported | | |
| **Sub** |14 |Does not support short integers. | |
| **Sum** |13 | | |
| **Tan** |7 | | |
| **Tanh** |13 | | |
| **TfdfVectorizer** |unsupported | | |
| **ThresholdRelu** |unsupported | | |
| **Tile** |13 | | |
| **TopK** |11 | | |
| **Transpose** |13 | | |
| **Trilu** |unsupported | | |
| **Unique** |11 | | |
| **Unsqueeze** |13, 11 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |10, 9, 7 | | |
| **Where** |16 | | |
| **Xor** |7 | | |
