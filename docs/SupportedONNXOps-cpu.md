<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops` -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 19. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


| Op |Min Opset |Max Opset |Limitations |Notes |
| --- |--- |--- |--- |--- |
| **Abs** |13 |18 | | |
| **Acos** |7 |18 | | |
| **Acosh** |9 |18 | | |
| **Adagrad** | | |unsupported | |
| **Adam** | | |unsupported | |
| **Add** |14 |18 |No support for short integers. | |
| **And** |7 |18 | | |
| **ArgMax** |13 |18 | | |
| **ArgMin** |13 |18 | | |
| **ArrayFeatureExtractor** | | |unsupported | |
| **Asin** |7 |18 | | |
| **Asinh** |9 |18 | | |
| **Atan** |7 |18 | | |
| **Atanh** |9 |18 | | |
| **AveragePool** |19 |19 | | |
| **BatchNormalization** |15 |18 |Training not supported. | |
| **Bernoulli** | | |unsupported | |
| **Binarizer** | | |unsupported | |
| **BitShift** | | |unsupported | |
| **BitwiseAnd** |18 |18 | | |
| **BitwiseNot** | | |unsupported | |
| **BitwiseOr** |18 |18 | | |
| **BitwiseXor** |18 |18 | | |
| **BlackmanWindow** | | |unsupported | |
| **Cast** |19 |19 |Cast only between float and double types. Only ppc64le and MacOS platforms support float16. | |
| **CastLike** | | |unsupported | |
| **CastMap** | | |unsupported | |
| **CategoryMapper** | | |unsupported | |
| **Ceil** |13 |18 | | |
| **Celu** | | |unsupported | |
| **CenterCropPad** | | |unsupported | |
| **Clip** |13 |18 |No support for short integers. | |
| **Col2Im** | | |unsupported | |
| **Compress** |11 |18 | | |
| **Concat** |13 |18 | | |
| **ConcatFromSequence** | | |unsupported | |
| **Constant** |19 |19 | | |
| **ConstantOfShape** |9 |18 | | |
| **Conv** |11 |18 | | |
| **ConvInteger** | | |unsupported | |
| **ConvTranspose** |11 |18 |Unknown dimension in spatial dimensions (such as H and W) not supported. | |
| **Cos** |7 |18 | | |
| **Cosh** |9 |18 | | |
| **CumSum** |14 |18 | | |
| **DFT** |17 |18 | | |
| **DepthToSpace** |13 |18 | | |
| **DequantizeLinear** |19 |18 |Only support for per-tensor or layer dequantization. Not support for per-axis dequantization. | |
| **Det** | | |unsupported | |
| **DictVectorizer** | | |unsupported | |
| **Div** |14 |18 |No support for short integers. | |
| **Dropout** |13 |18 |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 |18 | | |
| **Einsum** |12 |18 |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. | |
| **Elu** |6 |18 | | |
| **Equal** |19 |19 | | |
| **Erf** |13 |18 | | |
| **Exp** |13 |18 | | |
| **Expand** |13 |18 | | |
| **EyeLike** | | |unsupported | |
| **FeatureVectorizer** | | |unsupported | |
| **Flatten** |13 |18 | | |
| **Floor** |13 |18 | | |
| **GRU** |14 |18 | | |
| **Gather** |13 |18 | | |
| **GatherElements** |13 |18 | | |
| **GatherND** |13 |18 | | |
| **Gemm** |13 |18 | | |
| **GlobalAveragePool** |1 |18 | | |
| **GlobalLpPool** | | |unsupported | |
| **GlobalMaxPool** |1 |18 | | |
| **Gradient** | | |unsupported | |
| **Greater** |13 |18 | | |
| **GreaterOrEqual** |16 |18 | | |
| **GridSample** | | |unsupported | |
| **GroupNormalization** | | |unsupported | |
| **HammingWindow** | | |unsupported | |
| **HannWindow** | | |unsupported | |
| **HardSigmoid** |6 |18 | | |
| **HardSwish** | | |unsupported | |
| **Hardmax** |13 |18 | | |
| **Identity** |19 |19 |Sequence identity not supported. | |
| **If** |11 |18 |Sequence and Optional outputs are not supported. | |
| **Imputer** | | |unsupported | |
| **InstanceNormalization** |6 |18 | | |
| **IsInf** |10 |18 | | |
| **IsNaN** |13 |18 | | |
| **LRN** |13 |18 | | |
| **LSTM** |14 |18 | | |
| **LabelEncoder** | | |unsupported | |
| **LayerNormalization** | | |unsupported | |
| **LeakyRelu** |16 |18 | | |
| **Less** |13 |18 | | |
| **LessOrEqual** |16 |18 | | |
| **LinearClassifier** | | |unsupported | |
| **LinearRegressor** | | |unsupported | |
| **Log** |13 |18 | | |
| **LogSoftmax** |13 |18 |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |11 |11 |No support for opset 13 and 16 at this time. | |
| **LpNormalization** | | |unsupported | |
| **LpPool** | | |unsupported | |
| **MatMul** |13 |18 | | |
| **MatMulInteger** |10 |18 | | |
| **Max** |13 |18 |No support for unsigned int. Only ppc64le and MacOS platforms support float16. | |
| **MaxPool** |12 |18 |Does not support argmax and short ints. Support single output only. | |
| **MaxRoiPool** | | |unsupported | |
| **MaxUnpool** | | |unsupported | |
| **Mean** |13 |18 | | |
| **MeanVarianceNormalization** | | |unsupported | |
| **MelWeightMatrix** | | |unsupported | |
| **Min** |13 |18 |Does not support unsigned numbers. Only ppc64le and MacOS platforms support float16. | |
| **Mish** | | |unsupported | |
| **Mod** |13 |18 |Support float and double only. Only ppc64le and MacOS platforms support float16. | |
| **Momentum** | | |unsupported | |
| **Mul** |14 |18 |Does not support short integers. | |
| **Multinomial** | | |unsupported | |
| **Neg** |13 |18 | | |
| **NegativeLogLikelihoodLoss** | | |unsupported | |
| **NonMaxSuppression** |11 |18 | | |
| **NonZero** |13 |18 | | |
| **Normalizer** | | |unsupported | |
| **Not** |1 |18 | | |
| **OneHot** |11 |18 | | |
| **OneHotEncoder** | | |unsupported | |
| **Optional** | | |unsupported | |
| **OptionalGetElement** | | |unsupported | |
| **OptionalHasElement** | | |unsupported | |
| **Or** |7 |18 | | |
| **PRelu** |16 |18 | | |
| **Pad** |18 |18 |axes input not supported. | |
| **Pow** |15 |18 |No support for power with integer types. | |
| **QLinearConv** | | |unsupported | |
| **QLinearMatMul** | | |unsupported | |
| **QuantizeLinear** |19 |19 |Do not support per-axis and i8 quantization. | |
| **RNN** |14 |18 | | |
| **RandomNormal** | | |unsupported | |
| **RandomNormalLike** | | |unsupported | |
| **RandomUniform** | | |unsupported | |
| **RandomUniformLike** | | |unsupported | |
| **Range** |11 |18 | | |
| **Reciprocal** |13 |18 | | |
| **ReduceL1** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceL2** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceLogSum** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceLogSumExp** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceMax** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceMean** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceMin** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceProd** |18 |18 |do_not_keep_dim not supported. | |
| **ReduceSum** |13 |18 |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |18 |18 |Default axis and do_not_keep_dim not supported. | |
| **Relu** |14 |18 | | |
| **Reshape** |19 |19 |allowzero not supported. | |
| **Resize** |10 |18 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. | |
| **ReverseSequence** |10 |18 | | |
| **RoiAlign** | | |unsupported | |
| **Round** |11 |18 | | |
| **STFT** | | |unsupported | |
| **SVMClassifier** | | |unsupported | |
| **SVMRegressor** | | |unsupported | |
| **Scaler** | | |unsupported | |
| **Scan** |9 |18 |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **Scatter** | | |unsupported | |
| **ScatterElements** |18 |18 |Does not support duplicate indices. | |
| **ScatterND** |18 |18 |Does not support scatternd add/multiply. | |
| **Selu** |6 |18 | | |
| **SequenceAt** | | |unsupported | |
| **SequenceConstruct** | | |unsupported | |
| **SequenceEmpty** | | |unsupported | |
| **SequenceErase** | | |unsupported | |
| **SequenceInsert** |9 |18 |Does not support unranked sequence element. | |
| **SequenceLength** | | |unsupported | |
| **SequenceMap** | | |unsupported | |
| **Shape** |19 |19 |Does not support start and end attributes. | |
| **Shrink** | | |unsupported | |
| **Sigmoid** |13 |18 | | |
| **Sign** |13 |18 | | |
| **Sin** |7 |18 | | |
| **Sinh** |9 |18 | | |
| **Size** |19 |19 | | |
| **Slice** |13 |18 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |13 |18 | | |
| **SoftmaxCrossEntropyLoss** | | |unsupported | |
| **Softplus** |1 |18 | | |
| **Softsign** |1 |18 | | |
| **SpaceToDepth** |13 |18 | |Example works, the other is imprecise. To investigate. |
| **Split** |11 |18 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **SplitToSequence** | | |unsupported | |
| **Sqrt** |13 |18 | | |
| **Squeeze** |11 |18 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StringNormalizer** | | |unsupported | |
| **Sub** |14 |18 |Does not support short integers. | |
| **Sum** |13 |18 | | |
| **Tan** |7 |18 | | |
| **Tanh** |13 |18 | | |
| **TfIdfVectorizer** | | |unsupported | |
| **ThresholdedRelu** | | |unsupported | |
| **Tile** |13 |18 | | |
| **TopK** |11 |18 | | |
| **Transpose** |13 |18 | | |
| **TreeEnsembleClassifier** | | |unsupported | |
| **TreeEnsembleRegressor** | | |unsupported | |
| **Trilu** |14 |18 | | |
| **Unique** |11 |18 | | |
| **Unsqueeze** |11 |18 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |9 |18 | | |
| **Where** |16 |18 | | |
| **Xor** |7 |18 | | |
| **ZipMap** | | |unsupported | |
