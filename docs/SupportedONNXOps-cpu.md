<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops` -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 19. Limitations are listed when applicable. This documentation highlights the minimum and maximum opset versions that are fully supported by onnx-mlir and not the version changes.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.
* Please note an * in the "Max Opset" column indicates that this operation is at the current version.


| Op |Min Opset |Max Opset |Limitations |Notes |
| --- |--- |--- |--- |--- |
| **Abs** |6 |18 | | |
| **Acos** |7 |18 | | |
| **Acosh** |9 |18 | | |
| **Adagrad** | | |unsupported | |
| **Adam** | | |unsupported | |
| **Add** |6 |18 |No support for short integers. | |
| **And** |7 |18 | | |
| **ArgMax** |6 |18 | | |
| **ArgMin** |6 |18 | | |
| **ArrayFeatureExtractor** | | |unsupported | |
| **Asin** |7 |18 | | |
| **Asinh** |9 |18 | | |
| **Atan** |7 |18 | | |
| **Atanh** |9 |18 | | |
| **AveragePool** |7 |18 | | |
| **BatchNormalization** |6 |18 |Training not supported. | |
| **Bernoulli** | | |unsupported | |
| **Binarizer** | | |unsupported | |
| **BitShift** | | |unsupported | |
| **BitwiseAnd** |18 |* | | |
| **BitwiseNot** | | |unsupported | |
| **BitwiseOr** |18 |* | | |
| **BitwiseXor** |18 |* | | |
| **BlackmanWindow** | | |unsupported | |
| **Cast** |6 |19 |Cast only between float and double types. Only ppc64le and MacOS platforms support float16. | |
| **CastLike** | | |unsupported | |
| **CastMap** | | |unsupported | |
| **CategoryMapper** | | |unsupported | |
| **Ceil** |6 |18 | | |
| **Celu** | | |unsupported | |
| **CenterCropPad** | | |unsupported | |
| **Clip** |6 |18 |No support for short integers. | |
| **Col2Im** | | |unsupported | |
| **Compress** |9 |18 | | |
| **Concat** |6 |18 | | |
| **ConcatFromSequence** | | |unsupported | |
| **Constant** |6 |19 | | |
| **ConstantOfShape** |9 |18 | | |
| **Conv** |6 |18 | | |
| **ConvInteger** | | |unsupported | |
| **ConvTranspose** |6 |18 |Unknown dimension in spatial dimensions (such as H and W) not supported. | |
| **Cos** |7 |18 | | |
| **Cosh** |9 |18 | | |
| **CumSum** |11 |18 | | |
| **DFT** |17 |18 | | |
| **DepthToSpace** |6 |18 | | |
| **DequantizeLinear** |10 |18 |Only support for per-tensor or layer dequantization. Not support for per-axis dequantization. | |
| **Det** | | |unsupported | |
| **DictVectorizer** | | |unsupported | |
| **Div** |6 |18 |No support for short integers. | |
| **Dropout** |6 |18 |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 |18 | | |
| **Einsum** |12 |18 |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. | |
| **Elu** |6 |18 | | |
| **Equal** |7 |19 | | |
| **Erf** |9 |18 | | |
| **Exp** |6 |18 | | |
| **Expand** |8 |18 | | |
| **EyeLike** | | |unsupported | |
| **FeatureVectorizer** | | |unsupported | |
| **Flatten** |9 |18 | | |
| **Floor** |6 |18 | | |
| **GRU** |7 |18 | | |
| **Gather** |6 |18 | | |
| **GatherElements** |11 |18 | | |
| **GatherND** |11 |18 | | |
| **Gemm** |6 |18 | | |
| **GlobalAveragePool** |6 |18 | | |
| **GlobalLpPool** | | |unsupported | |
| **GlobalMaxPool** |6 |18 | | |
| **Gradient** | | |unsupported | |
| **Greater** |7 |18 | | |
| **GreaterOrEqual** |12 |18 | | |
| **GridSample** | | |unsupported | |
| **GroupNormalization** | | |unsupported | |
| **HammingWindow** | | |unsupported | |
| **HannWindow** | | |unsupported | |
| **HardSigmoid** |6 |18 | | |
| **HardSwish** | | |unsupported | |
| **Hardmax** |6 |18 | | |
| **Identity** |6 |18 |Sequence identity not supported. | |
| **If** |6 |18 |Sequence and Optional outputs are not supported. | |
| **Imputer** | | |unsupported | |
| **InstanceNormalization** |6 |18 | | |
| **IsInf** |10 |18 | | |
| **IsNaN** |9 |18 | | |
| **LRN** |6 |18 | | |
| **LSTM** |7 |18 | | |
| **LabelEncoder** | | |unsupported | |
| **LayerNormalization** | | |unsupported | |
| **LeakyRelu** |6 |18 | | |
| **Less** |7 |18 | | |
| **LessOrEqual** |12 |18 | | |
| **LinearClassifier** | | |unsupported | |
| **LinearRegressor** | | |unsupported | |
| **Log** |6 |18 | | |
| **LogSoftmax** |6 |18 |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |6 |11 |No support for opset 13 and 16 at this time. | |
| **LpNormalization** | | |unsupported | |
| **LpPool** | | |unsupported | |
| **MatMul** |9 |18 | | |
| **MatMulInteger** |10 |18 | | |
| **Max** |6 |18 |No support for unsigned int. Only ppc64le and MacOS platforms support float16. | |
| **MaxPool** |6 |18 |Does not support argmax and short ints. Support single output only. | |
| **MaxRoiPool** | | |unsupported | |
| **MaxUnpool** | | |unsupported | |
| **Mean** |6 |18 | | |
| **MeanVarianceNormalization** | | |unsupported | |
| **MelWeightMatrix** | | |unsupported | |
| **Min** |6 |18 |Does not support unsigned numbers. Only ppc64le and MacOS platforms support float16. | |
| **Mish** | | |unsupported | |
| **Mod** |10 |18 |Support float and double only. Only ppc64le and MacOS platforms support float16. | |
| **Momentum** | | |unsupported | |
| **Mul** |6 |18 |Does not support short integers. | |
| **Multinomial** | | |unsupported | |
| **Neg** |6 |18 | | |
| **NegativeLogLikelihoodLoss** | | |unsupported | |
| **NonMaxSuppression** |10 |18 | | |
| **NonZero** |9 |18 | | |
| **Normalizer** | | |unsupported | |
| **Not** |6 |18 | | |
| **OneHot** |9 |18 | | |
| **OneHotEncoder** | | |unsupported | |
| **Optional** | | |unsupported | |
| **OptionalGetElement** | | |unsupported | |
| **OptionalHasElement** | | |unsupported | |
| **Or** |7 |18 | | |
| **PRelu** |6 |18 | | |
| **Pad** |2 |18 |axes input not supported. | |
| **Pow** |7 |18 |No support for power with integer types. | |
| **QLinearConv** | | |unsupported | |
| **QLinearMatMul** | | |unsupported | |
| **QuantizeLinear** |10 |18 |Do not support per-axis and i8 quantization. | |
| **RNN** |7 |18 | | |
| **RandomNormal** | | |unsupported | |
| **RandomNormalLike** | | |unsupported | |
| **RandomUniform** | | |unsupported | |
| **RandomUniformLike** | | |unsupported | |
| **Range** |11 |18 | | |
| **Reciprocal** |6 |18 | | |
| **ReduceL1** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceL2** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceLogSum** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceLogSumExp** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceMax** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceMean** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceMin** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceProd** |6 |18 |do_not_keep_dim not supported. | |
| **ReduceSum** |6 |18 |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |6 |18 |Default axis and do_not_keep_dim not supported. | |
| **Relu** |6 |18 | | |
| **Reshape** |6 |18 |allowzero not supported. | |
| **Resize** |10 |18 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. | |
| **ReverseSequence** |10 |18 | | |
| **RoiAlign** | | |unsupported | |
| **Round** |11 |18 | | |
| **STFT** | | |unsupported | |
| **SVMClassifier** | | |unsupported | |
| **SVMRegressor** | | |unsupported | |
| **Scaler** | | |unsupported | |
| **Scan** |8 |18 |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **Scatter** | | |unsupported | |
| **ScatterElements** |11 |18 |Does not support duplicate indices. | |
| **ScatterND** |11 |18 |Does not support scatternd add/multiply. | |
| **Selu** |6 |18 | | |
| **SequenceAt** | | |unsupported | |
| **SequenceConstruct** | | |unsupported | |
| **SequenceEmpty** | | |unsupported | |
| **SequenceErase** | | |unsupported | |
| **SequenceInsert** |11 |18 |Does not support unranked sequence element. | |
| **SequenceLength** | | |unsupported | |
| **SequenceMap** | | |unsupported | |
| **Shape** |6 |18 |Does not support start and end attributes. | |
| **Shrink** | | |unsupported | |
| **Sigmoid** |6 |18 | | |
| **Sign** |9 |18 | | |
| **Sin** |7 |18 | | |
| **Sinh** |9 |18 | | |
| **Size** |6 |18 | | |
| **Slice** |6 |18 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |6 |18 | | |
| **SoftmaxCrossEntropyLoss** | | |unsupported | |
| **Softplus** |6 |18 | | |
| **Softsign** |6 |18 | | |
| **SpaceToDepth** |6 |18 | |Example works, the other is imprecise. To investigate. |
| **Split** |6 |18 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **SplitToSequence** | | |unsupported | |
| **Sqrt** |6 |18 | | |
| **Squeeze** |6 |18 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StringNormalizer** | | |unsupported | |
| **Sub** |6 |18 |Does not support short integers. | |
| **Sum** |6 |18 | | |
| **Tan** |7 |18 | | |
| **Tanh** |6 |18 | | |
| **TfIdfVectorizer** | | |unsupported | |
| **ThresholdedRelu** | | |unsupported | |
| **Tile** |6 |18 | | |
| **TopK** |6 |18 | | |
| **Transpose** |6 |18 | | |
| **TreeEnsembleClassifier** | | |unsupported | |
| **TreeEnsembleRegressor** | | |unsupported | |
| **Trilu** |14 |18 | | |
| **Unique** |11 |18 | | |
| **Unsqueeze** |6 |18 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |7 |18 | | |
| **Where** |9 |18 | | |
| **Xor** |7 |18 | | |
| **ZipMap** | | |unsupported | |
