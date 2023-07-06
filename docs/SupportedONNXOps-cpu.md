<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops` -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 19. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


| Op |Up to Opset |Limitations |Notes |
| --- |--- |--- |--- |
| **Abs** |13 | | |
| **Acos** |7 | | |
| **Acosh** |9 | | |
| **Adagrad** | |unsupported | |
| **Adam** | |unsupported | |
| **Add** |14 |No support for short integers. | |
| **And** |7 | | |
| **ArgMax** |13 | | |
| **ArgMin** |13 | | |
| **ArrayFeatureExtractor** | |unsupported | |
| **Asin** |7 | | |
| **Asinh** |9 | | |
| **Atan** |7 | | |
| **Atanh** |9 | | |
| **AveragePool** |11 | | |
| **BatchNormalization** |15 |Training not supported. | |
| **Bernoulli** | |unsupported | |
| **Binarizer** | |unsupported | |
| **BitShift** | |unsupported | |
| **BitwiseAnd** |18 | | |
| **BitwiseNot** | |unsupported | |
| **BitwiseOr** |18 | | |
| **BitwiseXor** |18 | | |
| **BlackmanWindow** | |unsupported | |
| **Cast** |13 |Cast only between float and double types. | |
| **CastLike** | |unsupported | |
| **CastMap** | |unsupported | |
| **CategoryMapper** | |unsupported | |
| **Ceil** |13 | | |
| **Celu** | |unsupported | |
| **CenterCropPad** | |unsupported | |
| **Clip** |13, 12, 11, 6 |No support for short integers. | |
| **Col2Im** | |unsupported | |
| **Compress** |11 | | |
| **Concat** |13 | | |
| **ConcatFromSequence** | |unsupported | |
| **Constant** |19 | | |
| **ConstantOfShape** |9 | | |
| **Conv** |11 | | |
| **ConvInteger** | |unsupported | |
| **ConvTranspose** |11 |Unknown dimension in spatial dimensions (such as H and W) not supported. | |
| **Cos** |7 | | |
| **Cosh** |9 | | |
| **CumSum** |14 | | |
| **DFT** | |unsupported | |
| **DepthToSpace** |13 | | |
| **DequantizeLinear** |13 |Only support for per-tensor or layer dequantization. Not support for per-axis dequantization. | |
| **Det** | |unsupported | |
| **DictVectorizer** | |unsupported | |
| **Div** |14 |No support for short integers. | |
| **Dropout** |13 |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 | | |
| **Einsum** |12 |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. | |
| **Elu** |6 | | |
| **Equal** |13 | | |
| **Erf** |13 | | |
| **Exp** |13 | | |
| **Expand** |13 | | |
| **EyeLike** | |unsupported | |
| **FeatureVectorizer** | |unsupported | |
| **Flatten** |13 | | |
| **Floor** |13 | | |
| **GRU** |14 | | |
| **Gather** |13 | | |
| **GatherElements** |13 | | |
| **GatherND** |13 | | |
| **Gemm** |13 | | |
| **GlobalAveragePool** |1 | | |
| **GlobalLpPool** | |unsupported | |
| **GlobalMaxPool** |1 | | |
| **Gradient** | |unsupported | |
| **Greater** |13 | | |
| **GreaterOrEqual** |16 | | |
| **GridSample** | |unsupported | |
| **GroupNormalization** | |unsupported | |
| **HammingWindow** | |unsupported | |
| **HannWindow** | |unsupported | |
| **HardSigmoid** |6 | | |
| **HardSwish** | |unsupported | |
| **Hardmax** |13 | | |
| **Identity** |16 |Sequence identity not supported. | |
| **If** |16 |Sequence and Optional outputs are not supported. | |
| **Imputer** | |unsupported | |
| **InstanceNormalization** |6 | | |
| **IsInf** |10 | | |
| **IsNaN** |13 | | |
| **LRN** |13 | | |
| **LSTM** |14 | | |
| **LabelEncoder** | |unsupported | |
| **LayerNormalization** | |unsupported | |
| **LeakyRelu** |16 | | |
| **Less** |13 | | |
| **LessOrEqual** |16 | | |
| **LinearClassifier** | |unsupported | |
| **LinearRegressor** | |unsupported | |
| **Log** |13 | | |
| **LogSoftmax** |13 |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |16 |No support for opset 13 and 16 at this time. | |
| **LpNormalization** | |unsupported | |
| **LpPool** | |unsupported | |
| **MatMul** |13 | | |
| **MatMulInteger** |10 | | |
| **Max** |13 |No support for short floats and unsigned int. | |
| **MaxPool** |12 |Does not support argmax and short ints. Support single output only. | |
| **MaxRoiPool** | |unsupported | |
| **MaxUnpool** | |unsupported | |
| **Mean** |13 | | |
| **MeanVarianceNormalization** | |unsupported | |
| **MelWeightMatrix** | |unsupported | |
| **Min** |13 |Does not support short floats and unsigned numbers. | |
| **Mish** | |unsupported | |
| **Mod** |13 |Support float and double only. | |
| **Momentum** | |unsupported | |
| **Mul** |14 |Does not support short integers. | |
| **Multinomial** | |unsupported | |
| **Neg** |13 | | |
| **NegativeLogLikelihoodLoss** | |unsupported | |
| **NonMaxSuppression** |11 | | |
| **NonZero** |13 | | |
| **Normalizer** | |unsupported | |
| **Not** |1 | | |
| **OneHot** |11 | | |
| **OneHotEncoder** | |unsupported | |
| **Optional** | |unsupported | |
| **OptionalGetElement** | |unsupported | |
| **OptionalHasElement** | |unsupported | |
| **Or** |7 | | |
| **PRelu** |16 | | |
| **Pad** |18, 13, 11, 2 |axes input not supported. | |
| **Pow** |15 |No support for power with integer types. | |
| **QLinearConv** | |unsupported | |
| **QLinearMatMul** | |unsupported | |
| **QuantizeLinear** |13 |Do not support per-axis and i8 quantization. | |
| **RNN** |14 | | |
| **RandomNormal** | |unsupported | |
| **RandomNormalLike** | |unsupported | |
| **RandomUniform** | |unsupported | |
| **RandomUniformLike** | |unsupported | |
| **Range** |11 | | |
| **Reciprocal** |13 | | |
| **ReduceL1** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceL2** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceLogSum** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceLogSumExp** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceMax** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceMean** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceMin** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceProd** |18, 13 |do_not_keep_dim not supported. | |
| **ReduceSum** |13, 11 |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |18, 13 |Default axis and do_not_keep_dim not supported. | |
| **Relu** |14 | | |
| **Reshape** |14 |allowzero not supported. | |
| **Resize** |18, 13, 11, 10 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. | |
| **ReverseSequence** |10 | | |
| **RoiAlign** | |unsupported | |
| **Round** |11 | | |
| **STFT** | |unsupported | |
| **SVMClassifier** | |unsupported | |
| **SVMRegressor** | |unsupported | |
| **Scaler** | |unsupported | |
| **Scan** |16 |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **Scatter** | |unsupported | |
| **ScatterElements** |18 |Does not support duplicate indices. | |
| **ScatterND** |18 |Does not support scatternd add/multiply. | |
| **Selu** |6 | | |
| **SequenceAt** | |unsupported | |
| **SequenceConstruct** | |unsupported | |
| **SequenceEmpty** | |unsupported | |
| **SequenceErase** | |unsupported | |
| **SequenceInsert** |11 |Does not support unranked sequence element. | |
| **SequenceLength** | |unsupported | |
| **SequenceMap** | |unsupported | |
| **Shape** |15 |Does not support start and end attributes. | |
| **Shrink** | |unsupported | |
| **Sigmoid** |13 | | |
| **Sign** |13 | | |
| **Sin** |7 | | |
| **Sinh** |9 | | |
| **Size** |13 | | |
| **Slice** |13 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |13, 11 | | |
| **SoftmaxCrossEntropyLoss** | |unsupported | |
| **Softplus** |1 | | |
| **Softsign** |1 | | |
| **SpaceToDepth** |13 | |Example works, the other is imprecise. To investigate. |
| **Split** |18, 13, 11 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **SplitToSequence** | |unsupported | |
| **Sqrt** |13 | | |
| **Squeeze** |13, 11 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StringNormalizer** | |unsupported | |
| **Sub** |14 |Does not support short integers. | |
| **Sum** |13 | | |
| **Tan** |7 | | |
| **Tanh** |13 | | |
| **TfIdfVectorizer** | |unsupported | |
| **ThresholdedRelu** | |unsupported | |
| **Tile** |13 | | |
| **TopK** |11 | | |
| **Transpose** |13 | | |
| **TreeEnsembleClassifier** | |unsupported | |
| **TreeEnsembleRegressor** | |unsupported | |
| **Trilu** |14 | | |
| **Unique** | |unsupported | |
| **Unsqueeze** |13, 11 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |9, 7 | | |
| **Where** |16 | | |
| **Xor** |7 | | |
| **ZipMap** | |unsupported | |
