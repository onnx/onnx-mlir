<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops_cpu' -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 19. Limitations are listed when applicable. This documentation highlights the minimum and maximum opset versions that are fully supported by onnx-mlir and not the version changes.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* **Supported Opsets** indicates the lowest and highest opset a model may have for onnx-mlir to support compiling a model with the operator. 
   * A * indicates onnx-mlir is compatible with the latest version of that operator available as of opset 19.


| Op |Supported Opsets (inclusive) |Limitations |Notes |
| --- |--- |--- |--- |
| **Abs** |6 - * | | |
| **Acos** |7 - * | | |
| **Acosh** |9 - * | | |
| **Adagrad** |none | | | |
| **Adam** |none | | | |
| **Add** |6 - * |No support for short integers. | |
| **And** |7 - * | | |
| **ArgMax** |6 - * | | |
| **ArgMin** |13 - * | | |
| **ArrayFeatureExtractor** |none | | | |
| **Asin** |7 - * | | |
| **Asinh** |9 - * | | |
| **Atan** |7 - * | | |
| **Atanh** |9 - * | | |
| **AveragePool** |6 - * | | |
| **BatchNormalization** |6 - * |Training not supported. | |
| **Bernoulli** |none | | | |
| **Binarizer** |none | | | |
| **BitShift** |none | | | |
| **BitwiseAnd** |18 - * | | |
| **BitwiseNot** |none | | | |
| **BitwiseOr** |18 - * | | |
| **BitwiseXor** |18 - * | | |
| **BlackmanWindow** |none | | | |
| **Cast** |6 - 18 |Cast only between float and double types. Only ppc64le and MacOS platforms support float16. | |
| **CastLike** |none | | | |
| **CastMap** |none | | | |
| **CategoryMapper** |none | | | |
| **Ceil** |6 - * | | |
| **Celu** |none | | | |
| **CenterCropPad** |none | | | |
| **Clip** |6 - * |No support for short integers. | |
| **Col2Im** |none | | | |
| **Compress** |9 - * | | |
| **Concat** |6 - * | | |
| **ConcatFromSequence** |none | | | |
| **Constant** |6 - 18 | | |
| **ConstantOfShape** |9 - * | | |
| **Conv** |6 - * | | |
| **ConvInteger** |none | | | |
| **ConvTranspose** |6 - * |Unknown dimension in spatial dimensions (such as H and W) not supported. | |
| **Cos** |7 - * | | |
| **Cosh** |9 - * | | |
| **CumSum** |11 - * | | |
| **DFT** |none | | | |
| **DeformConv** |none | | | |
| **DepthToSpace** |13 - * | | |
| **DequantizeLinear** |10 - * |Only support for per-tensor or layer dequantization. No support for per-axis dequantization. | |
| **Det** |none | | | |
| **DictVectorizer** |none | | | |
| **Div** |6 - * |No support for short integers. | |
| **Dropout** |6 - * |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 - * | | |
| **Einsum** |12 - * |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. | |
| **Elu** |6 - * | | |
| **Equal** |7 - 18 | | |
| **Erf** |9 - * | | |
| **Exp** |6 - * | | |
| **Expand** |8 - * | | |
| **EyeLike** |none | | | |
| **FeatureVectorizer** |none | | | |
| **Flatten** |6 - * | | |
| **Floor** |6 - * | | |
| **GRU** |7 - * | | |
| **Gather** |6 - * | | |
| **GatherElements** |11 - * | | |
| **GatherND** |11 - * | | |
| **Gemm** |6 - * | | |
| **GlobalAveragePool** |6 - * | | |
| **GlobalLpPool** |none | | | |
| **GlobalMaxPool** |6 - * | | |
| **Gradient** |none | | | |
| **Greater** |7 - * | | |
| **GreaterOrEqual** |12 - * | | |
| **GridSample** |none | | | |
| **GroupNormalization** |none | | | |
| **HammingWindow** |none | | | |
| **HannWindow** |none | | | |
| **HardSigmoid** |6 - * | | |
| **HardSwish** |none | | | |
| **Hardmax** |6 - * | | |
| **Identity** |16 - * |Sequence identity not supported. | |
| **If** |16 - * |Sequence and Optional outputs are not supported. | |
| **Imputer** |none | | | |
| **InstanceNormalization** |6 - * | | |
| **IsInf** |10 - * | | |
| **IsNaN** |9 - * | | |
| **LRN** |6 - * | | |
| **LSTM** |7 - * | | |
| **LabelEncoder** |none | | | |
| **LayerNormalization** |17 - * | | |
| **LeakyRelu** |6 - * | | |
| **Less** |7 - * | | |
| **LessOrEqual** |12 - * | | |
| **LinearClassifier** |none | | | |
| **LinearRegressor** |none | | | |
| **Log** |6 - * | | |
| **LogSoftmax** |13 - * |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |6 - * | | |
| **LpNormalization** |none | | | |
| **LpPool** |none | | | |
| **MatMul** |6 - * | | |
| **MatMulInteger** |10 - * | | |
| **Max** |6 - * |No support for unsigned int. Only ppc64le and MacOS platforms support float16. | |
| **MaxPool** |6 - * |Does not support argmax and short ints. Support single output only. | |
| **MaxRoiPool** |none | | | |
| **MaxUnpool** |none | | | |
| **Mean** |6 - * | | |
| **MeanVarianceNormalization** |none | | | |
| **MelWeightMatrix** |none | | | |
| **Min** |6 - * |Does not support unsigned numbers. Only ppc64le and MacOS platforms support float16. | |
| **Mish** |none | | | |
| **Mod** |10 - * |Support float and double only. Only ppc64le and MacOS platforms support float16. | |
| **Momentum** |none | | | |
| **Mul** |6 - * |Does not support short integers. | |
| **Multinomial** |none | | | |
| **Neg** |6 - * | | |
| **NegativeLogLikelihoodLoss** |none | | | |
| **NonMaxSuppression** |10 - * | | |
| **NonZero** |9 - * | | |
| **Normalizer** |none | | | |
| **Not** |6 - * | | |
| **OneHot** |9 - * | | |
| **OneHotEncoder** |none | | | |
| **Optional** |none | | | |
| **OptionalGetElement** |none | | | |
| **OptionalHasElement** |none | | | |
| **Or** |7 - * | | |
| **PRelu** |6 - * | | |
| **Pad** |6 - 18 |axes input not supported. | |
| **Pow** |7 - * |No support for power with integer types. | |
| **QLinearConv** |none | | | |
| **QLinearMatMul** |none | | | |
| **QuantizeLinear** |10 - 18 |Do not support per-axis and i8 quantization. | |
| **RNN** |7 - * | | |
| **RandomNormal** |none | | | |
| **RandomNormalLike** |none | | | |
| **RandomUniform** |none | | | |
| **RandomUniformLike** |none | | | |
| **Range** |11 - * | | |
| **Reciprocal** |6 - * | | |
| **ReduceL1** |13 - * |do_not_keep_dim not supported. | |
| **ReduceL2** |13 - * |do_not_keep_dim not supported. | |
| **ReduceLogSum** |13 - * |do_not_keep_dim not supported. | |
| **ReduceLogSumExp** |13 - * |do_not_keep_dim not supported. | |
| **ReduceMax** |6 - * |do_not_keep_dim not supported. | |
| **ReduceMean** |6 - * |do_not_keep_dim not supported. | |
| **ReduceMin** |6 - * |do_not_keep_dim not supported. | |
| **ReduceProd** |13 - * |do_not_keep_dim not supported. | |
| **ReduceSum** |6 - * |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |13 - * |Default axis and do_not_keep_dim not supported. | |
| **Relu** |6 - * | | |
| **Reshape** |6 - * |allowzero not supported. | |
| **Resize** |10 - 18 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. | |
| **ReverseSequence** |10 - * | | |
| **RoiAlign** |none | | | |
| **Round** |11 - * | | |
| **STFT** |none | | | |
| **SVMClassifier** |none | | | |
| **SVMRegressor** |none | | | |
| **Scaler** |none | | | |
| **Scan** |8 - * |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **Scatter** |none | | | |
| **ScatterElements** |11 - * |Does not support duplicate indices. | |
| **ScatterND** |11 - * |Does not support scatternd add/multiply. | |
| **Selu** |6 - * | | |
| **SequenceAt** |none | | | |
| **SequenceConstruct** |none | | | |
| **SequenceEmpty** |none | | | |
| **SequenceErase** |none | | | |
| **SequenceInsert** |11 - * |Does not support unranked sequence element. | |
| **SequenceLength** |none | | | |
| **SequenceMap** |none | | | |
| **Shape** |15 - * |Does not support start and end attributes. | |
| **Shrink** |none | | | |
| **Sigmoid** |6 - * | | |
| **Sign** |9 - * | | |
| **Sin** |7 - * | | |
| **Sinh** |9 - * | | |
| **Size** |13 - * | | |
| **Slice** |13 - 18 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |6 - * | | |
| **SoftmaxCrossEntropyLoss** |none | | | |
| **Softplus** |6 - * | | |
| **Softsign** |6 - * | | |
| **SpaceToDepth** |13 - * | |Example works, the other is imprecise. To investigate. |
| **Split** |6 - 18 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **SplitToSequence** |none | | | |
| **Sqrt** |6 - * | | |
| **Squeeze** |6 - * |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StringNormalizer** |none | | | |
| **Sub** |6 - * |Does not support short integers. | |
| **Sum** |6 - * | | |
| **Tan** |7 - * | | |
| **Tanh** |6 - * | | |
| **TfIdfVectorizer** |none | | | |
| **ThresholdedRelu** |none | | | |
| **Tile** |6 - * | | |
| **TopK** |10 - * | | |
| **Transpose** |6 - * | | |
| **TreeEnsembleClassifier** |none | | | |
| **TreeEnsembleRegressor** |none | | | |
| **Trilu** |14 - * | | |
| **Unique** |11 - * | | |
| **Unsqueeze** |6 - * |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |7 - * | | |
| **Where** |9 - * | | |
| **Xor** |7 - * | | |
| **ZipMap** |none | | | |
