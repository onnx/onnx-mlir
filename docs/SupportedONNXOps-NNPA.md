<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops` -->

# Supported ONNX Operation for Target *NNPA*.

Onnx-mlir currently supports ONNX operations targeting up to opset 18. Limitations are listed when applicable.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* Opset indicates, for each operation, the ONNX opset that (1) last modified that operation and (2) is supported by the current version of onnx-mlir. For example, "Add" was modified in Opset 14 and carries on unmodified to Opset 16. If onnx-mlir supports Opset 14, we thus list "14" as the Opset associated with the "Add" operation.


NNPA has hardware limitations in dimension index size and tensor size, which are described in [NNPALimit.h](../src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h). They are enough large for normal use cases, but if your model exceeds the limitations, CPU is used instead of NNPA.


| Op |Min Opset |Max Opset |Limitations |Notes |
| --- |--- |--- |--- |--- |
| **Abs** | | |unsupported | |
| **Acos** | | |unsupported | |
| **Acosh** | | |unsupported | |
| **Adagrad** | | |unsupported | |
| **Adam** | | |unsupported | |
| **Add** |14 |18 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **And** | | |unsupported | |
| **ArgMax** | | |unsupported | |
| **ArgMin** | | |unsupported | |
| **ArrayFeatureExtractor** | | |unsupported | |
| **Asin** | | |unsupported | |
| **Asinh** | | |unsupported | |
| **Atan** | | |unsupported | |
| **Atanh** | | |unsupported | |
| **AveragePool** |19 |19 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors (N x C x H x W).<br>- `kernel_shape` must be static.<br>- `count_include_pad` must be default value(0).<br>- `ceil_mode` must be default value(0). | |
| **BatchNormalization** |15 |18 |Input and output tensor must be 4D(N x C x H x W). | |
| **Bernoulli** | | |unsupported | |
| **Binarizer** | | |unsupported | |
| **BitShift** | | |unsupported | |
| **BitwiseAnd** | | |unsupported | |
| **BitwiseNot** | | |unsupported | |
| **BitwiseOr** | | |unsupported | |
| **BitwiseXor** | | |unsupported | |
| **BlackmanWindow** | | |unsupported | |
| **Cast** | | |unsupported | |
| **CastLike** | | |unsupported | |
| **CastMap** | | |unsupported | |
| **CategoryMapper** | | |unsupported | |
| **Ceil** | | |unsupported | |
| **Celu** | | |unsupported | |
| **CenterCropPad** | | |unsupported | |
| **Clip** | | |unsupported | |
| **Col2Im** | | |unsupported | |
| **Compress** | | |unsupported | |
| **Concat** | | |unsupported | |
| **ConcatFromSequence** | | |unsupported | |
| **Constant** | | |unsupported | |
| **ConstantOfShape** | | |unsupported | |
| **Conv** |11 |18 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- Dimension in Height and weight must be static.<br>- `group` must be default value(1).<br>- `dilations` must be default value(1).<br>- Input and output tensors must have 4D (N x C x H x W).<br>- `kernel_shape` must be static. | |
| **ConvInteger** | | |unsupported | |
| **ConvTranspose** |11 |18 |- 1D and 3D not supported because Conv1D and Conv3D not supported in zDNN. non-default `dilations` not supported because dilated convolution not supported in zDNN. | |
| **Cos** | | |unsupported | |
| **Cosh** | | |unsupported | |
| **CumSum** | | |unsupported | |
| **DFT** | | |unsupported | |
| **DepthToSpace** | | |unsupported | |
| **DequantizeLinear** | | |unsupported | |
| **Det** | | |unsupported | |
| **DictVectorizer** | | |unsupported | |
| **Div** |14 |18 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Dropout** | | |unsupported | |
| **DynamicQuantizeLinear** | | |unsupported | |
| **Einsum** | | |unsupported | |
| **Elu** | | |unsupported | |
| **Equal** | | |unsupported | |
| **Erf** | | |unsupported | |
| **Exp** |13 |18 |Input tensor must have 4 dimensions. | |
| **Expand** | | |unsupported | |
| **EyeLike** | | |unsupported | |
| **FeatureVectorizer** | | |unsupported | |
| **Flatten** | | |unsupported | |
| **Floor** | | |unsupported | |
| **GRU** |14 |18 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- If `B` and `initial_h` are given, they must have static dimensions.<br>- `sequence_lens` is not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `linear_before_reset` must be 1.<br>- `layout` is not supported. | |
| **Gather** | | |unsupported | |
| **GatherElements** | | |unsupported | |
| **GatherND** | | |unsupported | |
| **Gemm** |13 |18 |- `alpha` and `beta` must be default value(1).<br>- Rank of `C` must be 1 or 2. If the rank is 1, the dimension of `C` must be the same with the seconde dimension of `B`. | |
| **GlobalAveragePool** |1 |18 |- Input shape must be 4D tensor(NCHW).<br>- Dimensions in `H` and `W` must be static. | |
| **GlobalLpPool** | | |unsupported | |
| **GlobalMaxPool** | | |unsupported | |
| **Gradient** | | |unsupported | |
| **Greater** | | |unsupported | |
| **GreaterOrEqual** | | |unsupported | |
| **GridSample** | | |unsupported | |
| **GroupNormalization** | | |unsupported | |
| **HammingWindow** | | |unsupported | |
| **HannWindow** | | |unsupported | |
| **HardSigmoid** | | |unsupported | |
| **HardSwish** | | |unsupported | |
| **Hardmax** | | |unsupported | |
| **Identity** | | |unsupported | |
| **If** | | |unsupported | |
| **Imputer** | | |unsupported | |
| **InstanceNormalization** | | |unsupported | |
| **IsInf** | | |unsupported | |
| **IsNaN** | | |unsupported | |
| **LRN** | | |unsupported | |
| **LSTM** |14 |18 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- `B` and `initial_h` have static dimensions if given. `B`'s direction dim must be 1 or 2.<br>- `P`(peepholes), `activation_alpha`, and `activation_beta` are not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `input_forget` must be default value(0).<br>- `layout` is not supported. | |
| **LabelEncoder** | | |unsupported | |
| **LayerNormalization** | | |unsupported | |
| **LeakyRelu** |16 |18 |The operations immediately before and after the LeakyRelu operation must be executed on the NNPA. Otherwise, LeakyRelu is executed on the CPU. This limitation is set to avoid performance degradation. | |
| **Less** | | |unsupported | |
| **LessOrEqual** | | |unsupported | |
| **LinearClassifier** | | |unsupported | |
| **LinearRegressor** | | |unsupported | |
| **Log** |13 |18 |Input tensor must have 4 dimensions. | |
| **LogSoftmax** |13 |18 | | |
| **Loop** | | |unsupported | |
| **LpNormalization** | | |unsupported | |
| **LpPool** | | |unsupported | |
| **MatMul** |13 |18 |Ranks of input tensors must be (Rank of A, Rank of B) = (M, N), where M >= 2 and N >= 2. | |
| **MatMulInteger** | | |unsupported | |
| **Max** |13 |18 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **MaxPool** |12 |18 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors(N x C x H x W).<br>- `kernel_shape` must be static.<br>- `ceil_mode` must be default value(0).<br>- `dilations` must be default value(1). | |
| **MaxRoiPool** | | |unsupported | |
| **MaxUnpool** | | |unsupported | |
| **Mean** | | |unsupported | |
| **MeanVarianceNormalization** | | |unsupported | |
| **MelWeightMatrix** | | |unsupported | |
| **Min** |13 |18 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Mish** | | |unsupported | |
| **Mod** | | |unsupported | |
| **Momentum** | | |unsupported | |
| **Mul** |14 |18 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Multinomial** | | |unsupported | |
| **Neg** | | |unsupported | |
| **NegativeLogLikelihoodLoss** | | |unsupported | |
| **NonMaxSuppression** | | |unsupported | |
| **NonZero** | | |unsupported | |
| **Normalizer** | | |unsupported | |
| **Not** | | |unsupported | |
| **OneHot** | | |unsupported | |
| **OneHotEncoder** | | |unsupported | |
| **Optional** | | |unsupported | |
| **OptionalGetElement** | | |unsupported | |
| **OptionalHasElement** | | |unsupported | |
| **Or** | | |unsupported | |
| **PRelu** | | |unsupported | |
| **Pad** | | |unsupported | |
| **Pow** |15 |18 |- Exponent should be a scalar integer and less or equal to 64. | |
| **QLinearConv** | | |unsupported | |
| **QLinearMatMul** | | |unsupported | |
| **QuantizeLinear** | | |unsupported | |
| **RNN** | | |unsupported | |
| **RandomNormal** | | |unsupported | |
| **RandomNormalLike** | | |unsupported | |
| **RandomUniform** | | |unsupported | |
| **RandomUniformLike** | | |unsupported | |
| **Range** | | |unsupported | |
| **Reciprocal** | | |unsupported | |
| **ReduceL1** | | |unsupported | |
| **ReduceL2** | | |unsupported | |
| **ReduceLogSum** | | |unsupported | |
| **ReduceLogSumExp** | | |unsupported | |
| **ReduceMax** | | |unsupported | |
| **ReduceMean** |18 |18 |- `keepdims` must be 1.<br>- Input tensor must be 4D tensors and `axis` must be [2, 3]. | |
| **ReduceMin** | | |unsupported | |
| **ReduceProd** | | |unsupported | |
| **ReduceSum** | | |unsupported | |
| **ReduceSumSquare** | | |unsupported | |
| **Relu** |14 |18 |Input tensor must be less than or equal to 4 dimensions. | |
| **Reshape** | | |unsupported | |
| **Resize** | | |unsupported | |
| **ReverseSequence** | | |unsupported | |
| **RoiAlign** | | |unsupported | |
| **Round** | | |unsupported | |
| **STFT** | | |unsupported | |
| **SVMClassifier** | | |unsupported | |
| **SVMRegressor** | | |unsupported | |
| **Scaler** | | |unsupported | |
| **Scan** | | |unsupported | |
| **Scatter** | | |unsupported | |
| **ScatterElements** | | |unsupported | |
| **ScatterND** | | |unsupported | |
| **Selu** | | |unsupported | |
| **SequenceAt** | | |unsupported | |
| **SequenceConstruct** | | |unsupported | |
| **SequenceEmpty** | | |unsupported | |
| **SequenceErase** | | |unsupported | |
| **SequenceInsert** | | |unsupported | |
| **SequenceLength** | | |unsupported | |
| **SequenceMap** | | |unsupported | |
| **Shape** | | |unsupported | |
| **Shrink** | | |unsupported | |
| **Sigmoid** |13 |18 |Input tensor must be less than or equal to 4 dimensions. | |
| **Sign** | | |unsupported | |
| **Sin** | | |unsupported | |
| **Sinh** | | |unsupported | |
| **Size** | | |unsupported | |
| **Slice** | | |unsupported | |
| **Softmax** |13 |18 |- `axis` must be the last dimension, i.e. `rank - 1` or -1. | |
| **SoftmaxCrossEntropyLoss** | | |unsupported | |
| **Softplus** | | |unsupported | |
| **Softsign** | | |unsupported | |
| **SpaceToDepth** | | |unsupported | |
| **Split** | | |unsupported | |
| **SplitToSequence** | | |unsupported | |
| **Sqrt** | | |unsupported | |
| **Squeeze** | | |unsupported | |
| **StringNormalizer** | | |unsupported | |
| **Sub** |14 |18 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Sum** |13 |18 |- All inputs must have the same static shape (Broadcasting not supported.)<br>- Single input not supported. | |
| **Tan** | | |unsupported | |
| **Tanh** |13 |18 |Input tensor must be less than or equal to 4 dimensions. | |
| **TfIdfVectorizer** | | |unsupported | |
| **ThresholdedRelu** | | |unsupported | |
| **Tile** | | |unsupported | |
| **TopK** | | |unsupported | |
| **Transpose** | | |unsupported | |
| **TreeEnsembleClassifier** | | |unsupported | |
| **TreeEnsembleRegressor** | | |unsupported | |
| **Trilu** | | |unsupported | |
| **Unique** | | |unsupported | |
| **Unsqueeze** | | |unsupported | |
| **Upsample** | | |unsupported | |
| **Where** | | |unsupported | |
| **Xor** | | |unsupported | |
| **ZipMap** | | |unsupported | |
