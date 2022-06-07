<!--- Automatically generated, do not edit. -->
<!--- python documentOps.py --arch cpu --todo --unsupported --input /workdir/onnx-mlir/test/backend/inference_backend.py -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently support ONNX operations targeting opset 16. Limitations are listed when applicable.

| Op |Opset |Limitations |Todo |
| --- |--- |--- |--- |
| **Abs** |16 | | |
| **Acos** |16 | | |
| **Acosh** |16 | | |
| **Adagrad** |unsupported | | |
| **Adam** |unsupported | | |
| **Add** |16 |No support for short integers. | |
| **And** |16 | | |
| **Argmax** |16 | | |
| **Argmin** |unsupported | | |
| **Asin** |16 | | |
| **Asinh** |16 | | |
| **Atan** |16 | | |
| **Atanh** |16 | | |
| **AveragePool** |16 | | |
| **BatchNormalization** |16 |Training not supported. | |
| **Bernoulli** |unsupported | | |
| **Bitshift** |unsupported | | |
| **Cast** |16 |Cast only between float and double types. | |
| **CastLike** |unsupported | | |
| **Ceil** |16 | | |
| **Celu** |unsupported | | |
| **Clip** |16 |No support for short integers. | |
| **Compress** |16 | | |
| **Concat** |16 | | |
| **Constant** |16 | | |
| **ConstantOfShape** |16 | | |
| **Conv** |16 | | |
| **ConvInteger** |unsupported | | |
| **ConvTranspose** |unsupported | | |
| **Cos** |16 | | |
| **Cosh** |16 | | |
| **CumSum** |16 | | |
| **DepthOfSpace** |16 | | |
| **DequatizeLinear** |unsupported | | |
| **Det** |unsupported | | |
| **Div** |16 |No support for short integers. | |
| **Dropout** |16 |Does not support masked and training. | |
| **DynamicQuantizeLinear** |unsupported | | |
| **EinSum** |unsupported | | |
| **Elu** |16 | | |
| **Equal** |16 | | |
| **Erf** |16 | | |
| **Exp** |16 | | |
| **Expand** |16 | | |
| **Eyelike** |unsupported | | |
| **Flatten** |16 | | |
| **Floor** |16 | | |
| **GRU** |16 |Batchwise test is not supported. | |
| **Gather** |16 | | |
| **GatherElements** |16 | | |
| **GatherND** |16 | | |
| **Gemm** |16 | | |
| **GlobalAveragePool** |16 | | |
| **GlobalMaxPool** |16 | | |
| **Greater** |16 | | |
| **GreaterOrEqual** |16 | | |
| **GridSample** |unsupported | | |
| **HardMax** |16 | | |
| **HardSigmoid** |16 | | |
| **HardSwish** |unsupported | | |
| **Identity** |16 |Sequence identity not supported. | |
| **If** |unsupported | | |
| **InstanceNorm** |16 | | |
| **IsInf** |unsupported | | |
| **IsNan** |unsupported | | |
| **LRN** |16 | | |
| **LSTM** |16 |No support for batchwise examples. | |
| **LeakyRelu** |16 | | |
| **Less** |16 | | |
| **LessOrEqual** |16 | | |
| **Log** |16 | | |
| **LogSoftmax** |16 |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |Opset 11 |No support for opset 13 and 16 at this time. | |
| **Matmul** |16 | | |
| **MatmulInteger** |unsupported | | |
| **Max** |16 |No support for short floats and unsigned int. | |
| **MaxPoolSingleOut** |16 |Does not support argmax and short ints. | |
| **MaxUnpool** |unsupported | | |
| **Mean** |16 | | |
| **MeanVarianceNormalization** |unsupported | | |
| **Min** |16 |Does not support short floats and unsigned numbers. | |
| **Mod** |16 |Support float and double only. | |
| **Momentum** |unsupported | | |
| **Mul** |16 |Does not support short integers. | |
| **Neg** |16 | | |
| **NegativeLogLikelihoodLoss** |unsupported | | |
| **NonMaxSuppression** |16 | | |
| **NonZero** |16 | | |
| **Not** |16 | | |
| **OneHot** |16 | | |
| **OptionalGetElement** |unsupported | | |
| **OptionalHasElement** |unsupported | | |
| **Or** |16 | | |
| **PRelu** |16 | | |
| **Pad** |16 | | |
| **Pow** |16 |No support for power with integer types. | |
| **QLinearConv** |unsupported | | |
| **QLinearMatmul** |unsupported | | |
| **QuantizeLinear** |unsupported | | |
| **RNN** |16 |Batchwise not supported. | |
| **Range** |16 | | |
| **ReciprocalOp** |16 | | |
| **ReduceL1** |16 | | |
| **ReduceL2** |16 | | |
| **ReduceLogSum** |16 | | |
| **ReduceLogSumExp** |16 | | |
| **ReduceMax** |16 | | |
| **ReduceMean** |16 | | |
| **ReduceMin** |16 | | |
| **ReduceProd** |16 | | |
| **ReduceSum** |16 |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |16 | | |
| **Relu** |16 | | |
| **Reshape** |16 | | |
| **Resize** |16 |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. | |
| **Reverse** |Sequence current | | |
| **RoiAlign** |unsupported | | |
| **Round** |16 | | |
| **Scan** |Opset 9 |Does not support dynamic shapes. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **ScatterElements** |16 |Does not support duplicate indices. | |
| **ScatterND** |16 |Does not support scatternd add/multiply. | |
| **Selu** |16 | | |
| **SequenceInsert** |unsupported | | |
| **Shape** |16 | | |
| **Shrink** |unsupported | | |
| **Sigmoid** |16 | | |
| **Sign** |16 | | |
| **Sin** |16 | | |
| **Sinh** |16 | | |
| **Size** |16 | | |
| **Slice** |16 |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |16 | | |
| **SoftmaxCrossEntropyLoss** |unsupported | | |
| **Softplus** |16 | | |
| **Softsign** |16 | | |
| **SpaceToDepth** |unsupported | |Example works, the other is imprecise. To investigate. |
| **Split** |16 |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **Sqrt** |16 | | |
| **Squeeze** |16 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **StrNormalizer** |unsupported | | |
| **Sub** |16 |Does not support short integers. | |
| **Sum** |16 | | |
| **Tan** |16 | | |
| **Tanh** |16 | | |
| **TfdfVectorizer** |unsupported | | |
| **ThresholdRelu** |unsupported | | |
| **Tile** |16 | | |
| **TopK** |16 | | |
| **Transpose** |16 | | |
| **Trilu** |unsupported | | |
| **Unique** |unsupported | | |
| **Unsqueeze** |16 |Does not support static and dynamic shape. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |16 | | |
| **Where** |16 | | |
| **Xor** |16 | | |
