# Please keep up to date.

- As you start working on a new operation, move it from the lower table to the top table,
- write your name under 'person working on it,'
- once you have pushed the functionality in the master branch, mark it as complete.

# ONNX to KRNL work.

Update as you push code to the master branch.

| ONNX Operation | Person working on it | ONNX Def | Shape Inference | ONNX to KRNL | Optimized |Extended Functionality |
|----------------|-----------------|----|---|---|----|-------|
| Abs            |                 | v  | v | v |    |       |
| Add            |                 | v  | v | v |    | M     |
| And            |                 | v  | v | v |    | M     |
| AveragePool    |                 | v  | v | v |    | P     |
| BatchNorm      |                 | v  | v | v |    |       |
| Constant       |                 | v  | v | v |    |       |
| Conv           |                 | v  | v | v |    | nP    |
| Concat         |                 | v  | v | v |    | V     |
| Cos            |                 | v  | v | v |    |       |
| Cosh           |                 | v  | v | v |    |       |
| Div            |                 | v  | v | v |    | M     |
| Dropout        |                 | v  | v |   |    |       |
| Elu            |                 | v  | v | v |    |       |
| Erf            |                 | v  | v |   |    |       |
| Exp            |                 | v  | v | v |    |       |
| Expand         |                 | v  | v |   |    |       |
| Gemm           |                 | v  | v | v |    | U     |
| GlobalAvgPool  |                 | v  | v |   |    |       |
| GlobalLpPool   |                 | v  | v |   |    |       |
| GlobalMaxPool  |                 | v  | v |   |    |       |
| HardSigmoid    |                 | v  | v | v |    |       |
| Identity       |                 | v  | v | v |    |       |
| LeakyRelu      |                 | v  | v | v |    |       |
| Log            |                 | v  | v | v |    |       |
| MatMul         |                 | v  | v | v |    | M     |
| Max            |                 | v  | v | v |    | M V   |
| MaxPool        |                 | v  | v | v |    | P     |
| Min            |                 | v  | v | v |    | M V   |
| Mul            |                 | v  | v | v |    | M     |
| Or             |                 | v  | v | v |    | M     |
| Pad            |                 | v  | V | v |    | const only |
| Pow            |                 | v  | v |   |    | M     |
| Reciprocal     |                 | v  | v | v |    |       |
| ReduceMax      |                 | v  | v | v |    |       |
| ReduceL1       |                 | v  |   |   |    | be decomposed into ReduceSum and Abs |
| ReduceL2       |                 | v  |   |   |    | be decomposed into ReduceSumSquare and Sqrt |
| ReduceLogSum   |                 | v  |   |   |    | be decomposed into ReduceSum and Log |
| ReduceLogSumExp|                 | v  |   |   |    | be decomposed into ReduceLogSum and Exp |
| ReduceMean     |                 | v  | v |   |    |       |
| ReduceMin      |                 | v  | v | v |    |       |
| ReduceProd     |                 | v  | v | v |    |       |
| ReduceSum      |                 | v  | v | v |    |       |
| ReduceSumSquare|                 | v  |   |   |    | be decomposed into ReduceSum and Mul |
| Relu           |                 | v  | v | v |    |       |
| Reshape        |                 | v  | v | v |    | const only |
| Selu           |                 | v  | v | v |    |       |
| Sigmoid        |                 | v  | v | v |    |       |
| Sign           |                 | v  | v | v |    |       |
| Sinh           |                 | v  | v | v |    |       |
| Softmax        |                 | v  | v | v |    |       |
| Softplus       |                 | v  | v | v |    |       |
| Softsign       |                 | v  | v | v |    |       |
| Sqrt           |                 | v  | v | v |    |       |
| Sub            |                 | v  | v | v |    | M     |
| Sum            |                 | v  | v | v |    | M V   |
| Tanh           |                 | v  | v | v |    |       |
| Transpose      |                 | v  | v | v |    |       |
| Unsqueeze      |                 | v  | v | v |    |       |
| Xor            |                 | v  | v | v |    | M     |

### Extended functionality info (noXXX: additional work is needed). 

* M / noM: multi-broadcast imlemented / not implemented
* U / noU: unidir-broadcast implemented / not implemented
* P / noP: padding implemented / not implemented
* V / noV: variadic input implemented / non implemented

# Operation not supported yet.

Please pick new operations to work on, and make sure you utilize current code to minimize additional work needed to support new functionality.

Typical sequence of work.

1. Ensure the operation is properly parsed from ONNX and MLIR input.
2. Perform shape inference.
3. Lower to KRNL dialect.

And add literal tests at each step, and end to end tests once completed.

| ONNX Operation | Person working on it | ONNX Def | Shape Inference | ONNX to KRNL | Basic Functionality | Extended functionality |
|----------------|-----------------|----|---|---|----|-------|
| ACos           |                 |    |   |   |    |       |
| ACosh          |                 |    |   |   |    |       |
| ArgMax         |                 |    |   |   |    |       |
| ArgMin         |                 |    |   |   |    |       |
| ASin           |                 |    |   |   |    |       |
| ASinh          |                 |    |   |   |    |       |
| ATan           |                 |    |   |   |    |       |
| ATanh          |                 |    |   |   |    |       |
| BitShift       |                 |    |   |   |    | M     |
| Ceil           |                 |    |   |   |    |       |
| Celu           |                 |    |   |   |    |       |
| Clip           |                 |    |   |   |    |       |
| Compress       |                 |    |   |   |    |       |
| ConcatFromSeq  |                 |    |   |   |    |       |
| ConstOfShape   |                 |    |   |   |    |       |
| ConvInt        |                 |    |   |   |    | P     |
| ConvTrans      |                 |    |   |   |    | P     |
| CumSum         |                 |    |   |   |    |       |
| DepthToSpace   |                 |    |   |   |    |       |
| DequantizeLin  |                 |    |   |   |    |       |
| Det            |                 |    |   |   |    |       |
| DynQuantizeLin |                 |    |   |   |    |       |
| Einsum         |                 |    |   |   |    | V     |
| Equal          |                 |    |   |   |    | M     |
| EyeLike        |                 |    |   |   |    |       |
| Flatten        |                 |    |   |   |    |       |
| Floor          |                 |    |   |   |    |       |
| GRU            |                 |    |   |   |    |       |
| GatherElements |                 |    |   |   |    |       |
| GatherND       |                 |    |   |   |    |       |
| Greater        |                 |    |   |   |    | M     |
| GreaterOrEq    |                 |    |   |   |    | M     |
| HardMax        |                 |    |   |   |    |       |
| If             |                 |    |   |   |    | V     |
| InstanceNorm   |                 |    |   |   |    |       |
| Inverse        |                 |    |   |   |    |       |
| IsInf          |                 |    |   |   |    |       |
| IsNaN          |                 |    |   |   |    |       |
| Less           |                 |    |   |   |    | M     |
| LessOrEq       |                 |    |   |   |    | M     |
| LSTM           |                 |    |   |   |    |       |
| LSTM           |                 |    |   |   |    |       |
| LSTM           |                 |    |   |   |    |       |
| LogSoftmax     |                 |    |   |   |    |       |
| Loop           |                 |    |   |   |    | V     |
| LpNorm         |                 |    |   |   |    |       |
| LpPool         |                 |    |   |   |    |       |
| MatMulInteger  |                 |    |   |   |    | U     |
| MaxRoiPool     |                 |    |   |   |    |       |
| MaxUnpool      |                 |    |   |   |    |       |
| Mean           |                 |    |   |   |    | M V   |
| MeanSquareDist |                 |    |   |   |    | M     |
| MeanVarNorm    |                 |    |   |   |    |       |
| Mod            |                 |    |   |   |    | M     |
| Multinomial    |                 |    |   |   |    |       |
| Neg            |                 |    |   |   |    |       |
| NegLogLikeLoss |                 |    |   |   |    |       |
| NonMaxSupress  |                 |    |   |   |    |       |
| NonZero        |                 |    |   |   |    |       |
| Not            |                 |    |   |   |    |       |
| OneHot         |                 |    |   |   |    |       |
| PRelu          |                 |    |   |   |    | U     |
| QLinearConv    |                 |    |   |   |    | P     |
| QLinearMatMul  |                 |    |   |   |    | M     |
| QuantizeLinear |                 |    |   |   |    |       |
| RNN            |                 |    |   |   |    |       |
| RandomNormal   |                 |    |   |   |    |       |
| RandNormalLike |                 |    |   |   |    |       |
| RandUniform    |                 |    |   |   |    |       |
| RandUniformLike|                 |    |   |   |    |       |
| Range          |                 |    |   |   |    |       |
| Resize         |                 |    |   |   |    |       |
| ReverseSequence|                 |    |   |   |    |       |
| RoiAlign       |                 |    |   |   |    |       |
| Round          |                 |    |   |   |    |       |
| Scan           |                 |    |   |   |    | V     |
| Scatter        |                 |    |   |   |    | deprecated for Scatter Element |
| ScatterElem    |                 |    |   |   |    |       |
| ScatterND      |                 |    |   |   |    |       |
| SequenceAt     |                 |    |   |   |    |       |
| SeqConstruct   |                 |    |   |   |    | V     |
| SequenceErase  |                 |    |   |   |    |       |
| SequenceInsert |                 |    |   |   |    |       |
| SequenceLength |                 |    |   |   |    |       |
| Shape          |                 |    |   |   |    |       |
| Shrink         |                 |    |   |   |    |       |
| Slice          |                 |    |   |   |    |       |
| SoftMaxCross   |                 |    |   |   |    |       |
| SpaceToDepth   |                 |    |   |   |    |       |
| Split          |                 |    |   |   |    | V     |
| SplitToSequence|                 |    |   |   |    |       |
| Squeeze        |                 |    |   |   |    |       |
| StringNorm     |                 |    |   |   |    |       |
| Tan            |                 |    |   |   |    |       |
| TfldfVect      |                 |    |   |   |    |       |
| ThresholdRelu  |                 |    |   |   |    |       |
| Tile           |                 |    |   |   |    |       |
| TopK           |                 |    |   |   |    |       |
| UnfoldToDepth  |                 |    |   |   |    |       |
| Unique         |                 |    |   |   |    |       |
| Upsample       |                 |    |   |   |    | deprecated |
| Where          |                 |    |   |   |    | M     |
