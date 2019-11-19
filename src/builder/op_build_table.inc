    if (OpName == "Abs") {
       OneOut(Abs, 1, 1);
    }else if (OpName == "Acos") {
       OneOut(Acos, 1, 1);
    }else if (OpName == "Acosh") {
       OneOut(Acosh, 1, 1);
    }else if (OpName == "Add") {
       OneOut(Add, 2, 1);
    }else if (OpName == "And") {
       OneOut(And, 2, 1);
    }else if (OpName == "ArgMax") {
       OneOut(ArgMax, 1, 1);
    }else if (OpName == "ArgMin") {
       OneOut(ArgMin, 1, 1);
    }else if (OpName == "Asin") {
       OneOut(Asin, 1, 1);
    }else if (OpName == "Asinh") {
       OneOut(Asinh, 1, 1);
    }else if (OpName == "Atan") {
       OneOut(Atan, 1, 1);
    }else if (OpName == "Atanh") {
       OneOut(Atanh, 1, 1);
    }else if (OpName == "AveragePool") {
       OneOut(AveragePool, 1, 1);
    }else if (OpName == "BatchNormalization") {
       MultipleOuts(BatchNormalization, 5, 5);
    }else if (OpName == "BitShift") {
       OneOut(BitShift, 2, 1);
    }else if (OpName == "Cast") {
       OneOut(Cast, 1, 1);
    }else if (OpName == "Ceil") {
       OneOut(Ceil, 1, 1);
    }else if (OpName == "Clip") {
       OneOut(Clip, 3, 1);
    }else if (OpName == "Compress") {
       OneOut(Compress, 2, 1);
    }else if (OpName == "Concat") {
       OneOut(Concat, 1, 1);
    }else if (OpName == "ConcatFromSequence") {
       OneOut(ConcatFromSequence, 1, 1);
    }else if (OpName == "Constant") {
       OneOut(Constant, 0, 1);
    }else if (OpName == "ConstantOfShape") {
       OneOut(ConstantOfShape, 1, 1);
    }else if (OpName == "Conv") {
       OneOut(Conv, 3, 1);
    }else if (OpName == "ConvInteger") {
       OneOut(ConvInteger, 4, 1);
    }else if (OpName == "ConvTranspose") {
       OneOut(ConvTranspose, 3, 1);
    }else if (OpName == "Cos") {
       OneOut(Cos, 1, 1);
    }else if (OpName == "Cosh") {
       OneOut(Cosh, 1, 1);
    }else if (OpName == "CumSum") {
       OneOut(CumSum, 2, 1);
    }else if (OpName == "DepthToSpace") {
       OneOut(DepthToSpace, 1, 1);
    }else if (OpName == "DequantizeLinear") {
       OneOut(DequantizeLinear, 3, 1);
    }else if (OpName == "Det") {
       OneOut(Det, 1, 1);
    }else if (OpName == "Div") {
       OneOut(Div, 2, 1);
    }else if (OpName == "Dropout") {
       MultipleOuts(Dropout, 1, 2);
    }else if (OpName == "DynamicQuantizeLinear") {
       MultipleOuts(DynamicQuantizeLinear, 1, 3);
    }else if (OpName == "Elu") {
       OneOut(Elu, 1, 1);
    }else if (OpName == "Equal") {
       OneOut(Equal, 2, 1);
    }else if (OpName == "Erf") {
       OneOut(Erf, 1, 1);
    }else if (OpName == "Exp") {
       OneOut(Exp, 1, 1);
    }else if (OpName == "Expand") {
       OneOut(Expand, 2, 1);
    }else if (OpName == "EyeLike") {
       OneOut(EyeLike, 1, 1);
    }else if (OpName == "Flatten") {
       OneOut(Flatten, 1, 1);
    }else if (OpName == "Floor") {
       OneOut(Floor, 1, 1);
    }else if (OpName == "GRU") {
       MultipleOuts(GRU, 6, 2);
    }else if (OpName == "Gather") {
       OneOut(Gather, 2, 1);
    }else if (OpName == "GatherElements") {
       OneOut(GatherElements, 2, 1);
    }else if (OpName == "GatherND") {
       OneOut(GatherND, 2, 1);
    }else if (OpName == "Gemm") {
       OneOut(Gemm, 3, 1);
    }else if (OpName == "GlobalAveragePool") {
       OneOut(GlobalAveragePool, 1, 1);
    }else if (OpName == "GlobalLpPool") {
       OneOut(GlobalLpPool, 1, 1);
    }else if (OpName == "GlobalMaxPool") {
       OneOut(GlobalMaxPool, 1, 1);
    }else if (OpName == "Greater") {
       OneOut(Greater, 2, 1);
    }else if (OpName == "HardSigmoid") {
       OneOut(HardSigmoid, 1, 1);
    }else if (OpName == "Hardmax") {
       OneOut(Hardmax, 1, 1);
    }else if (OpName == "Identity") {
       OneOut(Identity, 1, 1);
    }else if (OpName == "If") {
       OneOut(If, 1, 1);
    }else if (OpName == "InstanceNormalization") {
       OneOut(InstanceNormalization, 3, 1);
    }else if (OpName == "IsInf") {
       OneOut(IsInf, 1, 1);
    }else if (OpName == "IsNaN") {
       OneOut(IsNaN, 1, 1);
    }else if (OpName == "LRN") {
       OneOut(LRN, 1, 1);
    }else if (OpName == "LSTM") {
       MultipleOuts(LSTM, 8, 3);
    }else if (OpName == "LeakyRelu") {
       OneOut(LeakyRelu, 1, 1);
    }else if (OpName == "Less") {
       OneOut(Less, 2, 1);
    }else if (OpName == "Log") {
       OneOut(Log, 1, 1);
    }else if (OpName == "LogSoftmax") {
       OneOut(LogSoftmax, 1, 1);
    }else if (OpName == "Loop") {
       OneOut(Loop, 3, 1);
    }else if (OpName == "LpNormalization") {
       OneOut(LpNormalization, 1, 1);
    }else if (OpName == "LpPool") {
       OneOut(LpPool, 1, 1);
    }else if (OpName == "MatMul") {
       OneOut(MatMul, 2, 1);
    }else if (OpName == "MatMulInteger") {
       OneOut(MatMulInteger, 4, 1);
    }else if (OpName == "Max") {
       OneOut(Max, 1, 1);
    }else if (OpName == "MaxPool") {
       MultipleOuts(MaxPool, 1, 2);
    }else if (OpName == "MaxRoiPool") {
       OneOut(MaxRoiPool, 2, 1);
    }else if (OpName == "MaxUnpool") {
       OneOut(MaxUnpool, 3, 1);
    }else if (OpName == "Mean") {
       OneOut(Mean, 1, 1);
    }else if (OpName == "MeanVarianceNormalization") {
       OneOut(MeanVarianceNormalization, 1, 1);
    }else if (OpName == "Min") {
       OneOut(Min, 1, 1);
    }else if (OpName == "Mod") {
       OneOut(Mod, 2, 1);
    }else if (OpName == "Mul") {
       OneOut(Mul, 2, 1);
    }else if (OpName == "Multinomial") {
       OneOut(Multinomial, 1, 1);
    }else if (OpName == "Neg") {
       OneOut(Neg, 1, 1);
    }else if (OpName == "NonMaxSuppression") {
       OneOut(NonMaxSuppression, 5, 1);
    }else if (OpName == "NonZero") {
       OneOut(NonZero, 1, 1);
    }else if (OpName == "Not") {
       OneOut(Not, 1, 1);
    }else if (OpName == "OneHot") {
       OneOut(OneHot, 3, 1);
    }else if (OpName == "Or") {
       OneOut(Or, 2, 1);
    }else if (OpName == "PRelu") {
       OneOut(PRelu, 2, 1);
    }else if (OpName == "Pad") {
       OneOut(Pad, 3, 1);
    }else if (OpName == "Pow") {
       OneOut(Pow, 2, 1);
    }else if (OpName == "QLinearConv") {
       OneOut(QLinearConv, 9, 1);
    }else if (OpName == "QLinearMatMul") {
       OneOut(QLinearMatMul, 8, 1);
    }else if (OpName == "QuantizeLinear") {
       OneOut(QuantizeLinear, 3, 1);
    }else if (OpName == "RNN") {
       MultipleOuts(RNN, 6, 2);
    }else if (OpName == "RandomNormal") {
       OneOut(RandomNormal, 0, 1);
    }else if (OpName == "RandomNormalLike") {
       OneOut(RandomNormalLike, 1, 1);
    }else if (OpName == "RandomUniform") {
       OneOut(RandomUniform, 0, 1);
    }else if (OpName == "RandomUniformLike") {
       OneOut(RandomUniformLike, 1, 1);
    }else if (OpName == "Range") {
       OneOut(Range, 3, 1);
    }else if (OpName == "Reciprocal") {
       OneOut(Reciprocal, 1, 1);
    }else if (OpName == "ReduceL1") {
       OneOut(ReduceL1, 1, 1);
    }else if (OpName == "ReduceL2") {
       OneOut(ReduceL2, 1, 1);
    }else if (OpName == "ReduceLogSum") {
       OneOut(ReduceLogSum, 1, 1);
    }else if (OpName == "ReduceLogSumExp") {
       OneOut(ReduceLogSumExp, 1, 1);
    }else if (OpName == "ReduceMax") {
       OneOut(ReduceMax, 1, 1);
    }else if (OpName == "ReduceMean") {
       OneOut(ReduceMean, 1, 1);
    }else if (OpName == "ReduceMin") {
       OneOut(ReduceMin, 1, 1);
    }else if (OpName == "ReduceProd") {
       OneOut(ReduceProd, 1, 1);
    }else if (OpName == "ReduceSum") {
       OneOut(ReduceSum, 1, 1);
    }else if (OpName == "ReduceSumSquare") {
       OneOut(ReduceSumSquare, 1, 1);
    }else if (OpName == "Relu") {
       OneOut(Relu, 1, 1);
    }else if (OpName == "Reshape") {
       OneOut(Reshape, 2, 1);
    }else if (OpName == "Resize") {
       OneOut(Resize, 4, 1);
    }else if (OpName == "ReverseSequence") {
       OneOut(ReverseSequence, 2, 1);
    }else if (OpName == "RoiAlign") {
       OneOut(RoiAlign, 3, 1);
    }else if (OpName == "Round") {
       OneOut(Round, 1, 1);
    }else if (OpName == "Scan") {
       OneOut(Scan, 1, 1);
    }else if (OpName == "Scatter") {
       OneOut(Scatter, 3, 1);
    }else if (OpName == "ScatterElements") {
       OneOut(ScatterElements, 3, 1);
    }else if (OpName == "ScatterND") {
       OneOut(ScatterND, 3, 1);
    }else if (OpName == "Selu") {
       OneOut(Selu, 1, 1);
    }else if (OpName == "SequenceAt") {
       OneOut(SequenceAt, 2, 1);
    }else if (OpName == "SequenceConstruct") {
       OneOut(SequenceConstruct, 1, 1);
    }else if (OpName == "SequenceEmpty") {
       OneOut(SequenceEmpty, 0, 1);
    }else if (OpName == "SequenceErase") {
       OneOut(SequenceErase, 2, 1);
    }else if (OpName == "SequenceInsert") {
       OneOut(SequenceInsert, 3, 1);
    }else if (OpName == "SequenceLength") {
       OneOut(SequenceLength, 1, 1);
    }else if (OpName == "Shape") {
       OneOut(Shape, 1, 1);
    }else if (OpName == "Shrink") {
       OneOut(Shrink, 1, 1);
    }else if (OpName == "Sigmoid") {
       OneOut(Sigmoid, 1, 1);
    }else if (OpName == "Sign") {
       OneOut(Sign, 1, 1);
    }else if (OpName == "Sin") {
       OneOut(Sin, 1, 1);
    }else if (OpName == "Sinh") {
       OneOut(Sinh, 1, 1);
    }else if (OpName == "Size") {
       OneOut(Size, 1, 1);
    }else if (OpName == "Slice") {
       OneOut(Slice, 5, 1);
    }else if (OpName == "Softmax") {
       OneOut(Softmax, 1, 1);
    }else if (OpName == "Softplus") {
       OneOut(Softplus, 1, 1);
    }else if (OpName == "Softsign") {
       OneOut(Softsign, 1, 1);
    }else if (OpName == "SpaceToDepth") {
       OneOut(SpaceToDepth, 1, 1);
    }else if (OpName == "Split") {
       OneOut(Split, 1, 1);
    }else if (OpName == "SplitToSequence") {
       OneOut(SplitToSequence, 2, 1);
    }else if (OpName == "Sqrt") {
       OneOut(Sqrt, 1, 1);
    }else if (OpName == "Squeeze") {
       OneOut(Squeeze, 1, 1);
    }else if (OpName == "StringNormalizer") {
       OneOut(StringNormalizer, 1, 1);
    }else if (OpName == "Sub") {
       OneOut(Sub, 2, 1);
    }else if (OpName == "Sum") {
       OneOut(Sum, 1, 1);
    }else if (OpName == "Tan") {
       OneOut(Tan, 1, 1);
    }else if (OpName == "Tanh") {
       OneOut(Tanh, 1, 1);
    }else if (OpName == "TfIdfVectorizer") {
       OneOut(TfIdfVectorizer, 1, 1);
    }else if (OpName == "ThresholdedRelu") {
       OneOut(ThresholdedRelu, 1, 1);
    }else if (OpName == "Tile") {
       OneOut(Tile, 2, 1);
    }else if (OpName == "TopK") {
       MultipleOuts(TopK, 2, 2);
    }else if (OpName == "Transpose") {
       OneOut(Transpose, 1, 1);
    }else if (OpName == "Unique") {
       MultipleOuts(Unique, 1, 4);
    }else if (OpName == "Unsqueeze") {
       OneOut(Unsqueeze, 1, 1);
    }else if (OpName == "Upsample") {
       OneOut(Upsample, 2, 1);
    }else if (OpName == "Where") {
       OneOut(Where, 3, 1);
    }else if (OpName == "Xor") {
       OneOut(Xor, 2, 1);
    }