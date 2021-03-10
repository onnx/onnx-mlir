def ONNXCustomOp:ONNX_Op<"Custom",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX Custom operation";
  let description = [{
  "Allow call-out to a user defined operation. A single attribute"
  "is a string which names the operation, other inputs are"
  "passed to the user operation."
  "The number of inputs and outputs can vary."
  }];
  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$input, StrAttr:$function_name);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$outputs);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return -1;
    }
    static int getNumberOfResults() {
      return -1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXGeluOp:ONNX_Op<"Gelu",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX Gelu operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$input);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$output);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 1;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXSparseSoftmaxCrossEntropyGradOp:ONNX_Op<"SparseSoftmaxCrossEntropyGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX SparseSoftmaxCrossEntropyGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$log_prob,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, AnyMemRef]>:$label,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$weight,
    DefaultValuedAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$d_logits);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 4;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXGatherNDGradOp:ONNX_Op<"GatherNDGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX GatherNDGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, AnyMemRef]>:$shape,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, AnyMemRef]>:$indices,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, AnyMemRef]>:$update,
    DefaultValuedAttr<SI64Attr, "0">:$batch_dims);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$output);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXGatherGradOp:ONNX_Op<"GatherGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX GatherGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, AnyMemRef]>:$shape,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, AnyMemRef]>:$indices,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    DefaultValuedAttr<SI64Attr, "0">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dX);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXGeluGradOp:ONNX_Op<"GeluGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX GeluGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dX);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXSoftmaxGradOp:ONNX_Op<"SoftmaxGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX SoftmaxGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$X,
    DefaultValuedAttr<SI64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dX);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXDropoutGradOp:ONNX_Op<"DropoutGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX DropoutGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, AnyMemRef]>:$mask,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$ratio,
    AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, NoneType, AnyMemRef]>:$training_mode);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[I64]>, AnyMemRef]>:$dx);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 4;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXLayerNormalizationGradOp:ONNX_Op<"LayerNormalizationGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX LayerNormalizationGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$Y_grad,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef]>:$scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$mean,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$inv_std_var,
    DefaultValuedAttr<SI64Attr, "-1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$X_grad,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$scale_grad,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, AnyMemRef]>:$bias_grad);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 5;
    }
    static int getNumberOfResults() {
      return 3;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXSoftmaxCrossEntropyOp:ONNX_Op<"SoftmaxCrossEntropy",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX SoftmaxCrossEntropy operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$logits,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$labels,
    DefaultValuedAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$log_prob);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 2;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}


def ONNXLayerNormalizationOp:ONNX_Op<"LayerNormalization",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX LayerNormalization operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$data,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, TensorOf<[I64]>]>:$weight,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, TensorOf<[I64]>, NoneType]>:$bias,
    DefaultValuedAttr<SI64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "1.0E-5">:$epsilon);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$out,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, AnyMemRef, TensorOf<[I64]>, NoneType]>:$saved_mean,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, AnyMemRef, TensorOf<[I64]>, NoneType]>:$saved_inv_std_var);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 3;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXSoftmaxCrossEntropyGradOp:ONNX_Op<"SoftmaxCrossEntropyGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX SoftmaxCrossEntropyGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$log_prob,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$label,
    DefaultValuedAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$d_logits);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}


def ONNXTrainableDropoutOp:ONNX_Op<"TrainableDropout",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX TrainableDropout operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$data,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$ratio);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$output,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$mask);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 2;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXAdamOptimizerOp:ONNX_Op<"AdamOptimizer",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX AdamOptimizer operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$R,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$T,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$weights,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$gradients,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$moment_1,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$moment_2,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, TensorOf<[I32]>, AnyMemRef]>:$mixed_precision_weights,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$loss_scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$global_gradient_norm,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$update_signal,
    DefaultValuedAttr<F32Attr, "0.9">:$alpha,
    DefaultValuedAttr<F32Attr, "0.999">:$beta,
    DefaultValuedAttr<F32Attr, "0.0">:$lambda,
    DefaultValuedAttr<F32Attr, "1e-8">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0">:$max_norm_clip,
    DefaultValuedAttr<SI64Attr, "1">:$do_bias_correction,
    DefaultValuedAttr<SI64Attr, "0">:$weight_decay_mode);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$new_T,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$new_moment_1,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$new_moment_2,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$new_weights,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$new_gradients,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$new_mixed_precision_weights);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 10;
    }
    static int getNumberOfResults() {
      return 6;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXSparseSoftmaxCrossEntropyOp:ONNX_Op<"SparseSoftmaxCrossEntropy",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX SparseSoftmaxCrossEntropy operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$logits,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$label,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$weight,
    DefaultValuedAttr<StrAttr, "mean">:$reduction);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef, NoneType]>:$log_prob);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 3;
    }
    static int getNumberOfResults() {
      return 2;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def ONNXReluGradOp:ONNX_Op<"ReluGrad",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX ReluGrad operation";
  let description = [{
  ""
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dY,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[BF16]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$X);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I64]>, AnyMemRef]>:$dX);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 2;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

