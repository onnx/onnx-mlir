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
