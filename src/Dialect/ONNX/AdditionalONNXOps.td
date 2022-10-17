// SPDX-License-Identifier: Apache-2.0

//===-- AdditionalONNXOps.td -- ONNX Dialect Additional Ops  -*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// Defines Additional ONNX Dialect operations that are introduced to assist
// onnx-mlir.
//
// Ops are listed in alphabetical order, in 2 sections.
//
// * First are ops that are used to assist the processing of ONNX dialect.
//   Example are CallOp, ReturnOp,...
//
// * Second are ops that are used to handle optional arguments/special cases
//   of original ONNX ops. These ops may possibly be removed in the future.
//   Example are ONNXBatchNormalizationInferenceModeOp, ...
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// ONNX Ops to assit lowering of ONNX.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ONNX CallOp
def ONNXCallOp : ONNX_Op<"ONNX_Call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  // TODO I would rather have <anyTypeOf<AnyTensor>> here, but the testcase for
  // CustomOps supplies an empty parameter which fails that test.
  let results = (outs Variadic<AnyTypeOf<[AnyTensor]>>);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    StringAttr getCalleeAttr() { return calleeAttr().getAttr(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
}

//===----------------------------------------------------------------------===//
// ONNX CustomOp
def ONNXCustomOp:ONNX_Op<"Custom",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX Custom operation";
  let description = [{
    Allow call-out to a user defined operation. A single attribute
    is a string which names the operation, other inputs are
    passed to the user operation.
    The number of inputs and outputs can vary.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$input,
                       StrAttr:$function_name);
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

//===----------------------------------------------------------------------===//
// ONNX DimOp
def ONNXDimOp : ONNX_Op<"Dim",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX dimensions operation.";
  let description = [{
    This operation is to obtain the dimension of a Tensor;

    ```
    "onnx.Dim"(%tensor) {axis = 0 : si64} : (tensor<?x3x5xf32>) -> tensor<1xi64>
    ```

    The axis identifies the dimension within the shape which is going to be obtained.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       DefaultValuedAttr<SI64Attr, "0">:$axis);
  let results = (outs TensorOf<[I64]>:$dim);
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ONNX EntryPoint: Indicate entry point functions of ONNX graph.
def ONNXEntryPointOp: ONNX_Op<"EntryPoint"> {
  let summary = "Indicate ONNX entry point";
  let description = [{
    The "onnx.EntryPoint" function indicates the main entry point of ONNX model.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins SymbolRefAttr:$func);

  let builders = [OpBuilder<(ins "func::FuncOp":$function)>];

  let extraClassDeclaration = [{
    static ONNXEntryPointOp create(Location location, func::FuncOp& func);

    static StringRef getEntryPointFuncAttrName() { return "func"; }
  }];
}

//===----------------------------------------------------------------------===//
// NoValueOp.
def ONNXNoneOp : ONNX_Op<"NoValue", [ConstantLike, NoSideEffect]> {
  let summary = "An operation representing the absence of a value.";
  let description = [{
    This operation can be used to represent the absence of a value. It is typically
    used as an argument to operators that have optional parameters.

    Example:
    ```MLIR
      %cst = "onnx.NoValue"() {value} : () -> none
      %0, %1 = "onnx.Split"(%arg0, %cst) { axis=1 : si64 } : (tensor<?xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
    ```

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins UnitAttr:$value);
  let results = (outs NoneType:$none_val);

  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins),[{
      build($_builder, $_state, $_builder.getNoneType(), $_builder.getUnitAttr());
    }]>];
}

//===----------------------------------------------------------------------===//
// ONNX PrintSignatureOp.
def ONNXPrintSignatureOp:ONNX_Op<"PrintSignature", []> {
  let summary = "ONNX Op to print type signature of its input operands";
  let description = [{
    Print type signature of the op's input operands. This operation is introduced early
    so as to preserve the name of the original ONNX op.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins StrAttr:$op_name, Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$input);
}

//===----------------------------------------------------------------------===//
// ONNX ReturnOp
def ONNXReturnOp : ONNX_Op<"Return", [NoSideEffect,
                                ReturnLike, Terminator]> {
  let summary = "ONNX return operation";
  let description = [{
    The `ONNX.Return` operation represents a return operation within an ONNX subgraph.
    The operation takes variable number of operands and produces no results.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// ONNX Operations for handling optional arguments
//===----------------------------------------------------------------------===//

// To allow pattern matching on operations with optional arguments/outputs we
// implement variants of the original ONNX dialect operations. The ONNX
// operations automatically generated by the `gen_doc.py` script and included
// in the `onnxop.inc` file have all optional arguments and outputs present.
// In the operations below we include the variants with missing operands
// or outputs. This decision affects only ONNX operations with optional
// arguments not ONNX operations with variadic operands.

//===----------------------------------------------------------------------===//
// BatchNorm in Inference mode.
def ONNXBatchNormalizationInferenceModeOp: ONNX_Op<"BatchNormalizationInferenceMode",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX BatchNormalization operation in test mode";
  let description = [{
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    there are multiple cases for the number of outputs, which we list below:

    Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    Output case #2: Y (test mode)"

    For previous (depreciated) non-spatial cases, implementors are suggested
    to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
    This operator has **optional** inputs/outputs. See [the doc](IR.md)
    for more details about the representation of optional arguments.
    An empty string may be used in the place of an actual argument's name to
    indicate a missing argument. Trailing optional arguments (those not followed
    by an argument that is present) may also be simply omitted.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                       AnyTypeOf<[AnyMemRef, AnyTensor]>:$scale,
                       AnyTypeOf<[AnyMemRef, AnyTensor]>:$B,
                       AnyTypeOf<[AnyMemRef, AnyTensor]>:$mean,
                       AnyTypeOf<[AnyMemRef, AnyTensor]>:$var,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                       DefaultValuedAttr<F32Attr, "0.9">:$momentum);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$o_Y);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 5; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {20}; }
  }];
}

//===----------------------------------------------------------------------===//
// MaxPoolSingleOutOp
def ONNXMaxPoolSingleOutOp: ONNX_Op<"MaxPoolSingleOut",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX MaxPool operation with a single output.";
  let description = [{
    ONNX MaxPool operation with a single output.
    See ONNXMaxPoolOp for a full description of the MaxPool semantics.

    This operation is not part of the standard and was added to assit onnx-mlir.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                       DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
                       DefaultValuedAttr<SI64Attr, "0">:$ceil_mode,
                       OptionalAttr<I64ArrayAttr>:$dilations,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$kernel_shape,
                       OptionalAttr<I64ArrayAttr>:$pads,
                       DefaultValuedAttr<SI64Attr, "0">:$storage_order,
                       OptionalAttr<I64ArrayAttr>:$strides);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$o_Y);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 1; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {20}; }
  }];
}
