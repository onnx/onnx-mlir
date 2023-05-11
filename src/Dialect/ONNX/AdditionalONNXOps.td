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
//   Example are CallOp, YieldOp,...
//
// * Second are ops that are used to handle optional arguments/special cases
//   of original ONNX ops. These ops may possibly be removed in the future.
//   Example are ONNXBatchNormalizationInferenceModeOp, ...
//
// After changes that impact ONNX, run "make OMONNXOpsIncTranslation".
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// ONNX Ops to assist lowering of ONNX.
//===----------------------------------------------------------------------===//

/*
  Note: each operation ONNXxxxOp that interact with shape helper should have
  the following def.

  let extraClassDeclaration = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXxxxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXxxxOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];

  Then, in ShapeHelper.hpp, the class

  ONNXxxxOpShapeHelper

  should also be defined accordingly. Using ONNXxxxOpShapeHelper as an alias
  for ONNXUnimplementedOpShapeHelper is always a possibility.
*/

//===----------------------------------------------------------------------===//
// ONNX CallOp
def ONNXCallOp : ONNX_Op<"ONNX_Call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
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

    This operation is not part of the standard and was added to assist onnx-mlir.
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

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXCallOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXCallOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}


//===----------------------------------------------------------------------===//
// ONNX FuncReturnOp
def ONNXFuncReturnOp : ONNX_Op<"FuncReturn",
    [Pure, HasParent<"func::FuncOp">, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `onnx.FuncReturn` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation, with the exception that shaped types may have
    more specific shapes than the function signature return types.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ONNX CustomOp
def ONNXCustomOp:ONNX_Op<"Custom",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX Custom operation";
  let description = [{
    Allow call-out to a user defined operation. A single attribute
    is a string which names the operation, other inputs are
    passed to the user operation.
    The number of inputs and outputs can vary.

    This operation is not part of the standard and was added to assist onnx-mlir.
    NoneType is allowed for input and output since the CustomOp may want
    the fixed number of input/output for external function call.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef, NoneType]>>:$input,
                       StrAttr:$function_name);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor, AnyMemRef, NoneType]>>:$outputs);

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

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXCustomOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXCustomOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ONNX DimOp
def ONNXDimOp : ONNX_Op<"Dim",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX dimensions operation.";
  let description = [{
    This operation is to obtain the dimension of a Tensor;

    ```
    "onnx.Dim"(%tensor) {axis = 0 : si64} : (tensor<?x3x5xf32>) -> tensor<1xi64>
    ```

    The axis identifies the dimension within the shape which is going to be obtained.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       DefaultValuedAttr<SI64Attr, "0">:$axis);
  let results = (outs TensorOf<[I64]>:$dim);
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXDimOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXDimOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];

}

// ONNX DimGroupOp
def ONNXDimGroupOp : ONNX_Op<"DimGroup"> {
  let summary = "ONNX dimension group operation.";
  let description = [{
    This operation is to link a compile-time unknown dimension of a Tensor
    to a group id. Two dimensions that have the same group id are expected
    to be equal at runtime.

    ```
    "onnx.DimGroup"(%tensor) {axis = 0 : si64, group_id = 1: si64} : (tensor<?x3x5xf32>) -> ()
    ```

    `axis` identifies the dimension position in the tensor.

    `group_id` identifies the group id of the dimension. It is non-negative.
    Value -1 for `group_id` means the dimension does not belong to any group.

    This operation is currently used in the pass `--onnx-dim-analysis`
    for testing the unknown dimension analysis class.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$data,
                       DefaultValuedAttr<SI64Attr, "0">:$axis,
                       DefaultValuedAttr<SI64Attr, "-1">:$group_id);
}

//===----------------------------------------------------------------------===//
// ONNX EntryPoint: Indicate entry point functions of ONNX graph.
def ONNXEntryPointOp: ONNX_Op<"EntryPoint"> {
  let summary = "Indicate ONNX entry point";
  let description = [{
    The "onnx.EntryPoint" function indicates the main entry point of ONNX model.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];

  let arguments = (ins SymbolRefAttr:$func);

  let builders = [OpBuilder<(ins "func::FuncOp":$function)>];

  let extraClassDeclaration = [{
    static ONNXEntryPointOp create(Location location, func::FuncOp& func);

    static StringRef getEntryPointFuncAttrName() { return "func"; }
  }];
}

//===----------------------------------------------------------------------===//
// LayoutTransform.
def ONNXLayoutTransformOp:ONNX_Op<"LayoutTransform", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "An operation that transforms data between different layout formats";
  let description = [{
    An operation that transforms a tensor from a layout to another layout. 
    A layout is defined by an attribute, i.e. `target_layout`, which allows this
    operation work with an arbitrary layout (e.g. a layout used for accelerators).

    `target_layout` is optional. If it is not given, the input tensor will be
    transformed to a normal tensor that does not have layout.

    If `target_layout` is the same as the input's layout, this operation will
    become an no-op by canonicalization. 

    The input and output tensors must have the same shape.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>]>:$data,
                       OptionalAttr<AnyAttrOf<[LayoutAttr]>>:$target_layout
  );
  let results = (outs AnyTypeOf<[TensorOf<[F32]>]>:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$data, "::mlir::Attribute":$target_layout)>, // Defined layouts.
  ];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXLayoutTransformOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXLayoutTransformOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// NoValueOp.
def ONNXNoneOp : ONNX_Op<"NoValue", [ConstantLike, Pure]> {
  let summary = "An operation representing the absence of a value.";
  let description = [{
    This operation can be used to represent the absence of a value. It is typically
    used as an argument to operators that have optional parameters.

    Example:
    ```MLIR
      %cst = "onnx.NoValue"() {value} : () -> none
      %0, %1 = "onnx.Split"(%arg0, %cst) { axis=1 : si64 } : (tensor<?xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
    ```

    This operation is not part of the standard and was added to assist onnx-mlir.
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

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];

  let arguments = (ins StrAttr:$op_name, Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$input);
}

//===----------------------------------------------------------------------===//
// ONNX YieldOp
def ONNXYieldOp : ONNX_Op<"Yield", [Pure, ReturnLike, Terminator]> {
  let summary = "ONNX yield operation";
  let description = [{
    The `onnx.Yield` operation represents a yield operation within an ONNX subgraph.
    The operation takes variable number of operands and produces no results.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, std::nullopt); }]>];

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
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
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

    This operation is not part of the standard and was added to assist onnx-mlir.
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
  
  let extraClassDefinition = [{  
    onnx_mlir::ONNXOpShapeHelper * ONNXBatchNormalizationInferenceModeOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXBatchNormalizationInferenceModeOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// MaxPoolSingleOutOp
def ONNXMaxPoolSingleOutOp: ONNX_Op<"MaxPoolSingleOut",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX MaxPool operation with a single output.";
  let description = [{
    ONNX MaxPool operation with a single output.
    See ONNXMaxPoolOp for a full description of the MaxPool semantics.

    This operation is not part of the standard and was added to assist onnx-mlir.
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
  
  let extraClassDefinition = [{  
    onnx_mlir::ONNXOpShapeHelper * ONNXMaxPoolSingleOutOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXMaxPoolSingleOutOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConcatShapeTransposeOp
def ONNXConcatShapeTransposeOp: ONNX_Op<"ConcatShapeTranspose", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX merged operation";
  let description = [{
    Merge the following sequence of ops into one op
    v1 = onnx.concat
    v2 = onnx.shape(v1)
    v3 = onnx.transpose(v1)

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>>:$inputs,
  SI64Attr:$axis,
  OptionalAttr<SI64Attr>:$end,
  DefaultValuedAttr<SI64Attr, "0">:$start,
  OptionalAttr<I64ArrayAttr>:$perm);
  let results = (outs TensorOf<[I64]>:$shape,
     AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>]>:$transposed);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXConcatShapeTransposeOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXConcatShapeTransposeOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}
 
//===----------------------------------------------------------------------===//
// ONNXShapeTransformOp
def ONNXShapeTransformOp: ONNX_Op<"ShapeTransform", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX Element-wise shape transformation operation";
  let description = [{
    This operator transforms a tensor into another tensor whose shape is changed
    by a given affine map. This is elemement-wise transformation, so each element
    in the input will be copied to an element in the output via the affine map.
    The affine map must be bijective.

    For example, the following code is using `onnx.ShapeTransform` to reshape
    a tensor from 2D to 4D.
    ```mlir
    #reshape = affine_map(d0, d1) -> (d0/32, d0%32, d1/64, d1%64)
    %Y = onnx.ShapeTransform(%arg0) {index_map = #reshape} :  (tensor<128x128xf32>) -> tensor<4x32x2x64xf32>
    ```

    `onnx.ShapeTransform` will be finally materialized into an `affine.for` via
    lowering to `krnl` dialect, e.g.
    ```mlir
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x32x2x64xf32>
    affine.for %arg1 = 0 to 128 {
      affine.for %arg2 = 0 to 128 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref< 128x128xf32 >
        affine.store %0, %alloc[%arg1 / 32, %arg1 % 32, %arg2 / 64, %arg2 % 64] : memref<4x32x2x64xf32>
      }
    }
    ```

    When being canonicalized, ShapeTransform operations are composed into
    a new ShapeTransform operation by composing their affine maps.

    At this moment, this operation only supports static dimensions.

    This operation is not part of the standard and was added to assist onnx-mlir.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>]>:$input,
                       AffineMapAttr:$index_map);
  let results = (outs AnyTypeOf<[TensorOf<[F32]>]>:$output);

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ONNXShapeTransformOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXShapeTransformOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

