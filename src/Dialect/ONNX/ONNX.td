// SPDX-License-Identifier: Apache-2.0

//===--- ONNXOps.td -- ONNX Dialect Operation Definitions ----*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// Defines ONNX Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_OPS
#define ONNX_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"

// Definition for rewrite rules for onnx dialect
// Can be used in other table gen files (.td) for onnx dialect

def StringType : Type<CPred<"$_self.isa<ONNXStringType>()">, "string type">;

def IsSeqTypePred : CPred<"$_self.isa<SeqType>()">;

class SeqOf<list<Type> allowedTypes> : 
  ContainerType<AnyTypeOf<allowedTypes>, IsSeqTypePred,
                "$_self.cast<SeqType>().getElementType()", "SeqType">;

def IsOptTypePred : CPred<"$_self.isa<OptType>()">;

class OptOf<Type type> :
  ContainerType<type, IsOptTypePred,
                "$_self.cast<OptType>().getElementType()", "OptType">;

def ONNXConstantOpFromDenseAttr: NativeCodeCall<
  "onnx_mlir::createONNXConstantOpWithDenseAttr($_builder, $_loc, $0)">;

def createNoneIntegerConstant : NativeCodeCall<
  "onnx_mlir::createNoneIntegerConstant($_builder, $0.getDefiningOp()->getLoc())">;

def createNoneFloatConstant : NativeCodeCall<
  "onnx_mlir::createNoneFloatConstant($_builder, $0.getDefiningOp()->getLoc())">;

def ONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "A high-level dialect for the ONNX specification";
  let cppNamespace = "::mlir";
  let useDefaultTypePrinterParser = 1;
}

// Attribute for Tensors that have a different data layout than normal ONNX tensors.
class ONNX_Attr<string name, list<Trait> traits = []>
	: AttrDef<ONNX_Dialect, name, traits>;

// ONNX tensor encoding attribute.
def ONNXTensorEncodingAttr : ONNX_Attr<"ONNXTensorEncoding"> {
  let mnemonic = "encoding";
  let hasCustomAssemblyFormat = 1;

  let description = [{
    An attribute to encode information on properties of ONNX tensors. A tensor with
    this encoding attribute indicates that it is a tensor whose type is
    the original type of the pre-transformed custom-layout data.

    `dataLayout` indicates the data layout of the pre-transformed data.

    `x` indicates the tiling factor for the first tiled dimension. Typical values
    are multiples of the SIMD vector length of the target machine.

    `y` indicates the tiling factor for the first tiled dimension. Typical values
    are multiples of the SIMD vector length of the target machine. For layout
    with only one tiling dimension, `y` should be zero.

    Example:

    ```mlir
    #ONNX_NCHWxC= #onnx.encoding<{
      dataLayout = "NCHWxC",
      xFactor = 4,
      yFactor = 0
    }>

    ... tensor<8x8x128x128xf32, #ONNX_NCHWxC> ...
    ```
  }];

  // Data in ONNX tensor encoding.
  let parameters = (ins
    // A original data layout of the pre-transformed custom-layout data.
    "ONNXTensorEncodingAttr::DataLayout":$dataLayout,
    "int64_t":$xFactor, // typical is vector length (VL).
    "int64_t":$yFactor  // typical is VL when used, 0 otherwise.
  );

  let extraClassDeclaration = [{
    enum class DataLayout {
      NCHWxC,    // Layout for conv image, with C=Cin tiled by x.
      KCNMxCyK   // Layout for conv weights, with C=Cin & K=Cout tiled by x & y.
    };
  }];

  let cppNamespace = "::mlir";
}

// Whether a ztensor type has the specified layout. 
//class ONNXCustomDataLayoutOfPred<string layout> :
// And<[
//  CPred<"($_self.cast<::mlir::RankedTensorType>()) &&"
//        "($_self.cast<::mlir::RankedTensorType>()."
//        "getEncoding().dyn_cast_or_null<mlir::ONNXTensorEncodingAttr>()) &&"
//        "($_self.cast<::mlir::RankedTensorType>()."
//        "getEncoding().cast<ONNXTensorEncodingAttr>().getDataLayout()"
//        " == ONNXTensorEncodingAttr::DataLayout::" # layout # ")"> 
//]>;

class ONNXCustomDataLayoutAndFactorsOfPred<
  string layout, int xFactor, int yFactor> :
 And<[
  CPred<"($_self.cast<::mlir::RankedTensorType>()) &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().dyn_cast_or_null<ONNXTensorEncodingAttr>()) &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().cast<ONNXTensorEncodingAttr>().getDataLayout()"
        " == ONNXTensorEncodingAttr::DataLayout::" # layout # ") &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().cast<ONNXTensorEncodingAttr>().getXFactor()"
        " == " # xFactor # ")  &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().cast<ONNXTensorEncodingAttr>().getYFactor()"
        " == " # yFactor # ")"> 
]>;

// So far ONNX Tensor supports only F32 for pre-tranformed custom-layout data.
class ONNXCustomTensorOf<string layout, int xFactor, int yFactor> :
    Type<And<[TensorOf<[F32]>.predicate, 
              ONNXCustomDataLayoutAndFactorsOfPred<layout, xFactor, yFactor>]>,
         TensorOf<[F32]>.summary # " with layout " # layout # 
         "and factors " # xFactor # ", " # yFactor,
         "::mlir::TensorType">;

def ONNXCustomTensor_unranked : UnrankedTensorOf<[F32]>;

// Definition of types for tiling factor of 4.
def ONNXTensor_NCHW4C: AnyTypeOf<[ONNXCustomTensor_unranked, 
    ONNXCustomTensorOf<"NCHWxC", 4, 0>]>;
def ONNXTensor_KCMN4C4K: AnyTypeOf<[ONNXCustomTensor_unranked, 
    ONNXCustomTensorOf<"KCNMxCyK", 4, 4>]>;

// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ONNX_Op<string mnemonic, list<Trait> traits = []> :
  Op<ONNX_Dialect, mnemonic, traits> ;

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//

//the tablegen code onnxop.in is generated with gen_doc.py
//clone and install onnx
//   git clone --recursive https://github.com/onnx/onnx.git
// set up env for anaconda3 and for ONNX-MLIR (BOOSTROOT, cmake, gcc ...)
//   cd onnx
//install onnx
//  CC=gcc CXX=g++ pip install -e .
//run the script
//  python onnx/defs/gen_doc.py
//result is in docs/onnx_ops.td.inc
//current limitations:
// 1. Attributes are not processed
// 2. output type inference not implemented except Add
// 3. Type Attribute: 'optional' and 'Variadic hetergeneous' are ignored
// 4. type of string, complex64 and complex128 for input/output are ignored 
// 5. unsigned int are treated as signed one

include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Dialect/ONNX/ONNXOps.td.inc"
include "src/Dialect/ONNX/AdditionalONNXOps.td"

// Indicate entry point functions of ONNX graph.
def ONNXEntryPointOp: ONNX_Op<"EntryPoint"> {
  let summary = "Indicate ONNX entry point";
  let description = [{
    The "onnx.EntryPoint" function indicates the main entry point of ONNX model.
  }];

  let arguments = (ins SymbolRefAttr:$func);

  let builders = [OpBuilder<(ins "func::FuncOp":$function)>];

  let extraClassDeclaration = [{
    static ONNXEntryPointOp create(Location location, func::FuncOp& func);

    static StringRef getEntryPointFuncAttrName() { return "func"; }
  }];
}

def ONNXReturnOp : ONNX_Op<"Return", [NoSideEffect,
                                ReturnLike, Terminator]> {
  let summary = "ONNX return operation";
  let description = [{
    The `ONNX.Return` operation represents a return operation within an ONNX subgraph.
    The operation takes variable number of operands and produces no results.

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

def ONNXMaxPoolSingleOutOp: ONNX_Op<"MaxPoolSingleOut",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX MaxPool operation with a single output.";
  let description = [{
    "ONNX MaxPool operation with a single output."
    "See ONNXMaxPoolOp for a full description of the MaxPool semantics."
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

def ONNXBatchNormalizationInferenceModeOp: ONNX_Op<"BatchNormalizationInferenceMode",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX BatchNormalization operation in test mode";
  let description = [{
    "Carries out batch normalization as described in the paper"
    "https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,"
    "there are multiple cases for the number of outputs, which we list below:"
    ""
    "Output case #1: Y, mean, var, saved_mean, saved_var (training mode)"
    "Output case #2: Y (test mode)"
    ""
    "For previous (depreciated) non-spatial cases, implementors are suggested"
    "to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op."
    "This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted."
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

def ONNXNoneOp : ONNX_Op<"NoValue", [ConstantLike, NoSideEffect]> {
  let summary = "An operation representing the absence of a value.";
  let description = [{
    This operation can be used to represent the absence of a value. It is typically 
    used as an argument to operators that have optional parameters.
    Example:
      %cst = "onnx.NoValue"() {value} : () -> none
      %0, %1 = "onnx.Split"(%arg0, %cst) { axis=1 : si64 } : (tensor<?xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  }];

  let arguments = (ins UnitAttr:$value);
  let results = (outs NoneType:$none_val);

  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins),[{ 
      build($_builder, $_state, $_builder.getNoneType(), $_builder.getUnitAttr());
    }]>];
}

class ONNX_Type<string name, string typeMnemonic,
                string baseCppClass = "::mlir::Type">
    : TypeDef<ONNX_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

def ONNX_StringType : ONNX_Type<"ONNXString", "String"> {
  let summary = " ONNX StringType";
  let description = [{
    An array of characters.
  }];
}

def ONNX_SeqType : ONNX_Type<"Seq", "Seq"> {
  let summary = " ONNX SeqType";
  let description = [{
    An list of tensors which may have different shape
  }];

  let parameters = (ins "::mlir::Type":$ElementType, "int64_t":$Length);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType,
                                        "int64_t":$length), [{
      return Base::get(elementType.getContext(), elementType, length);
    }]>
  ];
}

def ONNX_OptType : ONNX_Type<"Opt", "Opt"> {
  let summary = " ONNX OptType";
  let description = [{
    An optional tensor or sequence
  }];

  let parameters = (ins "::mlir::Type":$ElementType);

  let assemblyFormat = "`<` $ElementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}
#endif // ONNX_OPS
