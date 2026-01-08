// SPDX-License-Identifier: Apache-2.0

//===--- ONNXOps.td -- ONNX Dialect Operation Definitions ----*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// Defines ONNX Dialect Definitions, Types, and Operations.
//
// When adding non-standard ONNX ops, please add them in AdditionalONNXOps.td.
//
// After changes that impact ONNX, run "make OMONNXOpsIncTranslation".
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_OPS
#define ONNX_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// Definition of the ONNX dialect.
//===----------------------------------------------------------------------===//

def ONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "A high-level dialect for the ONNX specification";
  let cppNamespace = "::mlir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 0;
  let dependentDialects = ["func::FuncDialect"];
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Types.
    void registerTypes();
  public:
    ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const override;
    void printAttribute(::mlir::Attribute attr,
                        ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

//===----------------------------------------------------------------------===//
// Definition of the ONNX dialect attributes for layout encoding.
//===----------------------------------------------------------------------===//

// Attribute for Tensors that have a different data layout than normal ONNX tensors.
class ONNX_LayoutAttr<string name, list<Trait> traits = []>
	: BaseLayoutAttr<ONNX_Dialect, name, traits>;

// ONNX tensor encoding attribute.
def ONNXTensorEncodingAttr : ONNX_LayoutAttr<"ONNXTensorEncoding"> {
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
      STANDARD, // Standard ONNX layout as defined by the specs.
      NCHWxC,   // Layout for conv image, with C=Cin tiled by x.
      KCNMxCyK  // Layout for conv weights, with C=Cin & K=Cout tiled by x & y.
    };
  }];

  let cppNamespace = "::mlir";
}

// Whether a ONNX Tensor type has the specified layout.

class ONNXCustomDataLayoutAndFactorsOfPred<
  string layout, int xFactor, int yFactor> :
 And<[
  CPred<"(mlir::cast<::mlir::RankedTensorType>($_self)) &&"
        "(mlir::dyn_cast_or_null<ONNXTensorEncodingAttr>"
        "(mlir::cast<::mlir::RankedTensorType>($_self).getEncoding())) &&"
        "(mlir::cast<ONNXTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self)"
        ".getEncoding()).getDataLayout()"
        " == ONNXTensorEncodingAttr::DataLayout::" # layout # ") &&"
        "(mlir::cast<ONNXTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self)"
        ".getEncoding()).getXFactor()"
        " == " # xFactor # ")  &&"
        "(mlir::cast<ONNXTensorEncodingAttr>(mlir::cast<::mlir::RankedTensorType>($_self)"
        ".getEncoding()).getYFactor()"
        " == " # yFactor # ")">
]>;

// So far ONNX Tensor supports only F32 for pre-tranformed custom-layout data.
class ONNXCustomTensorOf<string layout, int xFactor, int yFactor> :
    Type<And<[TensorOf<[F32]>.predicate,
              ONNXCustomDataLayoutAndFactorsOfPred<layout, xFactor, yFactor>]>,
         TensorOf<[F32]>.summary # " with layout " # layout #
         " and factors " # xFactor # ", " # yFactor,
         "::mlir::TensorType">;

def ONNXCustomTensor_unranked : UnrankedTensorOf<[F32]>;

// Definition of types for tiling factor of 4.
def ONNXTensor_NCHW4C: AnyTypeOf<[ONNXCustomTensor_unranked,
    ONNXCustomTensorOf<"NCHWxC", 4, 0>]>;
def ONNXTensor_KCMN4C4K: AnyTypeOf<[ONNXCustomTensor_unranked,
    ONNXCustomTensorOf<"KCNMxCyK", 4, 4>]>;

//===----------------------------------------------------------------------===//
// Definition of types
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Definition for rewrite rules for onnx dialect
// Can be used in other table gen files (.td) for onnx dialect
//===----------------------------------------------------------------------===//

def StringType : Type<CPred<"mlir::isa<ONNXStringType>($_self)">, "string type">;

def IsSeqTypePred : CPred<"mlir::isa<SeqType>($_self)">;

class SeqOf<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, IsSeqTypePred,
                "mlir::cast<SeqType>($_self).getElementType()", "SeqType">;

def IsOptTypePred : CPred<"mlir::isa<OptType>($_self)">;

class OptOf<Type type> :
  ContainerType<type, IsOptTypePred,
                "mlir::cast<OptType>($_self).getElementType()", "OptType">;

def ONNXConstantOpFromDenseAttr: NativeCodeCall<
  "onnx_mlir::OnnxBuilder($_builder, $_loc).constant($0)">;

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//

// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ONNX_Op<string mnemonic, list<Trait> traits = []> :
  Op<ONNX_Dialect, mnemonic, traits> ;

// The tablegen code onnxop.in is generated with gen_doc.py
// clone and install onnx
//    git clone --recursive https://github.com/onnx/onnx.git
//  set up env for anaconda3 and for ONNX-MLIR (BOOSTROOT, cmake, gcc ...)
//    cd onnx
// install onnx
//   CC=gcc CXX=g++ pip install -e .
// run the script
//   python onnx/defs/gen_doc.py
// result is in docs/onnx_ops.td.inc
// current limitations:
//  1. Attributes are not processed
//  2. output type inference not implemented except Add
//  3. Type Attribute: 'optional' and 'Variadic hetergeneous' are ignored
//  4. type of string, complex64 and complex128 for input/output are ignored
//  5. unsigned int are treated as signed one

include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Dialect/ONNX/ONNXOps.td.inc"
include "src/Dialect/ONNX/AdditionalONNXOps.td"

#endif // ONNX_OPS
