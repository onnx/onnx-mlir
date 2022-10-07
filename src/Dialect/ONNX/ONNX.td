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

//===----------------------------------------------------------------------===//
// Definition of the dialect.
//===----------------------------------------------------------------------===//

def ONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "A high-level dialect for the ONNX specification";
  let cppNamespace = "::mlir";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["func::FuncDialect"];
}

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
