//===-------------------- ONNXOps.hpp - ONNX Operations -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "src/Interface/PromotableConstOperandsOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceInterface.hpp"

#include "ONNXOpsHelper.hpp"

namespace mlir {

class ONNXOpsDialect : public Dialect {
public:
  ONNXOpsDialect(MLIRContext *context);

  /// Parse an instance of a type registered to the onnx dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the onnx dialect.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static StringRef getDialectNamespace() { return "onnx"; }
};

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.hpp.inc"

namespace ONNXTypes {

enum Kind {
  FIRST_USED_ONNX_TYPE = Type::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
//#define HANDLE_TF_TYPE(tftype, enumerant, name) enumerant,
//#include "src/Dialect/ONNX/ONXTypes.def"
  STRING,
  LAST_USED_ONNX_TYPE,
};
}  // namespace ONNXTypes

class StringType : public mlir::Type::TypeBase<StringType, mlir::Type> { 
public:                                                             
  using Base::Base;                                              
  static bool kindof(unsigned kind) { return kind == ONNXTypes::STRING; } 

  static unsigned getTypeKind() { return ONNXTypes::STRING; }

  static StringType get(MLIRContext* ctx) { return Base::get(ctx, ONNXTypes::STRING);} 

};

}  // end namespace mlir


namespace onnx_mlir {}
