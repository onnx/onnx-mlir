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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
  void printType(
      mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static StringRef getDialectNamespace() { return "onnx"; }
};

} // end namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// ONNX operations.
#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.hpp.inc"

namespace mlir {

// The namespace onnxmlir is experimental.
// onnx_mlir has been used in KRNL. Other candidates are onnxops, onnxdialect.
// Should this namesapce for onnx mlir project or ONNXOp dialect?
// Or we need two namespace?
// Will put all the ONNXOps into this namespace
namespace onnxmlir {
class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static StringType get(MLIRContext *ctx) { return Base::get(ctx); }
};

namespace detail {
struct SeqTypeStorage;
} // namespace detail

class SeqType
    : public mlir::Type::TypeBase<SeqType, mlir::Type, detail::SeqTypeStorage> {
public:
  using Base::Base;

  static SeqType get(llvm::ArrayRef<mlir::Type> elementTypes);

  llvm::ArrayRef<mlir::Type> getElementTypes();

  mlir::Type getElementType();

  size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // end namespace onnxmlir
} // end namespace mlir
