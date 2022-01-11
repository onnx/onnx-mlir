/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {

// OpSet level supported by onnx-mlir
static constexpr int CURRENT_ONNX_OPSET = 13;

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
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage,
          mlir::MemRefElementTypeInterface::Trait> {

public:
  using Base::Base;
  using Base::getChecked;

  static StringType get(MLIRContext *ctx) { return Base::get(ctx); }
};

namespace detail {
struct SeqTypeStorage;
} // namespace detail

class SeqType
    : public mlir::Type::TypeBase<SeqType, mlir::Type, detail::SeqTypeStorage> {
public:
  using Base::Base;

  static SeqType get(mlir::Type elementType, int64_t length = -1);

  mlir::Type getElementType() const;

  // Return the length of the sequence.
  // 0 : if the seq is empty
  // -1  if unknown at compiler time
  int64_t getLength() const;
};

} // end namespace onnxmlir
} // end namespace mlir
