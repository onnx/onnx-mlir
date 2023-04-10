/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- ONNXTypes.cpp --------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXTypes.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNX Type: SeqType
//===----------------------------------------------------------------------===//

Type SeqType::parse(AsmParser &parser) {
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation())
        << "failed to parse !onnx.Seq type";
    return Type();
  }

  return get(elementType, ShapedType::kDynamic);
}

void SeqType::print(AsmPrinter &printer) const {
  // Previous implementation did not print/parse the length field
  // May add the field in future
  printer << "<" << getElementType() << ">";
}

//===----------------------------------------------------------------------===//
// ONNX Types: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/ONNX/ONNXTypes.cpp.inc"

// See explanation in ONNXDialect::initialize() in ONNXDialect.cpp.
void ONNXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/ONNX/ONNXTypes.cpp.inc"
      >();
}
