/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- StickifiedConstant.cpp - ZHigh Operations ------------------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

// Print DisposableElementsAttr as a DenseElementsAttr, because
// DisposableElementsAttr is an internal representation, so we hide it
// in this way.
static void printAttribute(OpAsmPrinter &printer, Attribute attr) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(attr))
    disposable.printAsDenseElementsAttr(printer);
  else
    printer.printAttribute(attr);
}

static void printNamedAttribute(
    OpAsmPrinter &printer, NamedAttribute namedAttr) {
  // Print the name without quotes if possible.
  printer.printKeywordOrString(namedAttr.getName().strref());

  // Pretty printing elides the attribute value for unit attributes.
  if (mlir::isa<UnitAttr>(namedAttr.getValue()))
    return;

  printer << " = ";
  printAttribute(printer, namedAttr.getValue());
}

static void printOptionalAttrDict(
    OpAsmPrinter &printer, ArrayRef<NamedAttribute> attrs) {
  // If there are no attributes, then there is nothing to be done.
  if (attrs.empty())
    return;

  // Otherwise, print them all out in braces.
  printer << " {";
  llvm::interleaveComma(attrs, printer.getStream(),
      [&](NamedAttribute attr) { printNamedAttribute(printer, attr); });
  printer << '}';
}

//===----------------------------------------------------------------------===//
// Parser and Printer
//===----------------------------------------------------------------------===//

void ZHighStickifiedConstantOp::print(OpAsmPrinter &printer) {
  // If the result type is dynamic then it won't match the attribute type and
  // we fall back to printing as attribute dictionary at the end.
  Type resultType = getResult().getType();
  if (auto attr = getValue()) {
    // ZHighStickifiedConstant value must be ElementsAttr, but not
    // SparseElementsAttr.
    auto elements = mlir::cast<ElementsAttr>(*attr);
    assert(!mlir::isa<SparseElementsAttr>(elements) &&
           "ZHighStickifiedConstant value cannot be sparse");
    if (elements.getType() == resultType) {
      printer << ' ';
      printAttribute(printer, elements);
      return;
    }
  }
  // Fallback if there's something funny: no value or sparse_value attribute,
  // or types mismatch.
  printOptionalAttrDict(printer, (*this)->getAttrs());
  printer << " : " << resultType;
}

ParseResult ZHighStickifiedConstantOp::parse(
    OpAsmParser &parser, OperationState &result) {
  Attribute attr;
  Type type;
  // First try to parse attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // If there is no attribute dictionary, the parse above succeeds parsing
  // nothing. We detect this case by the absence of result attributes.
  if (result.attributes.empty()) {
    // Try to parse an ElementsAttr.
    OptionalParseResult opt = parser.parseOptionalAttribute(attr, type);
    if (opt.has_value()) {
      if (*opt)
        return failure();
      const char *name = "value";
      result.addAttribute(name, attr);
      result.addTypes({mlir::cast<ElementsAttr>(attr).getType()});
      return success();
    }
    // No sparse_value or value attr, so attribute dictionary really is empty.
  }
  if (parser.parseColonType(type))
    return failure();
  result.addTypes({type});
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
