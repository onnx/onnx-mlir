/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXAttributes.cpp -----------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttributeStorage.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// ONNX Attribute: ONNXTensorEncodingAttr
//===----------------------------------------------------------------------===//

/*
  For the moment, the x and y factor are explicitly encoded in the
  ONNXLayoutHelper.hpp LAYOUT strings. These strings are used to recognize which
  layout is used. But once the pattern is recognized, we use the encoding's
  layout to represent the high level type of encoding, and the encoding's x and
  y factor integer to represent the unroll factors. That way, the code that use
  these encoding does not need to be specialized for a specific value of x or y
  factor, it just looks at the embedding x and y factor integers to perform the
  proper unrolling.

  In other words, the string to encoding is manually encoded by fixed string
  that needs to be customized for each x and y factor that are accepted. But
  once that is done, the code is fully parametric in terms of the encoding
  attribute xFactor and yFactor.
*/

Attribute ONNXTensorEncodingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};

  ONNXTensorEncodingAttr::DataLayout dataLayout =
      ONNXTensorEncodingAttr::DataLayout::STANDARD;
  int64_t xFactor = 0;
  int64_t yFactor = 0;

  // Process the data from the parsed dictionary value into struct-like data.
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "dataLayout") {
      StringAttr layoutAttr = mlir::dyn_cast<StringAttr>(attr.getValue());
      if (!layoutAttr) {
        parser.emitError(
            parser.getNameLoc(), "expected a string value for data layout");
        return {};
      }
      if (!onnx_mlir::convertStringToONNXCustomTensorDataLayout(
              layoutAttr, dataLayout, xFactor, yFactor)) {
        parser.emitError(
            parser.getNameLoc(), "unexpected data layout attribute value: ")
            << layoutAttr.getValue();
        return {};
      }
    } else { // Attribute different than "dataLayout".
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().str();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<ONNXTensorEncodingAttr>(
      parser.getContext(), dataLayout, xFactor, yFactor);
}

void ONNXTensorEncodingAttr::print(AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{dataLayout = ";
  StringRef layoutStr = onnx_mlir::convertONNXTensorDataLayoutToString(
      getDataLayout(), getXFactor(), getYFactor());
  printer << "\"" << layoutStr.str() << "\"";
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ONNX Attributes: TableGen generated implementation
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"

// See explanation in ONNXDialect::initialize() in ONNXDialect.cpp.
void ONNXDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
      >();
  addAttributes<DisposableElementsAttr>();
  addInterface<DisposablePool>(getContext());
}

/// Parse an attribute registered to this dialect.
Attribute ONNXDialect::parseAttribute(
    DialectAsmParser &parser, Type type) const {
  // generatedAttributeParser is generated in ONNXAttributes.cpp.inc
  StringRef attrTag;
  if (Attribute attr;
      generatedAttributeParser(parser, &attrTag, type, attr).has_value())
    return attr;
  if (attrTag == DisposableElementsAttr::getMnemonic()) {
    auto shapedTy = mlir::cast<ShapedType>(type);
    if (auto membuf = DisposableElementsAttr::parse(parser, shapedTy))
      return OnnxElementsAttrBuilder(type.getContext())
          .fromMemoryBuffer(shapedTy, std::move(membuf));
    else
      return {};
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown attribute `" << attrTag << "` in dialect `ONNX`";
  return {};
}

/// Print an attribute registered to this dialect.
void ONNXDialect::printAttribute(
    Attribute attr, DialectAsmPrinter &printer) const {
  // generatedAttributePrinter is generated in ONNXAttributes.cpp.inc
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
  if (auto elements = mlir::dyn_cast<DisposableElementsAttr>(attr))
    elements.printWithoutType(printer);
}
