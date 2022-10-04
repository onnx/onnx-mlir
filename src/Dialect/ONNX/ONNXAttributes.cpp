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
#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttributeStorage.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

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
      StringAttr layoutAttr = attr.getValue().dyn_cast<StringAttr>();
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
  addAttributes<DisposableElementsAttr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
      >();
}

/// Parse an attribute registered to this dialect.
Attribute ONNXDialect::parseAttribute(
    DialectAsmParser &parser, Type type) const {
  // generatedAttributeParser is generated in ONNXAttributes.cpp.inc
  StringRef attrTag;
  if (Attribute attr;
      generatedAttributeParser(parser, &attrTag, type, attr).has_value())
    return attr;
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
  if (auto elements = attr.dyn_cast<DisposableElementsAttr>()) {
    elements.printWithoutType(printer.getStream());
  }
}

namespace onnx_mlir {

namespace {
void printDenseFloatElement(const APFloat &value, raw_ostream &os, Type type) {
  FloatAttr::get(type, value).print(os, /*elideType=*/true);
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
void printDenseIntElement(const APInt &value, raw_ostream &os, Type type) {
  if (type.isInteger(1))
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, !type.isUnsignedInteger());
}

// Copied from mlir/lib/IR/AsmPrinter.cpp:
void printDenseElementsAttrImpl(bool isSplat, ShapedType type, raw_ostream &os,
    function_ref<void(unsigned)> printEltFn) {
  // Special case for 0-d and splat tensors.
  if (isSplat)
    return printEltFn(0);

  // Special case for degenerate tensors.
  auto numElements = type.getNumElements();
  if (numElements == 0)
    return;

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  int64_t rank = type.getRank();
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto shape = type.getShape();
  auto bumpCounter = [&] {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = numElements; idx != e; ++idx) {
    if (idx != 0)
      os << ", ";
    while (openBrackets++ < rank)
      os << '[';
    openBrackets = rank;
    printEltFn(idx);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

template <typename Iterator>
bool checkIfSplat(ElementsAttr attr, Iterator valueIt) {
  if (attr.isSplat())
    return true;
  if (attr.isa<DenseElementsAttr>()) {
    // DenseElementsAttr always reports accurate isSplat() so no need to check
    // contents when isSplat() returned false.
    return false;
  }
  int64_t numElements = attr.getNumElements();
  if (numElements == 0)
    return false;
  auto first = *valueIt;
  for (int64_t i = 1; i < numElements; ++i) {
    if (first != *++valueIt)
      return false;
  }
  return true;
}
} // namespace

// adapted from AsmPrinter::Impl::printDenseIntOrFPElementsAttr:
void printIntOrFPElementsAttrAsDenseWithoutType(
    ElementsAttr attr, raw_ostream &os) {
  auto type = attr.getType();
  auto elementType = type.getElementType();
  os << "dense<";
  if (elementType.isIntOrIndex()) {
    auto valueIt = attr.value_begin<APInt>();
    bool isSplat = checkIfSplat(attr, valueIt);
    printDenseElementsAttrImpl(isSplat, type, os, [&](unsigned index) {
      printDenseIntElement(*(valueIt + index), os, elementType);
    });
  } else {
    assert(elementType.isa<FloatType>() && "unexpected element type");
    auto valueIt = attr.value_begin<APFloat>();
    bool isSplat = checkIfSplat(attr, valueIt);
    printDenseElementsAttrImpl(isSplat, type, os, [&](unsigned index) {
      printDenseFloatElement(*(valueIt + index), os, elementType);
    });
  }
  os << '>';
}

void printIntOrFPElementsAttrAsDense(ElementsAttr attr, raw_ostream &os) {
  printIntOrFPElementsAttrAsDenseWithoutType(attr, os);
  os << " : " << attr.getType();
}

} // namespace onnx_mlir
