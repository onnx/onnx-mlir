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
    elements.printWithoutType(printer);
  }
}

namespace onnx_mlir {

namespace {
using EltFn = function_ref<void(unsigned)>;

// Copied from mlir/lib/IR/AsmPrinter.cpp:
void printDenseElementsAttrImpl(
    bool isSplat, ShapedType type, raw_ostream &os, EltFn printEltFn) {
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
} // namespace

// adapted from AsmPrinter::Impl::printDenseIntOrFPElementsAttr:
void printIntOrFPElementsAttrAsDenseWithoutType(
    ElementsAttr attr, AsmPrinter &printer) {
  printer << "dense<";
  auto &os = printer.getStream();
  auto type = attr.getType();
  auto elTy = type.getElementType();
  if (auto disposable = attr.dyn_cast<DisposableElementsAttr>()) {
    // Sadly DisposableElementsAttr::value_begin() is too slow so we need to
    // access the data in bulk with getWideNums() or getRawBytes().
#ifdef DISPOSABLE_ELEMENTS_ATTR_ALLOW_HEX_PRINT
    auto bytes = disposable.getRawBytes();
    os << "\"0x" << llvm::toHex(asStringRef(bytes.get())) << "\"";
#else
    auto dtype = disposable.getDType();
    auto buf = disposable.getWideNums();
    auto nums = buf.get();
    EltFn b = [&](unsigned idx) { os << (nums[idx].u64 ? "true" : "false"); };
    EltFn u = [&](unsigned idx) { os << nums[idx].u64; };
    EltFn s = [&](unsigned idx) { os << nums[idx].i64; };
    EltFn f = [&](unsigned idx) { printer << nums[idx].toAPFloat(dtype); };
    printDenseElementsAttrImpl(attr.isSplat(), type, os,
        elTy.isIntOrIndex()
            ? elTy.isInteger(1) ? b : (elTy.isUnsignedInteger() ? u : s)
            : f);
#endif
  } else {
    if (elTy.isIntOrIndex()) {
      auto it = attr.value_begin<APInt>();
      EltFn b = [&](unsigned idx) {
        os << ((*(it + idx)).getBoolValue() ? "true" : "false");
      };
      EltFn u = [&](unsigned idx) { os << (*(it + idx)).getZExtValue(); };
      EltFn s = [&](unsigned idx) { os << (*(it + idx)).getSExtValue(); };
      printDenseElementsAttrImpl(attr.isSplat(), type, os,
          elTy.isInteger(1) ? b : (elTy.isUnsignedInteger() ? u : s));
    } else {
      assert(elTy.isa<FloatType>() && "unexpected element type");
      auto it = attr.value_begin<APFloat>();
      EltFn f = [&](unsigned idx) { printer << *(it + idx); };
      printDenseElementsAttrImpl(attr.isSplat(), type, os, f);
    }
  }
  printer << '>';
}

void printIntOrFPElementsAttrAsDense(ElementsAttr attr, AsmPrinter &printer) {
  printIntOrFPElementsAttrAsDenseWithoutType(attr, printer);
  printer << " : " << attr.getType();
}

} // namespace onnx_mlir
