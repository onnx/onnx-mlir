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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Endian.h"

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
// ONNX Attribute: DisposablElementsAttr
//===----------------------------------------------------------------------===//

namespace {
constexpr StringLiteral getDisposablElementsAttrMnemonic() {
  return {"dense_disposable"};
}

#if 1
Attribute parseDisposablElementsAttr(AsmParser &parser, Type type) {
  llvm_unreachable("TODO: implement");
}

void printDisposablElementsAttr(
    AsmPrinter &printer, DisposableElementsAttr disposable) {
  llvm_unreachable("TODO: implement");
}
#else
Attribute parseDisposablElementsAttr(AsmParser &parser, Type type) {
  return DisposableElementsAttr::parse(
      parser, type, [&](size_t id, ElementsAttr &elms) -> ParseResult {
        return OnnxElementsAttrBuilder(type.getContext())
            .parseElements(parser, cast<ShapedType>(type), id, elms);
      });
}

void printDisposablElementsAttr(
    AsmPrinter &printer, DisposableElementsAttr disposable) {
  disposable.printWithoutType(printer);
}

static Attribute parse(AsmParser &parser, Type type,
    function_ref<ParseResult(size_t, ElementsAttr &)> parseElements);

void printWithoutType(AsmPrinter &printer) const;

void printAsDenseElementsAttr(AsmPrinter &printer) const;

namespace {
// Perform byte swap if system endianness is BE and elements are multi-byte.
bool shouldSwapLEBytes(unsigned elementByteWidth) {
  return elementByteWidth > 1 && llvm::support::endian::system_endianness() !=
                                     llvm::support::endianness::little;
}
} // namespace

/*static*/
Attribute DisposableElementsAttr::parse(AsmParser &parser, Type type,
    function_ref<ParseResult(size_t, ElementsAttr &)> parseElements) {
  size_t id = 0; // The parsed id.
  ElementsAttr elms;
  if (parser.parseLess() || parser.parseInteger(id) || parser.parseColon() ||
      parseElements(id, elms) || parser.parseGreater())
    return nullptr;

  return elms;
}

void DisposableElementsAttr::printWithoutType(AsmPrinter &printer) const {
  // It would be ideal if we could read the printer flags from printer instead
  // of constructing them here, because printer may have been constructed with
  // an override of elideLargeElementsAttrs which we cannot see here.
  // Oh well, at least OpPrintingFlags().shouldElideElementsAttr(ElementsAttr)
  // lets us respect the --mlir-elide-elementsattrs-if-larger command line flag.
  static OpPrintingFlags printerFlags{};
  printer << getMnemonic() << "<" << getImpl()->id << ":";
  if (!printerFlags.shouldElideElementsAttr(*this)) {
    auto rawBytes = getRawBytes();
    SmallVector<char> buffer;
    ArrayRef<char> bytes;
    if (!shouldSwapLEBytes(getIntOrFloatByteWidth(getElementType()))) {
      bytes = rawBytes.get();
    } else {
      // Reorder raw bytes to little-endian on big-endian platforms:
      buffer.resize_for_overwrite(rawBytes.get().size());
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawBytes.get(), buffer, getType());
      ArrayRef<char> bufferRef(buffer);
      bytes = bufferRef;
    }
    printer << "\"0x" << llvm::toHex(castArrayRef<uint8_t>(bytes)) << "\"";
  } else {
    printer << "__elided__";
  }
  printer << ">";
}

mlir::ParseResult parseElements(mlir::AsmParser &parser, mlir::ShapedType type,
    size_t id, mlir::ElementsAttr &elms);

ParseResult ElementsAttrBuilder::parseElements(
    AsmParser &parser, ShapedType type, size_t id, ElementsAttr &elms) {
  std::string str;
  if (parser.parseString(&str))
    return failure();
  if (!parser.parseOptionalColon()) {
    uint64_t offset = 0;
    uint64_t length = 0;
    if (parser.parseInteger(offset) || parser.parseColon() ||
        parser.parseInteger(length))
      return failure();
    return parser.emitError(parser.getCurrentLocation(), "TODO: implement");
  } else {
    StringRef hex = str;
    std::string bytes;
    if (!hex.consume_front("0x") || (hex.size() & 1) ||
        !llvm::tryGetFromHex(hex, bytes))
      return parser.emitError(
          parser.getCurrentLocation(), "ill-formed hex string");
    if (bytes.size() != static_cast<size_t>(getSizeInBytes(type)))
      return parser.emitError(
          parser.getCurrentLocation(), "data size doesn't match type size");
    if (!shouldSwapLEBytes(getIntOrFloatByteWidth(type.getElementType()))) {
      elms =
          fromMemoryBuffer(type, llvm::MemoryBuffer::getMemBufferCopy(bytes));
    } else {
      // Reorder bytes from little-endian on big-endian platforms:
      std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
          llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes.size());
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          {bytes.data(), bytes.size()}, writeBuffer->getBuffer(), type);
      elms = fromMemoryBuffer(type, std::move(writeBuffer));
    }
    return success();
  }
}
#endif
} // namespace

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
  Attribute attr;
  StringRef attrTag;
  if (generatedAttributeParser(parser, &attrTag, type, attr).has_value())
    return attr;
  if (attrTag == getDisposablElementsAttrMnemonic()) {
    return parseDisposablElementsAttr(parser, type);
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
  if (auto disposable = attr.dyn_cast<DisposableElementsAttr>())
    printDisposablElementsAttr(printer, disposable);
}
