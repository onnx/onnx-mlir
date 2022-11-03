/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//
// Explanation: the type implementation is used in dialect initialization.
// If ONNXTypes.cpp.inc is included in ONNXTypes.cpp, compilation error occurs.
#define GET_TYPEDEF_CLASSES
#include "src/Dialect/ONNX/ONNXTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ONNXDialect initialization
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void ONNXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/ONNX/ONNXTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ONNX Attribute
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
      if (!convertStringToONNXCustomTensorDataLayout(
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
  StringRef layoutStr = convertONNXTensorDataLayoutToString(
      getDataLayout(), getXFactor(), getYFactor());
  printer << "\"" << layoutStr.str() << "\"";
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ONNX Type: SeqType
//===---------------------------------------------------------------------===//

mlir::Type SeqType::parse(mlir::AsmParser &parser) {
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation())
        << "failed to parse !onnx.Seq type";
    return Type();
  }

  return get(elementType, -1);
}

void SeqType::print(mlir::AsmPrinter &printer) const {
  // Previous implementation did not print/parse the length field
  // May add the field in future
  printer << "<" << getElementType() << ">";
}

//===----------------------------------------------------------------------===//
// Unsupported Operations
//===---------------------------------------------------------------------===//

// Operations for which shape inference has not been implemented yet
// If you add the implementation for one op, move it out of this section
// Also please add test case in test/mlir/onnx/onnx_shape_inference.mlir
// Followed by the implementation of lowering to Krnl and
// Enable the corresponding node test in check-onnx-backend

#define NOT_IMPLEMENTED_INFER_SHAPE(T)                                         \
  LogicalResult T::inferShapes(                                                \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return emitError(NOT_IMPLEMENTED_MESSAGE);                                 \
  }

// Listed alphabetically.
NOT_IMPLEMENTED_INFER_SHAPE(ONNXAdagradOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXAdamOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXArrayFeatureExtractorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXBatchNormalizationOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXBinarizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXCastMapOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV12Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXClipV6Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXDetOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXDictVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXFeatureVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXGradientOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXGridSampleOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXImputerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXIsInfOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLabelEncoderOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLinearClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLinearRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXLpPoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMaxPoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMaxUnpoolOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMomentumOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXMultinomialOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXNegativeLogLikelihoodLossOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXPadV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXPadV2Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXRandomUniformLikeOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXRandomUniformOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXResizeV10Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXResizeV11Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSVMClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSVMRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXSoftmaxCrossEntropyLossOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXStringNormalizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTfIdfVectorizerOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTreeEnsembleClassifierOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXTreeEnsembleRegressorOp)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXUpsampleV7Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXUpsampleV9Op)
NOT_IMPLEMENTED_INFER_SHAPE(ONNXZipMapOp)

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
