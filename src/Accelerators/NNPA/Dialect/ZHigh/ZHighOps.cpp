/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZHighOps.cpp - ONNX Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZHigh operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include <math.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "zdnn.h"

using namespace mlir;

namespace mlir {

LogicalResult OpTrait::impl::verifySameOperandsAndResultLayout(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) ||
      failed(verifyAtLeastNResults(op, 1)))
    return failure();

  onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout layout =
      onnx_mlir::zhigh::getZTensorLayout(op->getResult(0).getType());

  if (layout == onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::UNDEFINED)
    return success();

  for (auto result : llvm::drop_begin(op->getResults())) {
    if (onnx_mlir::zhigh::getZTensorLayout(result.getType()) != layout)
      return op->emitOpError()
             << "requires the same layout for all operands and results";
  }
  for (auto oprd : op->getOperands()) {
    if (onnx_mlir::zhigh::getZTensorLayout(oprd.getType()) != layout)
      return op->emitOpError()
             << "requires the same layout for all operands and results";
  }
  return success();
}

} // namespace mlir

namespace onnx_mlir {
namespace zhigh {

std::vector<Type> getZHighAuxSplitResultType(
    Value input, int64_t axis, ArrayAttr split) {
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  std::vector<Type> outputTypes;
  if (split.size() == 0) {
    llvm_unreachable("Unsupported split (size==0)");
  } else {
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(input.getType()).getShape();
    int64_t splitNum = split.size();
    for (int i = 0; i < splitNum; i++) {
      SmallVector<int64_t> outputShape;
      for (unsigned int dim = 0; dim < inputShape.size(); dim++) {
        outputShape.emplace_back(
            (dim == axis) ? mlir::cast<IntegerAttr>(split[dim]).getInt()
                          : inputShape[dim]);
      }
      outputTypes.emplace_back(RankedTensorType::get(outputShape, elementType));
    }
  }
  return outputTypes;
}

//===----------------------------------------------------------------------===//
// ZHigh Attribute
//===----------------------------------------------------------------------===//

Attribute ZTensorEncodingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};

  ZTensorEncodingAttr::DataLayout dataLayout =
      ZTensorEncodingAttr::DataLayout::UNDEFINED;

  ZTensorEncodingAttr::QuantizedType quantizedType =
      ZTensorEncodingAttr::QuantizedType::UNDEFINED;

  // Process the data from the parsed dictionary value into struct-like data.
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "dataLayout") {
      StringAttr layoutAttr = mlir::dyn_cast<StringAttr>(attr.getValue());
      if (!layoutAttr) {
        parser.emitError(
            parser.getNameLoc(), "expected a string value for data layout");
        return {};
      }
      StringRef strVal = layoutAttr.getValue();
      if (strVal.equals_insensitive(LAYOUT_1D)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_1D;
      } else if (strVal.equals_insensitive(LAYOUT_2D)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_2D;
      } else if (strVal.equals_insensitive(LAYOUT_2DS)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_2DS;
      } else if (strVal.equals_insensitive(LAYOUT_3D)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_3D;
      } else if (strVal.equals_insensitive(LAYOUT_3DS)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_3DS;
      } else if (strVal.equals_insensitive(LAYOUT_4D)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_4D;
      } else if (strVal.equals_insensitive(LAYOUT_4DS)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::_4DS;
      } else if (strVal.equals_insensitive(LAYOUT_NCHW)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::NCHW;
      } else if (strVal.equals_insensitive(LAYOUT_NHWC)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::NHWC;
      } else if (strVal.equals_insensitive(LAYOUT_HWCK)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::HWCK;
      } else if (strVal.equals_insensitive(LAYOUT_FICO)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::FICO;
      } else if (strVal.equals_insensitive(LAYOUT_ZRH)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::ZRH;
      } else if (strVal.equals_insensitive(LAYOUT_BFICO)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::BFICO;
      } else if (strVal.equals_insensitive(LAYOUT_BZRH)) {
        dataLayout = ZTensorEncodingAttr::DataLayout::BZRH;
      } else {
        parser.emitError(
            parser.getNameLoc(), "unexpected dimension level type: ")
            << strVal;
        return {};
      }
    } else if (attr.getName() == "quantizedType") {
      StringAttr qtypeAttr = mlir::dyn_cast<StringAttr>(attr.getValue());
      if (!qtypeAttr) {
        parser.emitError(
            parser.getNameLoc(), "expected a string value for quantized type");
        return {};
      }
      StringRef strVal = qtypeAttr.getValue();
      if (strVal.equals_insensitive(QTYPE_DLFLOAT16)) {
        quantizedType = ZTensorEncodingAttr::QuantizedType::DLFLOAT16;
      } else if (strVal.equals_insensitive(QTYPE_INT8)) {
        quantizedType = ZTensorEncodingAttr::QuantizedType::INT8;
      } else if (strVal.equals_insensitive(QTYPE_WEIGHTS)) {
        quantizedType = ZTensorEncodingAttr::QuantizedType::WEIGHTS;
      } else if (strVal.equals_insensitive(QTYPE_UNDEFINED)) {
        quantizedType = ZTensorEncodingAttr::QuantizedType::UNDEFINED;
      } else {
        parser.emitError(parser.getNameLoc(), "unexpected quantized type: ")
            << strVal;
        return {};
      }
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().str();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<ZTensorEncodingAttr>(
      parser.getContext(), dataLayout, quantizedType);
}

void ZTensorEncodingAttr::print(AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{dataLayout = ";
  switch (getDataLayout()) {
  case DataLayout::_1D:
    printer << "\"" << LAYOUT_1D << "\"";
    break;
  case DataLayout::_2D:
    printer << "\"" << LAYOUT_2D << "\"";
    break;
  case DataLayout::_2DS:
    printer << "\"" << LAYOUT_2DS << "\"";
    break;
  case DataLayout::_3D:
    printer << "\"" << LAYOUT_3D << "\"";
    break;
  case DataLayout::_3DS:
    printer << "\"" << LAYOUT_3DS << "\"";
    break;
  case DataLayout::_4D:
    printer << "\"" << LAYOUT_4D << "\"";
    break;
  case DataLayout::_4DS:
    printer << "\"" << LAYOUT_4DS << "\"";
    break;
  case DataLayout::NCHW:
    printer << "\"" << LAYOUT_NCHW << "\"";
    break;
  case DataLayout::NHWC:
    printer << "\"" << LAYOUT_NHWC << "\"";
    break;
  case DataLayout::HWCK:
    printer << "\"" << LAYOUT_HWCK << "\"";
    break;
  case DataLayout::FICO:
    printer << "\"" << LAYOUT_FICO << "\"";
    break;
  case DataLayout::ZRH:
    printer << "\"" << LAYOUT_ZRH << "\"";
    break;
  case DataLayout::BFICO:
    printer << "\"" << LAYOUT_BFICO << "\"";
    break;
  case DataLayout::BZRH:
    printer << "\"" << LAYOUT_BZRH << "\"";
    break;
  case DataLayout::UNDEFINED:
    llvm_unreachable("Unexpected data layout");
    break;
  }

  // QuantizedType is optional.
  switch (getQuantizedType()) {
  case QuantizedType::DLFLOAT16:
    printer << ", quantizedType = ";
    printer << "\"" << QTYPE_DLFLOAT16 << "\"";
    break;
  case QuantizedType::INT8:
    printer << ", quantizedType = ";
    printer << "\"" << QTYPE_INT8 << "\"";
    break;
  case QuantizedType::WEIGHTS:
    printer << ", quantizedType = ";
    printer << "\"" << QTYPE_WEIGHTS << "\"";
    break;
  case QuantizedType::UNDEFINED:
    break;
  default:
    llvm_unreachable("Unexpected quantized type");
    break;
  }
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ZHighDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void ZHighDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.cpp.inc"
      >();
}

} // namespace zhigh
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
// Keep this part at the end of the file.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighAttributes.cpp.inc"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighDialect.cpp.inc"
