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

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighShapeHelper.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Support/Diagnostic.hpp"
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

std::vector<mlir::Type> getZHighAuxSplitResultType(
    Value input, int64_t axis, ArrayAttr split) {
  Type elementType = input.getType().cast<ShapedType>().getElementType();
  std::vector<mlir::Type> outputTypes;
  if (split.size() == 0) {
    llvm_unreachable("Unsupported split (size==0)");
  } else {
    ArrayRef<int64_t> inputShape =
        input.getType().cast<RankedTensorType>().getShape();
    int64_t splitNum = split.size();
    for (int i = 0; i < splitNum; i++) {
      SmallVector<int64_t> outputShape;
      for (unsigned int dim = 0; dim < inputShape.size(); dim++) {
        outputShape.emplace_back((dim == axis)
                                     ? split[dim].cast<IntegerAttr>().getInt()
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

  // Process the data from the parsed dictionary value into struct-like data.
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "dataLayout") {
      StringAttr layoutAttr = attr.getValue().dyn_cast<StringAttr>();
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
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().str();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<ZTensorEncodingAttr>(
      parser.getContext(), dataLayout);
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
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ZHighDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ZHighDialect::ZHighDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<ZHighDialect>()) {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Define ZHigh Op's functions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StickOp.

void ZHighStickOp::build(
    OpBuilder &builder, OperationState &state, Value input, StringAttr layout) {
  Type resType = builder.getNoneType();
  if (!input.getType().isa<NoneType>()) {
    ShapedType inputType = input.getType().cast<ShapedType>();
    int64_t rank = -1;
    if (inputType.hasRank()) {
      rank = inputType.getRank();
      ZTensorEncodingAttr::DataLayout dataLayout;
      if (layout)
        dataLayout = convertStringAttrToDataLayout(layout);
      else {
        dataLayout = getDataLayoutByRank(rank);
        // Create a layout attribute.
        layout = convertDataLayoutToStringAttr(builder, dataLayout);
      }
      resType = RankedTensorType::get(inputType.getShape(),
          inputType.getElementType(),
          ZTensorEncodingAttr::get(builder.getContext(), dataLayout));
    } else {
      resType = UnrankedTensorType::get(inputType.getElementType());
    }
  }
  build(builder, state, resType, input, layout);
}

LogicalResult ZHighStickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(In()))
    return success();

  OpBuilder builder(this->getContext());
  ShapedType inputType = In().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  StringAttr layout = layoutAttr();
  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToDataLayout(layout);
  else
    dataLayout = getDataLayoutByRank(inputShape.size());
  RankedTensorType resType =
      RankedTensorType::get(inputType.getShape(), inputType.getElementType(),
          ZTensorEncodingAttr::get(this->getContext(), dataLayout));
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// StickForLSTMOp.

LogicalResult ZHighStickForLSTMOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(f_gate()) && !hasRankedType(i_gate()) &&
      !hasRankedType(c_gate()) && !hasRankedType(o_gate()))
    return success();

  ZHighStickForLSTMOpAdaptor operandAdaptor(*this);
  ZHighStickForLSTMOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError(
        "Failed to scan ZHigh StickForLSTM parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::FICO);
  RankedTensorType resType =
      RankedTensorType::get(outputDims, elementType, encoding);
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// StickForGRUOp.

LogicalResult ZHighStickForGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(z_gate()) && !hasRankedType(r_gate()) &&
      !hasRankedType(h_gate()))
    return success();

  ZHighStickForGRUOpAdaptor operandAdaptor(*this);
  ZHighStickForGRUOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError(
        "Failed to scan ZHigh StickForGRU parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::ZRH);
  RankedTensorType resType =
      RankedTensorType::get(outputDims, elementType, encoding);
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// UnstickOp

void ZHighUnstickOp::build(
    OpBuilder &builder, OperationState &state, Value input) {
  Type resType;
  ShapedType inputType = input.getType().cast<ShapedType>();
  if (hasRankedType(input))
    resType =
        RankedTensorType::get(inputType.getShape(), inputType.getElementType());
  else
    resType = UnrankedTensorType::get(inputType.getElementType());
  build(builder, state, resType, input);
}

LogicalResult ZHighUnstickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(In()))
    return success();

  Builder builder(this->getContext());
  ShapedType inputType = In().getType().cast<ShapedType>();
  RankedTensorType resType =
      RankedTensorType::get(inputType.getShape(), inputType.getElementType());
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// AddOp

LogicalResult ZHighAddOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// SubOp

LogicalResult ZHighSubOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// MulOp

LogicalResult ZHighMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// DivOp

LogicalResult ZHighDivOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// MinOp

LogicalResult ZHighMinOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// MaxOp

LogicalResult ZHighMaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// LogOp

LogicalResult ZHighLogOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// ExpOp

LogicalResult ZHighExpOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// ReluOp

LogicalResult ZHighReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// TanhOp

LogicalResult ZHighTanhOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// SigmoiOp

LogicalResult ZHighSigmoidOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp

LogicalResult ZHighSoftmaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()))
    return success();

  getResult().setType(X().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormOp

LogicalResult ZHighBatchNormOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  getResult().setType(input().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// MeanReduce2DOp

LogicalResult ZHighMeanReduce2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = inputType.getShape();
  // Input is NHWC, and H and W are reduction dimensions.
  Type resType = RankedTensorType::get({shape[0], 1, 1, shape[3]},
      inputType.getElementType(), inputType.getEncoding());
  getResult().setType(resType);
  return success();
}

//===----------------------------------------------------------------------===//
// MatMulOp

LogicalResult ZHighMatMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(X()) || !hasRankedType(Y()))
    return success();

  ZHighMatMulOpAdaptor operandAdaptor(*this);
  ZHighMatMulOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh MatMul parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  Type elementType = getResult().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding;
  if (outputDims.size() == 2)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_2D);
  else if (outputDims.size() == 3)
    encoding = ZTensorEncodingAttr::get(
        this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS);
  RankedTensorType resType =
      RankedTensorType::get(outputDims, elementType, encoding);
  getResult().setType(resType);
  return success();
}

LogicalResult ZHighMatMulOp::verify() {
  ZHighMatMulOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value X = operandAdaptor.X();
  Value Y = operandAdaptor.Y();
  Value B = operandAdaptor.B();

  // Get layouts.
  ZTensorEncodingAttr::DataLayout xLayout = getZTensorLayout(X.getType());
  ZTensorEncodingAttr::DataLayout yLayout = getZTensorLayout(Y.getType());
  // Bias can be None.
  ZTensorEncodingAttr::DataLayout bLayout;
  bool hasBias = !B.getType().isa<NoneType>();
  if (hasBias)
    bLayout = getZTensorLayout(B.getType());

  // X must be 2D or 3DS.
  if (!((xLayout == ZTensorEncodingAttr::DataLayout::_2D) ||
          (xLayout == ZTensorEncodingAttr::DataLayout::_3DS)))
    return failure();

  // If X is 2D, Y must be 2D and B must be 1D
  if (xLayout == ZTensorEncodingAttr::DataLayout::_2D) {
    if (!(yLayout == ZTensorEncodingAttr::DataLayout::_2D))
      return failure();
    if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_1D))
      return failure();
  }

  // X is 3DS, valid types for (X, Y, B) are (3DS, 3DS, 2DS) or (3DS, 2D, 1D)
  if (xLayout == ZTensorEncodingAttr::DataLayout::_3DS) {
    if (yLayout == ZTensorEncodingAttr::DataLayout::_3DS) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_2DS))
        return failure();
    } else if (yLayout == ZTensorEncodingAttr::DataLayout::_2D) {
      if (hasBias && !(bLayout == ZTensorEncodingAttr::DataLayout::_1D))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LSTMOp

LogicalResult ZHighLSTMOp::verify() {
  ZHighLSTMOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value W = operandAdaptor.input_weights();
  Value R = operandAdaptor.hidden_weights();
  Value WB = operandAdaptor.input_bias();
  Value RB = operandAdaptor.hidden_bias();

  // Hidden size attribute.
  int64_t hiddenSize = hidden_size();

  // Verify hidden size in W.
  if (hasRankedType(W)) {
    int64_t dim2 = W.getType().cast<RankedTensorType>().getShape()[2];
    if (dim2 != -1 && dim2 != hiddenSize * 4)
      return failure();
  }

  // Verify hidden size in R.
  if (hasRankedType(R)) {
    int64_t dim1 = R.getType().cast<RankedTensorType>().getShape()[1];
    int64_t dim2 = R.getType().cast<RankedTensorType>().getShape()[2];
    if (dim1 != -1 && dim1 != hiddenSize)
      return failure();
    if (dim2 != -1 && dim2 != hiddenSize * 4)
      return failure();
  }

  // Verify hidden size in WB.
  if (!WB.getType().isa<NoneType>() && hasRankedType(WB)) {
    int64_t dim1 = WB.getType().cast<RankedTensorType>().getShape()[1];
    if (dim1 != -1 && dim1 != hiddenSize * 4)
      return failure();
  }

  // Verify hidden size in RB.
  if (!RB.getType().isa<NoneType>() && hasRankedType(RB)) {
    int64_t dim1 = RB.getType().cast<RankedTensorType>().getShape()[1];
    if (dim1 != -1 && dim1 != hiddenSize * 4)
      return failure();
  }

  return success();
}

LogicalResult ZHighLSTMOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(hidden_weights()))
    return success();

  Builder builder(getContext());
  ZHighLSTMOpAdaptor operandAdaptor(*this);
  ZHighLSTMOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh LSTM parameters successfully");

  SmallVector<int64_t, 4> hnOutputDims, cfOutputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), hnOutputDims);
  IndexExpr::getShape(shapeHelper.dimsForOutput(1), cfOutputDims);

  // Output type is 3DS.
  Type elementType = input().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_4DS);
  RankedTensorType hnType =
      RankedTensorType::get(hnOutputDims, elementType, encoding);
  RankedTensorType cfType =
      RankedTensorType::get(cfOutputDims, elementType, encoding);
  getResults()[0].setType(hnType);
  getResults()[1].setType(cfType);
  return success();
}

//===----------------------------------------------------------------------===//
// GRUOp

LogicalResult ZHighGRUOp::verify() {
  ZHighGRUOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value W = operandAdaptor.input_weights();
  Value R = operandAdaptor.hidden_weights();
  Value WB = operandAdaptor.input_bias();
  Value RB = operandAdaptor.hidden_bias();

  // Hidden size attribute.
  int64_t hiddenSize = hidden_size();

  // Verify hidden size in W.
  if (hasRankedType(W)) {
    int64_t dim2 = W.getType().cast<RankedTensorType>().getShape()[2];
    if (dim2 != -1 && dim2 != hiddenSize * 3)
      return failure();
  }

  // Verify hidden size in R.
  if (hasRankedType(R)) {
    int64_t dim1 = R.getType().cast<RankedTensorType>().getShape()[1];
    int64_t dim2 = R.getType().cast<RankedTensorType>().getShape()[2];
    if (dim1 != -1 && dim1 != hiddenSize)
      return failure();
    if (dim2 != -1 && dim2 != hiddenSize * 3)
      return failure();
  }

  // Verify hidden size in WB.
  if (!WB.getType().isa<NoneType>() && hasRankedType(WB)) {
    int64_t dim1 = WB.getType().cast<RankedTensorType>().getShape()[1];
    if (dim1 != -1 && dim1 != hiddenSize * 3)
      return failure();
  }

  // Verify hidden size in RB.
  if (!RB.getType().isa<NoneType>() && hasRankedType(RB)) {
    int64_t dim1 = RB.getType().cast<RankedTensorType>().getShape()[1];
    if (dim1 != -1 && dim1 != hiddenSize * 3)
      return failure();
  }

  return success();
}

LogicalResult ZHighGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(hidden_weights()))
    return success();

  Builder builder(getContext());
  ZHighGRUOpAdaptor operandAdaptor(*this);
  ZHighGRUOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh GRU parameters successfully");

  SmallVector<int64_t, 4> hnOutputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), hnOutputDims);
  Type elementType = input().getType().cast<ShapedType>().getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_4DS);
  RankedTensorType hnType =
      RankedTensorType::get(hnOutputDims, elementType, encoding);
  getResult().setType(hnType);

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp

LogicalResult ZHighConv2DOp::verify() {
  ZHighConv2DOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value K = operandAdaptor.input_kernel();
  Value B = operandAdaptor.input_bias();

  // Verify attributes.
  // - padding_type must be SAME_PADDING or VALID_PADDING.
  StringRef paddingType = padding_type();
  if (!(paddingType.equals_insensitive("SAME_PADDING") ||
          paddingType.equals_insensitive("VALID_PADDING")))
    return failure();
  // - act_func must be ACT_NONE or ACT_RELU.
  StringRef actFunc = act_func();
  if (!(actFunc.equals_insensitive("ACT_NONE") ||
          actFunc.equals_insensitive("ACT_RELU")))
    return failure();

  // Verify bias shape.
  if (!B.getType().isa<NoneType>() && hasRankedType(B) && hasRankedType(K)) {
    int64_t channelOutB = B.getType().cast<RankedTensorType>().getShape()[0];
    int64_t channelOutK = K.getType().cast<RankedTensorType>().getShape()[3];
    if ((channelOutB != -1) && (channelOutK != -1) &&
        (channelOutB != channelOutK))
      return failure();
  }

  // Verify kernel shape.
  ArrayAttr kernelShape = kernel_shape();
  int64_t attrKH = kernelShape[0].cast<IntegerAttr>().getInt();
  int64_t attrKW = kernelShape[1].cast<IntegerAttr>().getInt();
  if (hasRankedType(K)) {
    int64_t KH = K.getType().cast<RankedTensorType>().getShape()[0];
    int64_t KW = K.getType().cast<RankedTensorType>().getShape()[1];
    if (KH != -1 && KH != attrKH)
      return failure();
    if (KW != -1 && KW != attrKW)
      return failure();
  }

  return success();
}

LogicalResult ZHighConv2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(input_kernel()))
    return success();

  Builder builder(getContext());
  ZHighConv2DOpAdaptor operandAdaptor(*this);
  ZHighConv2DOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh Conv2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type resType = RankedTensorType::get(
      outputDims, inputType.getElementType(), inputType.getEncoding());
  getResult().setType(resType);

  return success();
}

//===----------------------------------------------------------------------===//
// MaxPool2DOp

LogicalResult ZHighMaxPool2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  Builder builder(getContext());
  ZHighMaxPool2DOpAdaptor operandAdaptor(*this);
  ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp, ZHighMaxPool2DOpAdaptor>
      shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh MaxPool2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type resType = RankedTensorType::get(
      outputDims, inputType.getElementType(), inputType.getEncoding());
  getResult().setType(resType);

  return success();
}

//===----------------------------------------------------------------------===//
// AvgPool2DOp

LogicalResult ZHighAvgPool2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  Builder builder(getContext());
  ZHighAvgPool2DOpAdaptor operandAdaptor(*this);
  ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp, ZHighAvgPool2DOpAdaptor>
      shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh AvgPool2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type resType = RankedTensorType::get(
      outputDims, inputType.getElementType(), inputType.getEncoding());
  getResult().setType(resType);

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp

LogicalResult ZHighConcatOp::verify() {
  ZHighConcatOpAdaptor operandAdaptor(*this);
  // Check all inputs.
  for (const auto &operand : operandAdaptor.getOperands()) {
    if (!hasRankedType(operand)) {
      // Won't be able to do any checking at this stage.
      return success();
    }
  }

  auto commonType =
      operandAdaptor.getOperands().front().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  if (axisIndex < -commonRank || axisIndex >= commonRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(**this, "axis",
        axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-commonRank, commonRank - 1));

  if (axisIndex < 0)
    axisIndex += commonRank;

  for (const auto &operand : operandAdaptor.getOperands()) {
    ArrayRef<int64_t> currShape =
        operand.getType().cast<RankedTensorType>().getShape();
    if ((int64_t)currShape.size() != commonRank)
      return emitError("Concat inputs must all have the same rank");
    for (int j = 0; j < commonRank; ++j) {
      if (j == axisIndex)
        continue;
      if (currShape[j] != -1 && commonShape[j] != -1 &&
          currShape[j] != commonShape[j]) {
        return emitError("Concat input dimensions must be all identical, "
                         "except for dimension on the axis of the "
                         "concatenation. Expected something compatible with: ")
               << commonType << " but got " << operand.getType() << " instead.";
      }
    }
  }

  return success();
}

LogicalResult ZHighConcatOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // The check of constraints is kept
  // However, current check handles dynamic dim only for the concat dim
  int inputNum = getNumOperands();
  for (int i = 0; i < inputNum; ++i) {
    if (!hasRankedType(getOperand(i)))
      return success();
  }
  // Checking value of axis parameter.
  auto commonType = getOperand(0).getType().cast<RankedTensorType>();
  auto commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = axis();
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
    // Tong Chen:
    // TOFIX: attribute modification should be into canonicalization
    // I did not move the code into ShapeHelper
    auto builder = mlir::Builder(getContext());
    axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  ZHighConcatOpAdaptor operandAdaptor(*this);
  ZHighConcatOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan Tile parameters successfully");
  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);
  Type resType = RankedTensorType::get(
      outputDims, commonType.getElementType(), commonType.getEncoding());
  getResult().setType(resType);

  return success();
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
