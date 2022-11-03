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

#include "src/Dialect/ONNX/ONNXEinsumOpHelper.hpp"
#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
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
// ONNXEntryPointOp
//===----------------------------------------------------------------------===//

void ONNXEntryPointOp::build(mlir::OpBuilder &builder,
    mlir::OperationState &state, mlir::func::FuncOp function) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
      SymbolRefAttr::get(function));
}

ONNXEntryPointOp ONNXEntryPointOp::create(
    mlir::Location location, mlir::func::FuncOp &func) {
  mlir::OperationState state(location, "onnx.EntryPoint");
  OpBuilder builder(location->getContext());
  mlir::ONNXEntryPointOp::build(builder, state, func);
  Operation *op = mlir::Operation::create(state);
  auto onnxEntryOp = llvm::cast<mlir::ONNXEntryPointOp>(op);
  return onnxEntryOp;
}

//===----------------------------------------------------------------------===//
// ONNXNoneOp
//===----------------------------------------------------------------------===//

OpFoldResult ONNXNoneOp::fold(ArrayRef<Attribute> operands) {
  return valueAttr();
}


//===----------------------------------------------------------------------===//
// PRelu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXPReluOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXPReluOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXPReluOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  auto xShape = X().getType().cast<ShapedType>().getShape();
  auto slopeShape = slope().getType().cast<ShapedType>().getShape();

  // To do unidirectional broadcasting, we first apply bidirectional
  // broadcasting. Then, fine-tune by getting constant dimensions from X.
  SmallVector<int64_t, 4> shape;
  // Bidirectional broadcasting rules.
  getBroadcastedShape(xShape, slopeShape, shape);
  // Fine-tune.
  for (unsigned int i = 0; i < shape.size(); ++i)
    if (xShape[i] != -1)
      shape[i] = xShape[i];

  getResult().setType(RankedTensorType::get(
      shape, X().getType().cast<ShapedType>().getElementType()));
  return success();
}

LogicalResult ONNXPReluOp::verify() {
  if (!hasShapeAndRank(X())) {
    return success();
  }
  if (!hasShapeAndRank(slope())) {
    return success();
  }

  ArrayRef<int64_t> xShape = X().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> slopeShape =
      slope().getType().cast<ShapedType>().getShape();

  // PRelu supports unidirectional broadcasting, that is slope should be
  // unidirectional broadcastable to input X.
  if (slopeShape.size() > xShape.size())
    return emitError("Slope tensor has a wrong shape");

  return success();
}





//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ShapedType inputType = input().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return success();
  }

  auto getOutputType = [&inputType](Type elementType) -> Type {
    if (inputType.hasRank()) {
      return RankedTensorType::get(inputType.getShape(), elementType);
    }
    return UnrankedTensorType::get(elementType);
  };

  mlir::Type targetType =
      (*this)->getAttr("to").cast<::mlir::TypeAttr>().getValue();
  OpBuilder builder(getContext());
  getResult().setType(getOutputType(targetType));
  return success();
}

//===----------------------------------------------------------------------===//
// CastLike
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastLikeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ShapedType inputType = input().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return success();
  }

  TensorType targetType = target_type().getType().dyn_cast<TensorType>();
  if (!inputType) {
    return success();
  }
  auto targetElementType = targetType.getElementType();

  auto getOutputType = [&inputType](Type elementType) -> Type {
    if (inputType.hasRank()) {
      return RankedTensorType::get(inputType.getShape(), elementType);
    }
    return UnrankedTensorType::get(elementType);
  };

  getResult().setType(getOutputType(targetElementType));
  return success();
}

//===----------------------------------------------------------------------===//
// Scaler
//===----------------------------------------------------------------------===//

LogicalResult ONNXScalerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto inputType = X().getType().dyn_cast<RankedTensorType>();

  if (!inputType)
    return success();

  updateType(
      getResult(), inputType.getShape(), FloatType::getF32(getContext()));
  return success();
}


//===----------------------------------------------------------------------===//
// SplitToSequence
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitToSequenceOp::verify() {
  Value inputValue = input();
  if (!hasShapeAndRank(inputValue))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = inputValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();

  int64_t axisIndex = axis();
  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));
  if (axisIndex < 0)
    axisIndex += inputRank;

  Value splitValue = split();
  if (isFromNone(splitValue)) {
    // since split is not specified, check the keepdims attribute
    int64_t keep = keepdims();
    // keepdims must be 0 or 1
    if (keep < 0 || keep > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "keepdims", keep,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
    return success();
  }
  auto splitType = splitValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> splitShape = splitType.getShape();
  int64_t splitRank = splitShape.size();
  if (splitRank > 1)
    return emitOpError() << ": split has rank " << splitRank << " > 1";
  if (DenseElementsAttr entries =
          getDenseElementAttributeFromONNXValue(splitValue)) {
    if (splitRank == 0) {
      auto scalar = getScalarValue<int64_t>(entries, splitType);
      if (scalar <= 0)
        return emitOpError() << ": split scalar " << scalar << " <= 0";
    } else {
      int64_t sum = 0;
      for (auto entry : entries.getValues<IntegerAttr>()) {
        int64_t i = entry.getInt();
        if (i < 0)
          return emitOpError() << ": split tensor has entry " << i << " < 0";
        sum += i;
      }
      int64_t dimSize = inputShape[axisIndex];
      if (dimSize != -1 && dimSize != sum)
        return emitOpError() << ": split tensor entries sum to " << sum
                             << " != axis dimension size " << dimSize;
    }
  }

  return success();
}

LogicalResult ONNXSplitToSequenceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Value inputValue = input();
  if (!hasShapeAndRank(inputValue))
    return success(); // Cannot infer output shape if input shape isn't known.

  // NOTE: all the asserts below are conditions checked in verify()

  auto inputType = inputValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t rank = shape.size();
  int64_t axisIndex = axis();
  assert((-rank <= axisIndex && axisIndex < rank) && "axis out of range");
  if (axisIndex < 0)
    axisIndex += rank;
  int64_t dimSize = shape[axisIndex];

  // start with length unknown and dims == shape with unknown dimension size
  // for axis (-1 is ShapedType::kDynamicSize), and edit it as needed below
  int64_t length = -1;
  SmallVector<int64_t, 4> dims(shape.begin(), shape.end());
  dims[axisIndex] = -1;

  Value splitValue = split();
  if (isFromNone(splitValue)) {
    // since split is not specified, check the keepdims attribute
    int64_t keep = keepdims();
    assert(0 <= keep && keep <= 1 && "keepdims out of range");
    length = dimSize;
    if (keep == 1) {
      // if dimSize is zero we can choose any value here, 1 is fine
      dims[axisIndex] = 1;
    } else {
      dims.erase(dims.begin() + axisIndex);
    }
  } else {
    auto splitType = splitValue.getType().cast<ShapedType>();
    ArrayRef<int64_t> splitShape = splitType.getShape();
    int64_t splitRank = splitShape.size();
    assert(splitRank <= 1 && "invalid split tensor rank");
    if (DenseElementsAttr entries =
            getDenseElementAttributeFromONNXValue(splitValue)) {
      if (splitRank == 0) {
        auto scalar = getScalarValue<int64_t>(entries, splitType);
        assert(scalar > 0 && "invalid split scalar");
        if (dimSize != -1) {
          length = dimSize / scalar;
          if ((dimSize % scalar) == 0)
            dims[axisIndex] = scalar;
        }
      } else {
        auto values = entries.getValues<IntegerAttr>();
        length = values.size();
        if (length > 0) {
          // in the (unlikely?) case that all entries are the same, we infer
          // that's the dimension size for axis
          int64_t first = values[0].getInt();
          assert(first >= 0 && "invalid split tensor entry");
          if (llvm::all_of(values, [first](IntegerAttr value) {
                return value.getInt() == first;
              }))
            dims[axisIndex] = first;
        }
      }
    } else if (splitRank == 1 && splitShape[0] != -1) {
      length = splitShape[0];
      // corner case: if the input dimension size for axis is zero, any tensors
      // in the output sequence must also be zero if the sequence is non-empty
      if (length > 0 && dimSize == 0)
        dims[axisIndex] = 0;
      // if length and dimSize are both zero, we can choose any value,
      // leaving it be -1 is fine
    }
  }
  getResult().setType(SeqType::get(
      RankedTensorType::get(dims, inputType.getElementType()), length));
  return success();
}



//===----------------------------------------------------------------------===//
// DynamicQuantizeLinear
//===----------------------------------------------------------------------===//

LogicalResult ONNXDynamicQuantizeLinearOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy) {
    return success();
  }

  auto yTy = y().getType().cast<ShapedType>();
  auto yScaleTy = y_scale().getType().cast<ShapedType>();
  auto yZPTy = y_zero_point().getType().cast<ShapedType>();

  IntegerType ui8Type =
      IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  FloatType f32Type = FloatType::getF32(getContext());

  RankedTensorType scalarType = RankedTensorType::get({}, f32Type);
  RankedTensorType y_zero_point_type = RankedTensorType::get({}, ui8Type);

  // Set the types for the scalars
  if (!yScaleTy.hasStaticShape()) {
    y_scale().setType(scalarType);
  }

  if (!yZPTy.hasStaticShape()) {
    y_zero_point().setType(y_zero_point_type);
  }

  if (!yTy.hasStaticShape()) {
    RankedTensorType outType = RankedTensorType::get(inTy.getShape(), ui8Type);
    y().setType(outType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// QuantizeLinear
//===----------------------------------------------------------------------===//

LogicalResult ONNXQuantizeLinearOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy) {
    return success();
  }

  auto yTy = y().getType().cast<ShapedType>();

  if (!yTy.hasStaticShape()) {
    // TODO: Unfortunately, we can't tell if this should be signed or
    // unsigned here...
    IntegerType i8Type = IntegerType::get(getContext(), 8);
    RankedTensorType outType = RankedTensorType::get(inTy.getShape(), i8Type);
    y().setType(outType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DequantizeLinear
//===----------------------------------------------------------------------===//

namespace {
// Returns known length if ty is a non-scalar 1-D vector, otherwise -1.
int64_t nonScalar1DLen(ShapedType ty) {
  if (!ty.hasRank() || ty.getRank() != 1 || ty.isDynamicDim(0))
    return -1;
  int64_t d = ty.getDimSize(0);
  return d == 1 ? -1 : d; // If dim size is 1 then it's considered a scalar.
}
} // namespace

LogicalResult ONNXDequantizeLinearOp::verify() {
  // Is tensor known to be a scalar (rank 0 or rank 1 with 1 element)?
  auto isScalar = [](RankedTensorType t) -> bool {
    return t.getRank() == 0 || (t.getRank() == 1 && t.getDimSize(0) == 1);
  };

  Value scale = x_scale();
  auto scaleTy = scale.getType().cast<ShapedType>();
  if (scaleTy.hasRank() && scaleTy.getRank() > 1)
    return emitOpError("x_scale must be a scalar or 1-D tensor");
  int64_t scaleLen = nonScalar1DLen(scaleTy);

  Value zero = x_zero_point();
  int64_t zeroLen = -1;
  if (!isFromNone(zero)) {
    if (auto zeroTy = zero.getType().dyn_cast<RankedTensorType>()) {
      if (zeroTy.getRank() > 1)
        return emitOpError("x_zero_point must be a scalar or 1-D tensor");
      zeroLen = nonScalar1DLen(zeroTy);
      if (auto scaleTy = scale.getType().dyn_cast<RankedTensorType>()) {
        if ((isScalar(scaleTy) && scaleLen != -1) ||
            (zeroLen != -1 && isScalar(zeroTy)) ||
            (zeroLen != -1 && scaleLen != -1 && zeroLen != scaleLen))
          return emitOpError(
              "x_scale and x_zero_point must have the same shape");
      }
    }

    // TODO: Figure out whether to introduce a variant of this check from the
    // spec ("'x_zero_point' and 'x' must have same type"). It is violated in
    // in the resnet50-v1-12-qdq model where x, x_zero_point are i8, ui8.
    //
    // if (getElementType(x().getType()) != getElementType(zero.getType()))
    //   return emitOpError("x and x_zero_point must have the same data type");

    if (getElementType(zero.getType()).isInteger(32) && zeroLen != 0)
      if (auto values = getDenseElementAttributeFromONNXValue(zero))
        if (!values.isSplat() || !values.getSplatValue<APInt>().isZero())
          return emitOpError("x_zero_point must be 0 for data type int32");
  }

  if (scaleLen == -1 && zeroLen == -1) {
    // Either x_scale or x_zero_point is scalar, so quantization is per-tensor /
    // per layer and axis is ignored and there is nothing more to verify, or
    // their 1-D rank is unknown and we cannot verify more until they are known.
  } else {
    // If x_scale or x_zero_point is a non-scalar 1-D tensor then quantization
    // is per-axis.
    int64_t d = scaleLen != -1 ? scaleLen : zeroLen;
    if (auto xTy = x().getType().dyn_cast<RankedTensorType>()) {
      int64_t r = xTy.getRank();
      // axis attribute must be in the range [-r,r-1].
      int64_t a = axis();
      if (a < -r || a >= r)
        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "axis", a,
            onnx_mlir::Diagnostic::Range<int64_t>(-r, r - 1));
      if (a < 0)
        a += r;
      if (!xTy.isDynamicDim(a) && xTy.getDimSize(a) != d)
        return emitOpError("x_scale and x_zero_point 1-D tensor length must "
                           "match the input axis dim size");
    } else {
      // Cannot verify more until x rank is known.
    }
  }

  return success();
}

LogicalResult ONNXDequantizeLinearOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {

  if (auto xTy = x().getType().dyn_cast<RankedTensorType>()) {
    auto xShape = xTy.getShape();
    SmallVector<int64_t, 4> yShape(xShape.begin(), xShape.end());
    int64_t d = nonScalar1DLen(x_scale().getType().cast<ShapedType>());
    if (d == -1 && !isFromNone(x_zero_point())) {
      d = nonScalar1DLen(x_zero_point().getType().cast<ShapedType>());
    }
    if (d != -1) {
      int64_t r = xTy.getRank();
      int64_t a = axis();
      // Checked in verify:
      assert(-r <= a && a < r && "axis out of range");
      if (a < 0)
        a += r;
      if (yShape[a] == -1) {
        yShape[a] = d;
      } else {
        // Checked in verify:
        assert(yShape[a] == d && "x_scale and x_zero_point 1-D tensor length "
                                 "must match the input axis dim size");
      }
    }
    updateType(y(), yShape);
  }

  return success();
}



//===----------------------------------------------------------------------===//
// Dropout
//===----------------------------------------------------------------------===//

LogicalResult ONNXDropoutOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!data().getType().isa<RankedTensorType>())
    return success();

  getResult(0).setType(data().getType());

  IntegerType i1Type = IntegerType::get(getContext(), 1, IntegerType::Signless);
  updateType(getResult(1), getShape(data().getType()), i1Type);
  return success();
}

//===----------------------------------------------------------------------===//
// OneHotEncoder
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotEncoderOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ShapedType inputType = X().getType().dyn_cast<RankedTensorType>();
  if (!inputType)
    return success();
  auto shape = inputType.getShape();
  int64_t outDim = 0;

  // If the input is a tensor of float, int32, or double,
  // the data will be cast to integers and
  // the cats_int64s category list will be used for the lookups.
  if (inputType.getElementType().isIntOrFloat()) {
    outDim = ArrayAttrSize(cats_int64s());
  } else {
    outDim = ArrayAttrSize(cats_strings());
  }

  // Encoded output data, having one more dimension than X
  // total category count will determine the size of the extra dimension
  SmallVector<int64_t, 2> dims;
  for (unsigned int i = 0; i != shape.size(); ++i)
    dims.emplace_back(shape[i]);
  dims.emplace_back(outDim);

  updateType(getResult(), dims, FloatType::getF32(getContext()));
  return success();
}

LogicalResult ONNXOneHotEncoderOp::verify() {
  ONNXOneHotEncoderOpAdaptor operandAdaptor = ONNXOneHotEncoderOpAdaptor(*this);

  // get operands
  auto input = operandAdaptor.X();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = input.getType().cast<ShapedType>();
  if (!inputType)
    return success();

  // If the input is a tensor of float, int32, or double,
  // the data will be cast to integers and
  // the cats_int64s category list will be used for the lookups.
  if (inputType.getElementType().isIntOrFloat()) {
    if (!operandAdaptor.cats_int64s()) {
      return emitOpError("input is a tensor of float, int32, or double, "
                         "but no cats_int64s attribute");
    }
  } else {
    if (!operandAdaptor.cats_strings()) {
      return emitOpError("input is not a tensor of float, int32, or double, "
                         "but no cats_strings attribute");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Less
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXLessOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXLessOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  for (unsigned int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return success();
  }
  Type lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  Type rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  ArrayRef<int64_t> dims =
      getBroadcastedType(lhsTy, rhsTy).cast<RankedTensorType>().getShape();

  updateType(getResult(), dims, IntegerType::get(getContext(), /*width=*/1));
  return success();
}

// Operations for which shape inference has not been implemented yet
// If you add the implementation for one op, move it out of this section
// Also please add test case in test/mlir/onnx/onnx_shape_inference.mlir
// Followed by the implementation of lowering to Krnl and
// Enable the corresponding node test in check-onnx-backend

LogicalResult ONNXBatchNormalizationOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXBernoulliOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto builder = mlir::OpBuilder(getContext());
  if (!hasShapeAndRank(input())) {
    return success();
  }
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type elementType;
  if (dtypeAttr()) {
    elementType = convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)dtypeAttr().getValue().getSExtValue());
  } else {
    elementType = inputType.getElementType();
  }
  getResult().setType(RankedTensorType::get(inputType.getShape(), elementType));
  return success();
}




LogicalResult ONNXDetOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXEinsumOp::verify() {
  einsum::ErrorFn errorFn = [this]() -> mlir::InFlightDiagnostic {
    return this->emitOpError() << "equation '" << this->equation() << "': ";
  };

  ONNXEinsumOpAdaptor operandAdaptor(*this);
  ValueRange inputs = operandAdaptor.Inputs();

  if (failed(einsum::verifyEquation(equation(), inputs.size(), errorFn))) {
    return failure();
  }

  Type firstElementType =
      inputs[0].getType().cast<ShapedType>().getElementType();
  for (Value input : inputs) {
    ShapedType type = input.getType().cast<ShapedType>();
    if (type.getElementType() != firstElementType) {
      return emitOpError() << "different input element types";
    }
  }
  if (!llvm::all_of(inputs, hasShapeAndRank))
    return success(); // Can only infer once operand shapes are known.
  return einsum::verifyShapes(operandAdaptor, errorFn);
}

LogicalResult ONNXEinsumOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXEinsumOpAdaptor operandAdaptor(*this);
  if (!llvm::all_of(operandAdaptor.Inputs(), hasShapeAndRank))
    return success(); // Can only infer once operand shapes are known.

  einsum::ErrorFn errorFn = [this]() {
    return this->emitOpError() << "equation '" << this->equation() << "': ";
  };
  FailureOr<einsum::Shape> shape =
      einsum::inferOutputShape(operandAdaptor, errorFn);
  assert(succeeded(shape) && "any failure should be caught in verify()");
  Type elementType =
      getOperand(0).getType().cast<ShapedType>().getElementType();

  updateType(getResult(), *shape, elementType);
  return success();
}

LogicalResult ONNXEyeLikeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto builder = mlir::OpBuilder(getContext());
  if (!hasShapeAndRank(input())) {
    return success();
  }
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  Type elementType;
  if (dtypeAttr()) {
    elementType = convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)dtypeAttr().getValue().getSExtValue());
  } else {
    elementType = inputType.getElementType();
  }

  updateType(getResult(), inputType.getShape(), elementType);
  return success();
}

//===------------------------------------------------------------------------===//
// IsInfOp
//===------------------------------------------------------------------------===//

LogicalResult ONNXIsInfOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

//===------------------------------------------------------------------------===//
// LayoutTransform
//===------------------------------------------------------------------------===//

void ONNXLayoutTransformOp::build(OpBuilder &builder, OperationState &state,
    Value data, StringAttr targetLayoutAttr) {
  Type resType = convertTensorTypeToTensorTypeWithONNXTensorEncoding(
      builder, data.getType(), targetLayoutAttr);
  build(builder, state, resType, data, targetLayoutAttr);
}

LogicalResult ONNXLayoutTransformOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXLayoutTransformOp operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.data()))
    return success();

  auto builder = mlir::Builder(getContext());
  Type resType = convertTensorTypeToTensorTypeWithONNXTensorEncoding(
      builder, data().getType(), target_layoutAttr());
  getResult().setType(resType);
  return success();
}

//===------------------------------------------------------------------------===//
// IsNaNOp
//===------------------------------------------------------------------------===//

LogicalResult ONNXIsNaNOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  ONNXIsNaNOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.X()))
    return success();

  IntegerType i1Type = IntegerType::get(getContext(), 1, IntegerType::Signless);
  updateType(getResult(), getShape(X().getType()), i1Type);
  return success();
}

//===----------------------------------------------------------------------===//
// ONNXLogSoftmax
//===----------------------------------------------------------------------===//

LogicalResult ONNXLogSoftmaxOp::verify() {
  ONNXLogSoftmaxOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.input()))
    return success(); // Won't be able to do any checking at this stage.

  int64_t inputRank =
      operandAdaptor.input().getType().cast<ShapedType>().getRank();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

LogicalResult ONNXLpPoolOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXMaxPoolOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXMaxUnpoolOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}


LogicalResult ONNXMultinomialOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}




LogicalResult ONNXOptionalOp::verify() {
  if (type().has_value() != input().getType().isa<NoneType>())
    return emitError(
        "Optional should have either type attribute or input value");
  return success();
}

LogicalResult ONNXOptionalOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Type ty;
  if (auto typeAttr = type()) {
    ty = typeAttr.value();
  } else {
    ty = input().getType();
    // checked in verify()
    assert(!ty.isa<NoneType>() && "type attribute or input value needed");
  }
  getResult().setType(OptType::get(ty));
  return success();
}

LogicalResult ONNXOptionalGetElementOp::verify() {
  if (!input().getType().isa<OptType>())
    return emitError("OptionalGetElement input should have optional type");
  return success();
}

LogicalResult ONNXOptionalGetElementOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Type elementType = input().getType().cast<OptType>().getElementType();
  getResult().setType(elementType);
  return success();
}

LogicalResult ONNXOptionalHasElementOp::verify() {
  if (!input().getType().isa<OptType>())
    return emitError("OptionalHasElement input should have optional type");
  return success();
}

LogicalResult ONNXOptionalHasElementOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Builder builder(getContext());
  Type scalarBoolType = RankedTensorType::get({}, builder.getI1Type());
  getResult().setType(scalarBoolType);
  return success();
}

LogicalResult ONNXRandomUniformOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXRandomUniformLikeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}




LogicalResult ONNXRoiAlignOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !batch_indices().getType().isa<RankedTensorType>())
    return success();

  auto elementType = X().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXRoiAlignOpShapeHelper, ONNXRoiAlignOp,
      ONNXRoiAlignOpAdaptor>(*this, elementType);
}

LogicalResult ONNXRoiAlignOp::verify() {
  ONNXRoiAlignOpAdaptor operandAdaptor = ONNXRoiAlignOpAdaptor(*this);
  // get input info.
  mlir::Value X = operandAdaptor.X();
  mlir::Value batch_indices = operandAdaptor.batch_indices();

  if (!hasShapeAndRank(X) || !hasShapeAndRank(batch_indices))
    return success();

  int64_t x_rank = X.getType().cast<mlir::ShapedType>().getRank();
  int64_t batch_indices_rank =
      batch_indices.getType().cast<mlir::ShapedType>().getRank();

  // Test ranks.
  if (x_rank != 4)
    return emitOpError("RoiAlign with X should be a 4D tensor");
  if (batch_indices_rank != 1)
    return emitOpError("RoiAlign with batch_indices should be a 1D tensor");

  return success();
}



LogicalResult ONNXStringNormalizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXTfIdfVectorizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}


//===----------------------------------------------------------------------===//
// Unique
//===----------------------------------------------------------------------===//

LogicalResult ONNXUniqueOp::verify() {
  Optional<int64_t> optionalSorted = sorted();
  if (optionalSorted.has_value()) {
    // optional sorted attribute must be zero or one.
    int64_t sorted = optionalSorted.value();
    if (sorted < 0 || sorted > 1)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "sorted", sorted,
          onnx_mlir::Diagnostic::Range<int64_t>(0, 1));
  }

  ONNXUniqueOpAdaptor operandAdaptor(*this);
  Value X = operandAdaptor.X();
  if (!hasShapeAndRank(X))
    return success(); // Too early to verify.

  int64_t XRank = X.getType().cast<ShapedType>().getRank();
  Optional<int64_t> optionalAxis = axis();

  if (optionalAxis.has_value()) {
    // axis attribute must be in the range [-r,r-1], where r = rank(X).
    int64_t axis = optionalAxis.value();
    if (axis < -XRank || axis >= XRank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-XRank, XRank - 1));
  }

  return success();
}

LogicalResult ONNXUniqueOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXUpsampleOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  if (!scales().getType().isa<RankedTensorType>()) {
    return success();
  }

  auto inputTy = X().getType().cast<RankedTensorType>();
  int32_t inputRank = inputTy.getShape().size();

  SmallVector<int64_t, 4> outputDims(inputRank, -1);

  // Extract the scale values
  auto scalesConstOp = getONNXConstantOp(scales());
  if (!scalesConstOp) {
    return success();
  }
  auto valueAttr = scalesConstOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!valueAttr) {
    return emitError("Scales constant is not a DenseElementsAttr");
  }
  int scaleIdx = 0;
  // Why are the scale values float's?
  for (auto it = valueAttr.getValues<FloatAttr>().begin();
       it != valueAttr.getValues<FloatAttr>().end(); ++it) {
    outputDims[scaleIdx++] = (int)((*it).getValueAsDouble());
  }

  // Compute and set the output shape
  for (int i = 0; i < inputRank; ++i) {
    outputDims[i] *= inputTy.getShape()[i];
  }
  getResult().setType(
      RankedTensorType::get(outputDims, inputTy.getElementType()));

  return success();
}

LogicalResult ONNXUpsampleOp::verify() {
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  if (!scales().getType().isa<RankedTensorType>()) {
    return success();
  }

  auto inputTy = X().getType().cast<RankedTensorType>();
  int32_t inputRank = inputTy.getShape().size();

  // Sanity checks on scale argument
  auto scalesTy = scales().getType().cast<RankedTensorType>();
  if (scalesTy.getShape().size() != 1) {
    return emitError("Scales tensor must be rank-1");
  }
  if (scalesTy.getShape()[0] != inputRank) {
    return emitError("Input tensor rank doesn't match scales tensor shape");
  }

  // Extract the scale values
  auto scalesConstOp = getONNXConstantOp(scales());
  if (!scalesConstOp) {
    return success();
  }
  auto valueAttr = scalesConstOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!valueAttr) {
    return emitError("Scales constant is not a DenseElementsAttr");
  }

  int scaleIdx = 0;
  for (auto it = valueAttr.getValues<FloatAttr>().begin();
       it != valueAttr.getValues<FloatAttr>().end(); ++it) {
    if (scaleIdx >= inputRank) {
      return emitError("Scales tensor shape doesn't match # of scale values");
    }
    scaleIdx++;
  }
  if (scaleIdx != inputRank) {
    return emitError("Scales tensor shape doesn't match # of scale values");
  }
  return success();
}

LogicalResult ONNXArrayFeatureExtractorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXBinarizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXCastMapOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}


LogicalResult ONNXDictVectorizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXFeatureVectorizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXImputerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXLabelEncoderOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXLinearClassifierOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXLinearRegressorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXNormalizerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXSVMClassifierOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXSVMRegressorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXTreeEnsembleClassifierOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXTreeEnsembleRegressorOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXZipMapOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

LogicalResult ONNXGridSampleOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

#define NOT_IMPLEMENTED_INFERSHAPE(T)                                          \
  LogicalResult T::inferShapes(                                                \
      std::function<void(mlir::Region &)> doShapeInference) {                  \
    return emitError(NOT_IMPLEMENTED_MESSAGE);                                 \
  }

NOT_IMPLEMENTED_INFERSHAPE(ONNXAdagradOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXAdamOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXClipV6Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXClipV11Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXClipV12Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXGradientOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXMomentumOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXNegativeLogLikelihoodLossOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXPadV2Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXPadV11Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXResizeV11Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXResizeV10Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXSoftmaxCrossEntropyLossOp)
NOT_IMPLEMENTED_INFERSHAPE(ONNXUpsampleV9Op)
NOT_IMPLEMENTED_INFERSHAPE(ONNXUpsampleV7Op)

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

/// Infer the output shape of the ONNXCustomOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXCustomOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  func::FuncOp fn =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType ONNXCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// SeqType
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
