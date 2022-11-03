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

//hi alex: remove when done
#include "src/Dialect/ONNX/NN/NNHelper.hpp"

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
// ONNX Helper functions for shape helpers
//===----------------------------------------------------------------------===//

#if 0 // hi alex

// Get reduction type.
static RankedTensorType getReductionOutputType(
    ShapedType operandTy, Optional<ArrayAttr> axesAttrs, uint64_t keepdims) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs != llvm::None)
    for (auto axisAttr : axesAttrs.value()) {
      int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (rank + axis);
      assert(axis >= -rank && axis <= rank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }
  else
    for (decltype(rank) i = 0; i < rank; ++i)
      axes.emplace_back(i);

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i)
    isReductionAxis.emplace_back(
        (std::find(axes.begin(), axes.end(), i) != axes.end()) ? true : false);

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else
      dims.emplace_back(operandTy.getShape()[i]);
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

// Reduction with axes is from ConstantOp.
// Only ReduceSum call this function now.
static RankedTensorType getReductionOutputType(ShapedType operandTy,
    DenseElementsAttr axesAttrs, uint64_t keepdims,
    uint64_t noop_with_empty_axes) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs)
    for (auto element : axesAttrs.getValues<IntegerAttr>()) {
      int64_t axis = element.getInt();
      if (axis < -rank || axis > rank - 1)
        return RankedTensorType();

      axis = axis >= 0 ? axis : (rank + axis);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }

  if (axes.size() == 0 && !noop_with_empty_axes)
    for (decltype(rank) i = 0; i < rank; ++i)
      axes.emplace_back(i);

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i)
    isReductionAxis.emplace_back(
        (std::find(axes.begin(), axes.end(), i) != axes.end()) ? true : false);

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else
      dims.emplace_back(operandTy.getShape()[i]);
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

// Handle shapes for operations with a single output.
template <class SHAPE_HELPER, class OP, class ADAPTOR>
static LogicalResult shapeHelperInferShapes(OP &op, Type elementType) {
  SHAPE_HELPER shapeHelper(&op);
  ADAPTOR operandAdaptor(op);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return op.emitError("Failed to scan " + OP::getOperationName() +
                        " parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);

  updateType(op.getResult(), outputDims, elementType);
  return success();
}
#endif

#if 0 // hi alex

// Handle shapes for operations with multiple outputs.
template <class SHAPE_HELPER, class OP, class ADAPTOR>
static LogicalResult shapeHelperInferMultipleShapes(
    OP &op, TypeRange elementTypes) {
  assert(elementTypes.size() == op.getNumResults() &&
         "Incorrect elementTypes size");

  SHAPE_HELPER shapeHelper(&op);
  ADAPTOR operandAdaptor(op);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return op.emitError("Failed to scan " + OP::getOperationName() +
                        " parameters successfully");

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    SmallVector<int64_t, 4> outputDims;
    IndexExpr::getShape(shapeHelper.dimsForOutput(i), outputDims);
    updateType(op.getResults()[i], outputDims, elementTypes[i]);
  }
  return success();
}

// Handle shape inference for reduction like operators.
template <class OP, class ADAPTOR>
static LogicalResult inferShapeForReductionOps(OP &op) {
  ADAPTOR operandAdaptor(op);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // cannot infer when the operands shape is not yet known.

  auto operandTy = op.getOperand().getType().template cast<ShapedType>();
  auto resultTy = getReductionOutputType(operandTy, op.axes(), op.keepdims());

  updateType(op.getResult(), getShape(resultTy), resultTy.getElementType());
  return success();
}
#endif

#define NOT_IMPLEMENTED_MESSAGE                                                \
  (getOperationName() +                                                        \
      ": is not supported at this time. Please open an issue on "              \
      "https://github.com/onnx/onnx-mlir and/or consider contribute code. "    \
      "Error encountered in shape inference.")

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//






namespace onnx_mlir {

#if 0 // hi alex conv
template LogicalResult processConvTypeParams<ONNXConvOp>(
    ONNXConvOp *op, Value inputOperand, Value W);
template LogicalResult processConvTypeParams<ONNXQLinearConvOp>(
    ONNXQLinearConvOp *op, Value inputOperand, Value W);
template LogicalResult processConvTypeParams<ONNXConvIntegerOp>(
    ONNXConvIntegerOp *op, Value inputOperand, Value W);
template LogicalResult processConvTypeParams<ONNXConvTransposeOp>(
    ONNXConvTransposeOp *op, Value inputOperand, Value W);

#endif

} // namespace onnx_mlir


//===----------------------------------------------------------------------===//
// Support function that infers shape for RNN operations.
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult RNNShapeInference(T *op, int gates) {
  bool batchwiseLayout = op->layout() == 1;

  Value X = op->X();
  Value W = op->W();
  Value R = op->R();

  if (!X.getType().isa<RankedTensorType>() ||
      !W.getType().isa<RankedTensorType>() ||
      !R.getType().isa<RankedTensorType>()) {
    return success();
  }

  auto xTy = X.getType().cast<RankedTensorType>();
  auto elementType = xTy.getElementType();

  // xShape :: [batch_size, seq_length, input_size] if batchwiseLayout
  // xShape :: [seq_length, batch_size, input_size] otherwise
  auto xShape = xTy.getShape();
  // wShape :: [num_dir, gates*hidden_size, input_size]
  auto wShape = W.getType().cast<RankedTensorType>().getShape();
  // rShape :: [num_dir, gates*hidden_size, hidden_size]
  auto rShape = R.getType().cast<RankedTensorType>().getShape();

  if (xShape.size() != 3) {
    return op->emitError("The first input tensor must have rank 3");
  }
  if (wShape.size() != 3) {
    return op->emitError("The second input tensor must have rank 3");
  }
  if (rShape.size() != 3) {
    return op->emitError("The third input tensor must have rank 3");
  }

  // Get sequence length, batch size.
  int64_t seqLength = batchwiseLayout ? xShape[1] : xShape[0];
  int64_t batchSize = batchwiseLayout ? xShape[0] : xShape[1];

  // Get hidden size from hidden_size attribute.
  int64_t hiddenSize = -1;
  if (op->hidden_size().has_value()) {
    hiddenSize = op->hidden_size().value();
  } else {
    // Infer hidden_size from wShape and rShape if possible.
    if (rShape[2] != -1)
      hiddenSize = rShape[2];
    else if (rShape[1] != -1)
      hiddenSize = rShape[1] / gates;
    else if (wShape[1] != -1)
      hiddenSize = wShape[1] / gates;
    // Update hidden_size attribute.
    if (hiddenSize != -1) {
      auto builder = mlir::Builder(op->getContext());
      auto hiddenSizeAttr =
          IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
              APInt(64, /*value=*/hiddenSize, /*isSigned=*/true));
      op->hidden_sizeAttr(hiddenSizeAttr);
    }
  }

  // Get direction.
  int64_t numDir;
  if ((op->direction() == "forward") || (op->direction() == "reverse"))
    numDir = 1;
  else if (op->direction() == "bidirectional")
    numDir = 2;
  else
    return op->emitError(
        "direction attribute must be one of the strings: forward, "
        "reverse, and bidirectional");

  // Set result types. There are always 2 (RNN, GRU) or 3 results
  // but they are sometimes optional in which case they have NoneType.
  assert((op->getNumResults() == 2 || op->getNumResults() == 3) &&
         "RNN, GRU have 2 results, LSTM has 3");
  // Y :: [batch_size, seq_length, num_dir, hidden_size] if batchwiseLayout
  // Y :: [seq_length, num_dir, batch_size, hidden_size] otherwise
  Type yTy = op->getResult(0).getType();
  if (!yTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yTy = RankedTensorType::get(
          {batchSize, seqLength, numDir, hiddenSize}, elementType);
    } else {
      yTy = RankedTensorType::get(
          {seqLength, numDir, batchSize, hiddenSize}, elementType);
    }
    op->getResult(0).setType(yTy);
  }
  // Y_h :: [batch_size, num_dir, hidden_size] if batchwiseLayout
  // Y_h :: [num_dir, batch_size, hidden_size] otherwise
  Type yhTy = op->getResult(1).getType();
  if (!yhTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yhTy =
          RankedTensorType::get({batchSize, numDir, hiddenSize}, elementType);
    } else {
      yhTy =
          RankedTensorType::get({numDir, batchSize, hiddenSize}, elementType);
    }
    op->getResult(1).setType(yhTy);
  }
  if (op->getNumResults() == 3) {
    // Y_c :: [batch_size, num_dir, hidden_size] if batchwiseLayout
    // Y_c :: [num_dir, batch_size, hidden_size] otherwise
    Type ycTy = op->getResult(2).getType();
    if (!ycTy.isa<NoneType>()) {
      if (batchwiseLayout) {
        ycTy =
            RankedTensorType::get({batchSize, numDir, hiddenSize}, elementType);
      } else {
        ycTy =
            RankedTensorType::get({numDir, batchSize, hiddenSize}, elementType);
      }
      op->getResult(2).setType(ycTy);
    }
  }

  return success();
}


//===----------------------------------------------------------------------===//
// ONNXArgMaxOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMaxOp::verify() {
  ONNXArgMaxOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  int64_t rank = data().getType().cast<ShapedType>().getRank();
  int64_t axisIndex = axis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return success();
}

LogicalResult ONNXArgMaxOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasShapeAndRank(data()))
    return success();

  // ONNX spec specifies the reduced type as an int64
  auto elementType = IntegerType::get(getContext(), 64);
  return shapeHelperInferShapes<ONNXArgMaxOpShapeHelper, ONNXArgMaxOp,
      ONNXArgMaxOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// ONNXArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXArgMinOp::verify() {
  ONNXArgMinOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  int64_t rank = data().getType().cast<ShapedType>().getRank();
  int64_t axisIndex = axis();

  // axis value must be in the range [-rank, rank-1].
  if (axisIndex < -rank || axisIndex >= rank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));

  return success();
}

LogicalResult ONNXArgMinOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasShapeAndRank(data()))
    return success();

  // ONNX spec specifies the reduced type as an int64
  auto elementType = IntegerType::get(getContext(), 64);
  return shapeHelperInferShapes<ONNXArgMinOpShapeHelper, ONNXArgMinOp,
      ONNXArgMinOpAdaptor>(*this, elementType);
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
// ONNX Operations
//===----------------------------------------------------------------------===//


















// Sequence related operations
// The general form for seq is seq<tensor<*xT>>
// Tensors will be add to or removed from a seq dynamically.
// The tensor type in a seq should be a summary of all the tensor type in
// the seq.
// It is possible seq<tensor<*xT>> can be refined into seq<RankedTensor>,
// or even seq<StaticShapedTensor> if all the tensors have common shape info
// It is important to refine the type for seq in onnx-mlir because static
// type is used. If seq of unranked tensor remains, onnx-mlir can not handle
// the unranked tensor retrieved from the seq.
// Here is the rules for shape inferences of seq-related ops:
// * A seq is started empty as the result of SequenceEmpty. We can track this
//   property with a tag in seq type or along dataflow.
// * When the an element is added, we can merge its shape with that in seq.
// * when an element is removed from seq, the seq becomes empty if it is the
//   last tenor in the seq (known statically).
// Since the seq is usually used as a parameter of a graph (e.g. for LoopOp),
// shape inference for region may need improvement.

namespace {
// Helper function used in Sequence ops shape inference
ShapedType sequenceAddType(
    ShapedType accumulatedType, ShapedType additionalType) {
  Type elementType = accumulatedType.getElementType();
  assert(elementType == additionalType.getElementType() &&
         "types to merge must have the same data type");
  // Pick the weaker attr: known dim > unknown dim > unranked
  if (!accumulatedType.hasRank())
    return accumulatedType;
  if (!additionalType.hasRank())
    return additionalType;
  int64_t rank = accumulatedType.getRank();
  if (rank != additionalType.getRank())
    return UnrankedTensorType::get(elementType);
  ArrayRef<int64_t> acc = accumulatedType.getShape();
  ArrayRef<int64_t> add = additionalType.getShape();
  SmallVector<int64_t, 4> dims;
  for (int64_t i = 0; i < rank; i++) {
    dims.push_back(acc[i] != add[i] ? -1 : add[i]);
  }
  return RankedTensorType::get(dims, elementType);
}
} // namespace

//===----------------------------------------------------------------------===//
// SequenceInsertOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceInsertOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Merge the tensor type for the seq and the inserted tensor
  SeqType seqType = input_sequence().getType().cast<mlir::SeqType>();
  ShapedType tensorType = tensor().getType().cast<ShapedType>();
  int64_t length = seqType.getLength();
  if (length == 0) {
    // When the input seq is empty, inherit the tensor type
    getResult().setType(SeqType::get(tensorType, 1));
  } else {
    int64_t newLength = length == -1 ? -1 : length + 1;
    ShapedType seqTensorType = seqType.getElementType().cast<ShapedType>();
    seqTensorType = sequenceAddType(seqTensorType, tensorType);
    getResult().setType(SeqType::get(seqTensorType, newLength));
  }
  return success();
}

LogicalResult ONNXSequenceInsertOp::verify() {
  ONNXSequenceInsertOpAdaptor operandAdaptor =
      ONNXSequenceInsertOpAdaptor(*this);

  // These cast should be guaranteed by default verifier
  Type seqElementType = operandAdaptor.input_sequence()
                            .getType()
                            .dyn_cast<mlir::SeqType>()
                            .getElementType();
  Type elementType1 = seqElementType.dyn_cast<ShapedType>().getElementType();
  ShapedType insertType =
      operandAdaptor.tensor().getType().dyn_cast<ShapedType>();
  Type elementType2 = insertType.getElementType();

  if (elementType1 != elementType2) {
    return emitError("Element types of the tensor in seqence and input "
                     "have to be the same");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceAtOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceAtOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto outputType = getResult().getType();
  auto inputElementType =
      input_sequence().getType().cast<SeqType>().getElementType();
  if (!inputElementType.isa<UnrankedTensorType>() &&
      outputType.isa<UnrankedTensorType>()) {
    getResult().setType(inputElementType);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceConstructOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceConstructOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto types = inputs().getTypes();
  ShapedType seqTensorType = types[0].cast<ShapedType>();
  for (size_t i = 1; i < types.size(); ++i) {
    seqTensorType = sequenceAddType(seqTensorType, types[i].cast<ShapedType>());
  }
  getResult().setType(SeqType::get(seqTensorType, types.size()));
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceEmptyOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceEmptyOp::verify() {
  // For the Optional dtypeAttr, the default type is F32
  auto builder = mlir::OpBuilder(getContext());
  Type elementType;
  if (dtypeAttr()) {
    elementType = convertONNXTypeToMLIRType(builder,
        (onnx::TensorProto_DataType)dtypeAttr().getValue().getSExtValue());
  } else {
    elementType = builder.getF32Type();
  }

  // Get element type for seq from the output
  ShapedType outputSeqElementType =
      getResult().getType().cast<SeqType>().getElementType();
  if (outputSeqElementType.getElementType() != elementType)
    return emitError("SequenceEmpty dtype() does not match the output type");
  return success();
}

LogicalResult ONNXSequenceEmptyOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto originTy = getResult().getType().cast<SeqType>();
  auto elementTy = originTy.getElementType();
  auto returnTy = SeqType::get(elementTy, 0);
  getResult().setType(returnTy);
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceEraseOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceEraseOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto inputTy = input_sequence().getType().cast<SeqType>();
  int64_t length = inputTy.getLength();

  if (length == 0)
    return emitError("SequenceErase from an empty seq");
  getResult().setType(
      SeqType::get(inputTy.getElementType(), length == -1 ? -1 : length - 1));
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceLengthOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXSequenceLengthOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Type outputTy = getResult().getType();
  if (!outputTy.isa<RankedTensorType>() ||
      outputTy.cast<RankedTensorType>().getRank() != 0) {
    SmallVector<int64_t, 1> dims;
    auto builder = mlir::Builder(getContext());
    Type scalarTy =
        mlir::RankedTensorType::get(dims, builder.getIntegerType(64));
    getResult().setType(scalarTy);
  }
  // ElementType of I64 will be checked by verifier
  return success();
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
// IdentityOp
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXIdentityOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXIdentityOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  getResult().setType(getOperand().getType());
  return success();
}






// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//
// Reshape
//===----------------------------------------------------------------------===//

LogicalResult ONNXReshapeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape tensor is specified.
  if (!data().getType().isa<RankedTensorType>())
    return success();

  if (!shape().getType().isa<RankedTensorType>())
    return success();

  // Only rank 1 shape tensors are supported.
  auto shapeTensorTy = shape().getType().cast<RankedTensorType>();
  if (shapeTensorTy.getShape().size() != 1)
    return emitError("Shape tensor must have rank one");

  // Shape tensor must have constant shape.
  int64_t outputRank = shapeTensorTy.getShape()[0];
  if (outputRank < 0)
    return emitError("Shape tensor must have constant shape");

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXReshapeOpShapeHelper, ONNXReshapeOp,
      ONNXReshapeOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

LogicalResult ONNXTransposeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXTransposeOpShapeHelper, ONNXTransposeOp,
      ONNXTransposeOpAdaptor>(*this, elementType);
}










//===----------------------------------------------------------------------===//
// Pad
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::verify() {
  ShapedType dataTy = data().getType().cast<ShapedType>();
  Type constTy = constant_value().getType();

  if (!constTy.isa<NoneType>()) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = constTy.cast<ShapedType>();
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }
  return success();
}

LogicalResult ONNXPadOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>() ||
      !pads().getType().isa<RankedTensorType>())
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXPadOpShapeHelper, ONNXPadOp,
      ONNXPadOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// Unsqueeze
//===----------------------------------------------------------------------===//

// Update axes attribute so that it contains only positive values.
// Helper functions for both Unsqueeze and Squeeze Ops
template <typename Op>
void updateNegativeAxis(Op *op, ArrayRef<int64_t> axes) {
  OpBuilder builder(op->getContext());
  if (auto axesConstOp = getONNXConstantOp(op->axes())) {
    auto tensorType = axesConstOp.getType().template cast<RankedTensorType>();
    auto constDenseAttr = mlir::DenseElementsAttr::get(tensorType, axes);
    builder.setInsertionPoint(*op);
    auto constOp = builder.create<mlir::ONNXConstantOp>(
        op->getLoc(), mlir::Attribute(), constDenseAttr);
    mlir::Value constRes = constOp.output();
    op->setOperand(1, constRes);
  } else {
    llvm_unreachable("cannot update axes for non-constant Op");
  }
}

template <typename Op>
void updateNegativeAxisV11(Op *op, ArrayRef<int64_t> axes) {
  auto builder = mlir::Builder(op->getContext());
  ArrayRef<int64_t> defaultRefs(axes);
  op->axesAttr(builder.getI64ArrayAttr(defaultRefs));
}

void updateUnsqueezeOpNegativeAxis(
    ONNXUnsqueezeOp *op, ArrayRef<int64_t> axes) {
  updateNegativeAxis(op, axes);
}

void updateUnsqueezeOpNegativeAxis(
    ONNXUnsqueezeV11Op *op, ArrayRef<int64_t> axes) {
  updateNegativeAxisV11(op, axes);
}

template <typename Op, typename Adaptor, typename ShapeHelper>
LogicalResult ONNXUnsqueezeOpInferShapesCommon(Op *op,
    llvm::Optional<ArrayAttr> axisAttrs,
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!op->data().getType().template isa<RankedTensorType>())
    return success();

  auto operandTy = op->data().getType().template cast<RankedTensorType>();
  auto elementType =
      op->data().getType().template cast<ShapedType>().getElementType();
  int64_t inRank = operandTy.getRank();

  if (!axisAttrs)
    return op->emitError("Axes attribute is required");

  SmallVector<int64_t, 4> axes;
  bool hasNegativeAxis = false;
  int64_t outRank = inRank + axisAttrs.value().size();
  for (auto axisAttr : axisAttrs.value()) {
    int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
    if (axis < -outRank || axis >= outRank)
      return op->emitError("Invalid axis value");
    if (axis < 0) {
      axis = outRank + axis;
      hasNegativeAxis = true;
    }
    if (std::find(axes.begin(), axes.end(), axis) == axes.end())
      axes.emplace_back(axis);
    else
      return op->emitError("Duplicated axes");
  }

  if (hasNegativeAxis) {
    updateUnsqueezeOpNegativeAxis(op, axes);
  }

  return shapeHelperInferShapes<ShapeHelper, Op, Adaptor>(*op, elementType);
}

LogicalResult ONNXUnsqueezeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto builder = mlir::Builder(getContext());
  llvm::Optional<ArrayAttr> optionalAttr;
  if (auto axesConstOp = getONNXConstantOp(axes())) {
    auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
    optionalAttr.emplace(axesAttr);
  } else if (!axes().getType().isa<NoneType>()) {
    // Cannot handle Non-constant axes
    // Hope further transformation may creat constant axes
    return success();
  }
  return ONNXUnsqueezeOpInferShapesCommon<ONNXUnsqueezeOp,
      ONNXUnsqueezeOpAdaptor, ONNXUnsqueezeOpShapeHelper>(
      this, optionalAttr, doShapeInference);
}

LogicalResult ONNXUnsqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return ONNXUnsqueezeOpInferShapesCommon<ONNXUnsqueezeV11Op,
      ONNXUnsqueezeV11OpAdaptor, ONNXUnsqueezeV11OpShapeHelper>(
      this, axes(), doShapeInference);
}

//===----------------------------------------------------------------------===//
// Squeeze
//===----------------------------------------------------------------------===//

// Update axes attribute so that it contains only positive values.
void updateSqueezeOpNegativeAxis(ONNXSqueezeOp *op, ArrayRef<int64_t> axes) {
  updateNegativeAxis(op, axes);
}

void updateSqueezeOpNegativeAxis(ONNXSqueezeV11Op *op, ArrayRef<int64_t> axes) {
  updateNegativeAxisV11(op, axes);
}

template <typename Op, typename Adaptor, typename ShapeHelper>
LogicalResult ONNXSqueezeOpInferShapesCommon(Op *op,
    llvm::Optional<ArrayAttr> axisAttrs,
    std::function<void(mlir::Region &)> doShapeInference) {
  auto operandTy = op->data().getType().template cast<RankedTensorType>();
  auto elementType =
      op->data().getType().template cast<ShapedType>().getElementType();
  int64_t inRank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  bool hasNegativeAxis = false;
  for (auto axisAttr : axisAttrs.value()) {
    int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
    if (axis < -inRank || axis >= inRank)
      return op->emitError("Invalid axis value");
    if (axis < 0) {
      axis = inRank + axis;
      hasNegativeAxis = true;
    }
    if (std::find(axes.begin(), axes.end(), axis) != axes.end())
      return op->emitError("Duplicated axes");
    axes.emplace_back(axis);
  }

  if (hasNegativeAxis) {
    updateSqueezeOpNegativeAxis(op, axes);
  }

  return shapeHelperInferShapes<ShapeHelper, Op, Adaptor>(*op, elementType);
}

// Helper function to return an ArrayAttr from an input shape
// All single dimensions will be returned
ArrayAttr getSqueezeOpAxesFromShape(
    OpBuilder builder, ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> axes;
  for (unsigned int i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) {
      axes.emplace_back(i);
    } else if (shape[i] == -1) {
      llvm_unreachable(
          "only static input shape currently supported with empty axes");
    }
  }
  return builder.getI64ArrayAttr(axes);
}

LogicalResult ONNXSqueezeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  OpBuilder builder(getContext());
  llvm::Optional<ArrayAttr> optionalAttr;

  if (isFromNone(axes())) {
    auto axesAttr = getSqueezeOpAxesFromShape(builder, dataType.getShape());
    optionalAttr.emplace(axesAttr);

    // Create a ConstantOp associated with this Squeeze Op
    auto tensorType =
        RankedTensorType::get(ArrayAttrSize(axesAttr), builder.getI64Type());
    SmallVector<int64_t, 4> values;
    for (auto attr : axesAttr.getValue()) {
      values.emplace_back(attr.cast<IntegerAttr>().getInt());
    }
    auto constDenseAttr =
        DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
    builder.setInsertionPoint(*this);
    auto constOp = builder.create<mlir::ONNXConstantOp>(
        getLoc(), mlir::Attribute(), constDenseAttr);
    mlir::Value constRes = constOp.output();
    setOperand(1, constRes);
  } else if (auto axesConstOp = getONNXConstantOp(axes())) {
    auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
    optionalAttr.emplace(axesAttr);
  } else {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXSqueezeOpInferShapesCommon<ONNXSqueezeOp, ONNXSqueezeOpAdaptor,
      ONNXSqueezeOpShapeHelper>(this, optionalAttr, doShapeInference);
}

LogicalResult ONNXSqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  if (!axes()) {
    OpBuilder builder(getContext());

    auto newAxesAttr = getSqueezeOpAxesFromShape(builder, dataType.getShape());

    // Update the axes attribute
    axesAttr(newAxesAttr);
  }

  return ONNXSqueezeOpInferShapesCommon<ONNXSqueezeV11Op,
      ONNXSqueezeV11OpAdaptor, ONNXSqueezeV11OpShapeHelper>(
      this, axes(), doShapeInference);
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
// Constant
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if ((sparse_value().has_value() && value().has_value()) ||
      (!sparse_value().has_value() && !value().has_value()))
    return emitError("Require exactly one of the two attributes, "
                     "either value or sparse_value");
  ElementsAttr valAttr;
  if (sparse_value().has_value())
    valAttr = sparse_valueAttr().cast<SparseElementsAttr>();
  else
    valAttr = valueAttr().cast<DenseElementsAttr>();
  getResult().setType(valAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatOp::verify() {
  // Cannot verify semantics if the operands do not have a known shape yet.
  ONNXConcatOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  auto commonType =
      operandAdaptor.getOperands().front().getType().cast<ShapedType>();
  ArrayRef<int64_t> commonShape = commonType.getShape();
  int64_t commonRank = commonShape.size();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  if (axisIndex < -commonRank || axisIndex >= commonRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-commonRank, commonRank - 1));

  if (axisIndex < 0)
    axisIndex += commonRank;

  // All input tensors must have the same shape, except for the dimension size
  // of the axis to concatenate on.
  for (Value operand : operandAdaptor.getOperands()) {
    ArrayRef<int64_t> operandShape =
        operand.getType().cast<ShapedType>().getShape();
    int64_t operandRank = operandShape.size();
    if (operandRank != commonRank)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), operand, operandRank,
          std::to_string(commonRank));

    for (int64_t dim = 0; dim < commonRank; ++dim) {
      if (dim == axisIndex)
        continue;
      if (operandShape[dim] != -1 && commonShape[dim] != -1 &&
          operandShape[dim] != commonShape[dim])
        return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
            *this->getOperation(), operand, dim, operandShape[dim],
            std::to_string(commonShape[dim]));
    }
  }

  return success();
}

LogicalResult ONNXConcatOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // The check of constraints is kept
  // However, current check handles dynamic dim only for the concat dim
  int inputNum = getNumOperands();
  for (int i = 0; i < inputNum; ++i) {
    if (!getOperand(i).getType().isa<RankedTensorType>())
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

  return shapeHelperInferShapes<ONNXConcatOpShapeHelper, ONNXConcatOp,
      ONNXConcatOpAdaptor>(*this, commonType.getElementType());
}

//===----------------------------------------------------------------------===//
// ConcatFromSequence
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatFromSequenceOp::verify() {
  ONNXConcatFromSequenceOpAdaptor operandAdaptor(*this);
  if (!hasShapeAndRank(operandAdaptor.input_sequence()))
    return success(); // Won't be able to do any checking at this stage.

  Value inputSequence = operandAdaptor.input_sequence();
  assert(inputSequence.getType().isa<SeqType>() &&
         "Incorrect type for a sequence");
  auto seqType = inputSequence.getType().cast<SeqType>();
  auto elemType = seqType.getElementType().cast<ShapedType>();
  int64_t rank = elemType.getShape().size();
  int64_t axisIndex = axis();
  int64_t newAxisIndex = new_axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(inputs).
  // When `new_axis` is 1, accepted range is [-r-1,r].
  if (newAxisIndex == 1) {
    if (axisIndex < (-rank - 1) || axisIndex > rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank - 1, rank));
  } else {
    if (axisIndex < -rank || axisIndex >= rank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axisIndex,
          onnx_mlir::Diagnostic::Range<int64_t>(-rank, rank - 1));
  }

  return success();
}

LogicalResult ONNXConcatFromSequenceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}

//===----------------------------------------------------------------------===//
// RNN
//===----------------------------------------------------------------------===//

LogicalResult ONNXRNNOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  int gates = 1;
  return RNNShapeInference(this, gates);
}

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

LogicalResult ONNXLSTMOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  int gates = 4;
  return RNNShapeInference(this, gates);
}

//===----------------------------------------------------------------------===//
// GRU
//===----------------------------------------------------------------------===//

LogicalResult ONNXGRUOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  int gates = 3;
  return RNNShapeInference(this, gates);
}

//===----------------------------------------------------------------------===//
// Split
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitOp::verify() {
  ONNXSplitOpAdaptor operandAdaptor(*this);
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input))
    return success(); // Won't be able to do any checking at this stage.

  auto inputType = input.getType().cast<ShapedType>();
  int64_t inputRank = inputType.getShape().size();
  int64_t axisIndex = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(input).
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisIndex,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));

  return success();
}

LogicalResult ONNXSplitOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  SmallVector<Type> elementTypes(getNumResults(), elementType);

  return shapeHelperInferMultipleShapes<ONNXSplitOpShapeHelper, ONNXSplitOp,
      ONNXSplitOpAdaptor>(*this, elementTypes);
}

LogicalResult ONNXSplitV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape isn't known yet.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  SmallVector<Type> elementTypes(getNumResults(), elementType);

  return shapeHelperInferMultipleShapes<ONNXSplitV11OpShapeHelper,
      ONNXSplitV11Op, ONNXSplitV11OpAdaptor>(*this, elementTypes);
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
// Flatten
//===----------------------------------------------------------------------===//

LogicalResult ONNXFlattenOp::verify() {
  // Cannot verify constraints if the input shape is not yet known.
  if (!hasShapeAndRank(input()))
    return success();

  auto inputType = input().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  int64_t axisValue = axis();

  // axis attribute must be in the range [-r,r], where r = rank(input).
  if (axisValue < -inputRank || axisValue > inputRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank));

  return success();
}

LogicalResult ONNXFlattenOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet known.
  if (!hasShapeAndRank(input()))
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXFlattenOpShapeHelper, ONNXFlattenOp,
      ONNXFlattenOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// Resize
//===----------------------------------------------------------------------===//

LogicalResult ONNXResizeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!X().getType().isa<RankedTensorType>()) {
    return success();
  }
  auto inputTy = X().getType().cast<RankedTensorType>();

  // Output should at least has the same rank as X input
  if (!getResult().getType().isa<RankedTensorType>()) {
    SmallVector<int64_t, 4> dims(inputTy.getRank(), -1);
    getResult().setType(RankedTensorType::get(dims, inputTy.getElementType()));
  }

  if (isFromNone(scales()) == isFromNone(sizes())) {
    if (isFromNone(scales()))
      return emitError("scales() and sizes() can not be both None");
    else
      return emitError("scales() and sizes() can not be both defined");
  }

  // Current implementation handles constant scales only
  if (!isFromNone(scales())) {
    DenseElementsAttr scalesAttrs =
        getDenseElementAttributeFromONNXValue(scales());
    if (!scalesAttrs) {
      return success();
    }

    SmallVector<float, 4> scalesConstant;
    for (auto scaleAttr : scalesAttrs.getValues<FloatAttr>()) {
      scalesConstant.emplace_back(scaleAttr.getValueAsDouble());
    }

    SmallVector<int64_t, 4> dims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      int newDim;
      if (inputTy.getShape()[i] == -1)
        newDim = -1;
      else
        newDim = inputTy.getShape()[i] * scalesConstant[i];
      dims.emplace_back(newDim);
    }

    updateType(getResult(), dims, inputTy.getElementType());
  } else {
    DenseElementsAttr sizesAttrs =
        getDenseElementAttributeFromONNXValue(sizes());
    if (!sizesAttrs) {
      return success();
    }

    SmallVector<int64_t, 4> sizesConstant;
    for (auto sizeAttr : sizesAttrs.getValues<IntegerAttr>()) {
      sizesConstant.emplace_back(sizeAttr.getInt());
    }

    updateType(getResult(), sizesConstant, inputTy.getElementType());
  }
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
// Shape
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return success();

  // Output is an 1D int64 tensor containing the shape of the input tensor.
  auto elementType = IntegerType::get(getContext(), 64);
  return shapeHelperInferShapes<ONNXShapeOpShapeHelper, ONNXShapeOp,
      ONNXShapeOpAdaptor>(*this, elementType);
}

LogicalResult ONNXShapeOp::verify() {
  if (!data().getType().isa<RankedTensorType>())
    return success();
  ONNXShapeOpAdaptor operandAdaptor(*this);
  int64_t start;
  int64_t end;
  std::tie(start, end) = getDataShapeBounds(operandAdaptor);
  if (start > end)
    return emitOpError() << "Start: " << start << " is after End: " << end;
  return success();
}

//===----------------------------------------------------------------------===//
// Size
//===----------------------------------------------------------------------===//

LogicalResult ONNXSizeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Output is scalar of int64 containing the size of the input tensor.
  SmallVector<int64_t, 1> outDims;
  getResult().setType(
      RankedTensorType::get(outDims, IntegerType::get(getContext(), 64)));
  return success();
}

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return success();

  // Read 'repeats' value.
  if (!repeats().getType().isa<RankedTensorType>())
    return success();

  // 'repeats' tensor is an 1D tensor.
  auto repeatsTensorTy = repeats().getType().cast<RankedTensorType>();
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXTileOpShapeHelper, ONNXTileOp,
      ONNXTileOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// Gather
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherOp::verify() {
  ONNXGatherOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  auto dataType = operandAdaptor.data().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataShape.size();
  int64_t axisValue = axis();

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axisValue < -dataRank || axisValue >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axisValue,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

  return success();
}

LogicalResult ONNXGatherOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (llvm::any_of(this->getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXGatherOpShapeHelper, ONNXGatherOp,
      ONNXGatherOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// GatherElements
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherElementsOp::verify() {
  ONNXGatherElementsOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t axis = this->axis();

  // All inputs must have the same rank, and the rank must be strictly greater
  // than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, std::to_string(dataRank));

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axis < -dataRank || axis >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));
  if (axis < 0)
    axis += dataRank;

  // All index values in 'indices' are expected to be within bounds [-s, s-1]
  // along axis of size s.
  ArrayRef<int64_t> dataShape = dataType.getShape();
  const int64_t dataDimAtAxis = dataShape[axis];
  if (dataDimAtAxis >= 0)
    if (DenseElementsAttr valueAttribute =
            getDenseElementAttributeFromONNXValue(indices))
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t index = value.getInt();
        if (index >= -dataDimAtAxis && index < dataDimAtAxis)
          continue;

        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "indices", index,
            onnx_mlir::Diagnostic::Range<int64_t>(
                -dataDimAtAxis, dataDimAtAxis - 1));
      }

  return success();
}

LogicalResult ONNXGatherElementsOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape is not yet known.
  if (llvm::any_of(this->getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXGatherElementsOpShapeHelper,
      ONNXGatherElementsOp, ONNXGatherElementsOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// GatherND
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherNDOp::verify() {
  ONNXGatherNDOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t b = batch_dims();

  // 'data' and 'indices' must have rank strictly greater than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, "> 0");

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  int64_t indicesLastDim = indicesShape[indicesRank - 1];

  // b must be smaller than min(rank(data), rank(indices).
  int64_t minDataAndIndicesRank = std::min(dataRank, indicesRank);
  if (b >= minDataAndIndicesRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "batch_dims", b,
        onnx_mlir::Diagnostic::Range<int64_t>(0, minDataAndIndicesRank - 1));

  // The first b dimensions of the shape of 'indices' and 'data' must be equal.
  for (int64_t i = 0; i < b; ++i) {
    int64_t dataDim = dataShape[i];
    int64_t indicesDim = indicesShape[i];
    if (indicesDim < 0 || dataDim < 0)
      continue;
    if (indicesDim != dataDim)
      return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
          *this->getOperation(), indices, i, indicesShape[i],
          std::to_string(dataShape[i]));
  }

  // Let r = rank(data), indices.shape[-1] must be in the range [1, r-b].
  if (indicesLastDim == 0)
    return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
        *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
        ">= 1");
  if (indicesLastDim > dataRank - b)
    return onnx_mlir::Diagnostic::emitDimensionHasUnexpectedValueError(
        *this->getOperation(), indices, indicesRank - 1, indicesLastDim,
        "<= " + std::to_string(dataRank - b));

  // All values in 'indices' are expected to satisfy the inequality:
  //   -data.shape[b + i] <= indices[...,i] <= (data.shape[b + i]-1)].
  if (DenseElementsAttr valueAttribute =
          getDenseElementAttributeFromONNXValue(indices)) {
    int flatIndex = 0;
    for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t indexValue = value.getInt();
      int64_t gatherAxis = b + (flatIndex % indicesLastDim);
      int64_t dataDimAtAxis = dataShape[gatherAxis];
      if (dataDimAtAxis >= 0) {
        if (indexValue < -dataDimAtAxis || indexValue > dataDimAtAxis - 1)
          return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
              *this->getOperation(),
              "indices[" + std::to_string(flatIndex) + "]", indexValue,
              onnx_mlir::Diagnostic::Range<int64_t>(
                  -dataDimAtAxis, dataDimAtAxis - 1));
      }
      flatIndex++;
    }
  }

  return success();
}

LogicalResult ONNXGatherNDOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the shape of the output if the inputs shape is not yet known.
  if (llvm::any_of(
          this->getOperands(), [](Value op) { return !hasShapeAndRank(op); }))
    return success();

  // The output rank is given by:
  //   rank(output) = rank(indices) + rank(data) - indices_shape[-1] - 1 - b.
  // Therefore 'indices.shape[-1]' must be known in order to compute the output
  // shape.
  ArrayRef<int64_t> indicesShape =
      indices().getType().cast<ShapedType>().getShape();
  int64_t indicesRank = indicesShape.size();
  if (indicesShape[indicesRank - 1] < 0)
    return success(); // cannot infer the oputput shape yet.

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXGatherNDOpShapeHelper, ONNXGatherNDOp,
      ONNXGatherNDOpAdaptor>(*this, elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOfShape
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOfShapeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {

  Type elementType;

  // 'value' attribute is a one-element tensor whose value and datatype are
  // used to set the output tensor value and datatype.
  if (value().has_value()) {
    elementType =
        valueAttr().cast<DenseElementsAttr>().getType().getElementType();
  } else {
    // If 'value' attribute is not specified, it defaults to a tensor of
    // value 0 and datatype float32.
    elementType = FloatType::getF32(getContext());

    llvm::SmallVector<int64_t, 2> dims(1, 1);
    auto tensorType = mlir::RankedTensorType::get(dims, elementType);

    llvm::SmallVector<float, 1> values(1, 0.);
    valueAttr(
        mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values)));
  }

  // 'input' must be a 1D tensor.
  auto inputShape = input().getType().cast<RankedTensorType>().getShape();
  if (inputShape[0] == 0) {
    // If 'input' is an empty tensor, the output would be a scalar.
    getResult().setType(RankedTensorType::get({}, elementType));
    return success();
  }

  // Calculate output dimensions.
  SmallVector<int64_t, 4> outputDims(inputShape[0], -1);
  // If 'input' is a constant, check whether its values are valid or not.
  // If the values are valid, it is possible to infer shape.
  if (auto constantOp = getONNXConstantOp(input())) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    // Get repeat values from valueAttribute.
    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i < inputShape[0]; ++i) {
      auto dim = (*valueIt++).cast<IntegerAttr>().getInt();
      outputDims[i] = dim;
    }
  }

  updateType(getResult(), outputDims, elementType);
  return success();
}

LogicalResult ONNXConstantOfShapeOp::verify() {
  ONNXConstantOfShapeOpAdaptor operandAdaptor(*this);
  auto input = operandAdaptor.input();
  if (!hasShapeAndRank(input))
    return success();

  auto inputShape = input.getType().cast<RankedTensorType>().getShape();
  if (inputShape.size() != 1)
    return emitOpError("Input tensor must be a 1D tensor");
  if (inputShape[0] == -1)
    return emitOpError("Input tensor must have static shape");

  // Calculate output dimensions.
  SmallVector<int64_t, 4> outputDims(inputShape[0], -1);
  // If 'input' is a constant, check whether its values are valid or not.
  // If the values are valid, it is possible to infer shape.
  if (auto constantOp = getONNXConstantOp(input)) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    // Get repeat values from valueAttribute.
    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i < inputShape[0]; ++i) {
      auto dim = (*valueIt++).cast<IntegerAttr>().getInt();
      if (dim < 0)
        return emitOpError("All values of the input tensor must be >=0");
    }
    // Unreachable error: Type error will trigger before this occurs
    // No test needed for this error -----
    if (valueIt != valueAttribute.getValues<IntegerAttr>().end())
      return emitOpError(
          "Constant value must have same length as output's rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Slice
//===----------------------------------------------------------------------===//

LogicalResult ONNXSliceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return success();

  const auto startsType =
      this->getOperand(1).getType().dyn_cast<RankedTensorType>();
  assert(startsType != nullptr && "starts type is not a RankedTensorType");
  auto startsDim = startsType.getShape()[0];
  {
    OpBuilder builder(this->getContext());
    const auto elementType = builder.getIntegerType(64);
    const auto tensorType =
        mlir::RankedTensorType::get({startsDim}, elementType);

    // If axes is not specified, default to [0, ..., ndim-1]
    if (this->getOperand(3).getType().isa<NoneType>()) {
      SmallVector<int64_t, 1> vals = {};
      for (size_t s = 0; s < (size_t)startsDim; ++s)
        vals.emplace_back(s);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(vals));
      builder.setInsertionPoint(*this);
      auto constantOp = builder.create<mlir::ONNXConstantOp>(
          this->getLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = constantOp.output();
      this->setOperand(3, constantResult);
    }

    // If steps is not specified, default to [1, ..., 1]
    if (this->getOperand(4).getType().isa<NoneType>()) {
      SmallVector<int64_t, 1> vals(startsDim, 1);
      auto constantDenseAttribute =
          mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(vals));
      builder.setInsertionPoint(*this);
      auto constantOp = builder.create<mlir::ONNXConstantOp>(
          this->getLoc(), mlir::Attribute(), constantDenseAttribute);
      mlir::Value constantResult = constantOp.output();
      this->setOperand(4, constantResult);
    }
  }

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXSliceOpShapeHelper, ONNXSliceOp,
      ONNXSliceOpAdaptor>(*this, elementType);
}

//===----------------------------------------------------------------------===//
// Expand
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::verify() {
  ONNXExpandOpAdaptor operandAdaptor = ONNXExpandOpAdaptor(*this);
  // Get operands.
  auto shape = operandAdaptor.shape();
  // Check input.
  auto shapeType = shape.getType().dyn_cast_or_null<ShapedType>();
  if (shapeType && shapeType.hasRank()) {
    if (shapeType.getRank() != 1)
      return emitOpError("Shape has a rank of 1");
  }
  return success();
}

LogicalResult ONNXExpandOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!input().getType().isa<RankedTensorType>())
    return success();
  if (!shape().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXExpandOpShapeHelper, ONNXExpandOp,
      ONNXExpandOpAdaptor>(*this, elementType);
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



//===----------------------------------------------------------------------===//
// ONNXCompressOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXCompressOp::verify() {
  // Cannot check constraints if the shape of the inputs is not yet knwon.
  ONNXCompressOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  int64_t inputRank = input().getType().cast<ShapedType>().getRank();
  Optional<int64_t> optionalAxis = axis();

  if (optionalAxis.has_value()) {
    // axis attribute must be in the range [-r,r-1], where r = rank(input).
    int64_t axis = optionalAxis.value();
    if (axis < -inputRank || axis >= inputRank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-inputRank, inputRank - 1));
  }

  int64_t condRank = condition().getType().cast<ShapedType>().getRank();
  if (condRank != 1)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "condition", condRank,
        onnx_mlir::Diagnostic::Range<int64_t>(1, 1));

  return success();
}

LogicalResult ONNXCompressOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the input shape is not yet knwon.
  if (!hasShapeAndRank(input()))
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXCompressOpShapeHelper, ONNXCompressOp,
      ONNXCompressOpAdaptor>(*this, elementType);
}

LogicalResult ONNXDepthToSpaceOp::verify() {
  ONNXDepthToSpaceOpAdaptor operandAdaptor(*this);

  // Check input.
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return emitOpError("Input should have a rank of four");

  // Check blocksize.
  int64_t blocksize = operandAdaptor.blocksize();
  if (blocksize < 0)
    return emitOpError("Blocksize should be non negative");

  int64_t C = inputShape[1];
  if (C != -1 && C % (blocksize * blocksize) != 0)
    return emitOpError("The input tensor depth must be divisible by the "
                       "(blocksize * blocksize)");

  // Check mode.
  StringRef mode = operandAdaptor.mode();
  if (mode != "DCR" && mode != "CRD")
    return emitOpError("Mode must be DCR or CRD");

  return success();
}

LogicalResult ONNXDepthToSpaceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXDepthToSpaceOpShapeHelper,
      ONNXDepthToSpaceOp, ONNXDepthToSpaceOpAdaptor>(*this, elementType);
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


LogicalResult ONNXModOp::verify() {
  Type elementType;
  if (A().getType().isa<ShapedType>())
    elementType = A().getType().cast<ShapedType>().getElementType();
  else
    return emitOpError("Input type must be TensorType or MemRefType");

  // Verify that when the input type is floating point, then `fmod` attribute
  // must be set to 1.
  if (elementType.isa<FloatType>() && (fmod() != 1))
    return emitOpError("fmod must be 1 when the input type is floating point");

  return success();
}

LogicalResult ONNXMultinomialOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return emitError(NOT_IMPLEMENTED_MESSAGE);
}


LogicalResult ONNXNonZeroOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto builder = mlir::Builder(getContext());
  Type inputType = getOperand().getType();
  if (!inputType.isa<RankedTensorType>())
    return success();
  SmallVector<int64_t, 2> dims;
  // The first dimension size is the rank of the input.
  dims.emplace_back(inputType.cast<RankedTensorType>().getRank());
  // The second dimension size is the number of nonzero values in the input.
  // So this dimension size is always unknown at compile time.
  dims.emplace_back(-1);
  getResult().setType(RankedTensorType::get(dims, builder.getI64Type()));
  return success();
}


LogicalResult ONNXOneHotOp::verify() {
  ONNXOneHotOpAdaptor operandAdaptor = ONNXOneHotOpAdaptor(*this);
  // Check indices.
  Value indices = operandAdaptor.indices();
  if (hasShapeAndRank(indices)) {
    // Get rank.
    int64_t indicesRank = indices.getType().cast<ShapedType>().getRank();
    // Verify axis.
    int64_t axisValue = axis();
    // Unusually, with a rank of 3, acceptable values are 0 (before first) to 3
    // (after last).
    if (axisValue < 0)
      axisValue += indicesRank + 1;
    if (!(axisValue >= 0 && axisValue <= indicesRank))
      return emitOpError("OneHot axis value is out of range");
  }
  // Check that values is a rank 2 with 2 elements
  Value values = operandAdaptor.values();
  if (hasShapeAndRank(values)) {
    ShapedType valuesShape = values.getType().cast<ShapedType>();
    if (valuesShape.getRank() != 1)
      return emitOpError("OneHot values must be 1D tensor");
    int64_t dim = valuesShape.getDimSize(0);
    if (dim >= 0 && dim != 2)
      return emitOpError("OneHot values must be 1D tensor with 2 elements");
  }
  // Depth is a scalar, check when its a tensor of rank 0 or 1.
  Value depth = operandAdaptor.depth();
  if (hasShapeAndRank(depth)) {
    ShapedType depthShape = depth.getType().cast<ShapedType>();
    if (depthShape.getRank() == 1) {
      int64_t dim = depthShape.getDimSize(0);
      if (dim >= 0 && dim != 1)
        return emitOpError("OneHot depth can be 1D tensor with 1 elements");
    } else {
      if (depthShape.getRank() > 1)
        return emitOpError("OneHot depth must be 0 or 1D tensor");
    }
  }
  return success();
}

LogicalResult ONNXOneHotOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!indices().getType().isa<RankedTensorType>())
    return success();

  auto elementType = values().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXOneHotOpShapeHelper, ONNXOneHotOp,
      ONNXOneHotOpAdaptor>(*this, elementType);
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

LogicalResult ONNXRangeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // All inputs must be valid ranked tensors.
  if (!start().getType().isa<RankedTensorType>())
    return success();

  if (!limit().getType().isa<RankedTensorType>())
    return success();

  if (!delta().getType().isa<RankedTensorType>())
    return success();

  auto startTensorTy = start().getType().cast<RankedTensorType>();
  auto limitTensorTy = limit().getType().cast<RankedTensorType>();
  auto deltaTensorTy = delta().getType().cast<RankedTensorType>();

  // Only rank 0 or 1 input tensors are supported.
  if (startTensorTy.getShape().size() > 1)
    return emitError("start tensor must have rank zero or one");
  if (limitTensorTy.getShape().size() > 1)
    return emitError("limit tensor must have rank zero or one");
  if (deltaTensorTy.getShape().size() > 1)
    return emitError("delta tensor must have rank zero or one");

  // If tensor is rank 1 then the dimension has to be 1.
  if (startTensorTy.getShape().size() == 1 && startTensorTy.getShape()[0] > 1)
    return emitError("start tensor of rank one must have size one");
  if (limitTensorTy.getShape().size() == 1 && limitTensorTy.getShape()[0] > 1)
    return emitError("limit tensor of rank one must have size one");
  if (deltaTensorTy.getShape().size() == 1 && deltaTensorTy.getShape()[0] > 1)
    return emitError("delta tensor of rank one must have size one");

  // Only int or float input types are supported:
  // tensor(float), tensor(double), tensor(int16), tensor(int32),
  // tensor(int64)
  if (!startTensorTy.getElementType().isIntOrFloat())
    return emitError("start tensor type is not int or float");
  if (!limitTensorTy.getElementType().isIntOrFloat())
    return emitError("limit tensor type is not int or float");
  if (!deltaTensorTy.getElementType().isIntOrFloat())
    return emitError("delta tensor type is not int or float");

  // Additional condition for simplicity, enforce that all inputs have the
  // exact same element type:
  if (startTensorTy.getElementType() != limitTensorTy.getElementType() ||
      startTensorTy.getElementType() != deltaTensorTy.getElementType())
    return emitError("all inputs must have the exact same input type");

  // Number of elements, default is unknown so -1:
  int64_t number_of_elements = -1;

  // Check if input is constant. All inputs must be
  // constant for this path to be used.
  auto constantStart = getONNXConstantOp(start());
  auto constantLimit = getONNXConstantOp(limit());
  auto constantDelta = getONNXConstantOp(delta());
  if (constantStart && constantLimit && constantDelta) {
    // Get all inputs:
    double start = getScalarValue<double>(constantStart, startTensorTy);
    double limit = getScalarValue<double>(constantLimit, limitTensorTy);
    double delta = getScalarValue<double>(constantDelta, deltaTensorTy);

    // Compute size:
    number_of_elements = (int64_t)ceil((limit - start) / delta);

    // When no elements are present create a dynamic tensor.
    // TODO: represent an empty tensor for this case.
    if (number_of_elements <= 0)
      number_of_elements = -1;
  }

  SmallVector<int64_t, 1> dims(1, number_of_elements);
  updateType(getResult(), dims, startTensorTy.getElementType());
  return success();
}

LogicalResult ONNXReverseSequenceOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!input().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXReverseSequenceOpShapeHelper,
      ONNXReverseSequenceOp, ONNXReverseSequenceOpAdaptor>(*this, elementType);
}

LogicalResult ONNXReverseSequenceOp::verify() {
  ONNXReverseSequenceOpAdaptor operandAdaptor =
      ONNXReverseSequenceOpAdaptor(*this);

  auto sequence_lensTy =
      operandAdaptor.sequence_lens().getType().dyn_cast<RankedTensorType>();
  auto inputTy = operandAdaptor.input().getType().dyn_cast<RankedTensorType>();

  // sequence_lens should be 1D tensor
  if (sequence_lensTy) {
    if (sequence_lensTy.getRank() != 1)
      return emitOpError("sequence_lens of ReverseSequnce should be 1D tensor");
  }

  if (inputTy) {
    if (inputTy.getRank() < 2)
      return emitOpError(
          "input of Reversesequence should be 2D or higher rank tensor");
  }

  if (sequence_lensTy && inputTy) {
    int64_t batchAxis = batch_axis();
    if (sequence_lensTy.getShape()[0] != -1 &&
        inputTy.getShape()[batchAxis] != -1) {
      if (sequence_lensTy.getShape()[0] != inputTy.getShape()[batchAxis]) {
        return emitOpError("Length of sequence_lens should match the sizeof  "
                           "batch axis of the input");
      }
    }
  }

  return success();
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

//===----------------------------------------------------------------------===//
// ONNXScatterElements
//===----------------------------------------------------------------------===//

LogicalResult ONNXScatterElementsOp::verify() {
  ONNXScatterElementsOpAdaptor operandAdaptor(*this);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // Won't be able to do any checking at this stage.

  // Get operands and attributes.
  Value data = operandAdaptor.data();
  Value indices = operandAdaptor.indices();
  Value updates = operandAdaptor.updates();
  auto dataType = data.getType().cast<ShapedType>();
  auto indicesType = indices.getType().cast<ShapedType>();
  auto updatesType = updates.getType().cast<ShapedType>();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  int64_t updatesRank = updatesType.getRank();
  int64_t axis = this->axis();

  // All inputs must have the same rank, and the rank must be strictly greater
  // than zero.
  if (dataRank < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), data, dataRank, "> 0");
  if (indicesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), indices, indicesRank, std::to_string(dataRank));
  if (updatesRank != dataRank)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), updates, updatesRank, std::to_string(dataRank));

  // axis attribute must be in the range [-r,r-1], where r = rank(data).
  if (axis < -dataRank || axis >= dataRank)
    return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
        *this->getOperation(), "axis", axis,
        onnx_mlir::Diagnostic::Range<int64_t>(-dataRank, dataRank - 1));

  if (axis < 0)
    axis += dataRank;

  // All index values in 'indices' are expected to be within bounds [-s, s-1]
  // along axis of size s.
  ArrayRef<int64_t> dataShape = dataType.getShape();
  const int64_t dataDimAtAxis = dataShape[axis];
  if (dataDimAtAxis >= 0) {
    if (DenseElementsAttr valueAttribute =
            getDenseElementAttributeFromONNXValue(indices)) {
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t index = value.getInt();
        if (index >= -dataDimAtAxis && index < dataDimAtAxis)
          continue;

        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "indices", index,
            onnx_mlir::Diagnostic::Range<int64_t>(
                -dataDimAtAxis, dataDimAtAxis - 1));
      }
    }
  }

  return success();
}





//===----------------------------------------------------------------------===//
// ONNXSpaceToDepth
//===----------------------------------------------------------------------===//

LogicalResult ONNXSpaceToDepthOp::verify() {
  ONNXSpaceToDepthOpAdaptor operandAdaptor(*this);

  // Check input.
  Value input = operandAdaptor.input();
  if (!hasShapeAndRank(input)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return emitOpError("Input should have a rank of four");

  // Check blocksize.
  int64_t blocksize = operandAdaptor.blocksize();
  if (blocksize < 0)
    return emitOpError("Blocksize should be non negative");

  int64_t H = inputShape[2];
  int64_t W = inputShape[3];

  if (H != -1 && H % blocksize != 0)
    return emitOpError(
        "The input tensor height must be divisible by the block size");
  if (W != -1 && W % blocksize != 0)
    return emitOpError(
        "The input tensor width must be divisible by the block size");

  return success();
}

LogicalResult ONNXSpaceToDepthOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no input shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return success();

  auto elementType = input().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXSpaceToDepthOpShapeHelper,
      ONNXSpaceToDepthOp, ONNXSpaceToDepthOpAdaptor>(*this, elementType);
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
// TopK
//===----------------------------------------------------------------------===//

#if 0 // hi alex
LogicalResult ONNXTopKOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape isn't known yet.
  if (llvm::any_of(this->getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  Builder b(getContext());
  auto elementType = X().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferMultipleShapes<ONNXTopKOpShapeHelper, ONNXTopKOp,
      ONNXTopKOpAdaptor>(*this, {elementType, b.getI64Type()});
}
#endif

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

#if 0 // hi alex

LogicalResult ONNXCategoryMapperOp::verify() {
  ONNXCategoryMapperOpAdaptor operandAdaptor(*this);

  // Check input.
  const Value X = operandAdaptor.X();
  if (!hasShapeAndRank(X)) {
    // Won't be able to do any checking at this stage.
    return success();
  }

  ShapedType inputType = X.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  if (!elementType.isInteger(64) && !elementType.isa<ONNXStringType>())
    return emitOpError("input must be a tensor of int64 or string");

  // Check attributes.
  if (!cats_int64s())
    return emitOpError("cats_int64 attribute must be present");
  if (!cats_strings())
    return emitOpError("cats_strings attribute must be present");
  if (ArrayAttrSize(cats_int64s()) != ArrayAttrSize(cats_strings()))
    return emitOpError("cats_int64 and cats_strings should have the same size");

  if (elementType.isInteger(64) && !default_stringAttr())
    return emitOpError("'default_string' attribute is missing.");
  if (elementType.isa<ONNXStringType>() && !default_int64Attr())
    return emitOpError("'default_int64' attribute is missing.");

  return success();
}

LogicalResult ONNXCategoryMapperOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>())
    return success();

  Type inputElementType = X().getType().cast<ShapedType>().getElementType();
  assert((inputElementType.isInteger(64) ||
             inputElementType.isa<ONNXStringType>()) &&
         "Input tensor must have int64 or string element type.");

  Type outputElementType;
  if (inputElementType.isInteger(64))
    outputElementType = ONNXStringType::get(getContext());
  else
    outputElementType = IntegerType::get(getContext(), /*width=*/64);

  return shapeHelperInferShapes<ONNXCategoryMapperOpShapeHelper,
      ONNXCategoryMapperOp, ONNXCategoryMapperOpAdaptor>(
      *this, outputElementType);
}

#endif

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
// DimOp
//===---------------------------------------------------------------------===//

LogicalResult ONNXDimOp::verify() {
  // Input data must be ranked.
  if (!hasShapeAndRank(this->data()))
    return failure();
  // Axis must be in [0, rank -1].
  int64_t axis = this->axis();
  return failure((axis < 0) || (axis >= getRank(this->data().getType())));
}

LogicalResult ONNXDimOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  OpBuilder b(getContext());
  getResult().setType(RankedTensorType::get({1}, b.getI64Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.cpp.inc"
