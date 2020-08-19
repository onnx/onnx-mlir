//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "ONNXOps.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace mlir::onnxmlir;

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//

static size_t ArrayAttrSize(ArrayAttr a) { return a.size(); }

static size_t ArrayAttrSize(Optional<ArrayAttr> a) {
  return a.getValue().size();
}

static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

static int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.getValue().getValue()[i]).cast<IntegerAttr>().getInt();
}

// Returns the ConstantOp which defines an MLIR Value or null.
static mlir::ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<mlir::ONNXConstantOp>(value.getDefiningOp());
}

// This method substitutes any uses of dimensions and symbols (e.g.
// dim#0 with dimReplacements[0]) in an affine map, simplifies the modified
// affine map, and returns an integer constant.
int64_t AffineMapIntConstant(Builder &builder, AffineMap map,
    ArrayRef<int64_t> dimReplacements, ArrayRef<int64_t> symReplacements,
    unsigned numResultDims, unsigned numResultSyms) {
  // Prepare affine expressions.
  SmallVector<AffineExpr, 4> dimExprs, symExprs;
  for (int64_t dim : dimReplacements) {
    AffineExpr exp = builder.getAffineConstantExpr(dim);
    dimExprs.emplace_back(exp);
  }
  for (int64_t sym : symReplacements) {
    AffineExpr exp = builder.getAffineConstantExpr(sym);
    symExprs.emplace_back(exp);
  }
  // Replace all the affine map's arguments with real values and evaluate the
  // map.
  AffineMap replacedDimMap = map.replaceDimsAndSymbols(
      dimExprs, symExprs, numResultDims, numResultSyms);
  AffineMap simplifiedMap = simplifyAffineMap(replacedDimMap);
  return simplifiedMap.getSingleConstantResult();
}

//===----------------------------------------------------------------------===//
// Get reduction type
//===----------------------------------------------------------------------===//
RankedTensorType getReductionOutputType(
    RankedTensorType operandTy, Optional<ArrayAttr> axesAttrs, APInt keepdims) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs != llvm::None) {
    for (auto axisAttr : axesAttrs.getValue()) {
      int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (rank + axis);
      assert(axis >= -rank && axis <= rank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }
  } else {
    for (decltype(rank) i = 0; i < rank; ++i) {
      axes.emplace_back(i);
    }
  }

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.emplace_back(true);
    else
      isReductionAxis.emplace_back(false);
  }

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else {
      dims.emplace_back(operandTy.getShape()[i]);
    }
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for dilations.
//===----------------------------------------------------------------------===//
template <class T>
static LogicalResult processConvDilationParam(
    T *op, Optional<ArrayAttr> kernelShape) {
  auto builder = mlir::Builder(op->getContext());
  auto kernelRank = ArrayAttrSize(kernelShape);

  auto dilationsOpt = op->dilations();
  if (dilationsOpt.hasValue()) {
    if (ArrayAttrSize(dilationsOpt) != kernelRank) {
      return op->emitError(
          "dialation rank is not the same as the spatial rank");
    }
    // Test values to be greater than 0.
    for (int i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(dilationsOpt, i) < 1) {
        return op->emitError("dialation value must be nonzero positive");
      }
    }
  } else {
    // Default dilatation is needed, all dimensions init with 1.
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->dilationsAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for strides.
//===----------------------------------------------------------------------===//
template <class T>
static LogicalResult processConvStrideParam(
    T *op, Optional<ArrayAttr> kernelShape) {
  auto builder = mlir::Builder(op->getContext());
  auto kernelRank = ArrayAttrSize(kernelShape);

  auto stridesOpt = op->strides();
  if (stridesOpt.hasValue()) {
    if (ArrayAttrSize(stridesOpt) != kernelRank)
      return op->emitError("strides rank is not the same as the spatial rank");
    // Check values to be greater than 0.
    for (int i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(stridesOpt, i) < 1)
        return op->emitError("strides value must be nonzero positive");
    }
  } else {
    // Default stride is needed, all dimensions init with 1.
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->stridesAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for pads.
//===----------------------------------------------------------------------===//
template <class T>
static LogicalResult processConvPadParam(T *op, ArrayRef<int64_t> inputShape,
    Optional<ArrayAttr> kernelShape, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> dilationsOpt = llvm::None) {
  auto builder = mlir::Builder(op->getContext());

  auto inputRank = inputShape.size();
  auto kernelRank = ArrayAttrSize(kernelShape);
  auto kernelOffset = inputRank - kernelRank;

  // Try to find padding, getting auto_pad attribute first.
  auto autoPad = op->auto_pad();
  // And then investigate the various different cases. Prefill pad values with
  // zeros, the most common case.
  SmallVector<int64_t, 4> actualPads(2 * kernelRank, 0);
  bool updatedPad = false;
  if (autoPad == "NOTSET") {
    auto padsOpt = op->pads();
    if (padsOpt.hasValue()) {
      // Only option where pads are not updated. Pads consists of two entries
      // for each spatial axis.
      if (ArrayAttrSize(padsOpt) != 2 * kernelRank) {
        return op->emitError("pads rank is not twice the spatial rank");
      }
      // Check values, pads cannot be negative.
      for (int i = 0; i < 2 * kernelRank; ++i) {
        if (ArrayAttrIntVal(padsOpt, i) < 0) {
          return op->emitError("pads value must be nonnegative");
        }
      }
    } else {
      // We have notset with no pads, they are assumed to be all zero.
      updatedPad = true;
    }
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    // Reload dialtion and strides as they may have gotten default values.
    updatedPad = true;
    int64_t dilationVal = 1;
    for (int i = 0; i < kernelRank; ++i) {
      auto inputSize = inputShape[kernelOffset + i];
      auto kernelSize = ArrayAttrIntVal(kernelShape, i);
      if (dilationsOpt.hasValue())
        dilationVal = ArrayAttrIntVal(dilationsOpt, i);
      auto strideVal = ArrayAttrIntVal(stridesOpt, i);
      // Output size is input size divided by stride. When stride is 1, then
      // input and output are the same size, which is the usual case. When
      // stride is greater than 1, take the ceil to be sure to have each input
      // value used, as padding will be used to fill the gaps.
      int64_t outputSize = ceil((1.0 * inputSize) / (1.0 * strideVal));
      // Forumla is from ONNX MaxPool, and can be explained as follows. Pads is
      // the difference between the needed values for the computations, minus
      // the input values. The needed values for the computation is the
      // effective side of the kernel plus the number of times we jump to the
      // next kernel. Number of time we jump is (outputSize - 1). That number is
      // multiplied with the size of the jump, namely strideVal. Now for the
      // effective kernel size. It is the kernelSize + the number of times we
      // have dilation holes time the dialtion. The number of dialtion holes is
      // (kernelSize -1). Thus the effective size is "kernelSize +
      // (kernelSize-1)*dialation". This simplifies to "(kernelSize
      // -1)*dialation + 1".
      auto sumOfPad = (outputSize - 1) * strideVal +
                      ((kernelSize - 1) * dilationVal + 1) - inputSize;
      // Pad values are assumed equal on both size, at half the total value.
      actualPads[i] = actualPads[kernelRank + i] = sumOfPad / 2;
      // But if the total pad value is odd, we add 1 to begining or end
      // depending on autoPad value.
      if (sumOfPad % 2 != 0) {
        if (autoPad == "SAME_UPPER") {
          actualPads[kernelRank + i] += 1;
        } else {
          actualPads[i] += 1;
        }
      }
    }
  } else if (autoPad == "VALID") {
    // No pad, default value was set to zero, we are all set.
    updatedPad = true;
  } else {
    return op->emitError("auto_pad of unknown / unsupported value");
  }
  // Set pads values in attributes, if it is needed.
  if (updatedPad) {
    ArrayRef<int64_t> defaultRefs(actualPads);
    op->padsAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  // In all cases now, the acutal pad values are found in the pads attribute.
  op->auto_padAttr(builder.getStringAttr("NOTSET"));
  return success();
}

//===----------------------------------------------------------------------===//
// Support function computing default values for dilations, strides, and pads.
//===----------------------------------------------------------------------===//
template <class T>
static LogicalResult processConvTypeParams(T *op, Value inputOperand) {
  auto builder = mlir::Builder(op->getContext());

  // 1) Get shape of input.
  auto inputShape = inputOperand.getType().cast<RankedTensorType>().getShape();
  auto inputRank = inputShape.size();

  // 2) Get kernel_shape attribute.
  auto kernelShape = op->kernel_shape();

  // Dilation.
  LogicalResult res = processConvDilationParam<T>(op, kernelShape);
  if (failed(res))
    return res;
  auto dilationsOpt = op->dilations();

  // Strides.
  res = processConvStrideParam<T>(op, kernelShape);
  if (failed(res))
    return res;
  auto stridesOpt = op->strides();

  // Pads.
  return processConvPadParam<T>(
      op, inputShape, kernelShape, stridesOpt, dilationsOpt);
}

//===----------------------------------------------------------------------===//
// Compute spatial dimensions given dilations, strides, pads, and ceil mode.
//===----------------------------------------------------------------------===//
static void insertConvSpatialDim(SmallVector<int64_t, 4> *outputDims,
    Builder &builder, ArrayRef<int64_t> xShape, Optional<ArrayAttr> kernelShape,
    Optional<ArrayAttr> padsOpt, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> dilationsOpt = llvm::None, bool ceilMode = false) {
  auto spatialRank = ArrayAttrSize(kernelShape);
  auto spatialOffset = xShape.size() - spatialRank;

  // Get an affine map to compute the output dimension.
  AffineMap dimMap = getConvDimMap(builder, ceilMode);
  for (int i = 0; i < spatialRank; ++i) {
    int64_t res = -1;
    if (xShape[spatialOffset + i] != -1) {
      auto inputSize = xShape[spatialOffset + i];
      auto kernelSize = ArrayAttrIntVal(kernelShape, i);
      auto sumOfPads = ArrayAttrIntVal(padsOpt, i) +
                       ArrayAttrIntVal(padsOpt, spatialRank + i);
      auto strideVal = ArrayAttrIntVal(stridesOpt, i);
      int64_t dilationVal = 1;
      if (dilationsOpt.hasValue())
        dilationVal = ArrayAttrIntVal(dilationsOpt, i);
      res = AffineMapIntConstant(builder, dimMap, {inputSize},
          {kernelSize, sumOfPads, strideVal, dilationVal}, 1, 4);
    }
    outputDims->emplace_back(res);
  }
}

//===----------------------------------------------------------------------===//
// Support function that infers shape for RNN operations.
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult RNNShapeInference(T *op) {
  Value X = op->X();
  Value W = op->W();
  Value R = op->R();

  if (!X.getType().isa<RankedTensorType>() ||
      !W.getType().isa<RankedTensorType>() ||
      !R.getType().isa<RankedTensorType>()) {
    return op->emitError("Input tensor not ranked");
  }

  auto xTy = X.getType().cast<RankedTensorType>();
  auto elementType = xTy.getElementType();

  // xShape :: [seq_length, batch_size, input_size]
  auto xShape = xTy.getShape();
  // wShape :: [num_directions, 4*hidden_size, input_size]
  auto wShape = W.getType().cast<RankedTensorType>().getShape();
  // rShape :: [num_directions, 4*hidden_size, hidden_size]
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

  // Get sequence length, batch size and input size.
  auto sequenceLength = xShape[0];
  auto batchSize = xShape[1];
  auto inputSize = xShape[2];

  // Get hidden size from hidden_size attribute.
  int64_t hiddenSize = -1;
  if (op->hidden_size().hasValue()) {
    hiddenSize = op->hidden_size().getValue().getSExtValue();
  } else {
    // Infer hidden_size from wShape and rShape if possible.
    if (rShape[2] != -1)
      hiddenSize = rShape[2];
    else if (rShape[1] != -1)
      hiddenSize = rShape[1] / 4;
    else if (wShape[1] != -1)
      hiddenSize = wShape[1] / 4;
    // Update hidden_size attribute.
    if (hiddenSize != -1) {
      auto builder = mlir::Builder(op->getContext());
      op->hidden_sizeAttr(builder.getI64IntegerAttr(hiddenSize));
    }
  }

  // Get direction.
  int numDirection;
  if ((op->direction() == "forward") || (op->direction() == "reverse"))
    numDirection = 1;
  else if (op->direction() == "bidirectional")
    numDirection = 2;
  else
    numDirection = -1;
  if (numDirection == -1) {
    return op->emitError(
        "direction attribute muse be one of the strings: forward, "
        "reverse, and bidirectional");
  }

  // Set result types.
  unsigned numOfResults = op->getNumResults();
  if (numOfResults > 0) {
    // Y :: [seq_length, num_directions, batch_size, hidden_size]
    Type yTy = op->getResults()[0].getType();
    if (!yTy.isa<NoneType>()) {
      yTy = RankedTensorType::get(
          {sequenceLength, numDirection, batchSize, hiddenSize}, elementType);
      op->getResults()[0].setType(yTy);
    }
  }
  if (numOfResults > 1) {
    // Y_h :: [num_directions, batch_size, hidden_size]
    Type yhTy = op->getResults()[1].getType();
    if (!yhTy.isa<NoneType>()) {
      yhTy = RankedTensorType::get(
          {numDirection, batchSize, hiddenSize}, elementType);
      op->getResults()[1].setType(yhTy);
    }
  }
  if (numOfResults > 2) {
    // Y_c :: [num_directions, batch_size, hidden_size]
    Type ycTy = op->getResults()[2].getType();
    if (!ycTy.isa<NoneType>()) {
      ycTy = RankedTensorType::get(
          {numDirection, batchSize, hiddenSize}, elementType);
      op->getResults()[2].setType(ycTy);
    }
  }
  return success();
}

static void insertConvTransposeSpatialDim(SmallVectorImpl<int64_t> &outputDims,
    ArrayRef<int64_t> xShape, Optional<ArrayAttr> kernelShape,
    Optional<ArrayAttr> padsOpt, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> outputPadsOpt, Optional<ArrayAttr> outputShapeOpt,
    Optional<ArrayAttr> dilationsOpt = llvm::None, bool ceilMode = false) {
  auto xRank = xShape.size();
  auto spatialRank = ArrayAttrSize(kernelShape);
  auto spatialOffset = xRank - spatialRank;

  int64_t dilationVal = 1;
  int64_t outputPadsVal = 0;
  // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +
  // ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
  for (int i = 0; i < spatialRank; ++i) {
    auto inputSize = xShape[spatialOffset + i];
    auto sumOfPads =
        ArrayAttrIntVal(padsOpt, i) + ArrayAttrIntVal(padsOpt, spatialRank + i);
    auto kernelSize = ArrayAttrIntVal(kernelShape, i);
    if (dilationsOpt.hasValue())
      dilationVal = ArrayAttrIntVal(dilationsOpt, i);
    auto strideVal = ArrayAttrIntVal(stridesOpt, i);
    if (outputPadsOpt.hasValue())
      outputPadsVal = ArrayAttrIntVal(outputPadsOpt, i);
    // Number of useful values: input plus pad - effective size of kernel (see
    // processConvTypeParams comments to see how this value is derived).
    int64_t res = strideVal * (inputSize - 1) + outputPadsVal +
                  ((kernelSize - 1) * dilationVal + 1) - sumOfPads;
    outputDims.emplace_back(res);
  }
}

//===----------------------------------------------------------------------===//
// ONNXOpsDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ONNXOpsDialect::ONNXOpsDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
      >();
  addTypes<StringType>();
  addTypes<SeqType>();
}

mlir::Type ONNXOpsDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "String")
    return StringType::get(getContext());
  if (keyword == "Seq") {
    if (parser.parseLess())
      return Type();

    SmallVector<mlir::Type, 1> elementTypes;
    do {
      llvm::SMLoc typeLoc = parser.getCurrentLocation();
      mlir::Type elementType;
      if (parser.parseType(elementType))
        return Type();

      // TOFIX: type limitation for Seq? similar but different shape??
      elementTypes.push_back(elementType);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater())
      return Type();
    return SeqType::get(elementTypes);
  }
}

void ONNXOpsDialect::printType(
    mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  if (auto stringType = type.dyn_cast<StringType>()) {
    printer << "String";
  } else if (auto seqType = type.dyn_cast<SeqType>()) {
    printer << "Seq<";
    llvm::interleaveComma(seqType.getElementTypes(), printer);
    printer << '>';
  } else {
    llvm_unreachable("Unexpected onnxmlir type");
  }
}

void ONNXEntryPointOp::build(mlir::OpBuilder &builder,
    mlir::OperationState &state, mlir::FuncOp function, int numInputs,
    int numOutputs) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
      builder.getSymbolRefAttr(function));
  state.addAttribute(ONNXEntryPointOp::getNumInputsAttrName(),
      builder.getI32IntegerAttr(numInputs));
  state.addAttribute(ONNXEntryPointOp::getNumOutputsAttrName(),
      builder.getI32IntegerAttr(numOutputs));
}

ONNXEntryPointOp ONNXEntryPointOp::create(mlir::Location location,
    mlir::FuncOp &func, int numInputs, int numOutputs) {
  mlir::OperationState state(location, "onnx.EntryPoint");
  OpBuilder builder(location->getContext());
  mlir::ONNXEntryPointOp::build(builder, state, func, numInputs, numOutputs);
  Operation *op = mlir::Operation::create(state);
  auto onnxEntryOp = llvm::cast<mlir::ONNXEntryPointOp>(op);
  return onnxEntryOp;
}

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//
// Exp
/// Infer the output shape of the ONNXExpOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXExpOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Atan
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXAtanOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXAtanOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Tan
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXTanOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXTanOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Tanh
/// Infer the output shape of the ONNXTanhOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXTanhOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Sin
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSinOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXSinOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Sinh
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSinhOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXSinhOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Cosh
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXCoshOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXCoshOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Cos
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXCosOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXCosOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Log
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXLogOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXLogOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// HardSigmoid
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXHardSigmoidOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXHardSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Sigmoid
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSigmoidOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Elu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXEluOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXEluOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Relu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXReluOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXReluOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// LeakyRelu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXLeakyReluOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXLeakyReluOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Selu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSeluOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSeluOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// PRelu
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXPReluOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXPReluOp::inferShapes() {
  getResult().setType(getOperand(0).getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Reciprocal
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXReciprocalOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXReciprocalOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Softmax
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSoftmaxOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSoftmaxOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Softplus
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSoftplusOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSoftplusOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Softsign
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSoftsignOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSoftsignOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Sqrt
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSqrtOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSqrtOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Sign
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSignOp. This method is required by
/// the shape inference interface.
LogicalResult ONNXSignOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Abs
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXAbsOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXAbsOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Erf
//===----------------------------------------------------------------------===//

LogicalResult ONNXErfOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Pow
//===----------------------------------------------------------------------===//

LogicalResult ONNXPowOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Add
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXAddOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXAddOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Mul
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXMulOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXMulOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Div
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXDivOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXDivOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Sub
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSubOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXSubOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// And
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXAndOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXAndOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Or
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXOrOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXOrOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Xor
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXXorOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXXorOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
  return success();
}

//===----------------------------------------------------------------------===//
// Sum
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXSumOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXSumOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return emitError("Input tensor(s) not ranked");
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// Max
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXMaxOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXMaxOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return emitError("Input tensor(s) not ranked");
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// Min
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXMinOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXMinOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return emitError("Input tensor(s) not ranked");
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// Neg
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXNegOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXNegOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Identity
//===----------------------------------------------------------------------===//
/// Infer the output shape of the ONNXIdentityOp. This method is required by the
/// shape inference interface.
LogicalResult ONNXIdentityOp::inferShapes() {
  getResult().setType(getOperand().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// MatMul
//===----------------------------------------------------------------------===//

LogicalResult ONNXMatMulOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!A().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");

  auto lhsTy = A().getType().cast<RankedTensorType>();
  auto rhsTy = B().getType().cast<RankedTensorType>();

  SmallVector<int64_t, 2> dims;
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  if (lhsShape.size() < 1 && rhsShape.size() < 1) {
    // Multiplication by scalars is not allowed.
    return emitError("Multiplication by scalar arguments not allowed");
  } else if (lhsShape.size() == 1 && rhsShape.size() == 1) {
    // Special case when both arrays are 1-dimensional and according to
    // numpy rules the types need to be extended to 1xN and Nx1. Helper sizes
    // need to be removed after the multiplication but cannot be removed if all
    // sizes are 1.
    if (lhsShape[0] != -1 && rhsShape[0] != -1 && lhsShape[0] != rhsShape[0])
      return emitError("Attempt to multiply incompatible matrices");
    dims.emplace_back(1);
  } else if (lhsShape.size() == 1 && rhsShape.size() >= 2) {
    // If the first argument is 1-D, it is promoted to a matrix by prepending a
    // 1 to its dimensions. After matrix multiplication the prepended 1 is
    // removed.
    //
    // N MATMUL (s1 x s2 x... x sK x N x P)
    // =>
    // (s1 x s2 x... x sK x P)

    // Check legality of matrix multiplication.
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[0] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[0] != rhsShape[rhsRank - 2])
      return emitError("Attempt to multiply incompatible matrices");
    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      dims.emplace_back(rhsShape[i]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else if (lhsShape.size() >= 2 && rhsShape.size() == 1) {
    // If the second argument is 1-D, it is promoted to a matrix by appending a
    // 1 to its dimensions. After matrix multiplication the appended 1 is
    // removed.
    //
    // (s1 x s2 x... x sK x M x N) MATMUL N
    // =>
    // (s1 x s2 x... x sK x M)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[0] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[0])
      return emitError("Attempt to multiply incompatible matrices");
    for (decltype(lhsRank) i = 0; i < lhsRank - 2; ++i)
      dims.emplace_back(lhsShape[i]);
    dims.emplace_back(lhsShape[lhsRank - 2]);
  } else if (lhsShape.size() > 2 && rhsShape.size() == 2) {
    // (s1 x s2 x... x sK x M x N) MATMUL (N x P)
    // =>
    // (s1 x s2 x... x sK x M x P)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[0] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[0])
      return emitError("Attempt to multiply incompatible matrices");
    for (decltype(lhsRank) i = 0; i < lhsRank - 1; ++i)
      dims.emplace_back(lhsShape[i]);
    dims.emplace_back(rhsShape[1]);
  } else if (lhsShape.size() == 2 && rhsShape.size() > 2) {
    // (M x N) MATMUL (s1 x s2 x... x sK x N x P)
    // =>
    // (s1 x s2 x... x sK x M x P)

    // Check legality of matrix multiplication.
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[1] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[1] != rhsShape[rhsRank - 2])
      return emitError("Attempt to multiply incompatible matrices");
    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      dims.emplace_back(rhsShape[i]);
    dims.emplace_back(lhsShape[0]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else if (lhsShape.size() > 2 && rhsShape.size() > 2) {
    // (s1 x s2 x... x sK x M x N) MATMUL (t1 x t2 x... x tK x N x P)
    // =>
    // (u1 x u2 x... x uK x M x P)

    // Check legality of matrix multiplication.
    unsigned lhsRank = lhsShape.size();
    unsigned rhsRank = rhsShape.size();
    if (lhsShape[lhsRank - 1] != -1 && rhsShape[rhsRank - 2] != -1 &&
        lhsShape[lhsRank - 1] != rhsShape[rhsRank - 2])
      return emitError("Attempt to multiply incompatible matrices");
    // Check and perform broadcasting for the shapes.
    SmallVector<int64_t, 2> lhsBcastShape;
    for (decltype(lhsRank) i = 0; i < lhsRank - 2; ++i)
      lhsBcastShape.emplace_back(lhsShape[i]);
    SmallVector<int64_t, 2> rhsBcastShape;
    for (decltype(rhsRank) i = 0; i < rhsRank - 2; ++i)
      rhsBcastShape.emplace_back(rhsShape[i]);
    if (!getBroadcastedShape(lhsBcastShape, rhsBcastShape, dims))
      return emitError("Broadcasted dimensions are incompatible");
    dims.emplace_back(lhsShape[lhsRank - 2]);
    dims.emplace_back(rhsShape[rhsRank - 1]);
  } else {
    // This case covers all remaining combinations of 1 and 2-D matrices.
    int64_t lhsDim = lhsShape[0];
    int64_t rhsDim = rhsShape[0];
    if (lhsShape.size() > 1) {
      lhsDim = lhsShape[1];
      dims.emplace_back(lhsShape[0]);
    }

    // Check legality of matrix multiplication.
    if (lhsDim != -1 && rhsDim != -1 && lhsDim != rhsDim)
      return emitError("Attempt to multiply incompatible matrices");
    if (rhsShape.size() > 1)
      dims.emplace_back(rhsShape[1]);
  }

  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
  return success();
}

// Gemm
LogicalResult ONNXGemmOp::inferShapes() {
  bool hasBias = !C().getType().isa<NoneType>();
  // Cannot infer shape if no shape exists.
  if (!A().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>() ||
      (hasBias && !C().getType().isa<RankedTensorType>()))
    return emitError("Input tensor(s) not ranked");
  auto lhsTy = A().getType().cast<RankedTensorType>();
  auto rhsTy = B().getType().cast<RankedTensorType>();

  int64_t M, N, K_A, K_B;
  M = (transA() == 0) ? lhsTy.getShape()[0] : lhsTy.getShape()[1];
  K_A = (transA() == 0) ? lhsTy.getShape()[1] : lhsTy.getShape()[0];
  N = (transB() == 0) ? rhsTy.getShape()[1] : rhsTy.getShape()[0];
  K_B = (transB() == 0) ? rhsTy.getShape()[0] : rhsTy.getShape()[1];

  if ((K_A != -1) && (K_B != -1) && (K_A != K_B))
    return emitError("Tensor shapes mismatched");

  if (hasBias) {
    // Check whether bias is unidirectional broadcasting or not.
    auto biasTy = C().getType().cast<RankedTensorType>();
    auto shape = biasTy.getShape();
    int rank = shape.size();
    if ((rank > 2) ||
        (rank >= 1 && shape[rank - 1] != -1 && N != -1 &&
            N != shape[rank - 1] && shape[rank - 1] != 1) ||
        (rank == 2 && shape[rank - 2] != -1 && M != -1 &&
            M != shape[rank - 2] && shape[rank - 2] != 1))
      return emitError("Bias shape mismatched");
  }

  SmallVector<int64_t, 2> dims;
  dims.emplace_back(M);
  dims.emplace_back(N);
  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
  return success();
}

/// BatchNormalizationTestMode
LogicalResult ONNXBatchNormalizationTestModeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !scale().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>() ||
      !mean().getType().isa<RankedTensorType>() ||
      !var().getType().isa<RankedTensorType>())
    return emitError("Input tensor(s) not ranked");

  auto inputTensorTy = X().getType().cast<RankedTensorType>();
  auto scaleTensorTy = scale().getType().cast<RankedTensorType>();
  auto biasTensorTy = B().getType().cast<RankedTensorType>();
  auto meanTensorTy = mean().getType().cast<RankedTensorType>();
  auto varianceTensorTy = var().getType().cast<RankedTensorType>();

  // Check whether the shapes of scale, bias, mean and variance are valid.
  // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
  // In case of N, C is assumed to be 1.
  // 2-D tensors are assumed to be of shape NxC
  // Shapes of scale, bias, mean and variance must be C.
  int64_t c = -1;
  if (inputTensorTy.getShape().size() == 1) {
    c = 1;
  } else if (inputTensorTy.getShape().size() >= 2) {
    c = (inputTensorTy.getShape()[1] != -1) ? inputTensorTy.getShape()[1] : -1;
  }

  if (c != -1) {
    auto s = scaleTensorTy.getShape();
    auto b = biasTensorTy.getShape();
    auto m = meanTensorTy.getShape();
    auto v = varianceTensorTy.getShape();

    if ((s.size() != 1) || (s[0] != -1 && s[0] != c))
      return emitError("Wrong rank for the scale");
    if ((b.size() != 1) || (b[0] != -1 && b[0] != c))
      return emitError("Wrong rank for the bias");
    if ((m.size() != 1) || (m[0] != -1 && m[0] != c))
      return emitError("Wrong rank for the mean");
    if ((v.size() != 1) || (v[0] != -1 && v[0] != c))
      return emitError("Wrong rank for the variance");
  }

  // The output tensor of the same shape as the input.
  getResult().setType(X().getType());
  return success();
}

// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//
// Reshape
//===----------------------------------------------------------------------===//

LogicalResult ONNXReshapeOp::inferShapes() {
  // Cannot infer shape if no shape tensor is specified.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input data tensor not ranked");

  if (!shape().getType().isa<RankedTensorType>())
    return emitError("Shape tensor not ranked");

  auto inputTensorTy = data().getType().cast<RankedTensorType>();
  auto shapeTensorTy = shape().getType().cast<RankedTensorType>();

  // Only rank 1 shape tensors are supported.
  if (shapeTensorTy.getShape().size() != 1)
    return emitError("Shape tensor must have rank one");
  int64_t outputRank = shapeTensorTy.getShape()[0];

  // Shape tensor must have constant shape.
  if (outputRank < 0)
    return emitError("Shape tensor must have constant shape");
  // Compute total number of elements.
  int64_t totalInputSize = 1;
  for (auto inputDim : inputTensorTy.getShape())
    totalInputSize *= inputDim;

  // Check if second argument of ReshapeOp is a constant.
  auto constantOp = getONNXConstantOp(shape());

  SmallVector<int64_t, 2> dims(outputRank, -1);
  if (constantOp) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    if (!valueAttribute)
      return emitError("DenseElementsAttr expected");
    // Get dims from valueAttribute.
    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i < outputRank; ++i)
      dims[i] = (*valueIt++).cast<IntegerAttr>().getInt();

    if (valueIt != valueAttribute.getValues<IntegerAttr>().end())
      return emitError("Constant value must have same rank as output");
    int64_t numberOfDynamicInputs = 0;
    int64_t totalKnownDimsSize = 1;
    int64_t dynamicValueIndex = -1;
    for (int i = 0; i < outputRank; ++i) {
      // Set output dimension.
      if (dims[i] == 0)
        dims[i] = inputTensorTy.getShape()[i];

      if (dims[i] < 0) {
        numberOfDynamicInputs++;
        dynamicValueIndex = i;
      } else {
        totalKnownDimsSize *= dims[i];
      }
    }

    // If the number of dynamic inputs is 1 then deduce the missing value
    // based on the total input size. The total input size must be greater
    // than 0 i.e. all constant dimensions.
    // TODO: Support dynamic input dimensons.
    if (numberOfDynamicInputs == 1 && totalKnownDimsSize > 0 &&
        totalInputSize > 0)
      dims[dynamicValueIndex] = totalInputSize / totalKnownDimsSize;
  }

  getResult().setType(
      RankedTensorType::get(dims, inputTensorTy.getElementType()));
  return success();
}

// Transpose

LogicalResult ONNXTransposeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  // Naive transposition which handles the default case of
  // reversing the shape of the tensor (similar to numpy.transpose).
  auto arrayTy = data().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  auto permutation = ONNXTransposeOp::permAttr();
  if (!permutation) {
    // Generate revese order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = mlir::Builder(getContext());
    auto rank = arrayTy.getShape().size();
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    permAttr(builder.getI64ArrayAttr(defaultRefs));
    permutation = permAttr();
  }
  // Perform transposition according to perm attribute.
  for (auto perm : permutation.getValue())
    dims.emplace_back(arrayTy.getShape()[perm.cast<IntegerAttr>().getInt()]);
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceMax
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMaxOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceMean
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMeanOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceMin
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMinOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceProd
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceProdOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumOp::inferShapes() {
  if (!getOperand().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = getOperand().getType().cast<RankedTensorType>();
  getResult().setType(getReductionOutputType(operandTy, axes(), keepdims()));
  return success();
}

//===----------------------------------------------------------------------===//
// Conv
//===----------------------------------------------------------------------===//

// For this operation, we define the attributes once in the original Conv
// operation class. There is no need to redefine the attribute names for the
// other classes based on Conv.
// Conv attributes output:
//   -  auto_pad set to NOTSET;
//   -  dilations, strides: set to 1 if not defined by user;
//   -  kernelShape: inferred from weight matrix if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

LogicalResult ONNXConvOp::inferShapes() {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  bool hasBias = !B().getType().isa<NoneType>();

  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !W().getType().isa<RankedTensorType>() ||
      (hasBias && !B().getType().isa<RankedTensorType>()))
    return emitError("Input tensor not ranked");

  auto xTy = X().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto weightTy = W().getType().cast<RankedTensorType>();
  auto weightShape = weightTy.getShape();
  auto builder = mlir::Builder(this->getContext());

  // Lowest supported convolution is a one dimensional convolution.
  if (xShape.size() < 3)
    return emitError("Data input shape must be at least (NxCxD1)");

  // Check that shape of weight and data have same length.
  if (xShape.size() != weightShape.size())
    return emitError("Weight size not compatible with data size");

  // Group is a required attribute and should have default value of 1.
  int64_t group = ONNXConvOp::group().getSExtValue();

  // Check if the attribute actually exists. If it does not then add it.
  if (!groupAttr())
    groupAttr(builder.getI64IntegerAttr(group));

  // Check that the X.shape[1] == (W.shape[1] * group) == C condition holds.
  if (xShape[1] != -1 && weightShape[1] != -1 &&
      xShape[1] != (weightShape[1] * group)) {
    return emitOpError("Channel dimension mismatch")
           << xTy << " " << weightTy << " " << group;
  }

  // Check the size of bias.
  if (hasBias) {
    auto bTx = B().getType().cast<RankedTensorType>();
    auto bShape = bTx.getShape();
    if (bShape.size() != 1)
      return emitError("bias should be one dimensional");
    if (bShape[0] != weightShape[0])
      return emitError("bias should have same dimensions "
                       "as weight's first dimension");
  }

  // Note: the value of the group attribut only impacts the way the
  // computation is carried out and not the actual output size.

  // Number of spatial dimensions.
  auto spatialOffset = 2;
  int32_t spatialRank = xShape.size() - spatialOffset;

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  auto kernelShape = kernel_shape();
  if (kernelShape.hasValue()) {
    if (ArrayAttrSize(kernelShape) != spatialRank)
      return emitError(
          "kernel_shape length incompatible with spatial dimensions");
    // Have the right number of values, check them.
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(kernelShape, i) < 1)
        return emitError("bad kernel_shape value");
  } else {
    // Deduce shape from weight input.
    SmallVector<int64_t, 2> defaultVals;
    for (int i = 0; i < spatialRank; ++i)
      defaultVals.emplace_back(weightShape[spatialOffset + i]);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto builder = mlir::Builder(getContext());
    kernel_shapeAttr(builder.getI64ArrayAttr(defaultRefs));
    kernelShape = kernel_shape();
  }

  // Process strides, dilations, and pads.
  processConvTypeParams<>(this, X());
  auto dilationsOpt = dilations();
  auto stridesOpt = strides();
  auto padsOpt = pads();

  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  // Insert number of filters being applied (number of output channels).
  outputDims.emplace_back(weightShape[0]);
  // Compute and insert spatial dims.
  insertConvSpatialDim(&outputDims, builder, xShape, kernelShape, padsOpt,
      stridesOpt, dilationsOpt);

  getResult().setType(RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConvTranspose
//===----------------------------------------------------------------------===//

// For this operation, we define the attributes once in the original Conv
// operation class. There is no need to redefine the attribute names for the
// other classes based on Conv.
// Conv attributes output:
//   -  auto_pad set to NOTSET;
//   -  dilations, strides: set to 1 if not defined by user;
//   -  kernelShape: inferred from weight matrix if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

LogicalResult ONNXConvTransposeOp::inferShapes() {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (C x M/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  bool hasBias = !B().getType().isa<NoneType>();

  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>() ||
      !W().getType().isa<RankedTensorType>() ||
      (hasBias && !B().getType().isa<RankedTensorType>())) {
    return emitError("Input tensor not ranked");
  }

  auto xTy = X().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto weightTy = W().getType().cast<RankedTensorType>();
  auto weightShape = weightTy.getShape();
  auto builder = mlir::Builder(this->getContext());

  // Lowest supported convolution is a one dimensional convolution.
  if (xShape.size() < 3) {
    return emitError("Data input shape must be at least (NxCxD1)");
  }

  // Check that shape of weight and data have same length.
  if (xShape.size() != weightShape.size()) {
    return emitError("Weight size not compatible with data size");
  }

  // Group is a required attribute and should have default value of 1.
  int64_t group = ONNXConvTransposeOp::group().getSExtValue();

  // Check if the attribute actually exists. If it does not then add it.
  if (!groupAttr())
    groupAttr(builder.getI64IntegerAttr(group));

  int64_t inChannels = weightShape[0];
  int64_t outChannels = weightShape[1] * group;

  // Check that the X.shape[1] == W.shape[0] == C && X.shape[1] % group == 0
  // condition holds.
  if (xShape[1] != -1 && inChannels != -1 && xShape[1] != inChannels &&
      xShape[1] % group != 0) {
    return emitOpError("Channel dimension mismatch")
           << xTy << " " << weightTy << " " << group;
  }

  // Check the size of bias.
  if (hasBias) {
    auto bTx = B().getType().cast<RankedTensorType>();
    auto bShape = bTx.getShape();
    if (bShape.size() != 1) {
      return emitError("bias should be one dimensional");
    }
    if (bShape[0] != outChannels) {
      return emitError(
          "bias should have same dimensions as number of output channels");
    }
  }

  // Note: the value of the group attribut only impacts the way the
  // computation is carried out and not the actual output size.

  // Number of spatial dimensions.
  auto spatialOffset = 2;
  int32_t spatialRank = xShape.size() - spatialOffset;

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  auto kernelShape = kernel_shape();
  if (kernelShape.hasValue()) {
    if (ArrayAttrSize(kernelShape) != spatialRank) {
      return emitError(
          "kernel_shape length incompatible with spatial dimensions");
    }
    // Have the right number of values, check them.
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(kernelShape, i) < 1) {
        return emitError("bad kernel_shape value");
      }
  } else {
    // Deduce shape from weight input.
    SmallVector<int64_t, 2> defaultVals;
    for (int i = 0; i < spatialRank; ++i)
      defaultVals.emplace_back(weightShape[spatialOffset + i]);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto builder = mlir::Builder(getContext());
    kernel_shapeAttr(builder.getI64ArrayAttr(defaultRefs));
    kernelShape = kernel_shape();
  }

  // Process strides, dilations, and pads.
  processConvTypeParams<>(this, X());
  auto dilationsOpt = dilations();
  auto stridesOpt = strides();
  auto padsOpt = pads();
  auto outputPads = output_padding();
  auto outputShape = output_shape();
  // TODO: handle the spatial dimension computation if output shape is specified
  assert(!outputShape.hasValue() && "unhandled option in ConvTranspose");

  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  // Insert number of filters being applied (number of output channels *
  // groups).
  outputDims.emplace_back(outChannels);
  // Compute and insert spatial dims.
  insertConvTransposeSpatialDim(outputDims, xShape, kernelShape, padsOpt,
      stridesOpt, outputPads, outputShape, dilationsOpt);

  // Set the output shape if it's not already set
  if (!outputShape.hasValue()) {
    output_shapeAttr(builder.getI64ArrayAttr(outputDims));
  }

  getResult().setType(RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// AveragePool
//===----------------------------------------------------------------------===//

// Infer shape attributes output:
//   -  auto_pad set to NOTSET;
//   -  strides: set to 1 if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

LogicalResult ONNXAveragePoolOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto builder = mlir::Builder(getContext());

  // Get shape of input.
  auto xTy = X().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();

  // Kernel shape.
  auto kernelShape = kernel_shape();
  if (!kernelShape)
    return emitError(
        "kernel_shape is a mandatory attribute for which there is no default");

  // Ceil mode.
  auto ceilMode = ceil_mode().getSExtValue();

  // Process strides and pads.
  LogicalResult res =
      processConvStrideParam<ONNXAveragePoolOp>(this, kernelShape);
  if (failed(res))
    return res;
  auto stridesOpt = strides();
  res = processConvPadParam<ONNXAveragePoolOp>(
      this, xShape, kernelShape, stridesOpt, llvm::None);
  if (failed(res))
    return res;
  auto padsOpt = pads();

  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  outputDims.emplace_back(xShape[1]);
  // Compute and insert spatial dims.
  insertConvSpatialDim(&outputDims, builder, xShape, kernelShape, padsOpt,
      stridesOpt, llvm::None, ceilMode);

  getResult().setType(RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// MaxPoolSingleOut
//===----------------------------------------------------------------------===//

// Infer shape attributes output:
//   -  auto_pad set to NOTSET;
//   -  dilations, strides: set to 1 if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

LogicalResult ONNXMaxPoolSingleOutOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!X().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto builder = mlir::Builder(getContext());

  // Get shape of input.
  auto xTy = X().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();

  // Kernel shape.
  auto kernelShape = kernel_shape();
  if (!kernelShape)
    return emitError(
        "kernel_shape is a mandatory attribute for which there is no default");

  // Storage order.
  auto storageOrder = storage_order().getSExtValue();
  if (storageOrder != 0)
    return emitError("column major storage order not supported at this time");

  // Process strides, dilations, and pads.
  processConvTypeParams<>(this, X());
  auto dilationsOpt = dilations();
  auto stridesOpt = strides();
  auto padsOpt = pads();

  // Ceil mode.
  auto ceilMode = ceil_mode().getSExtValue();

  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  outputDims.emplace_back(xShape[1]);
  // Compute and insert spatial dims.
  insertConvSpatialDim(&outputDims, builder, xShape, kernelShape, padsOpt,
      stridesOpt, dilationsOpt, ceilMode);

  getResult().setType(RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// Pad
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Pad: unknown input shape");

  auto dataTy = data().getType().cast<RankedTensorType>();
  auto dataShape = dataTy.getShape();
  auto dataRank = dataTy.getRank();
  SmallVector<int64_t, 4> outputShape(dataShape.begin(), dataShape.end());

  // Get pads from valueAttribute.
  Attribute padattr = getAttr("pads");
  SmallVector<int64_t, 2> pads(dataRank * 2, -1);
  // Sometimes it's an ArrayAttr and sometimes it's a DenseElementsAttr, so
  // handle both cases.
  if (ArrayAttr padsAttributes = padattr.dyn_cast_or_null<mlir::ArrayAttr>()) {
    auto valueIt = padsAttributes.getValue().begin();
    for (int64_t i = 0; i < dataRank * 2; ++i)
      pads[i] = (*valueIt++).cast<IntegerAttr>().getInt();
  } else if (DenseElementsAttr padsAttributes =
                 padattr.dyn_cast_or_null<mlir::DenseElementsAttr>()) {
    auto valueIt = padsAttributes.getValues<IntegerAttr>().begin();
    for (int64_t i = 0; i < dataRank * 2; ++i)
      pads[i] = (*valueIt++).getInt();
  } else {
    // Cannot infer if the pads is not constant
    return emitError("Pad: unknown pads ") << getAttr("pads");
  }

  // Pads consists of two values for each axis of data.
  // The two values specify the number of elements padded before and after
  // respectively.
  for (int64_t i = 0; i < dataRank; ++i) {
    int64_t p1 = pads[i];
    int64_t p2 = pads[i + dataRank];
    // Have to non-negative constant
    if (p1 < 0 || p2 < 0)
      return emitError("padding value can not be negative");
    if (outputShape[i] != -1)
      outputShape[i] += p1 + p2;
  }

  auto outputType = RankedTensorType::get(outputShape, dataTy.getElementType());
  getResult().setType(outputType);
  return success();
}

static Type padShapeInferenceHelper(Value data, ArrayAttr padsOpt) {
  // Cannot infer shape if no shape exists.
  if (!data.getType().isa<RankedTensorType>())
    return (Type)NULL;
  auto dataTy = data.getType().cast<RankedTensorType>();
  auto dataShape = dataTy.getShape();
  auto dataRank = dataShape.size();
  SmallVector<int64_t, 4> outputShape(dataShape.begin(), dataShape.end());
  if (padsOpt) {
    auto padsArray = padsOpt.getValue();
    // Pads consists of two values for each axis of data.
    // The two values specify the number of elements padded before and after
    // respectively.
    for (int i = 0; i < dataRank; ++i) {
      int64_t p1 = (padsArray[i]).cast<IntegerAttr>().getInt();
      int64_t p2 = (padsArray[i + dataRank]).cast<IntegerAttr>().getInt();
      // Have to non-negative constant
      if (p1 < 0 || p2 < 0)
        return (Type)NULL;
      if (outputShape[i] != -1)
        outputShape[i] += p1 + p2;
    }

    return (RankedTensorType::get(outputShape, dataTy.getElementType()));
  } else {
    return (Type)NULL;
  }
}

//===----------------------------------------------------------------------===//
// PadConstantPad
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadConstantPadOp::inferShapes() {
  auto outputType = padShapeInferenceHelper(data(), pads());
  if (!outputType)
    return emitError("missing output");
  getResult().setType(outputType);
  return success();
}

//===----------------------------------------------------------------------===//
// PadConstantValuePad
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadConstantValuePadOp::inferShapes() {
  auto outputType = padShapeInferenceHelper(data(), pads());
  if (!outputType)
    return emitError("missing output");
  getResult().setType(outputType);
  return success();
}

void ONNXPadConstantValuePadOp::build(OpBuilder &builder, OperationState &state,
    Value data, ArrayAttr pads, FloatAttr constant_value, StringAttr mode) {
  Type outputType = padShapeInferenceHelper(data, pads);
  if (!outputType) {
    auto elementType = data.getType().cast<TensorType>().getElementType();
    outputType = UnrankedTensorType::get(elementType);
  }
  build(builder, state, outputType, data, pads, constant_value, mode);
}

//===----------------------------------------------------------------------===//
// Unsqueeze
//===----------------------------------------------------------------------===//

LogicalResult ONNXUnsqueezeOp::inferShapes() {
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = data().getType().cast<RankedTensorType>();
  int inRank = operandTy.getRank();

  ArrayAttr axisAttrs = axesAttr();
  SmallVector<int, 4> axes;
  int outRank = 0;
  if (axisAttrs) {
    outRank = inRank + axisAttrs.getValue().size();
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (outRank + axis);
      // Valid range
      assert(axis >= -outRank && axis <= outRank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
      else
        return emitError("Duplicated axes");
    }
  } else
    return emitError("Axes attribute is required");

  SmallVector<int64_t, 4> dims;
  for (int i = 0, j = 0; i < outRank || j < inRank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      dims.emplace_back(1);
    } else {
      dims.emplace_back(operandTy.getShape()[j++]);
    }
  }
  getResult().setType(RankedTensorType::get(dims, operandTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//

// Squeeze

LogicalResult ONNXSqueezeOp::inferShapes() {
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto operandTy = data().getType().cast<RankedTensorType>();
  int64_t inRank = operandTy.getRank();

  ArrayAttr axisAttrs = axesAttr();
  if (!axisAttrs)
    return emitError("Axes attribute is required");

  SmallVector<int64_t, 4> axes;
  bool hasNegativeAxis = false;
  for (auto axisAttr : axisAttrs.getValue()) {
    int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
    if (axis < -inRank || axis >= inRank)
      return emitError("Invalid axis value");
    if (axis < 0) {
      axis = inRank + axis;
      hasNegativeAxis = true;
    }
    if (std::find(axes.begin(), axes.end(), axis) != axes.end())
      return emitError("Duplicated axes");
    axes.emplace_back(axis);
  }
  if (hasNegativeAxis) {
    // Update axes attribute so that it contains only positive values.
    auto builder = mlir::Builder(getContext());
    ArrayRef<int64_t> defaultRefs(axes);
    axesAttr(builder.getI64ArrayAttr(defaultRefs));
  }

  SmallVector<int64_t, 4> dims;
  for (int i = 0; i < inRank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      dims.emplace_back(operandTy.getShape()[i]);
    }
  }
  getResult().setType(RankedTensorType::get(dims, operandTy.getElementType()));
  return success();
}

// Cast
//===----------------------------------------------------------------------===//

LogicalResult ONNXCastOp::inferShapes() {
  ShapedType inputType = input().getType().dyn_cast<ShapedType>();
  if (!inputType) {
    return emitError("Non-shaped input type");
  }

  auto getOutputType = [&inputType](Type elementType) -> Type {
    if (inputType.hasRank()) {
      return RankedTensorType::get(inputType.getShape(), elementType);
    }
    return UnrankedTensorType::get(elementType);
  };

  int64_t targetType = toAttr().getInt();
  OpBuilder builder(getContext());
  if (auto elementType = convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(targetType))) {
    getResult().setType(getOutputType(elementType));
  } else {
    return emitOpError("Unable to get the element type for to = " +
                       std::to_string(targetType));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Scaler
//===----------------------------------------------------------------------===//

LogicalResult ONNXScalerOp::inferShapes() {
  ShapedType inputType = X().getType().dyn_cast<ShapedType>();
  getResult().setType(RankedTensorType::get(
      inputType.getShape(), FloatType::getF32(getContext())));
  return success();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOp::inferShapes() {
  if ((sparse_value().hasValue() && value().hasValue()) ||
      (!sparse_value().hasValue() && !value().hasValue()))
    return emitError("Require exactly one of the two attributes, "
                     "either value or sparse_value");
  ElementsAttr valAttr;
  if (sparse_value().hasValue())
    valAttr = sparse_valueAttr().cast<SparseElementsAttr>();
  else
    valAttr = valueAttr().cast<DenseElementsAttr>();
  getResult().setType(valAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//

LogicalResult ONNXConcatOp::inferShapes() {
  int inputNum = getNumOperands();
  for (int i = 0; i < inputNum; ++i) {
    if (!getOperand(i).getType().isa<RankedTensorType>())
      return emitError("Input tensor(s) not ranked");
  }
  // Checking value of axis parameter.
  auto commonType = getOperand(0).getType().cast<RankedTensorType>();
  auto commonShape = commonType.getShape();
  auto commonRank = commonShape.size();
  auto axisIndex = axis().getSExtValue();
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
    auto builder = mlir::Builder(getContext());
    axisAttr(builder.getI64IntegerAttr(axisIndex));
  }
  if (axisIndex >= commonRank)
    return emitError("Concat axis value out of bound");
  // Initial cummlative size is that of the first operand.
  int cummulativeAxisSize = commonShape[axisIndex];

  // Compute the cummlative size with all of the other ones, and make sure
  // that the other sizes are all alike.
  for (int i = 1; i < inputNum; ++i) {
    auto currShape =
        getOperand(i).getType().cast<RankedTensorType>().getShape();
    if (currShape.size() != commonRank)
      return emitError("Concat input must all have the same rank");
    for (int j = 0; j < commonRank; ++j) {
      if (j == axisIndex) {
        // Check that the value is positive.
        if (currShape[j] <= 0)
          return emitError("Concat axis being concatenated is "
                           "expected to be known at compile time for now");
      } else if (currShape[j] != commonShape[j]) {
        return emitError("Concat input dimensions must be all identical, "
                         "except for dimension on the axis of the "
                         "concatenation. Expected something compatible with: ")
               << commonType << " but got " << getOperand(i).getType()
               << " instead.";
      }
    }
    cummulativeAxisSize += currShape[axisIndex];
  }

  // Set output size and type
  SmallVector<int64_t, 4> outputDims;
  for (int j = 0; j < commonRank; ++j)
    outputDims.emplace_back(
        j == axisIndex ? cummulativeAxisSize : commonShape[j]);
  getResult().setType(
      RankedTensorType::get(outputDims, commonType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// RNN
//===----------------------------------------------------------------------===//

LogicalResult ONNXRNNOp::inferShapes() { return RNNShapeInference<>(this); }

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

LogicalResult ONNXLSTMOp::inferShapes() { return RNNShapeInference<>(this); }

//===----------------------------------------------------------------------===//
// GRU
//===----------------------------------------------------------------------===//

LogicalResult ONNXGRUOp::inferShapes() { return RNNShapeInference<>(this); }

//===----------------------------------------------------------------------===//
// Split
//===----------------------------------------------------------------------===//

LogicalResult ONNXSplitOp::inferShapes() {
  if (!getOperand().getType().cast<RankedTensorType>())
    return emitError("Input tensor not ranked");

  int numOfResults = getNumResults();
  auto inputType = getOperand().getType().cast<RankedTensorType>();
  auto inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();

  // Checking value of axis parameter.
  auto axisIndex = axis().getSExtValue();
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return emitError("Split axis value out of bound");
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = inputRank + axisIndex;
    auto builder = mlir::Builder(getContext());
    axisAttr(builder.getI64IntegerAttr(axisIndex));
  }

  // Checking value of split parameter.
  auto splitAttribute = split();
  SmallVector<int64_t, 4> splitLengths;
  if (splitAttribute.hasValue()) {
    if (ArrayAttrSize(splitAttribute) != numOfResults)
      return emitError("Split size not equal to the number of results");
    for (int i = 0; i < numOfResults; ++i)
      splitLengths.emplace_back(ArrayAttrIntVal(splitAttribute, i));

  } else {
    if (inputShape[axisIndex] <= 0)
      return emitError("The dimension at the split axis is "
                       "expected to be known at compile time");
    if (inputShape[axisIndex] % numOfResults != 0)
      return emitError("The dimension at the split axis is "
                       "expected to be divisible by the number of results");
    // If split parameter is not specified, the dimension is split to
    // equal-sized parts.
    for (int i = 0; i < numOfResults; ++i)
      splitLengths.emplace_back(inputShape[axisIndex] / numOfResults);
    // Build attribute and store attribute.
    auto builder = mlir::Builder(getContext());
    splitAttr(builder.getI64ArrayAttr(llvm::makeArrayRef(splitLengths)));
  }

  // Build result types.
  for (int i = 0; i < numOfResults; ++i) {
    SmallVector<int64_t, 3> resultShape;
    for (int j = 0; j < inputRank; ++j) {
      if (j == axisIndex) {
        resultShape.emplace_back(splitLengths[i]);
      } else {
        resultShape.emplace_back(inputShape[j]);
      }
    }
    getResults()[i].setType(
        RankedTensorType::get(resultShape, inputType.getElementType()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Flatten
//===----------------------------------------------------------------------===//

LogicalResult ONNXFlattenOp::inferShapes() {
  assert(axis() == 1 && "ONNXFlattenOp can only handle axis=1 for now");
  auto inTy = input().getType().dyn_cast<ShapedType>();
  if (!inTy) {
    return emitOpError("Input is a non-shaped type");
  }
  auto outTy = output().getType().dyn_cast<ShapedType>();
  if (!outTy) {
    return emitOpError("Output is a non-shaped type");
  }

  // TODO(tjingrant): Seems like we can also fairly easily support the case
  //                  where the batch dimension is dynamic
  if (!outTy.hasStaticShape()) {
    auto inShape = inTy.getShape();
    assert(inShape.size() >= 1 && "ONNXFlattenOp inShape.size() should be > 0");
    uint64_t outDim = 1;
    for (auto it = inShape.begin() + 1; it < inShape.end(); it++) {
      outDim *= *it;
    }

    SmallVector<int64_t, 2> dims;
    // https://pytorch.org/docs/master/generated/torch.nn.Flatten.html
    dims.emplace_back(inShape[0]);
    dims.emplace_back(outDim);
    getResult().setType(RankedTensorType::get(dims, outTy.getElementType()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicQuantizeLinear
//===----------------------------------------------------------------------===//

LogicalResult ONNXDynamicQuantizeLinearOp::inferShapes() {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy || !inTy.hasStaticShape()) {
    return emitOpError("Input is not a statically-shaped type");
  }

  auto yTy = y().getType().cast<ShapedType>();
  auto yScaleTy = y_scale().getType().cast<ShapedType>();
  auto yZPTy = y_zero_point().getType().cast<ShapedType>();

  IntegerType ui8Type =
      IntegerType::get(8, IntegerType::Unsigned, getContext());
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

LogicalResult ONNXQuantizeLinearOp::inferShapes() {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy || !inTy.hasStaticShape()) {
    return emitOpError("Input is not a statically-shaped type");
  }

  auto yTy = y().getType().cast<ShapedType>();

  if (!yTy.hasStaticShape()) {
    // TODO: Unfortunately, we can't tell if this should be signed or unsigned
    //       here...
    IntegerType i8Type = IntegerType::get(8, getContext());
    RankedTensorType outType = RankedTensorType::get(inTy.getShape(), i8Type);
    y().setType(outType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DequantizeLinear
//===----------------------------------------------------------------------===//

LogicalResult ONNXDequantizeLinearOp::inferShapes() {
  auto inTy = x().getType().dyn_cast<RankedTensorType>();
  if (!inTy || !inTy.hasStaticShape()) {
    return emitOpError("Input is not a statically-shaped type");
  }

  auto yTy = y().getType().cast<ShapedType>();

  if (!yTy.hasStaticShape()) {
    FloatType f32 = FloatType::getF32(getContext());
    RankedTensorType outType = RankedTensorType::get(inTy.getShape(), f32);
    y().setType(outType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConvInteger - copied almost exactly from Conv (X -> x, W -> w, no bias)
//===----------------------------------------------------------------------===//

LogicalResult ONNXConvIntegerOp::inferShapes() {
  // Generic shape for data input X, weight tensor W
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)

  // Cannot infer shape if no shape exists.
  if (!x().getType().isa<RankedTensorType>() ||
      !w().getType().isa<RankedTensorType>()) {
    return emitOpError("Input tensor not ranked");
  }

  auto xTy = x().getType().cast<RankedTensorType>();
  if (!xTy.getElementType().isInteger(8)) {
    return emitOpError("Invalid input type");
  }
  auto xShape = xTy.getShape();
  auto weightTy = w().getType().cast<RankedTensorType>();
  if (!weightTy.getElementType().isInteger(8)) {
    return emitOpError("Invalid input type");
  }
  auto weightShape = weightTy.getShape();
  auto builder = mlir::Builder(this->getContext());

  // Lowest supported convolution is a one dimensional convolution.
  if (xShape.size() < 3) {
    return emitOpError("Data input shape must be at least (NxCxD1)");
  }

  // Check that shape of weight and data have same length.
  if (xShape.size() != weightShape.size()) {
    return emitError("Weight size not compatible with data size");
  }

  // Group is a required attribute and should have default value of 1.
  int64_t group = ONNXConvIntegerOp::group().getSExtValue();

  // Check if the attribute actually exists. If it does not then add it.
  if (!groupAttr())
    groupAttr(builder.getI64IntegerAttr(group));

  // Check that the X.shape[1] == (W.shape[1] * group) == C condition holds.
  if (xShape[1] != -1 && weightShape[1] != -1 &&
      xShape[1] != (weightShape[1] * group)) {
    return emitOpError("Channel dimension mismatch");
  }

  // Note: the value of the group attribut only impacts the way the
  // computation is carried out and not the actual output size.

  // Number of spatial dimensions.
  auto spatialOffset = 2;
  int32_t spatialRank = xShape.size() - spatialOffset;

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  auto kernelShape = kernel_shape();
  if (kernelShape.hasValue()) {
    if (ArrayAttrSize(kernelShape) != spatialRank) {
      return emitOpError(
          "kernel_shape length incompatible with spatial dimensions");
    }
    // Have the right number of values, check them.
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(kernelShape, i) < 1) {
        return emitError("bad kernel_shape value");
      }
  } else {
    // Deduce shape from weight input.
    SmallVector<int64_t, 2> defaultVals;
    for (int i = 0; i < spatialRank; ++i)
      defaultVals.emplace_back(weightShape[spatialOffset + i]);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto builder = mlir::Builder(getContext());
    kernel_shapeAttr(builder.getI64ArrayAttr(defaultRefs));
    kernelShape = kernel_shape();
  }

  // Process strides, dilations, and pads.
  processConvTypeParams<>(this, x());
  auto dilationsOpt = dilations();
  auto stridesOpt = strides();
  auto padsOpt = pads();

  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  // Insert number of filters being applied (number of output channels).
  outputDims.emplace_back(weightShape[0]);
  // Compute and insert spatial dims.
  insertConvSpatialDim(&outputDims, builder, xShape, kernelShape, padsOpt,
      stridesOpt, dilationsOpt);

  // ONNX spec specifies the output type as an int32
  Type outputType = IntegerType::get(32, getContext());
  getResult().setType(RankedTensorType::get(outputDims, outputType));
  return success();
}

//===----------------------------------------------------------------------===//
// Shape
//===----------------------------------------------------------------------===//

LogicalResult ONNXShapeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  // Output is an 1D int64 tensor containing the shape of the input tensor.
  int64_t rank = data().getType().cast<RankedTensorType>().getRank();
  SmallVector<int64_t, 1> outDims(1, rank);
  getResult().setType(
      RankedTensorType::get(outDims, IntegerType::get(64, getContext())));
  return success();
}

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!input().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  // Read 'repeats' value.
  if (!repeats().getType().isa<RankedTensorType>())
    return emitError("Repeats tensor not ranked");

  auto inputTensorTy = input().getType().cast<RankedTensorType>();
  auto repeatsTensorTy = repeats().getType().cast<RankedTensorType>();

  // 'repeats' tensor is an 1D tensor.
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  // 'repeats' tensor must have constant shape.
  int64_t repeatsLength = repeatsTensorTy.getShape()[0];
  if (repeatsLength < 0)
    return emitError("Repeats tensor must have constant shape");

  // Check the 1D repeats tensor length.
  int64_t inputRank = inputTensorTy.getShape().size();
  if (inputRank != repeatsLength)
    return emitError("Repeats tensor must have the same length as the input's "
                     "dimension number.");

  // Check if second argument of TileOp is a constant.
  auto constantOp = getONNXConstantOp(repeats());

  // Compute output's dimensions: output_dim[i] = input_dim[i] * repeats[i]
  SmallVector<int64_t, 2> dims(inputRank, -1);
  if (constantOp) {
    // 1. Initialize output_dim with values from 'input'.
    //   output_dim[i] = input[i]
    for (decltype(inputRank) i = 0; i < inputRank; ++i)
      dims[i] = inputTensorTy.getShape()[i];

    // 2. Update output_dim using values from 'repeats'.
    // Do this only for static 'input_dim[i]'.
    //   if (output_dim[i] != -1) output_dim[i] *= repeats[i]
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    if (!valueAttribute)
      return emitError("DenseElementsAttr expected");
    // Get repeat values from valueAttribute.
    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i < inputRank; ++i)
      if (dims[i] != -1)
        dims[i] *= (*valueIt++).cast<IntegerAttr>().getInt();

    if (valueIt != valueAttribute.getValues<IntegerAttr>().end())
      return emitError("Constant value must have same length as output's rank");
  }

  getResult().setType(
      RankedTensorType::get(dims, inputTensorTy.getElementType()));

  return success();
}

//===----------------------------------------------------------------------===//
// Gather
//===----------------------------------------------------------------------===//

LogicalResult ONNXGatherOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");
  if (!indices().getType().isa<RankedTensorType>())
    return emitError("Indices tensor not ranked");

  auto inputShape = data().getType().cast<RankedTensorType>().getShape();
  auto indicesShape = indices().getType().cast<RankedTensorType>().getShape();
  int64_t inputRank = inputShape.size();
  int64_t indicesRank = indicesShape.size();

  if (inputRank < 1)
    return emitError("Input tensor must have rank >= 1");

  // Read 'axis' attribute.
  auto axisIndex = axis().getSExtValue();
  // 'axis' must be in [-rank, rank-1]
  if (axisIndex < -inputRank || axisIndex >= inputRank)
    return emitError("Gather axis value out of bound");
  // Convert a negative axis to a positive axis.
  if (axisIndex < 0) {
    axisIndex += inputRank;
    auto builder = mlir::Builder(getContext());
    axisAttr(builder.getI64IntegerAttr(axisIndex));
  }

  // If 'indices' is a constant, check whether its values are valid or not.
  auto constantOp = getONNXConstantOp(indices());
  if (constantOp && inputShape[axisIndex] != -1) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    if (!valueAttribute)
      return emitError("DenseElementsAttr expected");
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
      auto index = value.cast<IntegerAttr>().getInt();
      if (index < -inputShape[axisIndex] || index >= inputShape[axisIndex])
        return emitError("Indices tensor contains an out-of-bound index");
    }
  }

  // Output has rank of 'indicesRank + (inputRank - 1).
  // Output shape is constructed from 'input' by:
  //    replacing the dimension at 'axis' in 'input' by the shape of 'indices'.
  SmallVector<int64_t, 1> outDims;
  for (decltype(inputRank) i = 0; i < inputRank; ++i) {
    if (i == axisIndex)
      for (decltype(indicesRank) j = 0; j < indicesRank; ++j)
        outDims.emplace_back(indicesShape[j]);
    else
      outDims.emplace_back(inputShape[i]);
  }

  getResult().setType(RankedTensorType::get(
      outDims, data().getType().cast<RankedTensorType>().getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOfShape
//===----------------------------------------------------------------------===//

LogicalResult ONNXConstantOfShapeOp::inferShapes() {
  Type elementType;

  // 'value' attribute is a one-element tensor whose value and datatype are used
  // to set the output tensor's value and datatype..
  if (value().hasValue()) {
    elementType =
        valueAttr().cast<DenseElementsAttr>().getType().getElementType();
  } else {
    // If 'value' attribute is not specified, it defaults to a tensor of value 0
    // and datatype float32.
    elementType = FloatType::getF32(getContext());

    llvm::SmallVector<int64_t, 2> dims(1, 1);
    auto tensorType = mlir::RankedTensorType::get(dims, elementType);

    llvm::SmallVector<float, 1> values(1, 0.);
    valueAttr(
        mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values)));
  }

  // 'input' must be a 1D tensor.
  auto inputShape = input().getType().cast<RankedTensorType>().getShape();
  if (inputShape.size() != 1)
    return emitError("Input tensor must be a 1D tensor");
  if (inputShape[0] == -1)
    return emitError("Input tensor must have static shape");
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
      if (dim < 0)
        return emitError("All values of the input tensor must be >=0");
      outputDims[i] = dim;
    }

    if (valueIt != valueAttribute.getValues<IntegerAttr>().end())
      return emitError("Constant value must have same length as output's rank");
  }

  getResult().setType(RankedTensorType::get(outputDims, elementType));
  return success();
}

//===----------------------------------------------------------------------===//
// Slice
//===----------------------------------------------------------------------===//

LogicalResult ONNXSliceOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  auto dataShape = data().getType().cast<ShapedType>().getShape();
  int64_t numDims = dataShape.size();

  SmallVector<int64_t, 2> outputDims(numDims, -1);
  // If 'starts', 'ends', 'axes', and 'steps' are constants, check whether their
  // values are valid or not. If the values are valid, it is possible to infer
  // shape.
  //
  // 'starts', 'ends', and 'steps' are for each axis in the list of axes, so
  // processing 'axes' first.

  // Check and get 'axes' tensor.
  SmallVector<int64_t, 2> axesValue;
  if (axes().getType().isa<NoneType>()) {
    // If `axes` are omitted, they are set to `[0, ..., ndim-1]`."
    for (int i = 0; i < numDims; ++i)
      axesValue.emplace_back(i);
  } else if (auto constantOp = getONNXConstantOp(axes())) {
    // If `axes` are constants, read them."
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t axis = value.cast<IntegerAttr>().getInt();
      if (axis < -numDims || axis >= numDims)
        return emitError("Axes contains an out-of-bound index");
      if (axis < 0)
        axis += numDims;
      if (dataShape[axis] == -1) {
        // It is unsafe to infer shape for an axis with an unknown dimension,
        // since we can not validate 'start' and 'end' values from this
        // dimension.
        getResult().setType(RankedTensorType::get(outputDims, elementType));
        return success();
      }
      axesValue.emplace_back(axis);
    }
  } else {
    // Cannot infer a static shape.
    getResult().setType(RankedTensorType::get(outputDims, elementType));
    return success();
  }

  // Check 'starts' tensor.
  SmallVector<int64_t, 2> startsValue;
  if (auto constantOp = getONNXConstantOp(starts())) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    int i = 0;
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t axis = axesValue[i];
      int64_t index = value.cast<IntegerAttr>().getInt();
      if (index < -dataShape[axis])
        index = 0;
      else if (index > dataShape[axis])
        index = dataShape[axis];
      else if (index < 0)
        index += dataShape[axis];
      startsValue.emplace_back(index);
      i++;
    }
    if (i != axesValue.size())
      emitError("starts and axes tensors must have the same length");
  } else {
    // Cannot infer a static shape.
    getResult().setType(RankedTensorType::get(outputDims, elementType));
    return success();
  }

  // Check 'ends' tensor.
  SmallVector<int64_t, 2> endsValue;
  if (auto constantOp = getONNXConstantOp(ends())) {
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    int i = 0;
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t axis = axesValue[i];
      int64_t index = value.cast<IntegerAttr>().getInt();
      if (index < -dataShape[axis])
        index = 0;
      else if (index > dataShape[axis])
        index = dataShape[axis];
      else if (index < 0)
        index += dataShape[axis];
      endsValue.emplace_back(index);
      i++;
    }
    if (i != axesValue.size())
      emitError("ends and axes tensors must have the same length");
  } else {
    // Cannot infer a static shape.
    getResult().setType(RankedTensorType::get(outputDims, elementType));
    return success();
  }

  // Check and get 'steps' tensor.
  SmallVector<int64_t, 2> stepsValue;
  if (steps().getType().isa<NoneType>()) {
    // If `steps` are omitted, they are set to `[1, ..., 1]` of len(starts)."
    for (int i = 0; i < startsValue.size(); ++i)
      stepsValue.emplace_back(1);
  } else if (auto constantOp = getONNXConstantOp(steps())) {
    // If `steps` are constants, read them."
    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    int i = 0;
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t index = value.cast<IntegerAttr>().getInt();
      if (index == 0)
        emitError("step cannot be zero");
      stepsValue.emplace_back(index);
      i++;
    }
    if (i != axesValue.size())
      emitError("steps and axes tensors must have the same length");
  } else {
    // Cannot infer a static shape.
    getResult().setType(RankedTensorType::get(outputDims, elementType));
    return success();
  }

  // All 'starts', 'ends', 'steps' values are valid. Now calculate output
  // dimensions for axes in 'axes'.
  for (int i = 0; i < axesValue.size(); i++) {
    int64_t axis = axesValue[i];
    int64_t start = startsValue[i];
    int64_t end = endsValue[i];
    int64_t step = stepsValue[i];
    if (step < 0)
      step = -step;

    int64_t q = (end - start) / step;
    int64_t r = (end - start) % step;
    if (r != 0)
      q += 1;
    outputDims[axis] = q;
  }

  // Fill in the rest of the dimensions - assume they're untouched.
  for (int i = 0, e = outputDims.size(); i < e; ++i) {
    if (llvm::any_of(axesValue, [i](int64_t a) { return a == i; })) {
      continue;
    }
    outputDims[i] = dataShape[i];
  }

  getResult().setType(RankedTensorType::get(outputDims, elementType));
  return success();
}

//===----------------------------------------------------------------------===//
// Expand
//===----------------------------------------------------------------------===//

LogicalResult ONNXExpandOp::inferShapes() {
  if (!input().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  auto lhsTy = input().getType().cast<RankedTensorType>();

  auto elementType = lhsTy.getElementType();
  auto lhsShape = lhsTy.getShape();
  SmallVector<int64_t, 2> rhsShape;

  Operation *shapeDef = shape().getDefiningOp();

  if (mlir::ONNXShapeOp shapeOp =
          dyn_cast_or_null<mlir::ONNXShapeOp>(shapeDef)) {
    // If the shape operand is produced by a onnx.Shape operation, infer its
    // shape and use it as the requested shape.
    if (!shapeOp.data().getType().isa<RankedTensorType>())
      return emitError("Input tensor not ranked");

    ArrayRef<int64_t> rhsShapeRef =
        shapeOp.data().getType().cast<RankedTensorType>().getShape();
    rhsShape.assign(rhsShapeRef.begin(), rhsShapeRef.end());

  } else if (mlir::ONNXConstantOp constantOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(shapeDef)) {
    // If the shape operand is produced by a onnx.Constant operation, extract
    // the actual value of the constant and use it as the reqested shape.

    auto shapeTensorTy = shape().getType().cast<RankedTensorType>();

    if (shapeTensorTy.getRank() != 1)
      return emitError("Shape tensor must have rank one");

    DenseElementsAttr valueAttribute =
        constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
    if (!valueAttribute)
      return emitError("DenseElementsAttr expected");

    int64_t shapeRank = shapeTensorTy.getShape()[0];
    rhsShape.resize(shapeRank);

    auto valueIt = valueAttribute.getValues<IntegerAttr>().begin();
    for (int i = 0; i != shapeRank; ++i)
      rhsShape[i] = (*valueIt++).cast<IntegerAttr>().getInt();

    assert(valueIt == valueAttribute.getValues<IntegerAttr>().end() &&
           "Shape of constant does not match its actual value");
  } else {
    return emitError(
        "Shape argument of Expand is the output of an unexpected operation: " +
        shapeDef->getName().getStringRef() +
        ". Supported operations are: onnx.Constant and onnx.Shape");
  }

  SmallVector<int64_t, 2> resultShape;
  if (!getBroadcastedShape(lhsShape, rhsShape, resultShape)) {
    return emitError("Tensor not exapandable");
  }

  getResult().setType(RankedTensorType::get(resultShape, elementType));
  return success();
}

//===----------------------------------------------------------------------===//
// Dropout
//===----------------------------------------------------------------------===//

LogicalResult ONNXDropoutOp::inferShapes() {
  if (!data().getType().isa<RankedTensorType>())
    return emitError("Input tensor not ranked");

  getResult(0).setType(data().getType());

  auto inputShape = data().getType().cast<RankedTensorType>().getShape();

  IntegerType i1Type = IntegerType::get(1, IntegerType::Signless, getContext());
  getResult(1).setType(RankedTensorType::get(inputShape, i1Type));
  return success();
}

//===----------------------------------------------------------------------===//
// OneHotEncoder
//===----------------------------------------------------------------------===//

LogicalResult ONNXOneHotEncoderOp::inferShapes() {
  ShapedType inputType = X().getType().dyn_cast<ShapedType>();
  if (!inputType)
    return emitError("Non-shaped input type");
  auto shape = inputType.getShape();
  int64_t outDim = 0;

  // If the input is a tensor of float, int32, or double,
  // the data will be cast to integers and
  // the cats_int64s category list will be used for the lookups.
  if (inputType.getElementType().isIntOrFloat()) {
    if (!cats_int64s())
      return emitError("input is a tensor of float, int32, or double, but no "
                       "cats_int64s attribute");
    outDim = ArrayAttrSize(cats_int64s());
  } else {
    if (!cats_strings())
      return emitError("input is not a tensor of float, int32, or double, but "
                       "no cats_strings attribute");
    outDim = ArrayAttrSize(cats_strings());
  }

  // Encoded output data, having one more dimension than X
  // total category count will determine the size of the extra dimension
  SmallVector<int64_t, 2> dims;
  for (int i = 0; i != shape.size(); ++i) {
    dims.emplace_back(shape[i]);
  }
  dims.emplace_back(outDim);

  getResult().setType(
      RankedTensorType::get(dims, FloatType::getF32(getContext())));
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX type related code
//===----------------------------------------------------------------------===//

namespace mlir {
namespace onnxmlir {
namespace detail {
struct SeqTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  SeqTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }
  static llvm::hash_code hasKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  static SeqTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
    return new (allocator.allocate<SeqTypeStorage>())
        SeqTypeStorage(elementTypes);
  }
  llvm::ArrayRef<mlir::Type> elementTypes;
};
} // end namespace detail
} // end namespace onnxmlir
} // end namespace mlir

SeqType SeqType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected non-empty seq");
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, ONNXTypes::SEQ, elementTypes);
}

llvm::ArrayRef<mlir::Type> SeqType::getElementTypes() {
  return getImpl()->elementTypes;
}

mlir::Type SeqType::getElementType() { return getElementTypes()[0]; }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
