#include "src/Dialect/ONNX/ONNXFoldHelper.hpp"
namespace {
constexpr bool DEBUG = false;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation, also as an
// attribute.

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXAddOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal + rhsVal;
    // Could use the APFloat interface to emulate the results, are ok to simply
    // perform them in the highest possible precision.
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal + rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for AddOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXSubOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal - rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal - rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SubOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXMulOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal * rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = lhsVal * rhsVal;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for MulOp: unkonwn data type");
}

namespace {
APInt divideAPInts(const IntegerType type, const APInt a, const APInt b) {
  if (type.isUnsigned()) {
    return a.udiv(b);
  } else { // Signed or Signless are both treated as Signed in Onnx-Mlir
    return a.sdiv(b);
  }
}
} // namespace

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXDivOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    APFloat lhsVal = lhsAttr.cast<FloatAttr>().getValue();
    APFloat rhsVal = secondAttr.cast<FloatAttr>().getValue();
    APFloat res = lhsVal / rhsVal;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt lhsVal = lhsAttr.cast<IntegerAttr>().getValue();
    APInt rhsVal = secondAttr.cast<IntegerAttr>().getValue();
    APInt res = divideAPInts(elementType.cast<IntegerType>(), lhsVal, rhsVal);
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for DivOp: unkonwn data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXNegOp>(
    Builder &builder, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    APFloat val = attr.cast<FloatAttr>().getValue();
    APFloat res = -val;
    return builder.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    APInt val = attr.cast<IntegerAttr>().getValue();
    APInt res = -val;
    return builder.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for NegOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXSqrtOp>(
    Builder &builder, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    double val = attr.cast<FloatAttr>().getValueAsDouble();
    double res = sqrt(val);
    return builder.getFloatAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SqrtOp: unkonwn data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

void RecurseConstPropSplit(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, uint64_t splitAxis, uint64_t axisOffset,
    uint64_t axisSize, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    Attribute res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    ArrayRef<int64_t> shape = attr.getType().getShape();
    int rank = shape.size();
    int index = rank - freeRank;
    int start, size;
    if (index == splitAxis) {
      start = axisOffset;
      size = axisSize;
    } else {
      start = 0;
      size = attr.getType().getShape()[index];
    }
    for (int i = start; i < start + size; ++i) {
      indices[index] = i;
      RecurseConstPropSplit(rewriter, resVector, attr, indices, splitAxis,
          axisOffset, axisSize, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropSplit(PatternRewriter &rewriter, Value resOperand,
    Attribute attr, IntegerAttr axisAttr, ArrayAttr splitAttr,
    unsigned resIndex) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  unsigned rank = denseAttr.getType().getShape().size();
  // Read split axis.
  uint64_t splitAxis = axisAttr.getValue().getSExtValue();
  // Read split vector.
  SmallVector<uint64_t, 4> splits;
  assert(splitAttr && "split attribute expected to be defined here");
  for (Attribute splitVal : splitAttr.getValue())
    splits.emplace_back(splitVal.cast<IntegerAttr>().getInt());
  // Compute the range of elements of interest in the given axis.
  uint64_t axisOffset = 0, axisSize = splits[resIndex];
  for (int i = 0; i < resIndex; ++i)
    axisOffset += splits[i];
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, -1);
  std::vector<Attribute> resVector;
  // Copy.
  RecurseConstPropSplit(rewriter, resVector, denseAttr, indices, splitAxis,
      axisOffset, axisSize, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

static void RecurseConstPropTranspose(Builder &builder,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, SmallVector<uint64_t, 4> &perm,
    int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    auto shape = attr.getType().getShape();
    int rank = shape.size();
    int index = perm[rank - freeRank];
    int size = attr.getType().getShape()[index];
    for (int i = 0; i < size; ++i) {
      indices[index] = i;
      RecurseConstPropTranspose(
          builder, resVector, attr, indices, perm, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropTranspose(
    Builder &builder, Value resOperand, Attribute attr, ArrayAttr permAttr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto rank = denseAttr.getType().getShape().size();
  // Read permute vector.
  SmallVector<uint64_t, 4> perm;
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, 0);
  std::vector<Attribute> resVector;
  // Copy using permute order.
  RecurseConstPropTranspose(builder, resVector, denseAttr, indices, perm, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropUnsqueeze(
    Builder &builder, Value resOperand, Attribute attr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();

  // Unqueeze does not change the order of access, so just copy the whole data.
  std::vector<Attribute> resVector;
  for (auto value : denseAttr.getValues<Attribute>()) {
    resVector.emplace_back(value);
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for squeeze.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropSqueeze(
    Builder &builder, Value resOperand, Attribute attr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();

  // Squeeze does not change the order of access, so just copy the whole data.
  std::vector<Attribute> resVector;
  for (auto value : denseAttr.getValues<Attribute>()) {
    resVector.emplace_back(value);
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for concat.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropConcat(
    Builder &builder, Value resOperand, ArrayRef<Attribute> attrs) {
  // TODO: expand this to support more than 1D Tensor
  std::vector<Attribute> resVector;
  for (auto attr : attrs) {
    // Read dense attribute, the constant tensor we are transforming.
    DenseElementsAttr denseAttr =
        attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
    assert(denseAttr && "expected dense attribute");

    for (auto value : denseAttr.getValues<Attribute>()) {
      resVector.emplace_back(value);
    }
  }
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  assert(
      resType.getShape()[0] == resVector.size() && "Unmatching operand size");

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for slice.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropSlice(Builder &builder, Value resOperand,
    Attribute data, Attribute starts, Attribute ends, Attribute axes,
    Attribute steps) {
  // TODO: expand this to support more than 1D Tensor
  //       Also fix restrictions on parameters

  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr dataAttr = data.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(dataAttr && "expected dense attribute");
  DenseElementsAttr startsAttr =
      starts.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(startsAttr && "expected dense attribute");
  DenseElementsAttr endsAttr = ends.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(endsAttr && "expected dense attribute");
  DenseElementsAttr axesAttr = axes.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(axesAttr && "expected dense attribute");
  DenseElementsAttr stepsAttr =
      steps.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(stepsAttr && "expected dense attribute");

  int64_t startLocal = startsAttr.getValue<IntegerAttr>(0).getInt();
  int64_t endLocal = endsAttr.getValue<IntegerAttr>(0).getInt();
  int64_t axisLocal = axesAttr.getValue<IntegerAttr>(0).getInt();
  int64_t stepLocal = stepsAttr.getValue<IntegerAttr>(0).getInt();
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto dataSize = dataAttr.size();

  if (startLocal < 0) {
    startLocal = startLocal + dataSize;
  } else if (startLocal > dataSize) {
    startLocal = dataSize;
  }
  if (endLocal < 0) {
    endLocal = endLocal + dataSize;
  } else if (endLocal > dataSize) {
    endLocal = dataSize;
  }
  assert(startLocal <= endLocal && axisLocal == 0 && stepLocal == 1 &&
         "Unsupported slice operation mode");

  std::vector<Attribute> resVector(dataAttr.attr_value_begin() + startLocal,
      dataAttr.attr_value_begin() + endLocal);

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

namespace {
APInt castIntToInt(APInt inVal, IntegerType fromType, IntegerType toType) {
  unsigned toWidth = toType.getWidth();
  if (fromType.isUnsigned() || toType.isUnsigned()) {
    // If either fromType or toType is unsigned, then we're expecting a
    // positive value, hence zero-extention will always be correct.
    return inVal.zextOrTrunc(toWidth);
  } else {
    // If both fromType and toType are signed, then sign-extension should
    // yield correct result.
    return inVal.sextOrTrunc(toWidth);
  }
}

APFloat castFloatToFloat(APFloat inVal, FloatType fromType, FloatType toType) {
  if (fromType.isa<Float16Type>() && toType.isa<Float32Type>()) {
    bool losesInfo = false;
    inVal.convert(APFloat::IEEEsingle(), llvm::RoundingMode::NearestTiesToEven,
        &losesInfo);
  } else if (fromType.isa<Float32Type>() && toType.isa<Float16Type>()) {
    bool losesInfo = true;
    inVal.convert(
        APFloat::IEEEhalf(), llvm::RoundingMode::NearestTiesToEven, &losesInfo);
  }
  return inVal;
}
} // namespace

DenseElementsAttr ConstPropCastIntToInt(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {

  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  IntegerType fromElemType = constType.getElementType().cast<IntegerType>();

  auto toAttr = to.getValue().getSExtValue();
  IntegerType toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                               .getElementType()
                               .cast<IntegerType>();

  assert(fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>());

  auto inputElems = input.cast<mlir::DenseElementsAttr>();
  std::vector<Attribute> result;

  for (IntegerAttr inputElement : inputElems.getValues<IntegerAttr>()) {
    APInt inVal = inputElement.getValue();
    APInt outVal = castIntToInt(inVal, fromElemType, toElemType);
    IntegerAttr attr = builder.getIntegerAttr(toElemType, outVal);
    result.push_back(attr);
  }

  auto constShape = constType.getShape();
  auto resultType = mlir::RankedTensorType::get(constShape, toElemType);
  auto resultAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(result));
  return resultAttr;
}

bool canConstPropCastIntToInt(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {
  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  Type fromElemType = constType.getElementType();

  auto toAttr = to.getValue().getSExtValue();
  auto toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                        .getElementType();

  return fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for cast float to float.
//===----------------------------------------------------------------------===//
DenseElementsAttr ConstPropCastFloatToFloat(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {

  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  FloatType fromElemType = constType.getElementType().cast<FloatType>();

  auto toAttr = to.getValue().getSExtValue();
  FloatType toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                             .getElementType()
                             .cast<FloatType>();

  assert(fromElemType.isa<FloatType>() && toElemType.isa<FloatType>());

  auto inputElems = input.cast<mlir::DenseElementsAttr>();
  std::vector<Attribute> result;

  for (FloatAttr inputElement : inputElems.getValues<FloatAttr>()) {
    APFloat inVal = inputElement.getValue();
    APFloat outVal = castFloatToFloat(inVal, fromElemType, toElemType);
    FloatAttr attr = builder.getFloatAttr(toElemType, outVal);
    result.push_back(attr);
  }

  auto constShape = constType.getShape();
  auto resultType = mlir::RankedTensorType::get(constShape, toElemType);
  auto resultAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(result));
  return resultAttr;
}

bool canConstPropCastFloatToFloat(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {
  mlir::RankedTensorType constType =
      constOp.getType().cast<mlir::RankedTensorType>();
  Type fromElemType = constType.getElementType();

  auto toAttr = to.getValue().getSExtValue();
  auto toElemType = mlir::UnrankedTensorType::get(
      convertONNXTypeToMLIRType(
          builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
                        .getElementType();

  return (fromElemType.isa<Float16Type>() || fromElemType.isa<Float32Type>()) &&
         (toElemType.isa<Float16Type>() || toElemType.isa<Float32Type>());
}

bool isConstOfZeros(Builder &builder, Attribute attr) {

  DenseElementsAttr denseAttr = attr.cast<DenseElementsAttr>();
  mlir::Type constElemType = denseAttr.getType().getElementType();
  if (constElemType.isa<IntegerType>()) {
    for (IntegerAttr value : denseAttr.getValues<IntegerAttr>()) {
      APInt inVal = value.getValue();
      if (!inVal.isNullValue()) {
        return false;
      }
    }
  } else if (constElemType.isa<FloatType>()) {
    for (FloatAttr value : denseAttr.getValues<FloatAttr>()) {
      APFloat inVal = value.getValue();
      if (!inVal.isZero()) {
        return false;
      }
    }
  } else {
    return false;
  }

  return true;
}

DenseElementsAttr CreateZerosFromTemplate(
    Builder &builder, Value templateTensor) {
  ShapedType shapedType = templateTensor.getType().cast<ShapedType>();
  Type elementType = shapedType.getElementType();
  DenseElementsAttr resultAttr;
  elementType.dump();
  if (elementType == builder.getF32Type()) {
    float value = 0.0;
    resultAttr = DenseElementsAttr::get(shapedType, value);
  }
  else if (elementType == builder.getF16Type()) {  
    APFloat value = APFloat(0.0);
    bool loses_info = true;
    value.convert(APFloat::IEEEhalf(),llvm::RoundingMode::NearestTiesToEven, &loses_info);
    resultAttr = DenseElementsAttr::get(shapedType, value);
  }
  else if (elementType == builder.getIntegerType(8)) {  
    int8_t value = 0;
    resultAttr = DenseElementsAttr::get(shapedType, value);
  }
  else if (elementType == builder.getIntegerType(32)) {  
    int32_t value = 0;
    resultAttr = DenseElementsAttr::get(shapedType, value);
  }
  return resultAttr;
}

namespace {

/**
 * @brief Compute the normalized shapes of result, and original LHS and RHS
 * operands. This is to make the shapes appear cleaner to assist broadcasted
 * MatMul computation. A few things are done:
 * - Perform vector-to-matrix promotion to either 1D operand (if any)
 * - Prepend 1s to the lower-ranked operand so both operands appear to have the
 * same rank
 * - Deduce the "normalized MatMul result shape" based on normalized operand
 * shapes, described in steps above :) Read MatMul Shape Broadcast Rules:
 * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
 *
 * @param lhsShape Shape of the original LHS operand
 * @param rhsShape Shape of the original RHS operand
 * @return A tuple of normalized result shape, LHS shape, and RHS shape
 */
static auto normalizeMatMulShapes(
    ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape) {

  assert(lhsShape.size() > 0 && rhsShape.size() > 0);
  assert(lhsShape.size() > 1 ||
         rhsShape.size() > 1); // Must have one shape that's 2D or higher.

  std::vector<int64_t> normLhsShape;
  if (lhsShape.size() == 1) {
    normLhsShape = {1, lhsShape[0]};
  } else {
    normLhsShape = lhsShape;
  }
  std::reverse(normLhsShape.begin(), normLhsShape.end());

  std::vector<int64_t> normRhsShape;
  if (rhsShape.size() == 1) {
    normRhsShape = {rhsShape[0], 1};
  } else {
    normRhsShape = rhsShape;
  }
  std::reverse(normRhsShape.begin(), normRhsShape.end());

  const size_t normResultNumRanks =
      std::max(normLhsShape.size(), normRhsShape.size());

  // Extend the shorter shape with 1 to make them the same length, and reverse
  // back
  normLhsShape.resize(normResultNumRanks, 1);
  std::reverse(normLhsShape.begin(), normLhsShape.end());
  normRhsShape.resize(normResultNumRanks, 1);
  std::reverse(normRhsShape.begin(), normRhsShape.end());

  // Verify some assumptions:
  assert(normLhsShape.size() == normRhsShape.size());

  std::vector<int64_t> normResultShape(normResultNumRanks);
  // Populate Batch Dimensions
  for (size_t i = 0; i < normResultNumRanks - 2; ++i) {
    int64_t lhsDim = normLhsShape.at(i);
    int64_t rhsDim = normRhsShape.at(i);
    int64_t normResultDim = std::max(lhsDim, rhsDim);
    normResultShape[i] = normResultDim;
  }
  // Populate MatMul Dimensions
  normResultShape[normResultNumRanks - 2] =
      normLhsShape[normLhsShape.size() - 2];
  normResultShape[normResultNumRanks - 1] = normRhsShape.back();

  return std::make_tuple(std::move(normLhsShape), std::move(normRhsShape),
      std::move(normResultShape));
}

template <class T>
void printVec(const char *title, T vec) {
  llvm::outs() << title << ": {";
  for (const auto &v : vec) {
    llvm::outs() << v << ",";
  }
  llvm::outs() << "}\n";
}

static void verifyShapes(ArrayRef<int64_t> normResultShape,
    ArrayRef<int64_t> normLhsShape, ArrayRef<int64_t> normRhsShape,
    ArrayRef<int64_t> resultShape, ArrayRef<int64_t> lhsShape,
    ArrayRef<int64_t> rhsShape) {

  using ShapeVec = llvm::SmallVector<int64_t, 6>;

  if (DEBUG) {
    printVec("normResultShape", normResultShape);
    printVec("normLhsShape", normLhsShape);
    printVec("normRhsShape", normRhsShape);
    printVec("resultShape", resultShape);
    printVec("lhsShape", lhsShape);
    printVec("rhsShape", rhsShape);
  }

  assert(normResultShape.size() == normLhsShape.size());
  assert(normResultShape.size() == normRhsShape.size());

  if (lhsShape.size() > 1 && rhsShape.size() > 1) {
    assert(normResultShape == resultShape);
  } else if (lhsShape.size() == 1) {
    assert(rhsShape.size() > 1);
    ShapeVec expected(normResultShape.begin(), normResultShape.end());
    expected.erase(expected.end() - 2);
    assert(expected == resultShape);
  } else if (rhsShape.size() == 1) {
    assert(lhsShape.size() > 1);
    ShapeVec expected(normResultShape.begin(), normResultShape.end() - 1);
    assert(expected == resultShape);
  } else {
    assert(false);
  }
}

class BroadcastMatMulCalculator {
public:
  /**
   * @brief Calculate MatMul result.
   *
   * @param resultType Type of the result
   * @param _lhs Raw LHS operand of the MatMul
   * @param _rhs Raw RHS operand of the MatMul
   * @return Raw buffer of the result that you may assign to a dense attribute
   * of correct result type (that was set during construction of the
   * calculator).
   */
  static std::vector<APInt> calc(
      ShapedType resultType, Attribute _lhs, Attribute _rhs) {
    DenseElementsAttr lhs = _lhs.cast<DenseElementsAttr>();
    DenseElementsAttr rhs = _rhs.cast<DenseElementsAttr>();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    ArrayRef<int64_t> lhsShape = lhs.getType().getShape();
    ArrayRef<int64_t> rhsShape = rhs.getType().getShape();

    const auto normShapes = normalizeMatMulShapes(lhsShape, rhsShape);
    const auto &normLhsShape = std::get<0>(normShapes);
    const auto &normRhsShape = std::get<1>(normShapes);
    const auto &normResultShape = std::get<2>(normShapes);

    verifyShapes(normResultShape, normLhsShape, normRhsShape, resultShape,
        lhsShape, rhsShape);

    const auto dimOffset = calculateDimOffsets(normResultShape);

    int64_t m = normLhsShape.at(normLhsShape.size() - 2);
    int64_t k = normLhsShape.back();
    int64_t n = normRhsShape.back();

    BroadcastMatMulCalculator calculator(resultType, lhs, rhs, normResultShape,
        normLhsShape, normRhsShape, dimOffset, m, k, n);

    calculator._resultStorage = createInitialResultStorage(resultType);
    calculator.recurse(0, 0);
    return std::move(calculator._resultStorage);
  }

private:
  BroadcastMatMulCalculator(ShapedType resultType, DenseElementsAttr lhs,
      DenseElementsAttr rhs, ArrayRef<int64_t> normResultShape,
      ArrayRef<int64_t> normLhsShape, ArrayRef<int64_t> normRhsShape,
      ArrayRef<size_t> dimOffset, int64_t m, int64_t k, int64_t n)
      : _resultType(resultType), _lhs(lhs), _rhs(rhs),
        _normResultShape(normResultShape), _normLhsShape(normLhsShape),
        _normRhsShape(normRhsShape), _dimOffsets(dimOffset), _M(m), _K(k),
        _N(n) {}

  void recurse(size_t rank, size_t flattenedIdxBase) {
    if (DEBUG) {
      printVec("_idxStack", _idxStack);
    }
    int64_t lhsRank = _lhs.getType().getRank();
    int64_t rhsRank = _rhs.getType().getRank();
    IntegerType lhsEType = _lhs.getType().getElementType().cast<IntegerType>();
    IntegerType rhsEType = _rhs.getType().getElementType().cast<IntegerType>();
    if (rank + 2 == _normLhsShape.size()) {
      // Do 2D MatMul stuff
      const IntegerType resultElementType =
          _resultType.getElementType().cast<IntegerType>();
      for (uint64_t i = 0; (int64_t)i < _M; ++i) {
        for (uint64_t j = 0; (int64_t)j < _N; ++j) {
          const uint64_t flattenedIdx = flattenedIdxBase + i * _N + j;
          for (uint64_t k = 0; (int64_t)k < _K; ++k) {
            auto lhsIdx =
                getBroadcastIdx<true>(_idxStack, i, k, _normLhsShape, _lhs);
            auto rhsIdx =
                getBroadcastIdx<false>(_idxStack, k, j, _normRhsShape, _rhs);
            if (DEBUG) {
              printVec("lhsIdx", lhsIdx);
              printVec("rhsIdx", rhsIdx);
            }
            APInt aRaw = _lhs.getValue(lhsIdx).cast<IntegerAttr>().getValue();
            APInt bRaw = _rhs.getValue(rhsIdx).cast<IntegerAttr>().getValue();
            APInt a = castIntToInt(aRaw, lhsEType, resultElementType);
            APInt b = castIntToInt(bRaw, rhsEType, resultElementType);
            APInt ab = a * b;
            if (DEBUG) {
              llvm::outs() << "  [" << flattenedIdx << "] += " << a << " * "
                           << b << "\n";
            }
            _resultStorage.at(flattenedIdx) += ab;
          }
        }
      }
    } else {
      const int64_t rankDim = _normResultShape[rank];
      if (DEBUG) {
        llvm::outs() << "rank: " << rank << ", rankDim: " << rankDim << "\n";
      }
      for (int64_t i = 0; i < rankDim; ++i) {
        size_t nextRankIdx = flattenedIdxBase + i * _dimOffsets[rank + 1];
        _idxStack.push_back(i);
        recurse(rank + 1, nextRankIdx);
        _idxStack.pop_back();
      }
    }
  }

  /**
   * @brief Given Normalized Batch Dimensions and the MatMul Dimensions,
   * get indices you can use to query the original operand by applying broadcast
   * rules https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   *
   *
   * @tparam operandIsLhs
   * @param normResultBatchDimensions Batch Dimensions that maps to the
   * normalized result
   * @param a First MatMul Dimension (normalized) on the operand
   * @param b Second MatMul Dimension (normalized) on the operand
   * @param operandNormShape The normalized shape of the operand
   * @param operand The operand itself
   * @return The Indices you can use to query the operand
   */
  template <bool operandIsLhs>
  static llvm::SmallVector<uint64_t, 6> getBroadcastIdx(
      ArrayRef<int64_t> normResultBatchDimensions, uint64_t a, uint64_t b,
      ArrayRef<int64_t> operandNormShape, DenseElementsAttr operand) {

    using IdxVec = llvm::SmallVector<uint64_t, 6>;
    IdxVec idx;

    assert(normResultBatchDimensions.size() + 2 == operandNormShape.size());
    const int64_t operandRank = operand.getType().getRank();
    if (operandRank > 1) {
      const int64_t extraRanks =
          (int64_t)normResultBatchDimensions.size() - (operandRank - 2);
      assert(extraRanks >= 0);

      idx.assign(normResultBatchDimensions.begin() + extraRanks,
          normResultBatchDimensions.end());
      idx.push_back(a);
      idx.push_back(b);

      for (size_t i = 0; i < idx.size() - 2; ++i) {
        uint64_t maxIdx = operandNormShape[i + extraRanks];
        assert(maxIdx > 0);
        --maxIdx;
        idx[i] = std::min(idx[i], maxIdx);
      }
    } else {
      // If operand is 1D, the index is just the inner dimension
      assert(operandRank == 1);
      if (operandIsLhs) {
        idx.push_back(b);
      } else {
        idx.push_back(a);
      }
    }

    // Trust that move will be ellided. Don't temper with this statement.
    // Even surrounding `idx` with parenthesis will break it.
    // Go figure. https://godbolt.org/z/dxbeM9jdK
    return idx;
  }

  /**
   * @brief Creates a buffer of APInt objects corresponding to the element
   * type of resultType, and size just enough to accommodate a tensor
   * of shape represented by resultType
   *
   * @param resultType Shape and element type information used to create buffer
   * @return std::vector<APInt> APInt Buffer created, initialized to 0.
   */
  static std::vector<APInt> createInitialResultStorage(ShapedType resultType) {
    IntegerType resultElementType =
        resultType.getElementType().cast<IntegerType>();
    const APInt emptyInt(
        resultElementType.getWidth(), 0, !resultElementType.isUnsigned());
    std::vector<APInt> resultAttr(resultType.getNumElements(), emptyInt);
    return std::move(resultAttr);
  }

  /**
   * @brief
   * Given a shape, calculate the offset of each dimension.
   * E.g. with given input shape (1, 2, 3, 4, 5),
   * the output would be (120, 120, 60, 20, 5)
   *
   * @param normResultShape input shape
   * @return std::vector<size_t> output shape
   */
  static std::vector<size_t> calculateDimOffsets(
      ArrayRef<int64_t> normResultShape) {

    std::vector<int64_t> normResultShapeReversed(normResultShape);
    std::reverse(
        normResultShapeReversed.begin(), normResultShapeReversed.end());

    std::vector<size_t> offsets;
    size_t accumulator = 1;
    for (int64_t dimSize : normResultShapeReversed) {
      accumulator *= dimSize;
      offsets.push_back(accumulator);
    }
    std::reverse(offsets.begin(), offsets.end());
    return offsets;
  }

  const ShapedType _resultType;
  const DenseElementsAttr _lhs;
  const DenseElementsAttr _rhs;
  const ArrayRef<int64_t> _normResultShape;
  const ArrayRef<int64_t> _normLhsShape;
  const ArrayRef<int64_t> _normRhsShape;
  const ArrayRef<size_t> _dimOffsets;
  const int64_t _M, _K, _N;
  std::vector<APInt> _resultStorage;
  std::vector<int64_t> _idxStack;
};

} // namespace

DenseElementsAttr CreateMatMulIntegerOfConsts(
    Builder &builder, Value resultValue, Attribute lhs, Attribute rhs) {
  ShapedType resultType = resultValue.getType().cast<ShapedType>();
  std::vector<APInt> result =
      BroadcastMatMulCalculator::calc(resultType, lhs, rhs);
  if (DEBUG) {
    printVec("result", result);
  }
  return DenseElementsAttr::get(resultType, result);
}
