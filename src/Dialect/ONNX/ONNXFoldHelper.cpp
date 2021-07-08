#include "src/Dialect/ONNX/ONNXFoldHelper.hpp"

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
}

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
  APInt castIntToInt(APInt inVal, IntegerType toType) {
    unsigned toWidth = toType.getWidth();
    bool isUnsigned = toType.isUnsigned();
    if (isUnsigned) {
      return inVal.zextOrTrunc(toWidth);
    } else {
      return inVal.sextOrTrunc(toWidth);
    }
  }
}

DenseElementsAttr ConstPropCastIntToInt(
    Builder &builder, Value constOp, Attribute input, IntegerAttr to) {

  mlir::RankedTensorType constType = constOp.getType().cast<mlir::RankedTensorType>();
  Type fromElemType = constType.getElementType();

  auto toAttr = to.getValue().getSExtValue();
  auto toElemType = mlir::UnrankedTensorType::get(
    convertONNXTypeToMLIRType(builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
    .getElementType();

  assert(fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>());

  auto inputElems = input.cast<mlir::DenseElementsAttr>();
  std::vector<Attribute> result;

  for (IntegerAttr inputElement : inputElems.getValues<IntegerAttr>()) {
    APInt inVal = inputElement.getValue();
    APInt outVal = castIntToInt(inVal, toElemType.cast<IntegerType>());
    IntegerAttr attr = builder.getIntegerAttr(toElemType, outVal);
    result.push_back(attr);
  }

  auto constShape = constType.getShape();
  auto resultType = mlir::RankedTensorType::get(constShape, toElemType);
  auto resultAttr = DenseElementsAttr::get(resultType, llvm::makeArrayRef(result));
  return resultAttr;
}

bool canConstPropCastIntToInt(Builder &builder, Value constOp, Attribute input, IntegerAttr to) {
  mlir::RankedTensorType constType = constOp.getType().cast<mlir::RankedTensorType>();
  Type fromElemType = constType.getElementType();

  auto toAttr = to.getValue().getSExtValue();
  auto toElemType = mlir::UnrankedTensorType::get(
    convertONNXTypeToMLIRType(builder, static_cast<onnx::TensorProto_DataType>(toAttr)))
    .getElementType();

  return fromElemType.isa<IntegerType>() && toElemType.isa<IntegerType>();
}
