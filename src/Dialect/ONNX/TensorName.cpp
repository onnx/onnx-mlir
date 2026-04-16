// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <memory>
#include <numeric>

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeUtilities.h>

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Interface/TensorNameInference.hpp"

using namespace mlir;

namespace onnx_mlir {

std::unique_ptr<Transform> fromAttr(ArrayAttr arrayAttr) {
  if (arrayAttr.size()) {
    if (auto transType = dyn_cast<StringAttr>(arrayAttr[0])) {
      if (transType == "Reshape")
        return std::make_unique<ReshapeTransform>(arrayAttr);
      else if (transType == "Transpose")
        return std::make_unique<TransposeTransform>(arrayAttr);
      else if (transType == "Pad")
        return std::make_unique<PadTransform>(arrayAttr);
      else if (transType == "Slice")
        return std::make_unique<SliceTransform>(arrayAttr);
      else if (transType == "Dequantize")
        return std::make_unique<DequantizeTransform>(arrayAttr);
      else if (transType == "Quantize")
        return std::make_unique<QuantizeTransform>(arrayAttr);
    }
  }
  return {};
}

std::unique_ptr<Transform> fromOp(Operation *op) {
  if (auto nameInf = dyn_cast_if_present<mlir::TensorNameInference>(op))
    return nameInf.inferTensorNameTransform();
  return {};
}

SmallVector<int64_t> arrayToVector(ArrayAttr arrayAttr) {
  SmallVector<int64_t> vector;
  for (const APInt intVal : arrayAttr.getAsValueRange<IntegerAttr>())
    vector.push_back(intVal.getSExtValue());
  return vector;
}

SmallVector<int64_t> denseToVector(DenseIntElementsAttr denseAttr) {
  if (denseAttr.isSplat()) {
    SmallVector<int64_t> vector(denseAttr.getNumElements());
    for (int64_t &v : vector)
      v = denseAttr.getSplatValue<int64_t>();
    return vector;
  }
  return SmallVector<int64_t>(denseAttr.getValues<int64_t>());
}

SmallVector<int64_t> valToVector(Value val) {
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>()) {
    if (auto arrayAttr = dyn_cast<ArrayAttr>(constOp.getValueAttr()))
      return arrayToVector(arrayAttr);
    else if (auto denseAttr =
                 dyn_cast<DenseIntElementsAttr>(constOp.getValueAttr()))
      return denseToVector(denseAttr);
  }
  return {};
}

SmallVector<int64_t> axesToVector(Value val, size_t rank) {
  if (isa<NoneType>(val.getType())) {
    SmallVector<int64_t> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);
    return axes;
  } else {
    SmallVector<int64_t> axes = valToVector(val);
    for (int64_t &ax : axes)
      if (ax < 0 && rank > 0)
        ax += rank;
    return axes;
  }
}

ArrayAttr vecToAttr(MLIRContext *context, ArrayRef<int64_t> vector) {
  auto vecOfAttr =
      llvm::map_to_vector(vector, [context](int64_t v) -> Attribute {
        return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, v));
      });
  return ArrayAttr::get(context, vecOfAttr);
}

// == Reshape == //

ReshapeTransform::ReshapeTransform(
    ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape)
    : Transform(Kind::Reshape, inShape, outShape) {}

ReshapeTransform::ReshapeTransform(ArrayAttr attr)
    : ReshapeTransform(arrayToVector(cast<ArrayAttr>(attr[1])),
          arrayToVector(cast<ArrayAttr>(attr[2]))) {}

Attribute ReshapeTransform::toAttr(MLIRContext *context) const {
  return ArrayAttr::get(
      context, {StringAttr::get(context, "Reshape"),
                   vecToAttr(context, inShape), vecToAttr(context, outShape)});
}

std::unique_ptr<Transform> ReshapeTransform::invert() const {
  return std::make_unique<ReshapeTransform>(outShape, inShape);
}

// == Transpose == //

TransposeTransform::TransposeTransform(ArrayRef<int64_t> inShape,
    ArrayRef<int64_t> perm, ArrayRef<int64_t> outShape)
    : Transform(Kind::Transpose, inShape, outShape), perm(perm) {}

TransposeTransform::TransposeTransform(ArrayAttr attr)
    : TransposeTransform(arrayToVector(cast<ArrayAttr>(attr[1])),
          arrayToVector(cast<ArrayAttr>(attr[2])),
          arrayToVector(cast<ArrayAttr>(attr[3]))) {}

Attribute TransposeTransform::toAttr(MLIRContext *context) const {
  return ArrayAttr::get(context, {
                                     StringAttr::get(context, "Transpose"),
                                     vecToAttr(context, inShape),
                                     vecToAttr(context, perm),
                                     vecToAttr(context, outShape),
                                 });
}

std::unique_ptr<Transform> TransposeTransform::invert() const {
  SmallVector<int64_t> invPerm(perm.size());
  for (const auto [i, p] : llvm::enumerate(perm))
    invPerm[p] = i;

  return std::make_unique<TransposeTransform>(outShape, invPerm, inShape);
}

// == Pad == //

PadTransform::PadTransform(ArrayRef<int64_t> inShape, ArrayRef<int64_t> starts,
    ArrayRef<int64_t> ends, ArrayRef<int64_t> axes, Attribute constant,
    ArrayRef<int64_t> outShape)
    : Transform(Kind::Pad, inShape, outShape), starts(starts), ends(ends),
      axes(axes), constant(constant) {}

PadTransform::PadTransform(ArrayAttr attr)
    : PadTransform(arrayToVector(cast<ArrayAttr>(attr[1])),
          arrayToVector(cast<ArrayAttr>(attr[2])),
          arrayToVector(cast<ArrayAttr>(attr[3])),
          arrayToVector(cast<ArrayAttr>(attr[4])), attr[5],
          arrayToVector(cast<ArrayAttr>(attr[6]))) {}

Attribute PadTransform::toAttr(MLIRContext *context) const {
  return ArrayAttr::get(
      context, {
                   StringAttr::get(context, "Pad"),
                   vecToAttr(context, inShape),
                   vecToAttr(context, starts),
                   vecToAttr(context, ends),
                   vecToAttr(context, axes),
                   constant ? constant : UnitAttr::get(context),
                   vecToAttr(context, outShape),
               });
}

std::unique_ptr<Transform> PadTransform::invert() const {
  SmallVector<int64_t> sliceEnds(ends.size());
  for (const auto &[ax, pe, se] : llvm::zip_equal(axes, ends, sliceEnds))
    se = outShape[ax] - pe;

  return std::make_unique<SliceTransform>(
      outShape, starts, sliceEnds, axes, inShape);
}

// == Slice == //

SliceTransform::SliceTransform(ArrayRef<int64_t> inShape,
    ArrayRef<int64_t> starts, ArrayRef<int64_t> ends, ArrayRef<int64_t> axes,
    ArrayRef<int64_t> outShape)
    : Transform(Kind::Slice, inShape, outShape), starts(starts), ends(ends),
      axes(axes) {}

SliceTransform::SliceTransform(ArrayAttr attr)
    : SliceTransform(arrayToVector(cast<ArrayAttr>(attr[1])),
          arrayToVector(cast<ArrayAttr>(attr[2])),
          arrayToVector(cast<ArrayAttr>(attr[3])),
          arrayToVector(cast<ArrayAttr>(attr[4])),
          arrayToVector(cast<ArrayAttr>(attr[5]))) {}

Attribute SliceTransform::toAttr(MLIRContext *context) const {
  return ArrayAttr::get(context, {
                                     StringAttr::get(context, "Slice"),
                                     vecToAttr(context, inShape),
                                     vecToAttr(context, starts),
                                     vecToAttr(context, ends),
                                     vecToAttr(context, axes),
                                     vecToAttr(context, outShape),
                                 });
}

std::unique_ptr<Transform> SliceTransform::invert() const {
  SmallVector<int64_t> padEnds(ends.size());
  for (const auto &[ax, se, pe] : llvm::zip_equal(axes, ends, padEnds))
    pe = inShape[ax] - se;

  return std::make_unique<PadTransform>(
      outShape, starts, padEnds, axes, Attribute(), inShape);
}

// == QDQ == //

template <Transform::Kind QDQType>
QDQTransform<QDQType>::QDQTransform(mlir::ArrayRef<int64_t> shape, double scale,
    int64_t zeroPoint, mlir::Type scaleType, mlir::Type zpType)
    : Transform(QDQType, shape, shape), scale(scale), zeroPoint(zeroPoint),
      scaleType(scaleType), zpType(zpType) {}

template <Transform::Kind QDQType>
QDQTransform<QDQType>::QDQTransform(mlir::ArrayAttr attr)
    : QDQTransform(arrayToVector(cast<ArrayAttr>(attr[1])),
          cast<FloatAttr>(attr[2]).getValueAsDouble(),
          cast<IntegerAttr>(attr[3]).getInt(),
          cast<FloatAttr>(attr[2]).getType(),
          cast<IntegerAttr>(attr[3]).getType()) {}

template <Transform::Kind QDQType>
mlir::Attribute QDQTransform<QDQType>::toAttr(
    mlir::MLIRContext *context) const {
  StringRef transType = "Unknown";
  if constexpr (QDQType == Transform::Kind::Dequantize)
    transType = "Dequantize";
  else if constexpr (QDQType == Transform::Kind::Dequantize)
    transType = "Quantize";
  return ArrayAttr::get(context, {
                                     StringAttr::get(context, transType),
                                     vecToAttr(context, inShape),
                                     FloatAttr::get(scaleType, scale),
                                     IntegerAttr::get(zpType, zeroPoint),
                                 });
}

template <Transform::Kind QDQType>
std::unique_ptr<Transform> QDQTransform<QDQType>::invert() const {
  if constexpr (QDQType == Transform::Kind::Dequantize)
    return std::make_unique<QuantizeTransform>(
        inShape, scale, zeroPoint, scaleType, zpType);
  else if constexpr (QDQType == Transform::Kind::Dequantize)
    return std::make_unique<DequantizeTransform>(
        inShape, scale, zeroPoint, scaleType, zpType);
  return nullptr;
}

template <Transform::Kind QDQType>
mlir::Type QDQTransform<QDQType>::getFromDType() const {
  if constexpr (QDQType == Transform::Kind::Dequantize)
    return zpType;
  else if constexpr (QDQType == Transform::Kind::Quantize)
    return scaleType;
  return nullptr;
}

template <Transform::Kind QDQType>
mlir::Type QDQTransform<QDQType>::getToDType() const {
  if constexpr (QDQType == Transform::Kind::Dequantize)
    return scaleType;
  else if constexpr (QDQType == Transform::Kind::Quantize)
    return zpType;
  return nullptr;
}

// == List == //

ListTransform::ListTransform(
    SmallVector<std::unique_ptr<Transform>> &&transforms)
    : Transform(Kind::List, {}, {}), transforms(std::move(transforms)) {}

mlir::Attribute ListTransform::toAttr(MLIRContext * /*context*/) const {
  llvm_unreachable(
      "ListTransform should not be directly converted to attribute");
}

std::unique_ptr<Transform> ListTransform::invert() const {
  SmallVector<std::unique_ptr<Transform>> trans;
  for (const auto &transform : llvm::reverse(transforms))
    trans.push_back(transform->invert());
  return std::make_unique<ListTransform>(std::move(trans));
}

// == TensorName == //

TensorName::TensorName(std::string name) : name(std::move(name)) {}

TensorName::TensorName(Value value) {
  if (auto opRes = dyn_cast<OpResult>(value)) {
    auto resultNames =
        opRes.getOwner()->getAttrOfType<ArrayAttr>("ResultNames");
    if (resultNames && resultNames.size() > opRes.getResultNumber()) {
      auto resultName = resultNames[opRes.getResultNumber()];
      if (auto strAttr = dyn_cast<StringAttr>(resultName))
        name = strAttr.getValue().str();
      else if (auto arrayAttr = dyn_cast<ArrayAttr>(resultName)) {
        name = cast<StringAttr>(arrayAttr[0]).getValue().str();
        for (size_t i = 1; i < arrayAttr.size(); i++) {
          transforms.push_back(fromAttr(cast<ArrayAttr>(arrayAttr[i])));
        }
      }
    }
  } else if (auto blkArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blkArg.getOwner()->getParentOp();
    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      auto argIndex = blkArg.getArgNumber();
      if (auto strAttr =
              funcOp.getArgAttrOfType<StringAttr>(argIndex, "onnx.name"))
        name = strAttr.getValue().str();
    }
  }
}

TensorName TensorName::infer(Value value) {
  if (TensorName tname = inferWithUse(value))
    return tname;

  return inferWithDef(value);
}

TensorName TensorName::inferWithUse(Value value) {
  TensorName tname(value);
  if (tname || !value.hasOneUse() || value.use_begin()->getOperandNumber() != 0)
    return tname;

  Operation *op = *value.user_begin();
  if (auto transform = fromOp(op)) {
    tname = inferWithUse(op->getResult(0));
    if (tname) {
      tname.push_back(transform->invert());
      (void)tname.setTo(value);
    }
  }

  return tname;
}

TensorName TensorName::inferWithDef(Value value) {
  TensorName tname(value);
  if (tname)
    return tname;

  Operation *op = value.getDefiningOp();
  if (auto transform = fromOp(op)) {
    tname = inferWithDef(op->getOperand(0));
    if (tname) {
      tname.push_back(std::move(transform));
      (void)tname.setTo(value);
    }
  }

  return tname;
}

void TensorName::push_back(std::unique_ptr<Transform> transform) {
  if (auto *list = dyn_cast<ListTransform>(transform.get())) {
    for (auto &it : list->transforms) {
      push_back(std::move(it));
    }
    return;
  }
  transforms.push_back(std::move(transform));
}

Attribute TensorName::toAttr(MLIRContext *context) const {
  Attribute tnameAttr = StringAttr::get(context, llvm::Twine(name));
  if (!transforms.empty()) {
    SmallVector<Attribute> arrayAttr({tnameAttr});
    for (const auto &t : transforms)
      arrayAttr.push_back(t->toAttr(context));
    tnameAttr = ArrayAttr::get(context, arrayAttr);
  }
  return tnameAttr;
}

LogicalResult TensorName::setTo(Value value) const {
  if (auto result = dyn_cast<OpResult>(value)) {
    MLIRContext *ctx = value.getContext();
    Operation *op = result.getOwner();
    unsigned idx = result.getResultNumber();

    SmallVector<Attribute> resultNames(
        op->getNumResults(), StringAttr::get(ctx));
    if (auto resultNamesAttr = op->getAttrOfType<ArrayAttr>("ResultNames")) {
      // The existing ResultNames array may have fewer entries than the op has
      // results (e.g. when a prior setTo only populated a subset, or the
      // attribute was carried from an op with a different result count).
      // Copy what we have and leave the rest as the default empty StringAttr.
      auto existing = resultNamesAttr.getValue();
      for (unsigned i = 0, e = std::min((unsigned)existing.size(),
                               (unsigned)resultNames.size());
           i < e; ++i)
        resultNames[i] = existing[i];
    }

    resultNames[idx] = toAttr(ctx);
    op->setAttr("ResultNames", ArrayAttr::get(ctx, resultNames));
    return success();
  }
  return failure();
}

// == OpInterface == //

std::unique_ptr<Transform>
ReshapeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto reshapeOp = cast<ONNXReshapeOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(reshapeOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  return std::make_unique<ReshapeTransform>(
      inType.getShape(), outType.getShape());
}

std::unique_ptr<Transform>
TransposeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto transposeOp = cast<ONNXTransposeOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(transposeOp.getOperand().getType());
  auto outType = cast<RankedTensorType>(transposeOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  auto perm = arrayToVector(transposeOp.getPermAttr());

  return std::make_unique<TransposeTransform>(
      inType.getShape(), perm, outType.getShape());
}

std::unique_ptr<Transform> PadOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto padOp = cast<ONNXPadOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(padOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(padOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  // Only mode = "constant" is supported
  if (padOp.getMode() != "constant")
    return nullptr;
  auto constOp = padOp.getConstantValue().getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return nullptr;
  Attribute constant = constOp.getValueAttr();
  if (!constant)
    return nullptr;

  // Validate if pads is constant
  auto pads = valToVector(padOp.getPads());
  if (pads.size() == 0)
    return nullptr;

  auto splitAt = pads.size() / 2;
  auto starts = SmallVector<int64_t>(pads.begin(), pads.begin() + splitAt);
  auto ends = SmallVector<int64_t>(pads.begin() + splitAt, pads.end());
  auto axes = axesToVector(padOp.getAxes(), inType.getRank());

  return std::make_unique<PadTransform>(
      inType.getShape(), starts, ends, axes, constant, outType.getShape());
}

std::unique_ptr<Transform> SliceOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto sliceOp = cast<ONNXSliceOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(sliceOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(sliceOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  // Validate if starts & ends are constant and steps is always 1
  auto starts = valToVector(sliceOp.getStarts());
  auto ends = valToVector(sliceOp.getEnds());
  auto steps = valToVector(sliceOp.getSteps());
  if (starts.size() == 0 || ends.size() == 0 ||
      llvm::any_of(steps, [](int64_t s) { return s != 1; }))
    return nullptr;

  auto inShape = inType.getShape();
  auto axes = axesToVector(sliceOp.getAxes(), inShape.size());

  // Clip end values
  for (const auto &[ax, en] : llvm::zip_equal(axes, ends))
    en = std::min(en, inShape[ax]);

  return std::make_unique<SliceTransform>(
      inShape, starts, ends, axes, outType.getShape());
}

void registerTensorNameInferenceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension<ONNXDialect>(
      +[](MLIRContext *ctx, ONNXDialect * /*dialect*/) {
        ONNXTransposeOp::attachInterface<TransposeOpTensorNameInference>(*ctx);
        ONNXReshapeOp::attachInterface<ReshapeOpTensorNameInference>(*ctx);
        ONNXPadOp::attachInterface<PadOpTensorNameInference>(*ctx);
        ONNXSliceOp::attachInterface<SliceOpTensorNameInference>(*ctx);
      });
}

} // namespace onnx_mlir
