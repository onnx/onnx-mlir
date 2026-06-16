// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// FuseMatMulAddToXFEMatMulBiasPass
//
// Fuses ONNX MatMul followed by ONNX Add with a constant bias into
// onnx.XFEMatMulBias, mirroring the xcompiler ReplaceQDQMatmulPass idea:
// when the bias tensor has one value per output channel (the last dimension
// of the RHS of MatMul), the add is folded into a single fused op.
//
// Supports float and quantized tensors (uniform per-tensor and per-axis on
// the bias), and constant data stored as DenseElementsAttr or
// DisposableElementsAttr (via ElementsAttr).
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

using onnx_mlir::OnnxElementsAttrBuilder;

static Type elementTypeFor1DBias(Type elemType) {
  auto perAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(elemType);
  if (!perAxis)
    return elemType;

  auto scales = perAxis.getScales();
  auto zps = perAxis.getZeroPoints();
  if (scales.size() == 1) {
    return quant::UniformQuantizedType::get(perAxis.getFlags(),
        perAxis.getStorageType(), perAxis.getExpressedType(), scales[0],
        zps.empty() ? 0 : zps[0], perAxis.getStorageTypeMin(),
        perAxis.getStorageTypeMax());
  }

  return quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(),
      perAxis.getStorageType(), perAxis.getExpressedType(), scales, zps,
      /*quantizedDimension=*/0, perAxis.getStorageTypeMin(),
      perAxis.getStorageTypeMax());
}

static int64_t quantScaleCount(Type elemType) {
  if (auto perAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(elemType))
    return static_cast<int64_t>(perAxis.getScales().size());
  if (isa<quant::UniformQuantizedType>(elemType))
    return 1;
  return 0;
}

static bool isBiasGranularityCompatible(
    Type biasElemType, Type weightElemType) {
  return quantScaleCount(biasElemType) == quantScaleCount(weightElemType);
}

static ONNXConstantOp getDefiningConstantThroughDequant(Value v) {
  if (auto c = v.getDefiningOp<ONNXConstantOp>())
    return c;
  if (auto dq = v.getDefiningOp<ONNXDequantizeLinearOp>())
    return dq.getX().getDefiningOp<ONNXConstantOp>();
  return nullptr;
}

static bool isBiasCompatibleWithMatMul(Value biasVal, Value bVal) {
  auto bType = mlir::dyn_cast<RankedTensorType>(bVal.getType());
  if (!bType || bType.getRank() < 1)
    return false;
  int64_t nOut = bType.getShape().back();
  if (nOut == ShapedType::kDynamic)
    return false;

  auto constOp = getDefiningConstantThroughDequant(biasVal);
  if (!constOp)
    return false;

  auto elms = mlir::dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elms)
    return false;

  auto biasTy = mlir::dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!biasTy)
    return false;

  if (static_cast<int64_t>(elms.getNumElements()) != nOut)
    return false;

  ArrayRef<int64_t> biasShape = biasTy.getShape();
  if (biasShape.empty() || biasShape.back() != nOut)
    return false;
  for (int64_t d = 0, e = biasTy.getRank() - 1; d < e; ++d)
    if (biasShape[d] != 1)
      return false;

  return isBiasGranularityCompatible(
      biasTy.getElementType(), bType.getElementType());
}

static Value create1DBiasFromConstant(
    PatternRewriter &rewriter, Value biasVal, Location loc, int64_t n) {
  auto constOp = getDefiningConstantThroughDequant(biasVal);
  if (!constOp)
    return biasVal;

  auto elms = mlir::dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());
  if (!elms)
    return biasVal;

  auto resultTy =
      mlir::dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!resultTy)
    return biasVal;

  Type newElemType = elementTypeFor1DBias(resultTy.getElementType());
  auto newResultType =
      RankedTensorType::get(SmallVector<int64_t>{n}, newElemType);

  if (resultTy.getRank() == 1 && resultTy.getShape() == ArrayRef<int64_t>{n} &&
      resultTy.getElementType() == newElemType)
    return biasVal;

  OnnxElementsAttrBuilder eb(rewriter.getContext());
  ElementsAttr reshaped = eb.reshape(elms, SmallVector<int64_t>{n});
  DenseElementsAttr denseNew =
      onnx_mlir::ElementsAttrBuilder::toDenseElementsAttr(reshaped);

  auto valueAttr = rewriter.getNamedAttr("value", denseNew);
  return rewriter
      .create<ONNXConstantOp>(loc, newResultType, mlir::ValueRange{},
          mlir::ArrayRef<mlir::NamedAttribute>{valueAttr})
      .getResult();
}

static LogicalResult tryFuseMatMulBias(
    Operation *addLikeOp, Value lhs, Value rhs, PatternRewriter &rewriter) {
  ONNXMatMulOp matmulOp = nullptr;
  Value biasConstant = nullptr;

  if (auto mm = lhs.getDefiningOp<ONNXMatMulOp>()) {
    if (getDefiningConstantThroughDequant(rhs)) {
      matmulOp = mm;
      biasConstant = rhs;
    }
  }
  if (!matmulOp) {
    if (auto mm = rhs.getDefiningOp<ONNXMatMulOp>()) {
      if (getDefiningConstantThroughDequant(lhs)) {
        matmulOp = mm;
        biasConstant = lhs;
      }
    }
  }

  if (!matmulOp || !biasConstant)
    return failure();

  if (!matmulOp.getResult().hasOneUse())
    return failure();

  Value bVal = matmulOp.getB();
  auto bType = mlir::dyn_cast<RankedTensorType>(bVal.getType());
  if (!bType || bType.getShape().back() == ShapedType::kDynamic)
    return failure();
  int64_t nOut = bType.getShape().back();

  if (!isBiasCompatibleWithMatMul(biasConstant, bVal))
    return failure();

  Location loc = addLikeOp->getLoc();
  Value bias1D = create1DBiasFromConstant(rewriter, biasConstant, loc, nOut);

  auto fused =
      rewriter.create<XFEMatMulBiasOp>(loc, addLikeOp->getResult(0).getType(),
          matmulOp.getA(), matmulOp.getB(), bias1D);

  rewriter.replaceOp(addLikeOp, fused.getY());
  rewriter.eraseOp(matmulOp);
  return success();
}

struct FuseMatMulAddToXFEMatMulBiasPattern
    : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    return tryFuseMatMulBias(addOp, addOp.getA(), addOp.getB(), rewriter);
  }
};

struct FuseMatMulFusedEltwiseToXFEMatMulBiasPattern
    : public OpRewritePattern<XCOMPILERFusedEltwiseOp> {
  using OpRewritePattern<XCOMPILERFusedEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XCOMPILERFusedEltwiseOp eltwiseOp,
      PatternRewriter &rewriter) const override {
    auto typeAttr = eltwiseOp->getAttrOfType<StringAttr>("type");
    if (!typeAttr || typeAttr.getValue() != "ADD")
      return failure();
    auto nonlinearAttr = eltwiseOp->getAttrOfType<StringAttr>("nonlinear");
    if (nonlinearAttr && nonlinearAttr.getValue() != "NONE")
      return failure();

    Value b = eltwiseOp.getB();
    if (!b || mlir::isa<NoneType>(b.getType()))
      return failure();

    return tryFuseMatMulBias(eltwiseOp, eltwiseOp.getA(), b, rewriter);
  }
};

} // namespace

namespace onnx_mlir {

struct FuseMatMulAddToXFEMatMulBiasPass
    : public PassWrapper<FuseMatMulAddToXFEMatMulBiasPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "fuse-matmul-add-to-xfe-matmul-bias";
  }
  StringRef getDescription() const override {
    return "Fuse Add(MatMul(A, B), constant bias) -- or the lowered "
           "XCOMPILERFusedEltwise[ADD](MatMul, const) form -- into "
           "onnx.XFEMatMulBias when the bias is an effectively 1-D constant "
           "with one element per output channel (last dim of B) and is "
           "quantized at the same granularity as the weight (per-tensor bias "
           "with per-tensor weight, or matching per-channel bias with "
           "per-channel weight). A per-tensor bias on a per-channel-weight "
           "matmul, and a 2-D broadcast bias, are left as a separate Add, "
           "matching the xcompiler ReplaceQDQMatmulPass fusion gates.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseMatMulAddToXFEMatMulBiasPattern,
        FuseMatMulFusedEltwiseToXFEMatMulBiasPattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createFuseMatMulAddToXFEMatMulBiasPass() {
  return std::make_unique<FuseMatMulAddToXFEMatMulBiasPass>();
}

} // namespace onnx_mlir
