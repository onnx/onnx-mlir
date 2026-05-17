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

/// After flattening a bias constant to shape [N], per-axis quantization must
/// refer to axis 0. Per-axis with a single scale collapses to per-tensor
/// uniform quantization (same convention as other xmc passes).
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

/// Per-axis bias must broadcast one scale or one scale per output channel.
static bool isBiasQuantParamsCompatible(Type biasElemType, int64_t nOut) {
  auto perAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(biasElemType);
  if (!perAxis)
    return true;
  const int64_t nScales = static_cast<int64_t>(perAxis.getScales().size());
  return nScales == 1 || nScales == nOut;
}

/// Bias must be a constant with total element count == N (last dim of B), and
/// per-axis quant scale count must be 1 or N when applicable.
static bool isBiasCompatibleWithMatMul(Value biasVal, Value bVal) {
  auto bType = mlir::dyn_cast<RankedTensorType>(bVal.getType());
  if (!bType || bType.getRank() < 1)
    return false;
  int64_t nOut = bType.getShape().back();
  if (nOut == ShapedType::kDynamic)
    return false;

  auto constOp =
      mlir::dyn_cast_or_null<ONNXConstantOp>(biasVal.getDefiningOp());
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

  return isBiasQuantParamsCompatible(biasTy.getElementType(), nOut);
}

/// Reshape constant bias to rank-1 [N] and fix quantized element type for the
/// flattened tensor (per-axis on axis 0, or collapse single-scale per-axis to
/// per-tensor). Returns the original bias value if no rewrite is needed.
static Value create1DBiasFromConstant(
    PatternRewriter &rewriter, Value biasVal, Location loc, int64_t n) {
  auto constOp =
      mlir::dyn_cast_or_null<ONNXConstantOp>(biasVal.getDefiningOp());
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

struct FuseMatMulAddToXFEMatMulBiasPattern
    : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    Value lhs = addOp.getA();
    Value rhs = addOp.getB();

    ONNXMatMulOp matmulOp = nullptr;
    Value biasConstant = nullptr;

    if (auto mm = lhs.getDefiningOp<ONNXMatMulOp>()) {
      if (rhs.getDefiningOp<ONNXConstantOp>()) {
        matmulOp = mm;
        biasConstant = rhs;
      }
    }
    if (!matmulOp) {
      if (auto mm = rhs.getDefiningOp<ONNXMatMulOp>()) {
        if (lhs.getDefiningOp<ONNXConstantOp>()) {
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

    Location loc = addOp.getLoc();
    Value bias1D = create1DBiasFromConstant(rewriter, biasConstant, loc, nOut);

    auto fused = rewriter.create<XFEMatMulBiasOp>(loc,
        addOp.getResult().getType(), matmulOp.getA(), matmulOp.getB(), bias1D);

    rewriter.replaceOp(addOp, fused.getY());
    rewriter.eraseOp(matmulOp);
    return success();
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
    return "Fuse Add(MatMul(A, B), constant bias) into onnx.XFEMatMulBias when "
           "the constant has one element per output channel (size of last dim "
           "of B). Supports float and quantized tensors (per-tensor and "
           "per-axis bias); constant data may be dense or disposable elements.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseMatMulAddToXFEMatMulBiasPattern>(context);

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
