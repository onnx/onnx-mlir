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

/// Number of quantization scales along the quantized axis: 1 for a per-tensor
/// (uniform) quantized type, N for a per-axis (per-channel) type, and 0 for a
/// non-quantized (float) type.
static int64_t quantScaleCount(Type elemType) {
  if (auto perAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(elemType))
    return static_cast<int64_t>(perAxis.getScales().size());
  if (isa<quant::UniformQuantizedType>(elemType))
    return 1;
  return 0; // non-quantized (float)
}

/// The bias is foldable into the matmul only when it is quantized at the same
/// granularity as the weight along the output channel:
///   - per-tensor weight  (1 scale)  needs a per-tensor bias  (1 scale)
///   - per-channel weight (N scales) needs a matching per-channel bias
///     (N scales, one per output channel)
/// A per-channel-weight matmul with a per-tensor bias cannot be folded into the
/// INT32 accumulator (the requantization scale x_scale*w_scale[c] differs per
/// output channel), so it must be left as a separate Add. This reproduces the
/// xcompiler/xmodel non-fusion behavior, where exactly the matmuls whose bias
/// granularity matches the weight are fused. Float (non-quantized) tensors have
/// 0 scales on both sides and so remain foldable.
static bool isBiasGranularityCompatible(Type biasElemType, Type weightElemType) {
  return quantScaleCount(biasElemType) == quantScaleCount(weightElemType);
}

/// Returns the ONNXConstantOp feeding `v`, looking through a single
/// DequantizeLinear if present. Mirrors the xcompiler get_matmul_add_template
/// chain (const-fix -> dequantize-linear -> add), so a bias constant behind a
/// DequantizeLinear is still recognized.
static ONNXConstantOp getDefiningConstantThroughDequant(Value v) {
  if (auto c = v.getDefiningOp<ONNXConstantOp>())
    return c;
  if (auto dq = v.getDefiningOp<ONNXDequantizeLinearOp>())
    return dq.getX().getDefiningOp<ONNXConstantOp>();
  return nullptr;
}

/// Bias must be a constant whose total element count == N (last dim of B), be
/// effectively 1-D ([N], leading dims == 1), and be quantized at the same
/// granularity as the weight (per-tensor bias with per-tensor weight, or
/// matching per-channel bias with per-channel weight). Together these mirror
/// the xcompiler ReplaceQDQMatmulPass fusion gates: const.shape[0] == output
/// channel (1-D bias) plus a bias that can fold into the INT32 accumulator.
/// A 2-D [seq, N] broadcast bias, or a per-tensor bias on a per-channel-weight
/// matmul, is kept as a separate Add (non-fusion).
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

  // Require an effectively 1-D per-output-channel bias [N]: innermost dim == N
  // and every leading dim == 1. A genuine 2-D [seq, N] broadcast constant is
  // left as a separate Add (xcompiler keeps these unfused).
  ArrayRef<int64_t> biasShape = biasTy.getShape();
  if (biasShape.empty() || biasShape.back() != nOut)
    return false;
  for (int64_t d = 0, e = biasTy.getRank() - 1; d < e; ++d)
    if (biasShape[d] != 1)
      return false;

  return isBiasGranularityCompatible(
      biasTy.getElementType(), bType.getElementType());
}

/// Reshape constant bias to rank-1 [N] and fix quantized element type for the
/// flattened tensor (per-axis on axis 0, or collapse single-scale per-axis to
/// per-tensor). Returns the original bias value if no rewrite is needed.
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

/// Shared core: try to fuse an add-like op (`onnx.Add` or
/// `onnx.XCOMPILERFusedEltwise` with type=ADD) whose operands are
/// (MatMul, constant bias) into `onnx.XFEMatMulBias`.
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

  auto fused = rewriter.create<XFEMatMulBiasOp>(loc,
      addLikeOp->getResult(0).getType(), matmulOp.getA(), matmulOp.getB(),
      bias1D);

  rewriter.replaceOp(addLikeOp, fused.getY());
  rewriter.eraseOp(matmulOp);
  return success();
}

/// Float / element-type-quant form: fuse `Add(MatMul, const)`.
struct FuseMatMulAddToXFEMatMulBiasPattern
    : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    return tryFuseMatMulBias(addOp, addOp.getA(), addOp.getB(), rewriter);
  }
};

/// Quantized form after the Add -> XCOMPILERFusedEltwise lowering: fuse
/// `XCOMPILERFusedEltwise[ADD](MatMul, const)` with no fused activation.
struct FuseMatMulFusedEltwiseToXFEMatMulBiasPattern
    : public OpRewritePattern<XCOMPILERFusedEltwiseOp> {
  using OpRewritePattern<XCOMPILERFusedEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XCOMPILERFusedEltwiseOp eltwiseOp,
      PatternRewriter &rewriter) const override {
    // Only a plain quantized ADD (no fused activation) maps to a matmul bias.
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
