/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForZHigh.cpp - Rewrite ONNX ops for ZHigh lowering ----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements pass for rewriting of ONNX operations to generate
// combination of ONNX and ZHigh operations.
//
// - `ONNXBatchNormalizationInferenceModeOp`
// In this pass, `ONNXBatchNormalizationInferenceModeOp` is converted into
// `ZHigh.BatchNorm`, generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
// and `ONNX.Sqrt` to calculate inputs(`a` and `b`) for `ZHigh.BatchNorm`.
// `ONNXToZHighLoweringPass`(`--convert-onnx-to-zhigh`) is also able to generate
// the ONNX ops, but,they are lowered to ZHigh ops. So, constant
// propagation(`--constprop-onnx`) doesn't work. To enable to work it, this
// separate pass is needed. By using this pass, constant propagation works by
// running it just after this pass.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Calculate sqrt(var + epsilon) for batchnorm op A.
/// A = scale / sqrt(var + epsilon)
Value getSqrtResultBatchNormA(
    Location loc, PatternRewriter &rewriter, Value var, FloatAttr epsilon) {
  Type elementType = var.getType().cast<ShapedType>().getElementType();

  // epsilon
  RankedTensorType epsilonType = RankedTensorType::get({1}, elementType);
  DenseElementsAttr epsilonConstAttr =
      DenseElementsAttr::get<float>(epsilonType, epsilon.getValueAsDouble());
  Value epsilonConst = rewriter.create<ONNXConstantOp>(loc, epsilonType,
      nullptr, epsilonConstAttr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);

  // sqrt(var + epsilon)
  Value var_plus_epsilon = rewriter.create<ONNXAddOp>(loc, var, epsilonConst);
  Value sqrtResult =
      rewriter.create<ONNXSqrtOp>(loc, var.getType(), var_plus_epsilon);

  return sqrtResult;
}

// Reshape: B1xB2x...xBkxMxN to BxMxN
Value reshapeTo3D(PatternRewriter &rewriter, Location loc, Value val) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  return create.onnx.reshapeToNDim(val, 3, /*collapseMostSignificant*/ true);
}

// Reshape: B1xB2x...xBkxM to BxM
Value reshapeTo2DKeepLast(PatternRewriter &rewriter, Location loc, Value val) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  return create.onnx.reshapeToNDim(val, 2, /*collapseMostSignificant*/ true);
}

// Get a value that store the shape of the matmul result.
Value getMatMulResultShape(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  int64_t lhsRank = getRank(lhs.getType());
  int64_t rhsRank = getRank(rhs.getType());
  assert((lhsRank >= 2 && rhsRank >= 2) && "Input rank must be >= 2");
  // lhs shape: B1xB2x...xBkxMxN or MxN
  // rhs shape: B1xB2x...xBkxNxP or NxP

  int64_t rank = std::max(lhsRank, rhsRank);
  Type rI64Type = RankedTensorType::get({rank}, rewriter.getI64Type());
  Type lhsRType = RankedTensorType::get({lhsRank}, rewriter.getI64Type());
  Type lhsR1Type = RankedTensorType::get({lhsRank - 1}, rewriter.getI64Type());
  Type rhsRType = RankedTensorType::get({rhsRank}, rewriter.getI64Type());
  Type rhsR2Type = RankedTensorType::get({rhsRank - 2}, rewriter.getI64Type());
  Type oneI64Type = RankedTensorType::get({1}, rewriter.getI64Type());

  Value lhsShape = create.onnx.shape(lhsRType, lhs);
  Value rhsShape = create.onnx.shape(rhsRType, rhs);

  Value zero = create.onnx.constantInt64({0});
  Value one = create.onnx.constantInt64({1});
  Value lhsR1Const = create.onnx.constantInt64({lhsRank - 1});
  Value rhsRConst = create.onnx.constantInt64({rhsRank});
  Value rhsR1Const = create.onnx.constantInt64({rhsRank - 1});

  // if lhsRank >= rhsRank:
  //   - get B1xB2x...xBkxM from lhs shape, then append P from rhs shape.
  // else
  //   - get B1xB2x...xBk from rhs shape, then append M from lhs and append P
  //   from rhs shape.
  Value shapeVal;
  if (lhsRank >= rhsRank) {
    Value bmVal =
        create.onnx.slice(lhsR1Type, lhsShape, zero, lhsR1Const, zero, one);
    Value pVal = create.onnx.slice(
        oneI64Type, rhsShape, rhsR1Const, rhsRConst, zero, one);
    shapeVal = create.onnx.concat(rI64Type, ValueRange({bmVal, pVal}), 0);
  } else {
    Value lhsR2Const = create.onnx.constantInt64({lhsRank - 2});
    Value rhsR2Const = create.onnx.constantInt64({rhsRank - 2});
    Value bVal =
        create.onnx.slice(rhsR2Type, rhsShape, zero, rhsR2Const, zero, one);
    Value mVal = create.onnx.slice(
        oneI64Type, lhsShape, lhsR2Const, lhsR1Const, zero, one);
    Value pVal = create.onnx.slice(
        oneI64Type, rhsShape, rhsR1Const, rhsRConst, zero, one);
    shapeVal = create.onnx.concat(rI64Type, ValueRange({bVal, mVal, pVal}), 0);
  }
  return shapeVal;
}

// Get result type of matmul.
Type getMatMulResultType(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  Type elementType = getElementType(lhs.getType());
  int64_t lhsRank = getRank(lhs.getType());
  int64_t rhsRank = getRank(rhs.getType());
  assert((lhsRank >= 2 && rhsRank >= 2) && "Input rank must be >= 2");
  // lhs shape: B1xB2x...xBkxMxN or MxN
  // rhs shape: B1xB2x...xBkxNxP or NxP

  int64_t rank = std::max(lhsRank, rhsRank);
  ArrayRef<int64_t> lhsShape = getShape(lhs.getType());
  ArrayRef<int64_t> rhsShape = getShape(rhs.getType());

  // if lhsRank >= rhsRank:
  //   - get B1xB2x...xBkxM from lhs shape, then append P from rhs shape.
  // else
  //   - get B1xB2x...xBk from rhs shape, then append M from lhs and append P
  //   from rhs shape.
  if (lhsRank >= rhsRank) {
    SmallVector<int64_t, 4> resultShape(lhsShape.begin(), lhsShape.end());
    resultShape[rank - 1] = rhsShape[rhsRank - 1];
    return RankedTensorType::get(resultShape, elementType);
  }

  SmallVector<int64_t, 4> resultShape(rhsShape.begin(), rhsShape.end());
  resultShape[rank - 2] = lhsShape[lhsRank - 2];
  resultShape[rank - 1] = rhsShape[rhsRank - 1];
  return RankedTensorType::get(resultShape, elementType);
}

/// Check if A is unidirectionally broadcastable to B, e.g.
/// A: [256], B: [128x256]
/// A: [1], B: [128x256]
/// More info about unidirectional broadcasting:
/// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
/// Note: being differenct from ONNX broadcasting, we return false if A and B
/// have exactly the same static shape.
bool isUniBroadcatableFirstToSecond(Value A, Value B) {
  if (!hasStaticShape(A.getType()) || !hasStaticShape(B.getType()))
    return false;
  ArrayRef<int64_t> aDims = getShape(A.getType());
  ArrayRef<int64_t> bDims = getShape(B.getType());
  // A and B have exactly the same static shape.
  if (aDims == bDims)
    return false;
  // aDims size > bDims size: not unidirectional broadcasting from A to B, but B
  // to A.
  if (aDims.size() > bDims.size())
    return false;
  // Pre-pad A's shape with dims 1 so that two shapes have the same size.
  SmallVector<int64_t> paddedADims(bDims.size(), 1);
  for (unsigned i = 0; i < aDims.size(); ++i)
    paddedADims[i + bDims.size() - aDims.size()] = aDims[i];
  // Check unidirectional broadcasting.
  return llvm::all_of(llvm::zip(paddedADims, bDims), [](auto v) {
    return ((std::get<0>(v) == 1 && std::get<1>(v) != 1) ||
            (std::get<0>(v) == std::get<1>(v)));
  });
}

/// Check a value is defined by ONNXConstantOp or not.
bool isDefinedByONNXConstantOp(Value v) {
  if (v.isa<BlockArgument>())
    return false;
  return isa<ONNXConstantOp>(v.getDefiningOp());
}

bool CanExpandPowOpToMul(ONNXPowOp op) {
  Value exponent = op.Y();
  if (!isDefinedByONNXConstantOp(exponent))
    return false;

  auto constOp = dyn_cast<ONNXConstantOp>(exponent.getDefiningOp());
  if (DenseElementsAttr dataAttr =
          constOp.valueAttr().dyn_cast<DenseElementsAttr>()) {
    if (dataAttr.getNumElements() == 1) {
      Type elementType = dataAttr.getElementType();
      if (elementType.isa<FloatType>()) {
        auto valueIt = dataAttr.getValues<APFloat>().begin();
        double val = (*valueIt).convertToDouble();
        if (ceil(val) == val && val >= 0 && val <= 64)
          return true;
      }
      if (elementType.isa<IntegerType>()) {
        auto valueIt = dataAttr.getValues<APInt>().begin();
        int64_t val = (*valueIt).getSExtValue();
        if (val >= 0 && val <= 64)
          return true;
      }
    }
  }
  return false;
}

//
// Check if pads can be inferenced for ONNXConv op
//
bool canInferencePadsForNNPAConv(ONNXConvOp op) {
  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(op);
  ONNXConvOpShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)));
  return (shapeHelper.pads.size() == 4) &&
         (llvm::all_of(
             shapeHelper.pads, [](IndexExpr val) { return val.isLiteral(); }));
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
ArrayAttr getPadsForNNPAConv(PatternRewriter &rewriter, Value ret) {
  ONNXConvOp op = dyn_cast<ONNXConvOp>(ret.getDefiningOp());
  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(op);
  ONNXConvOpShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)));
  SmallVector<int64_t, 4> vals;
  IndexExpr::getShape(shapeHelper.pads, vals);
  return rewriter.getI64ArrayAttr(vals);
}

// Pad a ArrayAttr with zeros.
//
// pads = [B1, B2, ... Bk, E1, E2, ..., Ek]
//
// becomes:
//
// pads = [0,... 0, B1, B2, ... Bk, 0,... 0, E1, E2, ..., Ek]
//         |_____|                  |_____|
//                 nZeros                    nZeros
//
// This function is used for padding attribute in Conv.
DenseElementsAttr insertZerosForNonPaddedDims(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int extensionLength) {
  int nDims = (int)origAttrs.getValue().size() / 2;
  int nElements = (nDims + extensionLength) * 2;
  SmallVector<int64_t, 4> pads(nElements, 0);
  for (int i = 0; i < nDims; ++i) {
    int64_t beginPad = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    int64_t endPad =
        origAttrs.getValue()[nDims + i].cast<IntegerAttr>().getInt();
    pads[i + extensionLength] = beginPad;
    pads[nDims + extensionLength + i + extensionLength] = endPad;
  }
  return rewriter.getI64TensorAttr(llvm::makeArrayRef(pads));
}

DenseElementsAttr createDenseFloatAttrOfValue(
    PatternRewriter &rewriter, Value origValue, float constantValue) {
  Type elementType = origValue.getType().cast<TensorType>().getElementType();
  SmallVector<float, 1> wrapper(1, 0);
  wrapper[0] = constantValue;
  return DenseElementsAttr::get(
      RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
ArrayAttr createArrayAttrOfZeros(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  int nElements = origAttrs.getValue().size();
  SmallVector<int64_t, 4> vals(nElements, 0);
  return rewriter.getI64ArrayAttr(vals);
}

// Create Type for Padded input
Type CreatePaddedXType(Value x, ArrayAttr pads) {
  RankedTensorType inputType = x.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  Type elementType = inputType.getElementType();
  SmallVector<int64_t, 4> paddingShape(4, 0);
  if (pads) {
    for (int i = 0; i < 4; i++) {
      paddingShape[i] = pads.getValue()[i].cast<IntegerAttr>().getInt();
    }
  }
  SmallVector<int64_t, 4> paddedShape = {inputShape[0], inputShape[1],
      inputShape[2] + paddingShape[0] + paddingShape[2],
      inputShape[3] + paddingShape[1] + paddingShape[3]};
  Type paddedType = RankedTensorType::get(paddedShape, elementType);
  return paddedType;
}

//===----------------------------------------------------------------------===//
// Rewrite ONNX ops to ZHigh ops and ONNX ops for ZHigh.
//===----------------------------------------------------------------------===//

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXRewriteONNXForZHigh.inc"

struct ExpandPowToMulPattern : public ConversionPattern {
  ExpandPowToMulPattern(MLIRContext *context)
      : ConversionPattern(ONNXPowOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto powOp = llvm::dyn_cast<ONNXPowOp>(op);
    Location loc = powOp.getLoc();
    // Illegal conditions must be satisfied at this point.
    assert(CanExpandPowOpToMul(powOp) && "Illegal conditions failed");

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value input = powOp.X();
    int64_t exponent;

    // Get the scalar integer exponent.
    // powOp.Y() is exponent that must be a scalar integer tensor by the Match
    // phase.
    auto constOp = dyn_cast<ONNXConstantOp>(powOp.Y().getDefiningOp());
    auto dataAttr = constOp.valueAttr().dyn_cast<DenseElementsAttr>();
    Type elementType = dataAttr.getElementType();
    if (elementType.isa<FloatType>()) {
      auto valueIt = dataAttr.getValues<APFloat>().begin();
      exponent = (*valueIt).convertToDouble();
      assert((ceil(exponent) == exponent && exponent <= 64) &&
             "Exponent must be an integer and <= 64");
    } else if (elementType.isa<IntegerType>()) {
      auto valueIt = dataAttr.getValues<APInt>().begin();
      exponent = (*valueIt).getSExtValue();
      assert(exponent <= 64 && "Exponent must be an integer and <= 64");
    } else
      return failure();

    Value result;
    Type resultType = powOp.Z().getType();
    if (exponent == 0) {
      DenseElementsAttr valAttr;
      if (elementType.isa<FloatType>())
        valAttr = DenseElementsAttr::get(resultType, ArrayRef<float>({1.0}));
      else if (elementType.isa<IntegerType>())
        valAttr = DenseElementsAttr::get(resultType, ArrayRef<int64_t>({1}));
      else
        llvm_unreachable("Unsupported type");
      result = create.onnx.constant(valAttr);
    } else {
      // calculate pow(input,exponent) with "exponentiation by squaring" method
      bool result_initialized = false;
      while (exponent > 0) {
        if (exponent & 1) {
          result = result_initialized
                       ? create.onnx.mul(resultType, result, input)
                       : input;
          result_initialized = true;
        }
        input = create.onnx.mul(resultType, input, input);
        exponent >>= 1;
      }
    }

    rewriter.replaceOp(op, {result});
    return success();
  };
};

struct RewriteONNXForZHighPass
    : public PassWrapper<RewriteONNXForZHighPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "rewrite-onnx-for-zhigh"; }

  StringRef getDescription() const override {
    return "Rewrite ONNX ops for ZHigh.";
  }

  RewriteONNXForZHighPass() = default;
  RewriteONNXForZHighPass(mlir::ArrayRef<std::string> execNodesOnCpu)
      : execNodesOnCpu(execNodesOnCpu) {}
  void runOnOperation() final;

public:
  mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>();
};

void RewriteONNXForZHighPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, func::FuncDialect>();

  // `ONNXBatchNormalizationInferenceModeOp` to `ZHigh.BatchNorm`,
  // generating `ONNX.Add`, `ONNX.Sub`, `ONNX.Mul`, `ONNX.Div`,
  // and `ONNX.Sqrt` to calculate inputs(`a` and `b`)
  addDynamicallyLegalOpFor<ONNXBatchNormalizationInferenceModeOp>(
      &target, execNodesOnCpu);

  // Illegalize BinaryOp if one of the two inputs is a constant and
  // unidirectional broadcastable to the other input. Rewrite patterns will be
  // added to turn a broadcasting op into a non-broadcasting op.
  //
  // This is preferred for NNPA because NNPA BinaryOp does not support
  // broadcasting.
  target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXDivOp>([](ONNXDivOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });
  target.addDynamicallyLegalOp<ONNXSubOp>([](ONNXSubOp op) {
    return !((isDefinedByONNXConstantOp(op.A()) &&
                 isUniBroadcatableFirstToSecond(op.A(), op.B())) ||
             (isDefinedByONNXConstantOp(op.B()) &&
                 isUniBroadcatableFirstToSecond(op.B(), op.A())));
  });

  // Illegalize MatMulOp if
  // - both inputs are *the same* N-D, N > 3, or
  // - one input is N-D, N > 3 and the other is 2-D.
  // Rewrite patterns will be added to turn this MatMulOp into the one where N-D
  // will become 3-D.
  target.addDynamicallyLegalOp<ONNXMatMulOp>([](ONNXMatMulOp op) {
    Type aType = op.A().getType();
    Type bType = op.B().getType();
    if (!isRankedShapedType(aType) || !isRankedShapedType(bType))
      return true;

    int64_t aRank = getRank(aType);
    int64_t bRank = getRank(bType);
    if (aRank == 2 && bRank > 3)
      return false;
    if (bRank == 2 && aRank > 3)
      return false;
    if (aRank > 3 && (aRank == bRank))
      return false;

    return true;
  });

  // Illegalize PowOp if
  // - exponent is a scalar integer, and
  // - exponent is <= 64.
  // This PowOp will be rewritten by using multiple MulOp.
  target.addDynamicallyLegalOp<ONNXPowOp>(
      [](ONNXPowOp op) { return !CanExpandPowOpToMul(op); });

  // Illegalize SoftmaxOp if
  // - axis is the last dimension.
  // This SoftmaxOp will be rewritten in which its input is reshaped to 2D.
  target.addDynamicallyLegalOp<ONNXSoftmaxOp>([](ONNXSoftmaxOp op) {
    Value input = op.input();
    if (auto shapedType = input.getType().dyn_cast<RankedTensorType>()) {
      if ((shapedType.getRank() > 2) &&
          ((op.axis() == shapedType.getRank() - 1) || (op.axis() == -1))) {
        return false;
      }
    }
    return true;
  });

  target.addDynamicallyLegalOp<ONNXConvOp>([](ONNXConvOp op) {
    return isSuitableForZDNN<ONNXConvOp>(op) ||
           !canInferencePadsForNNPAConv(op);
  });

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.insert<ExpandPowToMulPattern>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass() {
  return std::make_unique<RewriteONNXForZHighPass>();
}

std::unique_ptr<Pass> createRewriteONNXForZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  return std::make_unique<RewriteONNXForZHighPass>(execNodesOnCpu);
}

} // namespace onnx_mlir
