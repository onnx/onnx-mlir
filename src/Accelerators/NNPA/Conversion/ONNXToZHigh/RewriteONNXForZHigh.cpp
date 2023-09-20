/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- RewriteONNXForZHigh.cpp - Rewrite ONNX ops for ZHigh lowering ----===//
//
// Copyright 2019-2023 The IBM Research Authors.
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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Calculate sqrt(var + epsilon) for batchnorm op A.
/// A = scale / sqrt(var + epsilon)
Value getSqrtResultBatchNormA(
    Location loc, PatternRewriter &rewriter, Value var, FloatAttr epsilon) {
  Type elementType = var.getType().cast<ShapedType>().getElementType();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  // epsilon
  RankedTensorType epsilonType = RankedTensorType::get({1}, elementType);
  DenseElementsAttr epsilonConstAttr =
      DenseElementsAttr::get<float>(epsilonType, epsilon.getValueAsDouble());
  Value epsilonConst = create.onnx.constant(epsilonConstAttr);

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
/// Note: being different from ONNX broadcasting, we return false if A and B
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
  return isa_and_present<ONNXConstantOp>(v.getDefiningOp());
}

//
// Check if pads can be inferenced for ONNXConv op.
//
bool canInferencePadsForNNPAConv(ONNXConvOp op) {
  ONNXConvOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  RankedTensorType inputType = op.getX().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // dimension of inferenced pads should be 4D
  if (shapeHelper.pads.size() != 4)
    return false;
  // all dimensions of pad should be literal
  if (llvm::any_of(
          shapeHelper.pads, [](IndexExpr val) { return !val.isLiteral(); }))
    return false;
  // auto_pad should not be "VALID"
  if (op.getAutoPad().equals_insensitive("VALID"))
    return false;
  // image dimensions of input shape should be static
  if ((inputShape[2] == ShapedType::kDynamic) ||
      (inputShape[3] == ShapedType::kDynamic))
    return false;
  return true;
}

// Create an ArrayAttr of IntegerAttr(s) of zero values.
// This function is used for padding attribute in Conv.
ArrayAttr getPadsForNNPAConv(PatternRewriter &rewriter, Value ret) {
  ONNXConvOp op = dyn_cast<ONNXConvOp>(ret.getDefiningOp());
  ONNXConvOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
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
  return rewriter.getI64TensorAttr(llvm::ArrayRef(pads));
}

DenseElementsAttr createDenseFloatAttrOfValue(
    PatternRewriter &rewriter, Value origValue, float constantValue) {
  Type elementType = origValue.getType().cast<TensorType>().getElementType();
  SmallVector<float, 1> wrapper(1, 0);
  wrapper[0] = constantValue;
  return DenseElementsAttr::get(
      RankedTensorType::get({}, elementType), llvm::ArrayRef(wrapper));
}

// Create an ArrayAttr of IntegerAttr(s) of zero values.
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

/// This pattern is to split a large MatMul into smaller ones that fit into
/// NNPA. Given (NxK) * (K*M), the pattern considers dimensions N and/or M to
/// split, if N and/or M is greater than NNPA_MAXIMUM_DIMENSION_INDEX_SIZE
/// (MDIS).
/// For example, given A(NxK) * B(KxM), we will split A and B as follows.
// clang-format off
///
///                   K                            MDIS        MDIS    M-2*MDIS
///           <----------------------->        <----------><----------><----->
///           +------------------------+       +-----------+-----------+-----+
///         ^ |                        |     ^ |           |           |     |
///    MDIS | |                        |     | |           |           |     |
///         | |       A1               |     | |           |           |     |
///         v |                        |   K | |  B1       |  B2       |  B3 |
///           +------------------------+     | |           |           |     |
///         ^ |       A2               |     | |           |           |     |
///  N-MDIS | |                        |     v |           |           |     |
///         v +------------------------+       +-----------+-----------+-----+
///                                                         
/// Then,
/// - for A1, do (A1 * B1), (A1 * B2), (A1 * B3), and concat the results to get (A1*B)
/// - for A2, do (A2 * B1), (A2 * B2), (A2 * B3), and concat the results to get (A2*B)
/// - finally, concat (A1*B) and (A2*B) to get (A*B)
///
///
// clang-format on
//
/// Tensors are splitted into chunks of the equal size of MDIS, except the last
/// chunk.
class SplitLargeMatMulPattern : public OpRewritePattern<ONNXMatMulOp> {
public:
  using OpRewritePattern<ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMatMulOp matmulOp, PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    Operation *op = matmulOp.getOperation();
    Value A = matmulOp.getA(); // NxK
    Value B = matmulOp.getB(); // KxM

    Type aType = A.getType();
    Type bType = B.getType();
    Type outputType = matmulOp.getY().getType();
    int64_t aRank = getRank(aType);
    int64_t bRank = getRank(bType);
    int64_t outputRank = getRank(outputType);
    ArrayRef<int64_t> aShape = getShape(aType);
    ArrayRef<int64_t> bShape = getShape(bType);
    ArrayRef<int64_t> outputShape = getShape(outputType);
    Type elementType = getElementType(bType);
    auto unrankedType = UnrankedTensorType::get(elementType);

    // Expect 2D or 3D input.
    if (!((aRank == 2 || aRank == 3) && (bRank == 2 || bRank == 3)))
      return failure();

    // Expect N or M exceeds NNPA limitation.
    int64_t N = aShape[aRank - 2];
    int64_t M = bShape[bRank - 1];
    bool nExceeded = N > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE;
    bool mExceeded = M > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE;
    if (!(nExceeded || mExceeded))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    ValueRange subAs(A), subBs(B);
    if (nExceeded) {
      // Split A along the dimension N.
      subAs = splitAlongAxis(create, A, aRank - 2);
    }
    if (mExceeded) {
      // Split B along the dimension M.
      subBs = splitAlongAxis(create, B, bRank - 1);
    }
    // Emit sub matrix multiplication.
    SmallVector<Value> resSubAs;
    for (Value a : subAs) {
      ArrayRef<int64_t> subAShape = getShape(a.getType());
      // For each matrix along dimension N, do MatMul for sub matrices along
      // dimension M.
      SmallVector<Value> subMatrices;
      for (Value b : subBs) {
        Value sm = create.onnx.matmul(unrankedType, a, b, false);
        subMatrices.emplace_back(sm);
      }
      Value res = subMatrices[0];
      if (subMatrices.size() > 1) {
        // Concat sub results along dimension M of B.
        SmallVector<int64_t> concatShape(outputShape);
        concatShape[outputRank - 2] = subAShape[aRank - 2];
        Type concatTy = RankedTensorType::get(concatShape, elementType);
        res = create.onnx.concat(concatTy, subMatrices, outputRank - 1);
      }
      resSubAs.emplace_back(res);
    }
    Value res = resSubAs[0];
    if (resSubAs.size() > 1)
      // Concat sub results along dimension N of A.
      res = create.onnx.concat(outputType, resSubAs, outputRank - 2);

    rewriter.replaceOp(op, res);
    return success();
  }
};

/// This pattern is to replace `C = add/sub(A, B)` by `A` when B is a zero
/// defined by Expand of scalar constant and C's shape is the same as A's shape.
/// In other words, the output does not depend on the second operand.
/// This pattern is similar to Add/SubZerosOnRhs in ConstProp.td but allows
/// dynamic shape.
template <typename OP_TYPE>
class AddSubWithRHSZeroExpandPattern : public OpRewritePattern<OP_TYPE> {
public:
  DimAnalysis *dimAnalysis;

  AddSubWithRHSZeroExpandPattern(MLIRContext *context, DimAnalysis *dimAnalysis)
      : OpRewritePattern<OP_TYPE>(context, 1001), dimAnalysis(dimAnalysis) {}

  LogicalResult matchAndRewrite(
      OP_TYPE binaryOp, PatternRewriter &rewriter) const override {
    // Match
    if (!canBeRewritten(binaryOp, dimAnalysis))
      return failure();
    // Rewrite
    rewriter.replaceOp(binaryOp.getOperation(), {binaryOp.getA()});
    return success();
  }

  static bool canBeRewritten(OP_TYPE binaryOp, DimAnalysis *dimAnalysis) {
    Value A = binaryOp.getA();
    Value B = binaryOp.getB();
    Value C = binaryOp.getC();

    // Match
    // C's shape is the same as A'shape.
    if (!dimAnalysis->sameShape(A, C))
      return false;
    // B is a zero defined by Expand.
    if (isa<BlockArgument>(B))
      return false;
    bool BIsZero = false;
    if (auto expandOp = dyn_cast<ONNXExpandOp>(B.getDefiningOp())) {
      Value input = expandOp.getInput();
      if (isDenseONNXConstant(input)) {
        // Expand's input is 0?
        ElementsAttr constElements = getElementAttributeFromONNXValue(input);
        Type elemType = constElements.getElementType();
        if (!elemType.isInteger(1)) { // Booleans are not supported.
          WideNum zeroWN = wideZeroDispatch(elemType, [](auto wideZero) {
            using cpptype = decltype(wideZero);
            constexpr BType TAG = toBType<cpptype>;
            return WideNum::widen<TAG>(static_cast<cpptype>(0.0));
          });
          BIsZero = ElementsAttrBuilder::allEqual(constElements, zeroWN);
        }
      }
    }
    return BIsZero;
  }
};

//===----------------------------------------------------------------------===//
// Rewrite ONNX ops to ZHigh ops and ONNX ops for ZHigh.
//===----------------------------------------------------------------------===//

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXRewriteONNXForZHigh.inc"

struct RewriteONNXForZHighPass
    : public PassWrapper<RewriteONNXForZHighPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "rewrite-onnx-for-zhigh"; }

  StringRef getDescription() const override {
    return "Rewrite ONNX ops for ZHigh.";
  }

  RewriteONNXForZHighPass() = default;
  RewriteONNXForZHighPass(
      mlir::ArrayRef<std::string> execNodesOnCpu, bool useCostModel)
      : execNodesOnCpu(execNodesOnCpu), useCostModel(useCostModel) {}
  void runOnOperation() final;

public:
  mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>();
  bool useCostModel = false;
};

void RewriteONNXForZHighPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

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
      &target, &dimAnalysis, useCostModel, execNodesOnCpu);

  // Illegalize BinaryOp if one of the two inputs is a constant and
  // unidirectional broadcastable to the other input. Rewrite patterns will be
  // added to turn a broadcasting op into a non-broadcasting op.
  //
  // This is preferred for NNPA because NNPA BinaryOp does not support
  // broadcasting.
  target.addDynamicallyLegalOp<ONNXAddOp>([&dimAnalysis](ONNXAddOp op) {
    return !((isDefinedByONNXConstantOp(op.getA()) &&
                 isUniBroadcatableFirstToSecond(op.getA(), op.getB())) ||
             (isDefinedByONNXConstantOp(op.getB()) &&
                 isUniBroadcatableFirstToSecond(op.getB(), op.getA())) ||
             AddSubWithRHSZeroExpandPattern<ONNXAddOp>::canBeRewritten(
                 op, &dimAnalysis));
  });
  target.addDynamicallyLegalOp<ONNXDivOp>([](ONNXDivOp op) {
    return !((isDefinedByONNXConstantOp(op.getA()) &&
                 isUniBroadcatableFirstToSecond(op.getA(), op.getB())) ||
             (isDefinedByONNXConstantOp(op.getB()) &&
                 isUniBroadcatableFirstToSecond(op.getB(), op.getA())));
  });
  target.addDynamicallyLegalOp<ONNXMulOp>([](ONNXMulOp op) {
    return !((isDefinedByONNXConstantOp(op.getA()) &&
                 isUniBroadcatableFirstToSecond(op.getA(), op.getB())) ||
             (isDefinedByONNXConstantOp(op.getB()) &&
                 isUniBroadcatableFirstToSecond(op.getB(), op.getA())));
  });
  target.addDynamicallyLegalOp<ONNXSubOp>([&dimAnalysis](ONNXSubOp op) {
    return !((isDefinedByONNXConstantOp(op.getA()) &&
                 isUniBroadcatableFirstToSecond(op.getA(), op.getB())) ||
             (isDefinedByONNXConstantOp(op.getB()) &&
                 isUniBroadcatableFirstToSecond(op.getB(), op.getA())) ||
             AddSubWithRHSZeroExpandPattern<ONNXSubOp>::canBeRewritten(
                 op, &dimAnalysis));
  });

  // Determine if MatMulOp is already legal (no need to rewrite) or need to
  // rewrite. The following cases must be rewritten:
  // - both inputs are *the same* N-D (N > 3) and there is no broadcasting, or
  // - one input is N-D (N > 3) and the other is 2-D, or
  // - no input is N-D (N > 3) but dimension size exceeds NNPA limitation.
  //
  // For such cases, rewrite patterns will be added to turn MatMulOp into the
  // one where N-D will become 3-D or to split MatMul into smaller MatMuls.
  target.addDynamicallyLegalOp<ONNXMatMulOp>([&dimAnalysis](ONNXMatMulOp op) {
    Type aType = op.getA().getType();
    Type bType = op.getB().getType();
    if (!isRankedShapedType(aType) || !isRankedShapedType(bType))
      return true;

    int64_t aRank = getRank(aType);
    int64_t bRank = getRank(bType);
    ArrayRef<int64_t> aShape = getShape(aType);
    ArrayRef<int64_t> bShape = getShape(bType);

    // - one input is N-D (N > 3) and the other is 2-D.
    if (aRank == 2 && bRank > 3)
      return false;
    if (bRank == 2 && aRank > 3)
      return false;
    // No input is N-D (N > 3) but dimension N or M (NxK * KxM) is dynamic or
    // exceeds NNPA limitation.
    if ((aRank == 2 || aRank == 3) && (bRank == 2 || bRank == 3) &&
        ((aShape[aRank - 2] > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE) ||
            (bShape[bRank - 1] > NNPA_MAXIMUM_DIMENSION_INDEX_SIZE)))
      return false;

    // - both inputs are *the same* N-D, N > 3 and there is no broadcasting
    if (aRank > 3 && (aRank == bRank)) {
      bool sameBatchDims = true;
      for (int64_t i = 0; i < aRank - 2; ++i) {
        sameBatchDims &= (aShape[i] == bShape[i]);
        if (sameBatchDims && ShapedType::isDynamic(aShape[i]))
          sameBatchDims = dimAnalysis.sameDynDim(op.getA(), i, op.getB(), i);
      }
      return !sameBatchDims;
    }

    // Make other cases legal.
    return true;
  });

  // Illegalize SoftmaxOp if
  // - axis is the last dimension.
  // This SoftmaxOp will be rewritten in which its input is reshaped to 3D.
  target.addDynamicallyLegalOp<ONNXSoftmaxOp>([](ONNXSoftmaxOp op) {
    Value input = op.getInput();
    if (auto shapedType = input.getType().dyn_cast<RankedTensorType>()) {
      if ((shapedType.getRank() > 3) &&
          ((op.getAxis() == shapedType.getRank() - 1) ||
              (op.getAxis() == -1))) {
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
  patterns.insert<SplitLargeMatMulPattern>(&getContext());
  patterns.insert<AddSubWithRHSZeroExpandPattern<ONNXAddOp>>(
      &getContext(), &dimAnalysis);
  patterns.insert<AddSubWithRHSZeroExpandPattern<ONNXSubOp>>(
      &getContext(), &dimAnalysis);

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
    mlir::ArrayRef<std::string> execNodesOnCpu, bool useCostModel) {
  return std::make_unique<RewriteONNXForZHighPass>(
      execNodesOnCpu, useCostModel);
}

} // namespace onnx_mlir
