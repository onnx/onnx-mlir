/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToZHigh.cpp - ONNX dialect to ZHigh lowering -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and ZHigh operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

//
// LSTM/GRU specific functions
//

namespace onnx_mlir {

ArrayAttr getLSTMGRUBiasSplitShape(
    Location loc, PatternRewriter &rewriter, ArrayRef<int64_t> shapeR) {
  int64_t hiddenSize = shapeR[2];
  int splitNum = (shapeR[1] / shapeR[2]) * 2;
  std::vector<int64_t> splitShape;
  for (int i = 0; i < splitNum; i++) {
    splitShape.emplace_back(hiddenSize);
  }
  return rewriter.getI64ArrayAttr(splitShape);
}

Value getLSTMGRUZDNNWeightFromONNXWeight(
    Location loc, PatternRewriter &rewriter, Value weight, int isLSTM) {
  int64_t splitNum = isLSTM ? 4 : 3;
  RankedTensorType weightType = weight.getType().cast<RankedTensorType>();
  Type elementType = weightType.getElementType();
  ArrayRef<int64_t> weightShape = weightType.getShape();
  int64_t direction = weightShape[0];
  int64_t hiddenSize = weightShape[1] / splitNum;
  int64_t weightHiddenSize = weightShape[1];
  int64_t feature = weightShape[2];
  SmallVector<int64_t, 3> transposeShape;
  transposeShape.emplace_back(direction);
  transposeShape.emplace_back(feature);
  transposeShape.emplace_back(weightHiddenSize);
  RankedTensorType transposeType =
      RankedTensorType::get(transposeShape, elementType);
  SmallVector<int64_t, 3> perms({0, 2, 1});
  ArrayRef<int64_t> permArrayW(perms);
  ArrayAttr permAttrW = rewriter.getI64ArrayAttr(permArrayW);
  Value transposeOp =
      rewriter.create<ONNXTransposeOp>(loc, transposeType, weight, permAttrW);
  SmallVector<int64_t, 3> splitShape;
  splitShape.emplace_back(direction);
  splitShape.emplace_back(feature);
  splitShape.emplace_back(hiddenSize);
  Type splitType = RankedTensorType::get(splitShape, elementType);
  int64_t axis = 2;
  Value stickForOp;
  if (isLSTM) {
    SmallVector<Type, 4> splitTypes(splitNum, splitType);
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, transposeOp, axis, nullptr);
    Value i_gate = splitOp.getResults()[0];
    Value o_gate = splitOp.getResults()[1];
    Value f_gate = splitOp.getResults()[2];
    Value c_gate = splitOp.getResults()[3];
    stickForOp = rewriter.create<zhigh::ZHighStickForLSTMOp>(
        loc, f_gate, i_gate, c_gate, o_gate);
  } else { // GRU
    SmallVector<Type, 3> splitTypes(splitNum, splitType);
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, transposeOp, axis, nullptr);
    Value z_gate = splitOp.getResults()[0];
    Value r_gate = splitOp.getResults()[1];
    Value h_gate = splitOp.getResults()[2];
    stickForOp =
        rewriter.create<zhigh::ZHighStickForGRUOp>(loc, z_gate, r_gate, h_gate);
  }
  return stickForOp;
}

Value getLSTMGRUGetY(
    Location loc, PatternRewriter &rewriter, Value val, Value resY) {
  Value noneValue;
  if (isNoneValue(resY)) {
    return noneValue;
  }
  return val;
}

Value getLSTMGRUGetYh(Location loc, PatternRewriter &rewriter, Value val,
    Value resY, Value resYh, Value X, StringAttr direction) {
  Value noneValue;
  if (isNoneValue(resYh) || isNoneValue(val))
    return noneValue;

  ArrayRef<int64_t> shapeX = X.getType().cast<ShapedType>().getShape();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  // Generate Y_h for onnx.LSTM from hn_output for all timestep
  Value minusOne = create.onnx.constantInt64({-1});
  Value zero = create.onnx.constantInt64({0});
  Value one = create.onnx.constantInt64({1});
  // Use INT_MAX to get timestep dimension because timestep is the end of a
  // dimension. INT_MAX is recommended in ONNX.Slice to slice to the end of a
  // dimension with unknown size.
  Value intMax = create.onnx.constantInt64({INT_MAX});
  StringRef directionStr = direction.getValue();
  ArrayRef<int64_t> resYhShape =
      resYh.getType().cast<RankedTensorType>().getShape();
  int64_t T = isNoneValue(resY) ? 1 : shapeX[0];
  int64_t D = resYhShape[0];
  int64_t B = resYhShape[1];
  int64_t H = resYhShape[2];
  Type elementType = resYh.getType().cast<ShapedType>().getElementType();
  Value axis = zero;
  Value step = one;
  Value ret;
  if (directionStr.equals_insensitive("forward") ||
      directionStr.equals_insensitive("reverse")) {
    Value start = directionStr.equals_insensitive("forward") ? minusOne : zero;
    Value end = directionStr.equals_insensitive("forward") ? intMax : one;

    Type sliceType = RankedTensorType::get({1, D, B, H}, elementType);
    ONNXSliceOp sliceOp = rewriter.create<ONNXSliceOp>(
        loc, sliceType, val, start, end, axis, step);
    return rewriter.create<ONNXSqueezeV11Op>(
        loc, resYh.getType(), sliceOp.getResult(), rewriter.getI64ArrayAttr(0));
  } else if (directionStr.equals_insensitive("bidirectional")) {
    Type splitType = RankedTensorType::get({T, 1, B, H}, elementType);
    SmallVector<Type> splitTypes = {splitType, splitType};
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, val, /*splitAxis=*/1, nullptr);
    Type sliceType = RankedTensorType::get({1, 1, B, H}, elementType);
    Value fwdLastSlice = rewriter.create<ONNXSliceOp>(
        loc, sliceType, splitOp.getResults()[0], minusOne, intMax, axis, step);
    Value bkwFirstSlice = rewriter.create<ONNXSliceOp>(
        loc, sliceType, splitOp.getResults()[1], zero, one, axis, step);
    Type concatType = RankedTensorType::get({1, D, B, H}, elementType);
    Value concatOp = rewriter.create<ONNXConcatOp>(loc, concatType,
        ValueRange({fwdLastSlice, bkwFirstSlice}), /*concatAxis=*/1);
    Type squeezeType = RankedTensorType::get({D, B, H}, elementType);
    return rewriter.create<ONNXSqueezeV11Op>(
        loc, squeezeType, concatOp, rewriter.getI64ArrayAttr(0));
  } else {
    llvm_unreachable("Invalid direction.");
  }
  return ret;
}

Value getLSTMGRUGetYc(
    Location loc, PatternRewriter &rewriter, Value val, Value resYc) {
  Value noneValue;
  if (isNoneValue(resYc))
    return noneValue;

  zhigh::ZHighUnstickOp unstickOp =
      rewriter.create<zhigh::ZHighUnstickOp>(loc, val.getType(), val);
  return rewriter.create<ONNXSqueezeV11Op>(
      loc, resYc.getType(), unstickOp.getResult(), rewriter.getI64ArrayAttr(0));
}

SmallVector<Value, 4> emitONNXSplitOp(Location loc, PatternRewriter &rewriter,
    Value input, IntegerAttr axis, ArrayAttr split) {
  Type elementType = input.getType().cast<ShapedType>().getElementType();
  SmallVector<mlir::Type> outputTypes;
  int64_t splitNum = split.size();
  ArrayRef<int64_t> inputShape =
      input.getType().cast<RankedTensorType>().getShape();
  int64_t splitAxis = axis.cast<IntegerAttr>().getSInt();
  assert(splitAxis >= 0 && "Negative axis");
  for (int i = 0; i < splitNum; i++) {
    SmallVector<int64_t> outputShape;
    for (size_t dim = 0; dim < inputShape.size(); dim++) {
      outputShape.emplace_back((dim == (unsigned int)splitAxis)
                                   ? split[dim].cast<IntegerAttr>().getInt()
                                   : inputShape[dim]);
    }
    outputTypes.emplace_back(RankedTensorType::get(outputShape, elementType));
  }
  ONNXSplitV11Op splitOp =
      rewriter.create<ONNXSplitV11Op>(loc, outputTypes, input, axis, split);
  return splitOp.getResults();
}

/// Get kernelShapes using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayKernelShape(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();

  // Check if kernelShape is literal. Only static value is supported.
  assert((llvm::any_of(shapeHelper.kernelShape, [](IndexExpr val) {
    return val.isLiteral();
  })) && "Only support static kernel_shape ");

  SmallVector<int64_t, 2> kernelShapesRet;
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[0].getLiteral());
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[1].getLiteral());
  return kernelShapesRet;
}

/// Get strides using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayStrides(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  return shapeHelper.strides;
}

//===----------------------------------------------------------------------===//
// ONNX to ZHigh Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXONNXToZHigh.inc"

// Enhance 'replaceONNXSumOpPatternRecursion' to allow operating recursively.
struct ONNXSumOpPatternEnhancedRecursion
    : public replaceONNXSumOpPatternRecursion {
  ONNXSumOpPatternEnhancedRecursion(MLIRContext *context)
      : replaceONNXSumOpPatternRecursion(context) {}
  void initialize() {
    // This pattern recursively unpacks one variadic operand at a time. The
    // recursion bounded as the number of variadic operands is strictly
    // decreasing.
    setHasBoundedRewriteRecursion(true);
  }
};

// TODO: Shared with rewriting
class ONNXMatMulAsyncExecutionPattern : public OpRewritePattern<ONNXMatMulOp> {
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
    // TODO : Change the way to set chunkSize
    int chunkSize = 512; // 2048; // 4096; // 8192;(2 zAIU) // 16384 (1 zAIU);
    bool nExceeded = N > chunkSize;
    bool mExceeded = M > chunkSize;
    if (!(nExceeded || mExceeded))
      return failure();
    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    ValueRange subAs(A), subBs(B);
    if (nExceeded) {
      // Split A along the dimension N.
      subAs = splitAlongAxis(create, A, aRank - 2, chunkSize);
    }
    if (mExceeded) {
      // Split B along the dimension M.
      subBs = splitAlongAxis(create, B, bRank - 1, chunkSize);
    }
    // Emit sub matrix multiplication.
    SmallVector<Value> resSubAs;
    for (Value a : subAs) {
      ArrayRef<int64_t> subAShape = getShape(a.getType());
      // For each matrix along dimension N, do MatMul for sub matrices along
      // dimension M.
      SmallVector<Value> subMatrices, tokens, subCZeros;
      for (Value b : subBs) {
        // Create ZHighMatMulAsync op
        // TODO: Temporary change
        RankedTensorType tokenType =
            RankedTensorType::get({32}, rewriter.getF32Type());
        // RankedTensorType tokenType =
        //    RankedTensorType::get({8}, rewriter.getI64Type());
        // Create C with zero
        SmallVector<int64_t, 3> cShape;
        Type subBType = b.getType();
        int64_t subBRank = getRank(subBType);
        ArrayRef<int64_t> subBShape = getShape(subBType);
        cShape.emplace_back(subBShape[subBRank - 1]);
        Type cType = RankedTensorType::get(cShape, elementType);
        Value cZero =
            onnx_mlir::zhigh::getConstantOfType(rewriter, loc, cType, 0.0);
        subCZeros.emplace_back(cZero);
        zhigh::ZHighMatMulAsyncOp asyncMatMulOp =
            rewriter.create<zhigh::ZHighMatMulAsyncOp>(
                loc, unrankedType, tokenType, a, b, cZero);
        (void)asyncMatMulOp.inferShapes([](Region &region) {});
        Value sm = asyncMatMulOp.getResults()[0];
        subMatrices.emplace_back(sm);
        Value token = asyncMatMulOp.getResults()[1];
        tokens.emplace_back(token);
      }
      // Wait op
      SmallVector<Value> waitBOps;
      for (auto it : llvm::zip(subMatrices, tokens, subBs, subCZeros)) {
        Value subMatrix = std::get<0>(it);
        Value token = std::get<1>(it);
        Value b = std::get<2>(it);
        Value c = std::get<3>(it);
        onnx_mlir::zhigh::ZHighMatMulWaitOp waitOp =
            rewriter.create<onnx_mlir::zhigh::ZHighMatMulWaitOp>(
                loc, subMatrix.getType(), subMatrix, token, a, b, c);
        waitBOps.emplace_back(waitOp.getResult());
      }
      Value res = waitBOps[0];
      if (subMatrices.size() > 1) {
        // Concat sub results along dimension M of B.
        SmallVector<int64_t> concatShape(outputShape);
        concatShape[outputRank - 2] = subAShape[aRank - 2];
        Type concatTy = RankedTensorType::get(concatShape, elementType);
        res = create.onnx.concat(concatTy, waitBOps, outputRank - 1);
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

struct ONNXToZHighLoweringPass
    : public PassWrapper<ONNXToZHighLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToZHighLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-zhigh"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to ZHigh ops.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ONNXToZHighLoweringPass() = default;
  ONNXToZHighLoweringPass(const ONNXToZHighLoweringPass &pass)
      : PassWrapper<ONNXToZHighLoweringPass, OperationPass<ModuleOp>>() {}
  ONNXToZHighLoweringPass(mlir::ArrayRef<std::string> execNodesOnCpu) {
    this->execNodesOnCpu = execNodesOnCpu;
  }
  void runOnOperation() final;

public:
  ListOption<std::string> execNodesOnCpu{*this, "execNodesOnCpu",
      llvm::cl::desc("Comma-separated list of node names in an onnx graph. The "
                     "specified nodes are forced to run on the CPU instead of "
                     "using the zDNN. The node name is an optional attribute "
                     "in onnx graph, which is `onnx_node_name` in ONNX IR"),
      llvm::cl::ZeroOrMore};
};
} // end anonymous namespace.

void ONNXToZHighLoweringPass::runOnOperation() {
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
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, KrnlDialect,
      func::FuncDialect, arith::ArithDialect>();

  // Combined ONNX ops to ZHigh lowering.
  // There are some combinations of ONNX ops that can be lowering into a single
  // ZHigh op, e.g. ONNXMatMul and ONNXAdd can be lowered to ZHighMatmul.
  // The lowering of such combinations should be done before the lowering of
  // a single ONNX Op, because the single op lowering might have conditions that
  // prohibit the combined ops lowering happened.
  RewritePatternSet combinedPatterns(&getContext());
  combinedPatterns.insert<replaceONNXMatMulAddPattern1>(&getContext());
  combinedPatterns.insert<replaceONNXMatMulAddPattern2>(&getContext());
  combinedPatterns.insert<replaceONNXReluConvPattern>(&getContext());
  combinedPatterns.insert<replaceONNXLogSoftmaxPattern>(&getContext());

  // It's ok to fail.
  (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.insert<ONNXSumOpPatternEnhancedRecursion>(&getContext());
  patterns.insert<ONNXMatMulAsyncExecutionPattern>(&getContext());

  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  target.addIllegalOp<mlir::memref::AllocOp>();
  target.addIllegalOp<mlir::memref::DeallocOp>();

  // ONNX ops to ZHigh dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  addDynamicallyLegalOpFor<ONNXAddOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSubOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMulOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXDivOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSumOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMinOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMaxOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXReluOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXTanhOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSigmoidOp>(
      &target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXLogOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXExpOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSoftmaxOp>(
      &target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMaxPoolSingleOutOp>(
      &target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXAveragePoolOp>(
      &target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMatMulOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXGemmOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXReduceMeanV13Op>(
      &target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXLSTMOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXGRUOp>(&target, &dimAnalysis, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXConvOp>(&target, &dimAnalysis, execNodesOnCpu);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToZHighPass() {
  return std::make_unique<ONNXToZHighLoweringPass>();
}

std::unique_ptr<Pass> createONNXToZHighPass(
    mlir::ArrayRef<std::string> execNodesOnCpu) {
  return std::make_unique<ONNXToZHighLoweringPass>(execNodesOnCpu);
}

} // namespace onnx_mlir
