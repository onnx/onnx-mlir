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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
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

Value getLSTMGRUGetYWithSequenceLens(Location loc, PatternRewriter &rewriter,
    Value val, Value resY, Value sequenceLens, Value initialH) {

  Value noneValue;
  if (isNoneValue(resY)) {
    return noneValue;
  }

  if (isNoneValue(sequenceLens))
    return getLSTMGRUGetY(loc, rewriter, val, resY);

  std::vector<Value> inputs = {val, sequenceLens, initialH};
  return rewriter.create<zhigh::ZHighFixGRUYOp>(loc, resY.getType(), inputs);
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

Value getLSTMGRUGetYhWithSequenceLens(Location loc, PatternRewriter &rewriter,
    Value val, Value resY, Value resYh, Value X, StringAttr direction,
    Value sequenceLens) {
  Value noneValue;
  if (isNoneValue(resYh) || isNoneValue(val))
    return noneValue;

  if (isNoneValue(sequenceLens))
    return getLSTMGRUGetYh(loc, rewriter, val, resY, resYh, X, direction);

  std::vector<Value> inputs = {val, sequenceLens};
  return rewriter.create<zhigh::ZHighFixGRUYhOp>(loc, resYh.getType(), inputs);
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
  void runOnOperation() final;
};
} // end anonymous namespace.

void getONNXToZHighOneOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  populateWithGenerated(patterns);
  patterns.insert<ONNXSumOpPatternEnhancedRecursion>(context);
}

void getONNXToZHighOneOpDynamicallyLegal(
    ConversionTarget *target, const DimAnalysis *dimAnalysis) {
  addDynamicallyLegalOpFor<ONNXAddOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSubOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXDivOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSumOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMinOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReluOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXTanhOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSigmoidOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXLogOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXExpOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSoftmaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMaxPoolSingleOutOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXAveragePoolOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMatMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGemmOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReduceMeanV13Op>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXLSTMOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGRUOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXConvOp>(target, dimAnalysis);
}

void getONNXToZHighMultipleOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<replaceONNXMatMulAddPattern1>(context);
  patterns.insert<replaceONNXMatMulAddPattern2>(context);
  patterns.insert<replaceONNXReluConvPattern>(context);
  patterns.insert<replaceONNXLogSoftmaxPattern>(context);
}

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

  // NOTE: if we change the order of calling combinedPatterns and single op
  // patterns, make sure to change the order in DevicePlacement.cpp also to make
  // them synced.

  // Combined ONNX ops to ZHigh lowering.
  // There are some combinations of ONNX ops that can be lowering into a single
  // ZHigh op, e.g. ONNXMatMul and ONNXAdd can be lowered to ZHighMatmul.
  // The lowering of such combinations should be done before the lowering of
  // a single ONNX Op, because the single op lowering might have conditions that
  // prohibit the combined ops lowering happened.
  RewritePatternSet combinedPatterns(&getContext());
  onnx_mlir::getONNXToZHighMultipleOpPatterns(combinedPatterns);

  // It's ok to fail.
  (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  onnx_mlir::getONNXToZHighOneOpPatterns(patterns);

  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  target.addIllegalOp<mlir::memref::AllocOp>();
  target.addIllegalOp<mlir::memref::DeallocOp>();

  // ONNX ops to ZHigh dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  getONNXToZHighOneOpDynamicallyLegal(&target, &dimAnalysis);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToZHighPass() {
  return std::make_unique<ONNXToZHighLoweringPass>();
}

} // namespace onnx_mlir
