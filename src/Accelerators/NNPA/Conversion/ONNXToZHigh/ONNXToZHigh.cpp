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
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

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
  if (isNoneType(resY)) {
    return noneValue;
  }
  return val;
}

Value getLSTMGRUGetYh(Location loc, PatternRewriter &rewriter, Value val,
    Value resY, Value resYh, StringAttr direction) {
  Value noneValue;
  if (isNoneType(resYh) || isNoneType(val))
    return noneValue;

  // Generate Y_h for onnx.LSTM from hn_output for all timestep
  Value minusOne =
      rewriter
          .create<ONNXConstantOp>(loc, nullptr, rewriter.getI64TensorAttr({-1}))
          .getResult();
  Value zero =
      rewriter
          .create<ONNXConstantOp>(loc, nullptr, rewriter.getI64TensorAttr({0}))
          .getResult();
  Value one =
      rewriter
          .create<ONNXConstantOp>(loc, nullptr, rewriter.getI64TensorAttr({1}))
          .getResult();
  // Use INT_MAX to get timestep dimension because timestep is the end of a
  // dimension. INT_MAX is recommended in ONNX.Slice to slice to the end of a
  // dimension with unknown size.
  Value intMax = rewriter
                     .create<ONNXConstantOp>(
                         loc, nullptr, rewriter.getI64TensorAttr({INT_MAX}))
                     .getResult();
  StringRef directionStr = direction.getValue();
  Value start, end;
  if (directionStr.equals_insensitive("forward")) {
    start = minusOne;
    end = intMax;
  } else if (directionStr.equals_insensitive("reverse")) {
    start = zero;
    end = one;
  } else {
    llvm_unreachable("Bidirectional is not supported.");
  }
  ArrayRef<int64_t> yhShape =
      resYh.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> sliceShape({1, yhShape[0], yhShape[1], yhShape[2]});
  Type elementType = resYh.getType().cast<RankedTensorType>().getElementType();
  Type sliceType = RankedTensorType::get(sliceShape, elementType);
  Value axis = zero;
  Value step = one;
  ONNXSliceOp sliceOp =
      rewriter.create<ONNXSliceOp>(loc, sliceType, val, start, end, axis, step);
  ONNXSqueezeV11Op squeezeOp = rewriter.create<ONNXSqueezeV11Op>(
      loc, resYh.getType(), sliceOp.getResult(), rewriter.getI64ArrayAttr(0));
  return squeezeOp.getResult();
}

Value getLSTMGRUGetYc(
    Location loc, PatternRewriter &rewriter, Value val, Value resYc) {
  Value noneValue;
  if (isNoneType(resYc))
    return noneValue;

  auto unstickOp =
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
  OPAdaptor operandAdaptor = OPAdaptor(op);
  OPShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan OP parameters successfully");

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
  OPAdaptor operandAdaptor = OPAdaptor(op);
  OPShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan OP parameters successfully");
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
      llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore};
};
} // end anonymous namespace.

void ONNXToZHighLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, KrnlOpsDialect,
      StandardOpsDialect, arith::ArithmeticDialect>();

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

  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  target.addIllegalOp<mlir::memref::AllocOp>();
  target.addIllegalOp<mlir::memref::DeallocOp>();

  // ONNX ops to ZHigh dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  addDynamicallyLegalOpFor<ONNXAddOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSubOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMulOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXDivOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSumOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMinOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMaxOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXReluOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXTanhOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSigmoidOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXLogOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXExpOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXSoftmaxOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMaxPoolSingleOutOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXAveragePoolOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXMatMulOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXGemmOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXReduceMeanOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXLSTMOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXGRUOp>(&target, execNodesOnCpu);
  addDynamicallyLegalOpFor<ONNXConvOp>(&target, execNodesOnCpu);

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
