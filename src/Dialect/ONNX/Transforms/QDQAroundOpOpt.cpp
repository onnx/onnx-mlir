//===- QDQAroundOpOpt.cpp - Remove DQ, Q operations around data movement ops
//--------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <src/Dialect/ONNX/ONNXOps.hpp>
#include <src/Dialect/ONNX/ONNXOps/OpHelper.hpp>

using namespace mlir;
using namespace onnx_mlir;

namespace {

/// Check if a value is defined by a constant operation
/// Returns false for NoValue (NoneType)
/// Uses recursive logic to check if all operands are constants (initializers)
bool isConstantOrInitializer(Value val) {
  if (!val)
    return false;

  // Return false for NoValue (which has NoneType)
  if (mlir::isa<NoneType>(val.getType())) {
    return false;
  }

  Operation *definingOp = val.getDefiningOp();
  if (!definingOp) {
    return false;
  }

  // Check if it's a constant op
  if (llvm::isa<ONNXConstantOp>(definingOp)) {
    return true;
  }

  // Recursively check if all operands are initializers
  // If all operands are constants, the result is effectively constant
  for (Value operand : definingOp->getOperands()) {
    if (!isConstantOrInitializer(operand)) {
      return false;
    }
  }
  return true;
}

struct InputAndOutput {
  Value input;
  Value output;
};

InputAndOutput getDataInputOutput(ONNXTransposeOp transposeOp) {
  return {transposeOp.getData(), transposeOp.getTransposed()};
}
InputAndOutput getDataInputOutput(ONNXUnsqueezeOp unsqueezeOp) {
  return {unsqueezeOp.getData(), unsqueezeOp.getExpanded()};
}
InputAndOutput getDataInputOutput(ONNXSqueezeOp squeezeOp) {
  return {squeezeOp.getData(), squeezeOp.getSqueezed()};
}
InputAndOutput getDataInputOutput(ONNXReshapeOp reshapeOp) {
  return {reshapeOp.getData(), reshapeOp.getReshaped()};
}
InputAndOutput getDataInputOutput(ONNXGatherOp gatherOp) {
  return {gatherOp.getData(), gatherOp.getOutput()};
}
InputAndOutput getDataInputOutput(ONNXSliceOp sliceOp) {
  return {sliceOp.getData(), sliceOp.getOutput()};
}
InputAndOutput getDataInputOutput(ONNXResizeOp resizeOp) {
  return {resizeOp.getX(), resizeOp.getY()};
}
InputAndOutput getDataInputOutput(ONNXFlattenOp flattenOp) {
  return {flattenOp.getInput(), flattenOp.getOutput()};
}

template <typename T>
class RemoveQDQAroundOpPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      T op, PatternRewriter &rewriter) const override {

    if (llvm::isa<ONNXResizeOp>(op)) {
      // Resize: only support "nearest" mode, require either scales or sizes to
      // be constant
      auto resizeOp = llvm::cast<ONNXResizeOp>(op);
      if (resizeOp.getMode() != "nearest") {
        return failure();
      }
      // At least one of scales or sizes must be from initializer
      bool hasScales = isConstantOrInitializer(resizeOp.getScales());
      bool hasSizes = isConstantOrInitializer(resizeOp.getSizes());
      if (!hasScales && !hasSizes) {
        return failure();
      }
    } else if (llvm::isa<ONNXUnsqueezeOp>(op)) {
      // Unsqueeze requires axes to be a constant
      auto unsqueezeOp = llvm::cast<ONNXUnsqueezeOp>(op);
      if (!isConstantOrInitializer(unsqueezeOp.getAxes())) {
        return failure();
      }
    } else if (llvm::isa<ONNXSqueezeOp>(op)) {
      // Squeeze requires axes to be a constant
      auto squeezeOp = llvm::cast<ONNXSqueezeOp>(op);
      if (!isConstantOrInitializer(squeezeOp.getAxes())) {
        return failure();
      }
    } else if (llvm::isa<ONNXReshapeOp>(op)) {
      // Reshape requires shape to be a constant
      auto reshapeOp = llvm::cast<ONNXReshapeOp>(op);
      if (!isConstantOrInitializer(reshapeOp.getShape())) {
        return failure();
      }
    } else if (llvm::isa<ONNXGatherOp>(op)) {
      // Gather requires indices to be a constant
      auto gatherOp = llvm::cast<ONNXGatherOp>(op);
      if (!isConstantOrInitializer(gatherOp.getIndices())) {
        return failure();
      }
    } else if (llvm::isa<ONNXSliceOp>(op)) {
      // Slice requires all control parameters to be constants
      auto sliceOp = llvm::cast<ONNXSliceOp>(op);
      if (!isConstantOrInitializer(sliceOp.getStarts()) ||
          !isConstantOrInitializer(sliceOp.getEnds()) ||
          !isConstantOrInitializer(sliceOp.getAxes()) ||
          !isConstantOrInitializer(sliceOp.getSteps())) {
        return failure();
      }
    }

    InputAndOutput opIO = getDataInputOutput(op);

    auto dqOp = opIO.input.getDefiningOp<ONNXDequantizeLinearOp>();
    // Only run this pass if Quantizelization is on tensor
    if (!dqOp || !isScalarConstantTensor(dqOp.getXScale()) ||
        !isScalarConstantTensor(dqOp.getXZeroPoint())) {
      return failure();
    }
    if (!opIO.output.hasOneUse()) {
      return failure();
    }

    Operation *firstOp = *(opIO.output.getUsers().begin());
    if (auto qOp = dyn_cast<ONNXQuantizeLinearOp>(firstOp)) {
      if (!isScalarConstantTensor(qOp.getYScale()) ||
          !isScalarConstantTensor(qOp.getYZeroPoint())) {
        return failure();
      }
      if (!isDequantQuantSame(dqOp, qOp))
        return failure();

      // Map dqOp inputs to dqOp's inputs
      IRMapping irMapping;
      irMapping.map(dqOp, dqOp.getX());

      SmallVector<Value> newInputs;
      transform(op->getOperands(), std::back_inserter(newInputs),
          [&](Value operand) { return irMapping.lookupOrDefault(operand); });

      SmallVector<NamedAttribute> filteredAttrs(op->getAttrs());
      if (auto foundIter = llvm::find_if(filteredAttrs,
              [](NamedAttribute kv) { return kv.getName() == "ResultNames"; });
          foundIter != filteredAttrs.end())
        filteredAttrs.erase(foundIter);

      auto newOp =
          rewriter.create<T>(op.getLoc(), TypeRange{qOp.getResult().getType()},
              ValueRange{newInputs}, filteredAttrs);
      rewriter.replaceOp(qOp, newOp.getResult());
      return success();
    }
    return failure();
  };
};

} // namespace

namespace onnx_mlir {

void getRemoveQDQAroundOpPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RemoveQDQAroundOpPattern<ONNXTransposeOp>,
      RemoveQDQAroundOpPattern<ONNXUnsqueezeOp>,
      RemoveQDQAroundOpPattern<ONNXSqueezeOp>,
      RemoveQDQAroundOpPattern<ONNXReshapeOp>,
      RemoveQDQAroundOpPattern<ONNXResizeOp>,
      RemoveQDQAroundOpPattern<ONNXGatherOp>,
      RemoveQDQAroundOpPattern<ONNXSliceOp>,
      RemoveQDQAroundOpPattern<ONNXFlattenOp>>(context);
}

} // namespace onnx_mlir
