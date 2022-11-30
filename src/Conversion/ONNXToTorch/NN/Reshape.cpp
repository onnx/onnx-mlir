
//===------- ReshapeOp.cpp - ONNX Op Transform ------------------===//
// ===================================================================
//
// This file implements a combined pass that dynamically invokes several
// transformation on ONNX ops.
//
//===-------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::torch;

static llvm::Optional<std::pair<unsigned, unsigned>> isEquivalentToFlatten(const llvm::ArrayRef<int64_t> &inputShape, const llvm::ArrayRef<int64_t> &targetShape) {
  llvm::Optional<std::pair<unsigned, unsigned>> savedResult;
  int targetSize = targetShape.size();
  int inputSize = inputShape.size();
  int dimDiff = inputSize - targetSize; 
  if (dimDiff > 0) {
    //Case of ending with ones
    int startDim = 0;
    while (startDim < targetSize - 1) {
      if (inputShape[startDim] != targetShape[startDim]) {
        break;
      }
      ++startDim;
    }
    // Found a difference, we check that the difference can be the result of a flattening
    unsigned acc = 1;
    for (int endDim = startDim; (endDim < inputSize && (endDim - startDim) <= dimDiff); ++endDim) {
      unsigned tmpAcc = acc * inputShape[endDim];
      if (tmpAcc == targetShape[startDim]) {
        //Result saved but not returned in case it's not the longest result to match the flattening
        savedResult = std::pair<unsigned, unsigned>(startDim, endDim);
      }
      acc = tmpAcc;
    }
  }
  return savedResult;
}

// Reshape input into a different-sized tensor.
// It can be semantically equivalent to a Flatten operation if
// the result shape is a multiplication of two or more dimensions from
// the input shape.
// TODO: For Resnet18 purposes, only the flattening part of this operator
// is implemented. If you need to generate a AtenReshapeOp out of it, please
// replace the assert from the second part by your implementation.
//
// Operands:
//    input:
//              data: The input tensor to be reshaped
//              shape: The expected shape of the output
//
// Results:
//    output:   A tensor of the shape of the `shape` argument
//              based on the `data` argument.

class ONNXReshapeOpToTorchLowering : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value inputData = op.data();
    Value inputValue = adaptor.data();
    TensorType inputTensorType = inputData.getType().cast<TensorType>();

    TensorType resultTensorType = op.getResult().getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(context,
        resultTensorType.getShape(), resultTensorType.getElementType());
    TensorType targetTensorType = resultType.toBuiltinTensor();

      
    ArrayRef<int64_t> targetShape = targetTensorType.getShape();
    ArrayRef<int64_t> inputShape = inputTensorType.getShape();
    // Generate a Flatten operator when reshape is equivalent to a Flatten
    // i.e when the resulting shape is the result of merging two or more
    // dimensions together.
    if (auto dimPair = isEquivalentToFlatten(inputShape, targetShape)) {
        IntegerType ty = IntegerType::get(op.getContext(), 64);
        IntegerAttr startDimInt = IntegerAttr::get(ty, (dimPair->first));
        Value startDimConstInt =
            rewriter.create<Torch::ConstantIntOp>(loc, startDimInt);
        IntegerAttr endDimInt = IntegerAttr::get(ty, (dimPair->second));
        Value endDimConstInt = rewriter.create<Torch::ConstantIntOp>(loc, endDimInt);
        auto flattenedElem = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
            loc, resultType, inputValue, startDimConstInt, endDimConstInt);
        setLayerNameAttr(op, flattenedElem);
        rewriter.replaceOpWithNewOp<Torch::TensorStaticInfoCastOp>(
            op, resultType, flattenedElem);
    }
    else {
      assert(false && "Generation of ReshapeOP is not implemented.");
    }
    return success();
  }
};

void populateLoweringONNXToTorchReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpToTorchLowering>(typeConverter, ctx);
}
