
//===------- ReshapeOp.cpp - ONNX Op Transform ------------------===//
// ===================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::torch;

static llvm::Optional<std::pair<unsigned, unsigned>> isEquivalentToFlatten(const llvm::ArrayRef<int64_t> &inputShape, const llvm::ArrayRef<int64_t> &targetShape) {
  llvm::Optional<std::pair<unsigned, unsigned>> savedResult;
  int dimDiff = inputShape.size() - targetShape.size(); 
  if (dimDiff > 0) {
    //Case of ending with ones
    unsigned startDim = 0;
    while (startDim < targetShape.size() - 1) {
      if (inputShape[startDim] != targetShape[startDim]) {
        break;
      }
      ++startDim;
    }
    // Found a difference, we check that the difference can be the result of a flattening
    unsigned acc = 1;
    for (unsigned endDim = startDim; (endDim < inputShape.size() && (endDim - startDim) <= dimDiff); ++endDim) {
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

class ONNXReshapeOpToTorchLowering : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value inputData = op.data();
    Value inputValue = adaptor.data();
    //Value targetData = adaptor.shape();
    TensorType inputTensorType = inputData.getType().cast<TensorType>();
    //TensorType targetTensorType = targetData.getType().cast<TensorType>();

    TensorType resultTensorType = op.getResult().getType().cast<TensorType>();
    auto resultType = Torch::ValueTensorType::get(context,
        resultTensorType.getShape(), resultTensorType.getElementType());
    TensorType targetTensorType = resultType.toBuiltinTensor();

      
    ArrayRef<int64_t> targetShape = targetTensorType.getShape();
    ArrayRef<int64_t> inputShape = inputTensorType.getShape();
    // Flatten
    if (auto dimPair = isEquivalentToFlatten(inputShape, targetShape)) {
        IntegerType ty = IntegerType::get(op.getContext(), 64);
        IntegerAttr startDimInt = IntegerAttr::get(ty, (dimPair->first));
        Value startDimConstInt =
            rewriter.create<Torch::ConstantIntOp>(loc, startDimInt);
        IntegerAttr endDimInt = IntegerAttr::get(ty, (dimPair->second));
        Value endDimConstInt = rewriter.create<Torch::ConstantIntOp>(loc, endDimInt);
        auto flattenedElem = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
            loc, resultType, inputValue, startDimConstInt, endDimConstInt);
        rewriter.replaceOpWithNewOp<Torch::TensorStaticInfoCastOp>(
            op, resultType, flattenedElem);
    }
    else {
    //Reshape
        //auto TargetType = Torch::ValueTensorType::get(context,
        //  llvm::makeArrayRef(targetShape), inputTensorType.getElementType());
        //Value targetValue = rewriter.create<PrimListConstructOp>(loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{TargetType});
        //Value Shape = TargetType;
        //rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(op, resultType, inputValue, targetValue);
    }
    return success();
  }
};

void populateLoweringONNXToTorchReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpToTorchLowering>(typeConverter, ctx);
}
