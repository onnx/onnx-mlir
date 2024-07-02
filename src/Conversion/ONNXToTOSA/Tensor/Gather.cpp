/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Gather Op ------------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX GatherOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXGatherLoweringToTOSA : public OpConversionPattern<ONNXGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXGatherOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    TosaBuilder tosaBuilder(rewriter, loc);

    Value input = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axis = adaptor.getAxis();

    auto result = op.getResult();

    auto inputType = dyn_cast<TensorType>(input.getType());
    if (!onnx_mlir::isRankedShapedType(inputType))
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");
    auto resultTy = dyn_cast<TensorType>(op.getType());
    if (!onnx_mlir::isRankedShapedType(resultTy))
      return rewriter.notifyMatchFailure(op, "result is not a ranked tensor");
    int64_t inputRank = onnx_mlir::getRank(inputType);

    // onnx allows values beetween [-r, r-1] where r is the rank
    axis = tosa::convertNegativeAxis(axis, inputRank);

    auto indicesType = indices.getType().cast<ShapedType>();

    APInt indicesVal;
    /*std::cout << "indicesType.getRank():"<< indicesType.getRank() 
    << " matchPattern(indices, m_ConstantInt(&indicesVal):" << matchPattern(indices, m_ConstantInt(&indicesVal))
    << " indicesVal.getSExtValue():" << indicesVal.getSExtValue() << "\n";*/
    if(indicesType.getRank() == 0 && matchPattern(indices, m_ConstantInt(&indicesVal)) && indicesVal.getSExtValue() >= 0) {
      llvm::SmallVector<int64_t, 4> starts(inputType.getRank(), 0);
      llvm::SmallVector<int64_t, 4> size{inputType.getShape()};
      starts[axis] = indicesVal.getSExtValue();
      size[axis] = 1;
      Value sliceOp = tosaBuilder.slice(input, size, starts);
      auto reshape = tosaBuilder.reshape(sliceOp, resultTy.getShape());
      rewriter.replaceOp(op, reshape);
      return success();
    }

    SmallVector<int32_t, 4> newIndicesValues;
    newIndicesValues.resize(indicesType.getNumElements());

    ArrayRef<int64_t> inputShape = inputType.cast<ShapedType>().getShape();

    // ONNX allows negative indices and TOSA doesn't.
    // We will emit ops to compute
    //   newIndices = indices >= 0 ? indices : indices + dimSize
    // element-wise.

    // Create an 1x..x1 constant containing the size of the gathered dimension.
    auto dimSize =
        tosaBuilder.getSplattedConst(inputShape[axis], indicesType.getRank());
    if (!indicesType.getElementType().isInteger(64)) {
      dimSize = tosaBuilder.castToNewTensorElementType(
          dimSize, indicesType.getElementType());
    }
    auto indicesPlusDimSize =
        tosaBuilder.binaryOp<mlir::tosa::AddOp>(indices, dimSize);

    auto zero = tosaBuilder.getSplattedConst((int64_t)0, indicesType.getRank());
    if (!indicesType.getElementType().isInteger(64)) {
      zero = tosaBuilder.castToNewTensorElementType(
          zero, indicesType.getElementType());
    }
    auto indicesPositive = tosaBuilder.greaterEqual(indices, zero);

    auto newIndices =
        tosaBuilder.select(indicesPositive, indices, indicesPlusDimSize);

    auto newGather =
        tosaBuilder.gather(result, input, newIndices, 0, (int32_t)axis);

    if (!newGather.has_value()) {
      return failure();
    }
    rewriter.replaceOp(op, newGather.value());

    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGatherLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
