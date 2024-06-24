/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ONNXMatMulOp.cpp - ONNXMatMulOp --------------===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNXMatMulOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// TOSA matmul is performed on two 3D inputs and generates a 3D output.
// Lower ranked tensors are dim-1 reshaped up to 3D
Value reshapeUpTo3DTensor(Value tensor, TosaBuilder &builder) {
  auto tensorTy = cast<TensorType>(tensor.getType());
  auto rank = tensorTy.getRank();

  assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
  if (rank == 3)
    return tensor;

  ArrayRef<int64_t> shape = tensorTy.getShape();
  SmallVector<int64_t> newShape({1, 1, 1});

  if (rank == 2) { // batchsize = 1
    newShape[1] = shape[0];
    newShape[2] = shape[1];
  } else { // rank 1
    newShape[2] = shape[0];
  }

  return builder.reshape(tensor, newShape);
};

// Obtaining the rank broadcasted shapes of tensors makes it easier to
// construct the input and output reshaping logic.
void getRankBroadcastedShape(Value tensor, const int64_t maxInputRank,
    bool isRHS, SmallVectorImpl<int64_t> &bcastedShape) {
  auto tensorTy = cast<TensorType>(tensor.getType());
  ArrayRef<int64_t> tensorShape = tensorTy.getShape();
  int64_t tensorRank = tensorTy.getRank();

  const int64_t bcastDims = maxInputRank - tensorRank;

  if (isRHS && (tensorRank == 1) && bcastDims) {
    // RHS with rank1 is special. It be synthetically transposed to dim[:-2]
    for (int32_t i = 0; i < bcastDims - 1; i++)
      bcastedShape.push_back(1);
    bcastedShape.push_back(tensorShape[0]);
    bcastedShape.push_back(1);
  } else {
    if (bcastDims > 0) { // rank broadcast
      for (uint32_t i = 0; i < bcastDims; i++)
        bcastedShape.push_back(1);
    }
    for (const auto &dim : tensorShape)
      bcastedShape.push_back(dim);
  }
}

Type getMatMulOutputType(Type inputElemTy, PatternRewriter &rewriter) {
  Type outputElemTy;
  if (auto floatTy = dyn_cast<FloatType>(inputElemTy)) {
    if (floatTy.isBF16() || floatTy.isF16() || floatTy.isF32()) {
      // Always accumulate on f32
      outputElemTy = rewriter.getF32Type();
    }
  } else if (auto integerTy = dyn_cast<IntegerType>(inputElemTy)) {
    if (integerTy.isInteger(/*width=*/8)) {
      outputElemTy = rewriter.getIntegerType(/*width=*/32);
    } else if (integerTy.isInteger(/*width=*/16)) {
      outputElemTy = rewriter.getIntegerType(/*width=*/48);
    }
  }
  return outputElemTy;
}

// Lowering based on the lowering of torch-mlir
class ONNXMatMulOpLoweringToTOSA : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;
  using Adaptor = ONNXMatMulOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    TosaBuilder builder(rewriter, op->getLoc());

    auto lhs = adaptor.getA();
    auto rhs = adaptor.getB();

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy || !lhsTy.hasStaticShape() ||
        !rhsTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only ranked tensor types are supported");
    }

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = lhsTy.getShape();
    auto rhsShape = rhsTy.getShape();

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy) {
      return rewriter.notifyMatchFailure(
          op, "expected both inputs to have same element type");
    }

    auto outputElemType = getMatMulOutputType(lhsElemTy, rewriter);
    if (!outputElemType) {
      return rewriter.notifyMatchFailure(op,
          "Only i8 and i16 integer and bf16, f16 and "
          "f32 float types are valid");
    }

    int64_t maxInputRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    // If performing dot product on vectors, the RHS is synthetically transposed
    if (maxInputRank == 1)
      maxInputRank++;

    // Step: Rank broadcast the two inputs.
    SmallVector<int64_t, 3> lhsBroadcastedShape;
    SmallVector<int64_t, 3> rhsBroadcastedShape;
    getRankBroadcastedShape(lhs, maxInputRank, false, lhsBroadcastedShape);
    getRankBroadcastedShape(rhs, maxInputRank, true, rhsBroadcastedShape);

    auto rankBroadcastedLhs = lhsRank == maxInputRank
                                  ? lhs
                                  : builder.reshape(lhs, lhsBroadcastedShape);

    auto rankBroadcastedRhs = rhsRank == maxInputRank
                                  ? rhs
                                  : builder.reshape(rhs, rhsBroadcastedShape);

    // Where broadcasting is required in one or more batch dims, the following
    // is done.
    // Where all batch dims are involved in broadcasting:
    // Given A: 3x1x5x6 and B: 1x4x6x7
    // 1. Reshape A to 1x15x6 (squeeze all batchdims into dim1)
    // 2. Transpose B to 6x1x4x7, Reshape to 1x6x28
    // 3. tosa.Matmul 1x15x6 1x6x28 = 1x15x28
    // 4. Reshape out to 3x5x4x7, Transpose to 3x4x5x7
    // Where there are batch dimensions that are broadcast and not, the
    // treatment is to have dim0 correspond to product of all non-broadcast
    // dimsizes:
    // Given A: 4x8x16x32 B: 8x32x17
    // 1. Reshape A to 8x64x32 (squeeze all unbroadcasted dims into dim0,
    // broadcasted dims into dim1)
    // 2. No transpose or reshape of B as its batchdims are not broadcast to.
    // 3. tosa.Matmul 8x64x32 8x32x17 = 8x64x17
    // 4. Reshape to 8x4x16x17, Transpose to 4x8x16x17

    // Inputs to the tosa.matmul
    Value matmulLhs;
    Value matmulRhs;

    using TensorShape_t = struct {
      int64_t dim;
      int64_t shape;
    };

    // Transpose needs to done if transposeDims are not non-monotonically
    // increasing. E.g. [0, 1, 2, 3]: No transpose [1, 0, 2, 3]: Transpose dim0
    // and dim1 The order need not be sequential, since one or more dims may
    // have been removed due to broadcasting.
    auto isTransposeRequired = [](ArrayRef<int32_t> transposedDims) -> bool {
      int32_t lastDim = -1;
      for (int32_t dim : transposedDims) {
        if (lastDim > dim)
          return true;
        lastDim = dim;
      }
      return false;
    };

    SmallVector<TensorShape_t> commonElems;
    SmallVector<TensorShape_t> lhsSqueezedElems;
    SmallVector<TensorShape_t> rhsSqueezedElems;

    // Check if we need to perform the broadcast on batch dim
    // Not needed if max rank < 3, or if maxrank == 3 and dim[0] matches
    auto needsBatchDimBroadcast = [&]() -> bool {
      if (maxInputRank < 3) {
        return false;
      }
      return maxInputRank != 3 ||
             lhsBroadcastedShape[0] != rhsBroadcastedShape[0];
    };

    const bool performBatchDimBroadcast = needsBatchDimBroadcast();
    if (!performBatchDimBroadcast) {
      // Simple with no broadcasting artifacts. Just reshape up to 3D
      matmulLhs = reshapeUpTo3DTensor(rankBroadcastedLhs, builder);
      matmulRhs = reshapeUpTo3DTensor(rankBroadcastedRhs, builder);
    } else {
      // In this case, either or both input matrices involve broadcasting on
      // their batch dimensions. For example:
      // 4x5x6, 1x6x7 -> 4x5x7
      // 4x1x5x6, 1x3x6x7 -> 4x3x5x7
      // Though maxInputRank is necessarily >=3 here, individual matrices may be
      // lower rank.
      // E.g. 3x4x5x6, 6 -> 3x4x5

      // These are the accumulated products of the shape of each dim:
      // 1. common dimensions: upper dimensions (dims other than two rightmost)
      // whose shapes are the same for both LHS and RHS.
      // 2. LHS squeezed dimensions: all dimensions of LHS that involve
      // broadcasting in either direction, plus the LHS[-2] shape
      // 3. RHS squeezed dimensions: all dimensions of RHS that involve
      // broadcasting in either direction, plus the RHS[-1] shape
      int64_t commonValue = 1;
      int64_t lhsSqueezedValue = 1;
      int64_t rhsSqueezedValue = 1;

      // For both LHS and RHS, the dimensions are separated into the common,
      // squeezed and remaining dim. E.g. given
      // LHS = 3x4x5x6
      // RHS = 1x4x6x7
      // common = {{dim=1, shape=4}}
      // lhs squeezed = {{dim=0, shape=3},
      //                 {dim=2, shape=5}}
      // rhs squeezed = {{dim=0, shape=1},
      //                 {dim=2, shape=7}}
      // The matmul dim is LHS[-1] and RHS[-2], i.e. 6.
      // Once this is obtained, LHS and RHS are expressed as:
      // LHS = {common, lhs_squeezed, matmul_dim}
      // RHS = {common, matmul_dim, rhs_squeezed}
      // The matmul is then performed to obtain output:
      // matmul_out = {common, lhs_squeezed, rhs_squeezed}
      // Finally, we reshape to 'unsqueeze' the LHS and RHS parts and transpose
      // them back to their correct positions.

      SmallVector<int64_t> transposedLhsShape;
      SmallVector<int32_t> transposedLhsDims;

      // Step: generate the common dim/shape information
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        if (lhsBroadcastedShape[dim] == rhsBroadcastedShape[dim]) {
          commonValue *= lhsBroadcastedShape[dim];
          commonElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }

      // Step: generate the LHS squeezed dim/shape information.
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        if (!isDynamicDim &&
            lhsBroadcastedShape[dim] != rhsBroadcastedShape[dim]) {
          lhsSqueezedValue *= lhsBroadcastedShape[dim];
          lhsSqueezedElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      // including LHS[-2]
      lhsSqueezedElems.push_back(
          {maxInputRank - 2, lhsBroadcastedShape[maxInputRank - 2]});
      lhsSqueezedValue *= lhsBroadcastedShape[maxInputRank - 2];

      // Step: Create the tosa.transpose array. If this array has a
      // non-monotonic series of dims, perform transpose.
      // First the common_elems
      for (uint32_t i = 0; i < commonElems.size(); i++) {
        transposedLhsShape.push_back(commonElems[i].shape);
        transposedLhsDims.push_back(commonElems[i].dim);
      }
      // then the lhs_squeezed elems
      for (uint32_t i = 0; i < lhsSqueezedElems.size(); i++) {
        transposedLhsShape.push_back(lhsSqueezedElems[i].shape);
        transposedLhsDims.push_back(lhsSqueezedElems[i].dim);
      }
      // then the final dim
      transposedLhsDims.push_back(maxInputRank - 1);
      transposedLhsShape.push_back(lhsBroadcastedShape[maxInputRank - 1]);

      Value lhsReshapeInput = rankBroadcastedLhs;
      if (isTransposeRequired(transposedLhsDims)) {
        lhsReshapeInput =
            builder.transpose(rankBroadcastedLhs, transposedLhsDims);
      }

      // LHS = {common, lhs_squeezed, matmul_dim}
      SmallVector<int64_t> newLhsShape(
          {1, 1, lhsBroadcastedShape[maxInputRank - 1]});
      newLhsShape[0] = commonValue;
      newLhsShape[1] = lhsSqueezedValue;

      matmulLhs = builder.reshape(lhsReshapeInput, newLhsShape);

      SmallVector<int64_t> transposedRhsShape;
      SmallVector<int32_t> transposedRhsDims;

      // Step: Create the RHS transpose sequence
      // RHS = {common, matmul_dim, rhs_squeezed}
      // first the common_dims
      for (uint32_t i = 0; i < commonElems.size(); i++) {
        transposedRhsShape.push_back(commonElems[i].shape);
        transposedRhsDims.push_back(commonElems[i].dim);
      }
      // The matmul_dim of RHS
      transposedRhsDims.push_back(maxInputRank - 2);
      transposedRhsShape.push_back(rhsBroadcastedShape[maxInputRank - 2]);
      // finally all the rhs_squeeze dims
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        if (rhsBroadcastedShape[dim] != lhsBroadcastedShape[dim]) {
          rhsSqueezedElems.push_back({dim, rhsBroadcastedShape[dim]});
          rhsSqueezedValue *= rhsBroadcastedShape[dim];
        }
      }
      rhsSqueezedElems.push_back(
          {maxInputRank - 1, rhsBroadcastedShape[maxInputRank - 1]});
      rhsSqueezedValue *= rhsBroadcastedShape[maxInputRank - 1];
      for (uint32_t i = 0; i < rhsSqueezedElems.size(); i++) {
        transposedRhsShape.push_back(rhsSqueezedElems[i].shape);
        transposedRhsDims.push_back(rhsSqueezedElems[i].dim);
      }

      auto transposedRhsValue = rankBroadcastedRhs;
      if (isTransposeRequired(transposedRhsDims)) {
        transposedRhsValue =
            builder.transpose(rankBroadcastedRhs, transposedRhsDims);
      }

      // reshape
      SmallVector<int64_t> newRhsShape({commonValue,
          rhsBroadcastedShape[maxInputRank - 2], rhsSqueezedValue});
      matmulRhs = builder.reshape(transposedRhsValue, newRhsShape);
    }

    auto matmulLhsShape =
        cast<RankedTensorType>(matmulLhs.getType()).getShape();
    auto matmulRhsShape =
        cast<RankedTensorType>(matmulRhs.getType()).getShape();

    // The reshape/transpose should ensure the tosa.matmul always has same
    // batch size for either matrix. If if shapes are dynamic, they'll be
    // appropriately handled.
    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "tosa.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});

    auto mmOutputTy = RankedTensorType::get(matmulOutputShape, outputElemType);
    auto mmOpResult = tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(
        rewriter, op->getLoc(), mmOutputTy, matmulLhs, matmulRhs)
                          ->getResult(0);
    auto castToOrigOp =
        builder.castToNewTensorElementType(mmOpResult, lhsElemTy);

    // Perform the reshape to output shape. This is always required unless max
    // input rank=3 and there was no broadcasting, in which case the tosa.matmul
    // output itself is correctly shaped.
    bool performOpReshape = !(maxInputRank == 3 && !performBatchDimBroadcast);
    Value output = castToOrigOp;
    if (performOpReshape) {
      // Since the output shape may be unknown, we construct it
      // independently and reshape. Otherwise reshape may be expressed for
      // an unknown to-be-inferred output shape. The final tensor.cast
      // reshapes the known shape to the desired output shape.
      auto computeOpShape = [&](SmallVector<int64_t> &reshapedOpShape,
                                SmallVector<int64_t> &transposedOpShapes) {
        if (maxInputRank == 1)
          return;

        if (maxInputRank == 2) {
          if (lhsRank == 2)
            reshapedOpShape.push_back(lhsShape[0]);
          if (rhsRank == 2)
            reshapedOpShape.push_back(rhsShape[1]);
          return;
        }

        // Step: Construct the output transpose/reshape information
        // First the common_dims
        for (uint32_t i = 0; i < commonElems.size(); i++) {
          reshapedOpShape.push_back(commonElems[i].shape);
        }

        // Then the LHS squeezed dims
        for (uint32_t i = 0; i < lhsSqueezedElems.size() - 1; i++) {
          // Only dims that don't broadcast - broadcasting ones come from the
          // other input.
          if (lhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(lhsSqueezedElems[i].shape);
          }
        }
        // The last squeezed dim is lhs[-2] which needs to be
        // checked separately for broadcasting
        if (lhsRank > 1) {
          reshapedOpShape.push_back(lhsBroadcastedShape[maxInputRank - 2]);
        }

        // then the RHS squeezed dims except rhs[-1] which is handled like
        // lhs[-2]
        for (uint32_t i = 0; i < rhsSqueezedElems.size() - 1; i++) {
          if (rhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(rhsSqueezedElems[i].shape);
          }
        }
        // rhs[-1]
        if (rhsRank > 1) {
          reshapedOpShape.push_back(rhsBroadcastedShape[maxInputRank - 1]);
        }

        // Final transposed output shape construction
        for (uint32_t i = 0; i < maxInputRank - 2; i++) {
          if (lhsBroadcastedShape[i] == rhsBroadcastedShape[i]) {
            transposedOpShapes.push_back(lhsBroadcastedShape[i]);
          } else {
            transposedOpShapes.push_back(lhsBroadcastedShape[i] == 1
                                             ? rhsBroadcastedShape[i]
                                             : lhsBroadcastedShape[i]);
          }
        }
        if (lhsRank > 1)
          transposedOpShapes.push_back(lhsBroadcastedShape[maxInputRank - 2]);
        if (rhsRank > 1)
          transposedOpShapes.push_back(rhsBroadcastedShape[maxInputRank - 1]);

        return;
      };

      // Calculated output shapes for reshape and transpose
      SmallVector<int64_t> reshapedOpShape;
      SmallVector<int64_t> transposedOpShape;
      computeOpShape(reshapedOpShape, transposedOpShape);

      // Perform reshape
      auto reshapeOp = builder.reshape(castToOrigOp, reshapedOpShape);

      // Calculate transmutation required
      SetVector<int32_t> transmutationSetVec;
      for (unsigned i = 0; i < transposedOpShape.size(); i++) {
        for (unsigned j = 0; j < reshapedOpShape.size(); j++) {
          if (!transmutationSetVec.contains(j) &&
              transposedOpShape[i] == reshapedOpShape[j]) {
            transmutationSetVec.insert(j);
            break;
          }
        }
      }
      ArrayRef<int32_t> transVec = transmutationSetVec.getArrayRef();

      // Perform final reshape
      output = isTransposeRequired(transVec)
                   ? builder.transpose(reshapeOp, transVec)
                   : reshapeOp;
    }

    rewriter.replaceOp(op, {output});
    return success();
  }
};

} // namespace

void populateLoweringONNXMatMulOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
