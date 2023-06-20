/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Unique.cpp - Lowering Unique Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unique Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Runtime/OMTensor.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"
using namespace mlir;

namespace onnx_mlir {

struct ONNXUniqueOpLowering : public ConversionPattern {
  ONNXUniqueOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXUniqueOp::getOperationName(), 1, ctx) {}

  ///
  /// Intermediate data are presented below for better understanding:
  ///
  /// there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  /// A: [[1, 1], [1, 1]],
  ///    [[0, 1], [0, 1]],
  ///    [[2, 1], [2, 1]],
  ///    [[0, 1], [0, 1]].
  ///
  /// there are 3 unique subtensors:
  /// [[1, 1], [1, 1]],
  /// [[0, 1], [0, 1]],
  /// [[2, 1], [2, 1]].
  ///
  /// sorted unique subtensors:
  /// B: [[0, 1], [0, 1]],
  ///    [[1, 1], [1, 1]],
  ///    [[2, 1], [2, 1]].
  ///
  /// output_Y is constructed from B:
  /// [[[0. 1.], [1. 1.], [2. 1.]],
  /// [[0. 1.], [1. 1.], [2. 1.]]]
  ///
  /// output_indices is to map from B to A:
  /// [1, 0, 2]
  ///
  /// output_inverse_indices is to map from A to B:
  /// [1, 0, 2, 0]
  ///
  /// output_counts = [2 1 1]
  ///

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXUniqueOp uniqueOp = llvm::cast<ONNXUniqueOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);
    ONNXUniqueOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    Value X = operandAdaptor.getX();
    ArrayRef<int64_t> xShape = getShape(X.getType());
    Type elementType = X.getType().cast<MemRefType>().getElementType();
    int64_t rank = create.krnlIE.getShapedTypeRank(X);
    int64_t sorted = operandAdaptor.getSorted();
    Optional<int64_t> optionalAxis = uniqueOp.getAxis();
    int64_t axis = -1;
    if (optionalAxis.has_value()) {
      axis = optionalAxis.value();
      axis = axis < 0 ? axis + rank : axis;
      assert(axis >= 0 && axis < rank && "axis is out of bound");
    }
#if 0
    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();
#endif
    // Count unique subtensors of X along axis.
    Type indexTy = rewriter.getIndexType();
    Value iZero = create.math.constantIndex(0);
    // Emit a variable for the total number of nonzero values.
    Value uniqueCount = create.mem.alloca(MemRefType::get({}, indexTy));
    create.krnl.store(iZero, uniqueCount, {});
    Value noneValue;
    //printf("YYYYY: axis=%ld\n", axis);
    emitArgUnique(rewriter, loc, uniqueCount, X, axis, /*sorted=*/sorted,
        noneValue, noneValue, noneValue, noneValue, /*count_only=*/true);
    // Calculate output shapes for ouputs according to the results
    Value total = create.krnl.load(uniqueCount, {});
    NonAffineIndexExpr totalDimExpr = DimIndexExpr(total);
    DimsExpr outputYDims;
    DimsExpr outputIndexDims;
    DimsExpr outputInverseIndexDims;
    if (axis < 0) {
      outputYDims.emplace_back(totalDimExpr);
      outputIndexDims.emplace_back(totalDimExpr);
      DimIndexExpr inputDimExpr = LiteralIndexExpr(xShape[0]);
      for (int64_t i = 1; i < rank; i++) {
        inputDimExpr = inputDimExpr * LiteralIndexExpr(xShape[i]);
      }
      outputInverseIndexDims.emplace_back(inputDimExpr);
    } else {
      printf("BBBBBB rank=%ld, axis=%ld\n", rank, axis);
      for (int64_t i = 0; i < rank; i++) {
        DimIndexExpr tDimExpr = LiteralIndexExpr(xShape[i]);
        if (i == axis)
          tDimExpr = totalDimExpr;
        outputYDims.emplace_back(tDimExpr);
      }
      outputIndexDims.emplace_back(totalDimExpr);
      outputInverseIndexDims.emplace_back(LiteralIndexExpr(xShape[axis]));
    }

    // Insert an allocation and deallocation for the results of this operation.
    // For Y output
    Value outputY;
    if (axis < 0) {
      MemRefType memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
      outputY = create.mem.alignedAlloc(memrefType, outputYDims);
    } else {
      ArrayRef<int64_t> xShape = getShape(X.getType());
      SmallVector<int64_t> yShape;
      for (int i = 0; i < rank; i++)
        yShape.emplace_back((i == axis) ? ShapedType::kDynamic : xShape[i]);
      MemRefType memrefType = MemRefType::get(yShape, elementType);
      outputY = create.mem.alignedAlloc(memrefType, outputYDims);
    }
    Type i64Type = rewriter.getI64Type();
    MemRefType memrefType = MemRefType::get({ShapedType::kDynamic}, i64Type);
    Value indices = create.mem.alignedAlloc(memrefType, outputIndexDims);
    Value inverse_indices = create.mem.alignedAlloc(memrefType,
        outputInverseIndexDims);
    Value counts = create.mem.alignedAlloc(memrefType, outputIndexDims);
    // Compute argUnique of X along axis.
    create.krnl.store(iZero, uniqueCount, {});
#if 0
    llvm::dbgs() << "XXXX BEFORE calling emitArgUnique: [\noutputY=" << outputY <<
      ",\n indices=" << indices << ",\n inverse_indices=" << inverse_indices <<
      ",\n counts=" << counts << "]\n";
#endif
    emitArgUnique(rewriter, loc, uniqueCount, X, axis, /*sorted=*/sorted,
        outputY, indices, inverse_indices, counts);
#if 0
    llvm::dbgs() << "XXXX AFTER calling emitArgUnique: [\noutputY=" << outputY <<
      ",\n indices=" << indices << ",\n inverse_indices=" << inverse_indices <<
      ",\n counts=" << counts << "]\n";
#endif
    rewriter.replaceOp(op, {outputY, indices, inverse_indices, counts});
    return success();
  }
};

void populateLoweringONNXUniqueOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUniqueOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
