/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- GatherElements.cpp - Lowering GatherElements Op ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GatherElements Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherElementsOpLowering
    : public OpConversionPattern<ONNXGatherElementsOp> {
  ONNXGatherElementsOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXGatherElementsOp gatherElementsOp,
      ONNXGatherElementsOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gatherElementsOp.getOperation();
    Location loc = ONNXLoc<ONNXGatherElementsOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXGatherElementsOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the result of this operation.
    Value output =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Operands and attributes.
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axis = adaptor.getAxis();
    int64_t dataRank = mlir::cast<MemRefType>(data.getType()).getRank();
    int64_t indicesRank = mlir::cast<MemRefType>(indices.getType()).getRank();
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(indicesRank == dataRank && "Input tensors must have the same rank");
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + dataRank : axis;

    DimsExpr dataDims, indicesDims;
    create.krnlIE.getShapeAsDims(data, dataDims);
    create.krnlIE.getShapeAsDims(indices, indicesDims);

    // Gather elements from the 'data' tensor, store them into the output.
    //   index = indices[i][j]...[n]
    //   output[i][j]...[n] = data[i][j]..[index]..[n] (index used at axis dim.)
    //
    ValueRange loopDef = create.krnl.defineLoops(indicesRank);
    DimsExpr lbs(indicesRank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, indicesDims,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for indices and output.
          DimsExpr accessFct;
          getIndexExprList<DimIndexExpr>(loopInd, accessFct);

          // Compute index = indices[i][j]...[n]
          Value indexVal = createKrnl.loadIE(indices, accessFct);
          IndexExpr index = NonAffineIndexExpr(indexVal);

          if (indicesMayBeNegative) {
            LiteralIndexExpr zero(0);
            SymbolIndexExpr axisDim(dataDims[axis]);
            index = index.selectOrSelf(index < zero, index + axisDim);
          }

          // Check the dynamic requirement of GatherElement Op
          // Refer to the comments in Gather.cpp
          if (enableSafeCodeGen) {
            // From onnx document:
            // All index values are expected to be within bounds [-s, s-1]
            // along axis of size s. It is an error if any of the index values
            // are out of bounds.
            // After the negative correction, the range should be [0, s-1]
            Value upperBound = create.mem.dim(data, axis);
            Value compareUpperBound =
                create.math.slt(index.getValue(), upperBound);
            std::string nodeNameStr = op->getName().getStringRef().str() + " ";
            StringAttr nodeName =
                op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
            if (nodeName && !nodeName.getValue().empty()) {
              nodeNameStr = nodeNameStr + nodeName.getValue().str();
            }
            rewriter.create<cf::AssertOp>(loc, compareUpperBound,
                "indices of GatherOp is larger than the upper bound");
            LiteralIndexExpr zero(0);
            Value compareLowerBound =
                create.math.sge(index.getValue(), zero.getValue());
            rewriter.create<cf::AssertOp>(loc, compareLowerBound,
                "indices of GatherOp is less than the lower bound");
          }

          // Access function for the 'data' tensor.
          DimsExpr dataAccessFct;
          for (int64_t i = 0; i < dataRank; ++i)
            dataAccessFct.emplace_back((i == axis) ? index : accessFct[i]);

          // Gather values from the 'data' tensor and save them.
          Value dataVal = createKrnl.loadIE(data, dataAccessFct);
          createKrnl.storeIE(dataVal, output, accessFct);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXGatherElementsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGatherElementsOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
