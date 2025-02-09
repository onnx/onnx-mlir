/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXGatherOpLowering : public OpConversionPattern<ONNXGatherOp> {
  ONNXGatherOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXGatherOp::getOperationName());
  }
  bool enableParallel;

  LogicalResult matchAndRewrite(ONNXGatherOp gatherOp,
      ONNXGatherOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gatherOp.getOperation();
    Location loc = ONNXLoc<ONNXGatherOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXGatherOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Operands and attributes.
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axisLit = adaptor.getAxis();
    int64_t dataRank = mlir::cast<MemRefType>(data.getType()).getRank();
    int64_t indicesRank = mlir::cast<MemRefType>(indices.getType()).getRank();

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

    int64_t outputRank = shapeHelper.getOutputDims().size();
    int iIndexStart = 0;
    int jIndexStart = iIndexStart + axisLit;
    int kIndexStart = jIndexStart + indicesRank - (axisLit + 1);

    LiteralIndexExpr zeroIE(0);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);

    /*
      The pattern that we are using is that of numpy.take.

      Ni, Nk = data.shape[:axis], data.shape[axis+1:]
      Nj = indices.shape
      for ii in ndindex(Ni):
        for jj in ndindex(Nj):
          for kk in ndindex(Nk):
            out[ii + jj + kk] = data[ii + (indices[jj],) + kk]
    */
    // Define loops and iteration trip counts (equivalent to size of output)
    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    DimsExpr lbs(outputRank, zeroIE);
    DimsExpr ubs = shapeHelper.getOutputDims();
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, ubs, 0, outputRank, parId)) {
        create.krnl.parallel(loopDef[parId]);
        onnxToKrnlParallelReport(
            op, true, parId, lbs[parId], ubs[parId], "gather");
      } else {
        onnxToKrnlParallelReport(
            op, false, -1, -1, "dim with not enough work in gather");
      }
    }
    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);
          SymbolIndexExpr axisDim(dataDims[axisLit]);

          // compute the loop indices for the output
          SmallVector<IndexExpr, 4> outputAccessFct;
          getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

          // Compute access function for indices[jjs].
          SmallVector<IndexExpr, 4> indicesAccessFct;
          for (int j = 0; j < indicesRank; ++j)
            indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
          Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
          // Loaded value is an index that is not affine
          IndexExpr index = NonAffineIndexExpr(indexVal);
          // When index may be negative, add axis Dim to it.
          if (indicesMayBeNegative)
            index = index.selectOrSelf(index < zeroIE, index + axisDim);

          // The Gather op is data dependent: the value of index should be
          // within the input data size.
          // Add runtime check if enableSafeCodeGen is set true
          // Implementation comments vs. createGenerateRuntimeVerificationPass
          // This check is according to onnx op semantics, not general bound
          // check for memref. Implementation of RuntimeVerification could be
          // borrowed. Slightly difference is that onnx semenatics check is for
          // each dimension independently, not the final address is within
          // the memref bound.
          if (enableSafeCodeGen) {
            // From onnx document:
            // All index values are expected to be within bounds [-s, s-1]
            // along axis of size s. It is an error if any of the index values
            // are out of bounds.
            // After the negative correction, the range should be [0, s-1]
            Value upperBound = create.mem.dim(data, axisLit);
            Value compareUpperBound =
                create.math.slt(index.getValue(), upperBound);
            // Report onnx_node_name if the op has the attribute
            std::string nodeNameStr = op->getName().getStringRef().str() + " ";
            StringAttr nodeName =
                op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
            if (nodeName && !nodeName.getValue().empty()) {
              nodeNameStr = nodeNameStr + nodeName.getValue().str();
            }
            rewriter.create<cf::AssertOp>(loc, compareUpperBound,
                nodeNameStr +
                    " indices of GatherOp is larger than the upper bound");
            Value compareLowerBound =
                create.math.sge(index.getValue(), zeroIE.getValue());
            rewriter.create<cf::AssertOp>(loc, compareLowerBound,
                nodeNameStr +
                    " indices of GatherOp is less than the lower bound");
          }

          // Compute access function of data: data[ii + (indices[jj],) + kk]
          SmallVector<IndexExpr, 4> dataAccessFct;
          // First add indices iis
          for (int i = 0; i < axisLit; ++i)
            dataAccessFct.emplace_back(outputAccessFct[iIndexStart + i]);
          // Then add indices[jj] (indexVal).
          dataAccessFct.emplace_back(index);
          // Then add kks.
          for (int k = axisLit + 1; k < dataRank; ++k)
            dataAccessFct.emplace_back(outputAccessFct[kIndexStart + k]);
          Value dataVal = createKrnl.loadIE(data, dataAccessFct);

          // Save data into output
          createKrnl.storeIE(dataVal, alloc, outputAccessFct);
        });
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXGatherOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXGatherOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
