/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ScatterElements.cpp - Lowering ScatterElements Op ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterElements Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterElementsOpLowering
    : public OpConversionPattern<ONNXScatterElementsOp> {
  ONNXScatterElementsOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCollapse)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXScatterElementsOp::getOperationName());
    // Not and-ed with this->enableParallel: see Slice.cpp for why the two bools
    // stay independent.
    this->enableCollapse = enableCollapse;
  }
  bool enableParallel = false;
  bool enableCollapse = false;

  LogicalResult matchAndRewrite(ONNXScatterElementsOp scatterElementsOp,
      ONNXScatterElementsOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scatterElementsOp.getOperation();
    Location loc = ONNXLoc<ONNXScatterElementsOp>(op);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Operands and attributes.
    Value data = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    int64_t axis = adaptor.getAxis();
    int64_t dataRank = mlir::cast<MemRefType>(data.getType()).getRank();
    int64_t updatesRank = mlir::cast<MemRefType>(updates.getType()).getRank();
    int64_t indicesRank = mlir::cast<MemRefType>(indices.getType()).getRank();
    assert(updatesRank == dataRank && indicesRank == dataRank &&
           "All input tensors must have the same rank");
    StringRef reduction = adaptor.getReduction();

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + dataRank : axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);

    // Step1: make `output` hold the values of `data`, either by scattering
    // into the buffer of `data` itself, or by copying `data` into a new one.
    Value output;
    if (canWriteInPlace(rewriter, op, /*operandIndex=*/0, outputMemRefType)) {
      output = data;
    } else {
      output = create.mem.alignedAlloc(outputMemRefType, dataDims);
      emitMemcpy(rewriter, loc, op, output, data, dataDims, enableParallel,
          "scatterElements copy");
    }

    // Step2: scatter the updates array into the output array.
    //   index = indices[i][j]...[n]
    //   val = updates[i][j]...[n]
    //   output[i][j]..[index]..[n] = val (index used at position axis)
    //
    ValueRange loopDef = create.krnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LitIE(0)), ubs;
    create.krnlIE.getShapeAsDims(updates, ubs);
    // Enable parallelism if required.
    // An update at position p is stored at p with p[axis] replaced by its
    // index, so two iterations that differ at any level other than `axis`
    // store to distinct output elements. With reduction "none" the spec
    // requires indices to have no duplicate entries, so iterations that differ
    // only at `axis` store to distinct elements too, and the whole nest is
    // order-independent. With a reduction, duplicate indices are allowed and
    // iterations that differ only at `axis` may read-modify-write the same
    // element, so the `axis` level is excluded: it then runs sequentially
    // inside each parallel iteration, in its original order.
    //
    // bodyCost 10: one innermost iteration loads an index, optionally
    // normalizes it with a compare and a select, loads the update, and stores
    // through a non-affine access function.
    SmallVector<int64_t, 1> exclusiveDims;
    if (reduction != "none")
      exclusiveDims.emplace_back(axis);
    KrnlParallelPlan plan(loopDef, enableCollapse, /*parFirstInclusiveDim=*/0,
        /*parLastExclusiveDim=*/updatesRank,
        /*collapseLastExclusiveDim=*/updatesRank,
        {.minTripCountForParallel = 4, .bodyCost = 10}, exclusiveDims);
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "scatterElements", lbs, ubs);
    create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Insert code inside the loop.
          IndexExprScope innerLoopScope(createKrnl);

          // Access function for updates and indices.
          SmallVector<IndexExpr, 4> accessFct;
          getIndexExprList<DimIndexExpr>(loopInd, accessFct);

          Value updateVal = createKrnl.loadIE(updates, accessFct);
          Value indexVal = createKrnl.loadIE(indices, accessFct);
          IndexExpr index = NonAffineIndexExpr(indexVal);

          // When index may be negative, add axis dim to it.
          if (indicesMayBeNegative) {
            LiteralIndexExpr zero(0);
            SymbolIndexExpr axisDim(dataDims[axis]);
            index = index.selectOrSelf(index < zero, index + axisDim);
          }

          // Access function for the output.
          SmallVector<IndexExpr, 4> outputAccessFct;
          for (int i = 0; i < dataRank; ++i)
            outputAccessFct.emplace_back((i == axis) ? index : accessFct[i]);

          // Scatter updateVal into the output tensor with the specified
          // reduction.
          Value result = updateVal;
          if (reduction == "add") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.add(current, updateVal);
          } else if (reduction == "mul") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.mul(current, updateVal);
          } else if (reduction == "max") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.max(current, updateVal);
          } else if (reduction == "min") {
            Value current = createKrnl.loadIE(output, outputAccessFct);
            result = create.math.min(current, updateVal);
          } else if (reduction != "none") {
            llvm_unreachable("Unknown reduction type");
          }
          createKrnl.storeIE(result, output, outputAccessFct);
        });

    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXScatterElementsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    bool enableCollapse) {
  patterns.insert<ONNXScatterElementsOpLowering>(
      typeConverter, ctx, enableParallel, enableCollapse);
}

} // namespace onnx_mlir
