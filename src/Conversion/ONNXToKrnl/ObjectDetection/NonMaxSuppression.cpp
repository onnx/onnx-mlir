/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- NonMaxSuppression.cpp - Lowering NonMaxSuppression Op ----------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonMaxSuppression Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Compute the intersection-over-union (IOU) score between two boxes.
/// IOU tells us how much two boxes are overlapped.
static Value emitIOU(MathBuilder &createMath, SmallVectorImpl<Value> &box1,
    SmallVectorImpl<Value> &box2, int64_t centerPointBox = 0) {
  Value area1, area2;
  Value y1_min, x1_min, y1_max, x1_max;
  Value y2_min, x2_min, y2_max, x2_max;
  if (centerPointBox == 0) {
    // The box data is supplied as [y1, x1, y2, x2]. (y1, x1) and (y2, x2)
    // are the coordinates of the diagonal pair of bottom-left and top-right
    // corners.
    y1_min = box1[0];
    x1_min = box1[1];
    y1_max = box1[2];
    x1_max = box1[3];

    y2_min = box2[0];
    x2_min = box2[1];
    y2_max = box2[2];
    x2_max = box2[3];

    area1 = createMath.mul(
        createMath.sub(y1_max, y1_min), createMath.sub(x1_max, x1_min));
    area2 = createMath.mul(
        createMath.sub(y2_max, y2_min), createMath.sub(x2_max, x2_min));
  } else {
    // The box data is supplied as [x_center, y_center, width, height].
    Value x1_center = box1[0];
    Value y1_center = box1[1];
    Value w1 = box1[2];
    Value h1 = box1[3];

    Value x2_center = box2[0];
    Value y2_center = box2[1];
    Value w2 = box2[2];
    Value h2 = box2[3];

    Value two = createMath.constant(w1.getType(), 2);
    x1_min = createMath.sub(x1_center, createMath.div(w1, two));
    x1_max = createMath.add(x1_center, createMath.div(w1, two));
    y1_min = createMath.sub(y1_center, createMath.div(h1, two));
    y1_max = createMath.add(y1_center, createMath.div(h1, two));

    y2_min = createMath.sub(y2_center, createMath.div(h2, two));
    y2_max = createMath.add(y2_center, createMath.div(h2, two));
    x2_min = createMath.sub(x2_center, createMath.div(w2, two));
    x2_max = createMath.add(x2_center, createMath.div(w2, two));

    area1 = createMath.mul(h1, w1);
    area2 = createMath.mul(h2, w2);
  }

  Value intersection_x_min = createMath.max(x1_min, x2_min);
  Value intersection_y_min = createMath.max(y1_min, y2_min);
  Value intersection_x_max = createMath.min(x1_max, x2_max);
  Value intersection_y_max = createMath.min(y1_max, y2_max);

  Value zero = createMath.constant(intersection_x_min.getType(), 0);
  Value intersection_w = createMath.sub(intersection_x_max, intersection_x_min);
  Value intersection_h = createMath.sub(intersection_y_max, intersection_y_min);
  Value intersection_area = createMath.mul(createMath.max(intersection_w, zero),
      createMath.max(intersection_h, zero));

  Value union_area = createMath.add(area1, area2);
  union_area = createMath.sub(union_area, intersection_area);
  // Avoid zero division.
  Value epsilon = createMath.constant(zero.getType(), 1e-8);
  union_area = createMath.add(union_area, epsilon);
  return createMath.div(intersection_area, union_area);
}

/// Suppress the number of output bounding boxes per class by scores.
static void suppressByScores(ConversionPatternRewriter &rewriter, Location loc,
    Value scores, Value scoreThreshold, Value maxOutputPerClass) {

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
      MemRefBuilder>
      create(rewriter, loc);
  IndexExprScope scope(create.krnl);
  Type indexType = rewriter.getIndexType();

  IndexExpr bsIE = create.krnlIE.getShapeAsDim(scores, 0); // batch size.
  IndexExpr csIE = create.krnlIE.getShapeAsDim(scores, 1); // class size.
  IndexExpr ssIE = create.krnlIE.getShapeAsDim(scores, 2); // spatial size.
  Value bs = bsIE.getValue();
  Value cs = csIE.getValue();
  Value ss = ssIE.getValue();
  Value zero = create.math.constantIndex(0);
  Value one = create.math.constantIndex(1);
  // Store the number of scores whose value is greater than the threshold.
  // Scalar, ok to use alloca.
  Value topk = create.mem.alloca(MemRefType::get({}, indexType));

  // Compute the effective max output per class.
  Value effectiveMaxPerClass =
      create.mem.alloca(MemRefType::get({}, indexType));
  create.krnl.store(zero, effectiveMaxPerClass);

  ValueRange bcLoopDef = create.krnl.defineLoops(2);
  create.krnl.iterate(bcLoopDef, bcLoopDef, {zero, zero}, {bs, cs},
      [&](const KrnlBuilder &createKrnl, ValueRange bcLoopInd) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
            createKrnl);
        Value b(bcLoopInd[0]), c(bcLoopInd[1]);

        // Reset the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        create.krnl.store(zero, topk);

        // Count the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        ValueRange sLoopDef = create.krnl.defineLoops(1);
        create.krnl.iterate(sLoopDef, sLoopDef, {zero}, {ss},
            [&](const KrnlBuilder &createKrnl, ValueRange sLoopInd) {
              Value s(sLoopInd[0]);
              MathBuilder createMath(createKrnl);

              Value score = createKrnl.load(scores, {b, c, s});
              // Increase the counter if score > threshold.
              Value gt = createMath.sgt(score, scoreThreshold);
              Value topkVal = createKrnl.load(topk);
              Value topkPlusOneVal = createMath.add(topkVal, one);
              topkVal = createMath.select(gt, topkPlusOneVal, topkVal);
              createKrnl.store(topkVal, topk);
            });

        // Update the effective max output per class.
        Value x = create.krnl.load(topk);
        Value y = create.krnl.load(effectiveMaxPerClass);
        create.krnl.store(create.math.max(x, y), effectiveMaxPerClass);
      });

  // Suppress the number of output bounding boxes per class.
  Value x = create.krnl.load(maxOutputPerClass);
  Value y = create.krnl.load(effectiveMaxPerClass);
  create.krnl.store(create.math.min(x, y), maxOutputPerClass);
}

/// Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
/// flip the flipped boxes back.
/// BoundingBoxes: [num_of_batch, spatial_dimension, 4]
static Value tryToUnflip(
    ConversionPatternRewriter &rewriter, Location loc, Value boundingBoxes) {
  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
      create(rewriter, loc);
  SmallVector<IndexExpr, 4> ubs;
  create.krnlIE.getShapeAsDims(boundingBoxes, ubs);
  IndexExpr bs = ubs[0]; // batch size.
  IndexExpr ss = ubs[1]; // spatial size.
  LiteralIndexExpr zeroIE(0), oneIE(1), twoIE(2), threeIE(3);

  Value resMemRef = create.mem.alignedAlloc(
      mlir::cast<MemRefType>(boundingBoxes.getType()), ubs);

  ValueRange loopDef = create.krnl.defineLoops(2);
  create.krnl.iterateIE(loopDef, loopDef, {zeroIE, zeroIE}, {bs, ss},
      [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
        MathBuilder createMath(createKrnl);
        DimIndexExpr b(loopInd[0]), s(loopInd[1]);
        // Load a bounding box.
        Value y_min = createKrnl.loadIE(boundingBoxes, {b, s, zeroIE});
        Value x_min = createKrnl.loadIE(boundingBoxes, {b, s, oneIE});
        Value y_max = createKrnl.loadIE(boundingBoxes, {b, s, twoIE});
        Value x_max = createKrnl.loadIE(boundingBoxes, {b, s, threeIE});

        // Flip x.
        Value gtX = createMath.sgt(x_min, x_max);
        Value newXMin = createMath.select(gtX, x_max, x_min);
        Value newXMax = createMath.select(gtX, x_min, x_max);

        // Flip y.
        Value gtY = createMath.sgt(y_min, y_max);
        Value newYMin = createMath.select(gtY, y_max, y_min);
        Value newYMax = createMath.select(gtY, y_min, y_max);

        // Update the bounding box.
        createKrnl.storeIE(newYMin, resMemRef, {b, s, zeroIE});
        createKrnl.storeIE(newXMin, resMemRef, {b, s, oneIE});
        createKrnl.storeIE(newYMax, resMemRef, {b, s, twoIE});
        createKrnl.storeIE(newXMax, resMemRef, {b, s, threeIE});
      });
  return resMemRef;
}

struct ONNXNonMaxSuppressionOpLowering
    : public OpConversionPattern<ONNXNonMaxSuppressionOp> {
  ONNXNonMaxSuppressionOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  /// To understand how code is generated for NonMaxSuppression, look at the
  /// python implementation at the end of this file.
  LogicalResult matchAndRewrite(ONNXNonMaxSuppressionOp nmsOp,
      ONNXNonMaxSuppressionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = nmsOp.getOperation();
    Location loc = ONNXLoc<ONNXNonMaxSuppressionOp>(op);

    // Builder helper.
    IndexExprScope mainScope(&rewriter, loc);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    // Common information.
    Type elementType = memRefType.getElementType();
    Type indexType = rewriter.getIndexType();
    Type boolType = rewriter.getI1Type();
    Type i64Type = rewriter.getI64Type();

    // Operation's operands.
    // Bounding boxes.
    Value boxes = adaptor.getBoxes();
    // Scores.
    Value scores = adaptor.getScores();
    // Maximum number of output boxes per class.
    Value maxOutputBoxPerClass = getOptionalScalarValue(
        rewriter, loc, adaptor.getMaxOutputBoxesPerClass(), i64Type, 0);
    // Score threshold.
    Type scoreType = mlir::cast<MemRefType>(scores.getType()).getElementType();
    Value scoreTH = getOptionalScalarValue(
        rewriter, loc, adaptor.getScoreThreshold(), scoreType, 0);
    // IOU threshold.
    Value iouTH = getOptionalScalarValue(
        rewriter, loc, adaptor.getIouThreshold(), scoreType, 0);
    // Mode: diagonal corners or center point.
    int64_t centerPointBox = nmsOp.getCenterPointBox();

    // boxes: [num_of_batch, spatial_dimension, 4]
    // scores: [num_of_batch, num_of_class, spatial_dimension]
    IndexExpr bsIE = create.krnlIE.getShapeAsDim(scores, 0); // batch size.
    IndexExpr csIE = create.krnlIE.getShapeAsDim(scores, 1); // class size.
    IndexExpr ssIE = create.krnlIE.getShapeAsDim(scores, 2); // spatial size.
    Value bs = bsIE.getValue();
    Value cs = csIE.getValue();
    Value ss = ssIE.getValue();

    // Frequently used constants.
    Value zero = create.math.constantIndex(0);
    Value one = create.math.constantIndex(1);
    Value two = create.math.constantIndex(2);
    Value three = create.math.constantIndex(3);
    Value falseVal = create.math.constant(boolType, 0);
    Value trueVal = create.math.constant(boolType, 1);

    // Refine the number of output boxes per class by suppressing it using
    // spatial dimension size and score threshold.
    // Scalar, ok to use alloca.
    Value maxOutputPerClass = create.mem.alloca(MemRefType::get({}, indexType));
    // 1. Suppress by using spatial dimension size.
    Value x = create.math.castToIndex(maxOutputBoxPerClass);
    create.krnl.store(create.math.min(x, ss), maxOutputPerClass);
    // 2. Suppress by score threshold.
    suppressByScores(rewriter, loc, scores, scoreTH, maxOutputPerClass);
    Value MOPC = create.krnl.load(maxOutputPerClass);

    // Sort scores in the descending order.
    Value order = emitArgSort(rewriter, loc, scores, /*axis=*/2,
        /*ascending=*/false);

    // Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
    // unflip the flipped boxes.
    if (centerPointBox == 0)
      boxes = tryToUnflip(rewriter, loc, boxes);

    // The total number of output selected indices.
    IndexExpr numSelectedIndicesIE = bsIE * csIE * DimIE(MOPC);

    // Allocate a MemRef for the output. This MemRef is NOT the final output
    // since the number of selected indices has yet not suppressed by IOU. So
    // the first dimension size is larger than necessary.
    // Output shape : [num_selected_indices, 3]
    SmallVector<IndexExpr, 2> outputDims = {numSelectedIndicesIE, LitIE(3)};
    SmallVector<int64_t, 2> outputShape;
    if (numSelectedIndicesIE.isLiteral())
      outputShape.emplace_back(numSelectedIndicesIE.getLiteral());
    else
      outputShape.emplace_back(ShapedType::kDynamic);
    outputShape.emplace_back(3);
    Value selectedMemRef = create.mem.alignedAlloc(
        MemRefType::get(outputShape, indexType), outputDims);
    // Initialize with value -1.
    create.krnl.memset(selectedMemRef, create.math.constantIndex(-1));

    // Effective number of selected indices. This is the final value for the 1st
    // dim of the output, which is suppressed by IOU during computation and
    // cannot be computed in advance.
    // Final output shape : [effective_num_selected_indices, 3]
    // Scalar, ok to use alloca.
    Value effectiveNumSelectedIndices =
        create.mem.alloca(MemRefType::get({}, indexType));
    create.krnl.store(zero, effectiveNumSelectedIndices);

    // Suppress by using IOU.
    // Iterate over all bounding boxes in the descending order of scores.
    Value effectiveMaxOutputPerClass =
        create.mem.alloca(MemRefType::get({}, indexType));
    ValueRange bcLoopDef = create.krnl.defineLoops(2);
    create.krnl.iterate(bcLoopDef, bcLoopDef, {zero, zero}, {bs, cs},
        [&](const KrnlBuilder &createKrnl, ValueRange bcLoopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
              createKrnl);
          // Keep trace of the number of output boxes per class.
          create.krnl.store(zero, effectiveMaxOutputPerClass);
          // Keep trace of removed indices per class.
          DimIndexExpr ssIE(ss);
          SmallVector<IndexExpr, 1> dims = {ssIE};
          SmallVector<int64_t, 1> shapes = {ShapedType::kDynamic};
          if (ssIE.isLiteral())
            shapes[0] = ssIE.getLiteral();
          Value removedIndices =
              create.mem.alignedAlloc(MemRefType::get(shapes, boolType), dims);
          create.krnl.memset(removedIndices, falseVal);

          // Iterate in the descending order of scores.
          ValueRange sLoopDef = create.krnl.defineLoops(1);
          create.krnl.iterate(sLoopDef, sLoopDef, {zero}, {ss},
              [&](const KrnlBuilder &createKrnl, ValueRange sLoopInd) {
                Value b(bcLoopInd[0]), c(bcLoopInd[1]), s(sLoopInd[0]);
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);

                // Index of the bounding box with the largest score.
                Value selectedBI = create.krnl.load(order, {b, c, s});

                // Check conditions to select a bounding box.
                // 1. Only bounding boxes whose score > score_threshold.
                Value score = create.krnl.load(scores, {b, c, selectedBI});
                Value checkScore = create.math.sgt(score, scoreTH);
                // 2. Have not yet got enough outputs.
                Value currentMOPC =
                    create.krnl.load(effectiveMaxOutputPerClass);
                Value checkMOPC = create.math.slt(currentMOPC, MOPC);
                // 3. Bounding box has not yet been removed.
                Value isRemoved =
                    create.krnl.load(removedIndices, {selectedBI});
                Value isNotRemoved = create.math.eq(isRemoved, falseVal);

                // Only proceed if the box satisfies the above conditions.
                Value canSelectBox = create.math.andi(
                    create.math.andi(checkScore, checkMOPC), isNotRemoved);
                auto ifOp = rewriter.create<scf::IfOp>(
                    loc, canSelectBox, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(
                    &ifOp.getThenRegion().front());

                // Select the bounding box with the largest score.
                SmallVector<Value, 4> selectedBox;
                for (int i = 0; i < 4; ++i) {
                  Value iVal = create.math.constantIndex(i);
                  Value x = create.krnl.load(boxes, {b, selectedBI, iVal});
                  selectedBox.emplace_back(x);
                }

                // Store the index of the selected box to the output.
                // out_index = effective_num_selected_indices
                // selected_indices[out_index] = [b, c, selected_box_index]
                Value soVal = create.krnl.load(effectiveNumSelectedIndices);
                create.krnl.store(b, selectedMemRef, {soVal, zero});
                create.krnl.store(c, selectedMemRef, {soVal, one});
                create.krnl.store(selectedBI, selectedMemRef, {soVal, two});

                // Update the number of output boxes per class.
                // effective_max_output_per_class += 1
                create.krnl.store(create.math.add(currentMOPC, one),
                    effectiveMaxOutputPerClass, {});

                // Update the effective number of selected indices.
                // effective_num_selected_indices += 1
                create.krnl.store(create.math.add(soVal, one),
                    effectiveNumSelectedIndices, {});

                // Remove boxes overlapped too much with the selected box,
                // using IOU.
                ValueRange oLoopDef = create.krnl.defineLoops(1);
                create.krnl.iterate(oLoopDef, oLoopDef, {zero}, {ss},
                    [&](const KrnlBuilder &createKrnl, ValueRange oLoopInd) {
                      Value o(oLoopInd[0]);
                      MathBuilder createMath(createKrnl);

                      // Only proceed if a box has not yet been removed.
                      Value isRemoved = createKrnl.load(removedIndices, {o});
                      Value isNotRemoved = createMath.eq(isRemoved, falseVal);
                      auto if1Op = rewriter.create<scf::IfOp>(
                          loc, isNotRemoved, /*withElseRegion=*/false);
                      rewriter.setInsertionPointToStart(
                          &if1Op.getThenRegion().front());

                      // Pick the current box.
                      SmallVector<Value, 4> otherBox;
                      for (int i = 0; i < 4; ++i) {
                        Value iVal = createMath.constantIndex(i);
                        Value x = createKrnl.load(boxes, {b, o, iVal});
                        otherBox.emplace_back(x);
                      }

                      // Compute IOU between the selected and current boxes.
                      Value iou = emitIOU(
                          createMath, selectedBox, otherBox, centerPointBox);

                      // Only proceed if IOU >= iou_threshold.
                      Value checkIOU = createMath.sge(iou, iouTH);
                      auto if2Op = rewriter.create<scf::IfOp>(
                          loc, checkIOU, /*withElseRegion=*/false);
                      rewriter.setInsertionPointToStart(
                          &if2Op.getThenRegion().front());

                      // If IOU is satisfied, mark the current box as removed.
                      createKrnl.store(trueVal, removedIndices, {o});
                    });
              });
        });

    // Insert allocation and deallocation for the final output.
    Value effectiveNSI = create.krnl.load(effectiveNumSelectedIndices);
    SmallVector<IndexExpr, 2> resDims = {DimIE(effectiveNSI), LitIE(3)};
    Value resMemRef = create.mem.alignedAlloc(
        MemRefType::get({ShapedType::kDynamic, 3}, elementType), resDims);

    // Copy data to the final ouput.
    ValueRange resLoopDef = create.krnl.defineLoops(2);
    create.krnl.iterate(resLoopDef, resLoopDef, {zero, zero},
        {effectiveNSI, three},
        [&](const KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          MathBuilder createMath(createKrnl);
          Value load = createKrnl.load(selectedMemRef, resLoopInd);
          Value res = createMath.cast(elementType, load);
          createKrnl.store(res, resMemRef, resLoopInd);
        });

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXNonMaxSuppressionOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXNonMaxSuppressionOpLowering>(typeConverter, ctx);
}

// clang-format off
// Below is a python implementation of NonMaxSuppression.
// import numpy as np
//
// def IOU(box1, box2, center_point_box=0):
//     if center_point_box == 0:
//         # The box data is supplied as [y1, x1, y2, x2]. (y1, x1) and (y2, x2)
//         # are the coordinates of the diagonal pair of bottom-left and top-right
//         # corners.
//         y1_min, x1_min, y1_max, x1_max = box1
//         y2_min, x2_min, y2_max, x2_max = box2
//
//         area1 = (y1_max - y1_min) * (x1_max - x1_min)
//         area2 = (y2_max - y2_min) * (x2_max - x2_min)
//     else:
//         # The box data is supplied as [x_center, y_center, width, height].
//         x1_center, y1_center, w1, h1 = box1
//         x2_center, y2_center, w2, h2 = box2
//
//         x1_min = x1_center - w1 / 2
//         x1_max = x1_center + w1 / 2
//         x2_min = x2_center - w2 / 2
//         x2_max = x2_center + w2 / 2
//
//         y1_min = y1_center - h1 / 2
//         y1_max = y1_center + h1 / 2
//         y2_min = y2_center - h2 / 2
//         y2_max = y2_center + h2 / 2
//
//         area1 = h1 * w1
//         area2 = h2 * w2
//
//     intersection_x_min = max(x1_min, x2_min)
//     intersection_y_min = max(y1_min, y2_min)
//     intersection_x_max = min(x1_max, x2_max)
//     intersection_y_max = min(y1_max, y2_max)
//     intersection_area = max(intersection_x_max - intersection_x_min, 0) * \
//         max(intersection_y_max - intersection_y_min, 0)
//
//     union_area = area1 + area2 - intersection_area + 1e-8
//     return intersection_area / union_area
//
//
// '''
// boxes :: [num_batch, spatial_dimension, 4]
// scores :: [num_batch, num_class, spatial_dimension]
// '''
//
//
// def nms(boxes, scores, max_output_boxes_per_class, iou_threshold,
//         score_threshold, center_point_box=0):
//     batch_size = scores.shape[0]
//     class_size = scores.shape[1]
//     spatial_size = boxes.shape[1]
//
//     score_threshold = score_threshold[0]
//     iou_threshold = iou_threshold[0]
//     # Suppress by spatial dimension.
//     max_output_per_class = min(spatial_size, max_output_boxes_per_class[0])
//     # Suppress by scores
//     max_per_class_by_score = 0
//     for b in range(batch_size):
//         for c in range(class_size):
//             topk = 0
//             for s in range(spatial_size):
//                 if scores[b, c, s] > score_threshold:
//                     topk += 1
//             max_per_class_by_score = max(max_per_class_by_score, topk)
//     max_output_per_class = min(
//         max_output_per_class, max_per_class_by_score)
//
//     # Sort scores in the descending order and get the sorted indices.
//     # order = np.argsort(-scores, axis=2)
//     order = np.full(scores.shape, -1)
//     for b in range(batch_size):
//         for c in range(class_size):
//             for i in range(spatial_size):
//                 order[b, c, i] = i
//     for b in range(batch_size):
//         for c in range(class_size):
//             for i in range(spatial_size - 1):
//                 for k in range(i+1, spatial_size):
//                      xOrd = order[b, c, i]
//                      yOrd = order[b, c, k]
//                      if (scores[b, c, xOrd] < scores[b, c, yOrd]):
//                          tmp = order[b, c, i]
//                          order[b, c, i] = order[b, c, k]
//                          order[b, c, k] = tmp
//
//
//     # Check if the coordinates are flipped. If so, flip them back.
//     if (center_point_box == 0):
//         new_boxes = np.empty(boxes.shape)
//         for b in range(batch_size):
//             for s in range(spatial_size):
//                 box = boxes[b, s]
//                 # Check whether the coordinates are flipped.
//                 y1_min, x1_min, y1_max, x1_max = box
//                 if (y1_min > y1_max):
//                     tmp = y1_min
//                     y1_min = y1_max
//                     y1_max = tmp
//                 if (x1_min > x1_max):
//                     tmp = x1_min
//                     x1_min = x1_max
//                     x1_max = tmp
//                 new_boxes[b, s] = [y1_min, x1_min, y1_max, x1_max]
//         boxes = new_boxes
//
//     # Output: [num_selected_indices, 3]
//     # The selected index format is [batch_index, class_index, box_index].
//     num_selected_indices = batch_size * max_output_per_class * class_size
//     selected_indices_shape = (num_selected_indices, 3)
//     selected_indices = np.full(selected_indices_shape, -1).astype(np.int64)
//     effective_num_selected_indices = 0
//     for b in range(batch_size):
//         for c in range(class_size):
//             effective_max_output_per_class = 0
//             removed_indices = np.full((spatial_size), False)
//             for s in range(spatial_size):
//                 # Index of the bounding box with the largest score.
//                 selected_box_index = order[b, c, s]
//
//                 # Discard bounding boxes using score threshold.
//                 if (scores[b, c, selected_box_index] <= score_threshold):
//                     continue
//                 # Have enough the number of outputs.
//                 if (effective_max_output_per_class >= max_output_per_class):
//                     continue
//                 # Removed, ignore.
//                 if removed_indices[selected_box_index]:
//                     continue
//
//                 # Pick the bounding box with the largest score.
//                 selected_box = boxes[b, selected_box_index, :]
//
//                 # Store the index of the picked box to the output.
//                 selected_indices[effective_num_selected_indices] = [b, c, selected_box_index]
//
//                 # Update counters.
//                 effective_max_output_per_class += 1
//                 effective_num_selected_indices += 1
//
//                 # Remove boxes overlapped too much with the selected box, using
//                 # IOU.
//                 for o in range(spatial_size):
//                     other_box = boxes[b, o, :]
//                     iou = IOU(selected_box, other_box, center_point_box)
//                     if (not removed_indices[o]) and (iou >= iou_threshold):
//                         removed_indices[o] = True
//                     else:
//                         removed_indices[o] = removed_indices[o]
//
//     # Since we cannot suppress by IOU in advance, so remove redundant score
//     # now.
//     res = np.empty((effective_num_selected_indices, 3))
//     for i in range(effective_num_selected_indices):
//         res[i] = selected_indices[i]
//     return res
//
//
// print("testing nonmaxsuppression_center_point_box_format")
// center_point_box = 1
// boxes = np.array([[
//     [0.5, 0.5, 1.0, 1.0],
//     [0.5, 0.6, 1.0, 1.0],
//     [0.5, 0.4, 1.0, 1.0],
//     [0.5, 10.5, 1.0, 1.0],
//     [0.5, 10.6, 1.0, 1.0],
//     [0.5, 100.5, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold, center_point_box)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_flipped_coordinates")
// boxes = np.array([[
//     [1.0, 1.0, 0.0, 0.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, 0.9, 1.0, -0.1],
//     [0.0, 10.0, 1.0, 11.0],
//     [1.0, 10.1, 0.0, 11.1],
//     [1.0, 101.0, 0.0, 100.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_identical_boxes")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array(
//     [[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_limit_output_size")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_single_box")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_suppress_by_IOU")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_suppress_by_IOU_and_scores")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.4]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_two_batches")
// boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
//                    [0.0, 0.1, 1.0, 1.1],
//                    [0.0, -0.1, 1.0, 0.9],
//                    [0.0, 10.0, 1.0, 11.0],
//                    [0.0, 10.1, 1.0, 11.1],
//                    [0.0, 100.0, 1.0, 101.0]],
//                   [[0.0, 0.0, 1.0, 1.0],
//                    [0.0, 0.1, 1.0, 1.1],
//                    [0.0, -0.1, 1.0, 0.9],
//                    [0.0, 10.0, 1.0, 11.0],
//                    [0.0, 10.1, 1.0, 11.1],
//                    [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
//                    [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array(
//     [[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
// print("testing nonmaxsuppression_two_classes")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
//                     [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array(
//     [[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
//
//
// # if __name__ == "__main__":
// #     main()
// clang-format on

} // namespace onnx_mlir
