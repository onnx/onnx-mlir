/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- NonMaxSuppression.cpp - Lowering NonMaxSuppression Op ----------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonMaxSuppression Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

/// Compute the intersection-over-union (IOU) score between two boxes.
/// IOU tells us how much two boxes are overlapped.
static Value emitIOU(MathBuilder &createMath, SmallVectorImpl<Value> box1,
    SmallVectorImpl<Value> box2, uint32_t centerPointBox = 0) {
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

  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemref(createKrnl);
  IndexExprScope scope(createKrnl);

  Type elementType =
      maxOutputPerClass.getType().cast<MemRefType>().getElementType();
  MemRefBoundsIndexCapture scoreBounds(scores);
  IndexExpr bsIE = scoreBounds.getDim(0); // batch size.
  IndexExpr csIE = scoreBounds.getDim(1); // class size.
  IndexExpr ssIE = scoreBounds.getDim(2); // spatial size.
  LiteralIndexExpr zeroIE(0);
  Value zero = createMath.constant(elementType, 0);
  Value one = createMath.constant(elementType, 1);

  // Compute the effective max output per class.
  Value effectiveMaxPerClass =
      createMemref.alloca(MemRefType::get({}, elementType));
  createKrnl.store(zero, effectiveMaxPerClass, {});

  ValueRange bcLoopDef = createKrnl.defineLoops(2);
  createKrnl.iterateIE(bcLoopDef, bcLoopDef, {zeroIE, zeroIE}, {bsIE, csIE},
      [&](KrnlBuilder &createKrnl, ValueRange bcLoopInd) {
        MathBuilder createMath(createKrnl);
        MemRefBuilder createMemref(createKrnl);
        IndexExprScope bcScope(createKrnl);
        Value b(bcLoopInd[0]), c(bcLoopInd[1]);

        // Store the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        Value topk = createMemref.alloca(MemRefType::get({}, elementType));
        createKrnl.store(zero, topk, {});

        // Load the score threshold.
        Value threshold = createKrnl.loadIE(scoreThreshold, {zeroIE});

        // Count the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        ValueRange sLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(sLoopDef, sLoopDef, {zeroIE}, {ssIE},
            [&](KrnlBuilder &createKrnl, ValueRange sLoopInd) {
              Value s(sLoopInd[0]);
              MathBuilder createMath(createKrnl);
              IndexExprScope sScope(createKrnl);

              Value score = createKrnl.load(scores, {b, c, s});
              // Increase the counter if score > threshold.
              Value gt = createMath.sgt(score, threshold);
              Value topkVal = createKrnl.load(topk, {});
              Value topkPlusOneVal = createMath.add(topkVal, one);
              topkVal = createMath.select(gt, topkPlusOneVal, topkVal);
              createKrnl.store(topkVal, topk, {});
            });

        // Update the effective max output per class.
        Value x = createKrnl.load(topk, {});
        Value y = createKrnl.load(effectiveMaxPerClass, {});
        createKrnl.store(createMath.min(x, y), effectiveMaxPerClass, {});
      });

  // Suppress the number of output bounding boxes per class.
  Value x = createKrnl.loadIE(maxOutputPerClass, {zeroIE});
  Value y = createKrnl.load(effectiveMaxPerClass, {});
  createKrnl.storeIE(createMath.min(x, y), maxOutputPerClass, {zeroIE});
}

/// Returns the indices that would sort the score tensor.
/// Scores :: [num_of_batch, num_of_class, spatial_dimension]
/// Sort along `spatial_dimension` axis.
static Value emitArgSort(
    ConversionPatternRewriter &rewriter, Location loc, Value scores) {
  KrnlBuilder createKrnl(rewriter, loc);
  MemRefBuilder createMemref(createKrnl);
  MathBuilder createMath(createKrnl);
  IndexExprScope scope(createKrnl);

  MemRefType scoreMemRefType = scores.getType().cast<MemRefType>();
  Type elementType = scoreMemRefType.getElementType();
  MemRefBoundsIndexCapture scoreBounds(scores);
  SmallVector<IndexExpr, 4> dimsSize;
  scoreBounds.getDimList(dimsSize);
  IndexExpr bsIE = dimsSize[0]; // batch size.
  IndexExpr csIE = dimsSize[1]; // class size.
  IndexExpr ssIE = dimsSize[2]; // spatial size.

  // Create and initialize the result.
  Value zero = createMath.constant(elementType, 0);
  Value order = insertAllocAndDeallocSimple(rewriter, nullptr, scoreMemRefType,
      loc, dimsSize, /*insertDealloc=*/true);
  createKrnl.memset(order, zero);

  // Do sorting in the descending order of scores and return their indices.
  // Using bubble sort.
  LiteralIndexExpr zeroIE(0), oneIE(1);
  ValueRange loopDef = createKrnl.defineLoops(3);
  createKrnl.iterateIE(loopDef, loopDef, {zeroIE, zeroIE, zeroIE},
      {bsIE, csIE, ssIE - oneIE},
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        Value b(loopInd[0]), c(loopInd[1]), i(loopInd[2]);
        IndexExpr i1 = DimIndexExpr(i) + LiteralIndexExpr(1);

        ValueRange swapLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(swapLoopDef, swapLoopDef, {i1}, {ssIE},
            [&](KrnlBuilder &createKrnl, ValueRange swapLoopInd) {
              MathBuilder createMath(createKrnl);
              Value k(swapLoopInd[0]);
              Value x = createKrnl.load(order, {b, c, i});
              Value y = createKrnl.load(order, {b, c, k});
              Value lt = createMath.slt(x, y);
              Value newX = createMath.select(lt, y, x);
              Value newY = createMath.select(lt, x, y);
              createKrnl.store(newX, order, {b, c, i});
              createKrnl.store(newY, order, {b, c, k});
            });
      });

  return order;
}

/// Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
/// flip the flipped boxes back.
/// BoundingBoxes: [num_of_batch, spatial_dimension, 4]
static void tryToUnflip(
    ConversionPatternRewriter &rewriter, Location loc, Value boundingBoxes) {
  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);

  MemRefBoundsIndexCapture bbBounds(boundingBoxes);
  IndexExpr bsIE = bbBounds.getDim(0); // batch size.
  IndexExpr ssIE = bbBounds.getDim(1); // spatial size.
  LiteralIndexExpr zeroIE(0);
  SmallVector<Value, 4> indices;
  for (int i = 0; i < 4; ++i) {
    Value iVal = createMath.constant(rewriter.getIndexType(), i);
    indices.emplace_back(iVal);
  }

  ValueRange loopDef = createKrnl.defineLoops(2);
  createKrnl.iterateIE(loopDef, loopDef, {zeroIE, zeroIE}, {bsIE, ssIE},
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        MathBuilder createMath(createKrnl);
        Value b(loopInd[0]), c(loopInd[1]);
        // Load a bounding box.
        Value y_min = createKrnl.load(boundingBoxes, {b, c, indices[0]});
        Value x_min = createKrnl.load(boundingBoxes, {b, c, indices[1]});
        Value y_max = createKrnl.load(boundingBoxes, {b, c, indices[2]});
        Value x_max = createKrnl.load(boundingBoxes, {b, c, indices[3]});

        // Flip x.
        Value gtX = createMath.sgt(x_min, x_max);
        Value newXMin = createMath.select(gtX, x_max, x_min);
        Value newXMax = createMath.select(gtX, x_min, x_max);

        // Flip y.
        Value gtY = createMath.sgt(y_min, y_max);
        Value newYMin = createMath.select(gtY, y_max, y_min);
        Value newYMax = createMath.select(gtY, y_min, y_max);

        // Update the bounding box.
        createKrnl.store(newYMin, boundingBoxes, {b, c, indices[0]});
        createKrnl.store(newXMin, boundingBoxes, {b, c, indices[1]});
        createKrnl.store(newYMax, boundingBoxes, {b, c, indices[2]});
        createKrnl.store(newXMax, boundingBoxes, {b, c, indices[3]});
      });
}

struct ONNXNonMaxSuppressionOpLowering : public ConversionPattern {
  ONNXNonMaxSuppressionOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXNonMaxSuppressionOp::getOperationName(), 1, ctx) {
  }

  /// To understand how code is generated for NonMaxSuppression, look at the
  /// python implementation at the end of this file.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXNonMaxSuppressionOp nmsOp = llvm::cast<ONNXNonMaxSuppressionOp>(op);
    ONNXNonMaxSuppressionOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope mainScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    MemRefBuilder createMemref(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();
    Type indexType = rewriter.getIndexType();

    Value scores = operandAdaptor.scores();
    Value boxes = operandAdaptor.boxes();
    Value maxOutputPerClass = operandAdaptor.max_output_boxes_per_class();
    Value iouThreshold = operandAdaptor.iou_threshold();
    Value scoreThreshold = operandAdaptor.score_threshold();
    int64_t centerPointBox = nmsOp.center_point_box();

    // boxes: [num_of_batch, spatial_dimension, 4]
    // scores: [num_of_batch, num_of_class, spatial_dimension]
    MemRefBoundsIndexCapture boxBounds(boxes);
    MemRefBoundsIndexCapture scoreBounds(scores);
    IndexExpr bsIE = scoreBounds.getDim(0); // batch size.
    IndexExpr csIE = scoreBounds.getDim(1); // class size.
    IndexExpr ssIE = scoreBounds.getDim(2); // spatial size.
    LiteralIndexExpr zeroIE(0), oneIE(1), threeIE(3);

    // Refine the number of output boxes per class by suppress it using spatial
    // dimension size and score threshold.
    // 1. Suppress by using spatial dimension size.
    Value x = createKrnl.loadIE(maxOutputPerClass, {zeroIE});
    Value y = rewriter.create<IndexCastOp>(loc, elementType, ssIE.getValue());
    createKrnl.storeIE(createMath.min(x, y), maxOutputPerClass, {zeroIE});
    // 2. Suppress by score threshold.
    suppressByScores(rewriter, loc, scores, scoreThreshold, maxOutputPerClass);

    // Sort scores in the descending order.
    Value order = emitArgSort(rewriter, loc, scores);

    // Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
    // unflip the flipped boxes.
    if (centerPointBox == 1)
      tryToUnflip(rewriter, loc, boxes);

    // The total number of output selected indices.
    IndexExpr maxPerClassIE =
        DimIndexExpr(createKrnl.loadIE(maxOutputPerClass, {zeroIE}));
    IndexExpr numSelectedIndicesIE = bsIE * csIE * maxPerClassIE;

    // Allocate a MemRef for the output. This MemRef is NOT the final output
    // since the number of selected indices has yet not suppressed by IOU.
    // Output shape : [num_selected_indices, 3]
    SmallVector<IndexExpr, 2> outputDims = {numSelectedIndicesIE, threeIE};
    Value bufMemRef = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, outputDims, /*insertDealloc=*/false);
    // Initialize with value -1.
    createKrnl.memset(bufMemRef, createMath.constant(elementType, -1));

    // Effective number of selected indices. This is the final value for the 1st
    // dim of the output, which is suppressed by IOU during computation and
    // cannot be computed in advance.
    // Final output shape : [effective_num_selected_indices, 3]
    Value effectiveNumSelectedIndices =
        createMemref.alloca(MemRefType::get({}, indexType));
    createKrnl.store(zeroIE.getValue(), effectiveNumSelectedIndices);

    // Suppress by using IOU.
    ValueRange bcLoopDef = createKrnl.defineLoops(2);
    createKrnl.iterateIE(bcLoopDef, bcLoopDef, {zeroIE, zeroIE}, {bsIE, csIE},
        [&](KrnlBuilder &createKrnl, ValueRange bcLoopInd) {});

    rewriter.replaceOp(op, bufMemRef);
    return success();
  }
};

void populateLoweringONNXNonMaxSuppressionOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXNonMaxSuppressionOpLowering>(ctx);
}

// Below is a python implementation of NonMaxSuppression.
// def IOU(box1, box2, center_point_box=0):
//     if center_point_box == 0:
//         # The box data is supplied as [y1, x1, y2, x2]. (y1, x1) and (y2, x2)
//         # are the coordinates of the diagonal pair of bottom-left and
//         top-right # corners. y1_min, x1_min, y1_max, x1_max = box1 y2_min,
//         x2_min, y2_max, x2_max = box2
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
//                     if (scores[b, c, i] < scores[b, c, k]):
//                         tmp = order[b, c, i]
//                         order[b, c, i] = order[b, c, k]
//                         order[b, c, k] = tmp
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
//             num_output_per_class = 0
//             removed_indices = np.full((spatial_size), False)
//             for s in range(spatial_size):
//                 # Discard bounding boxes using score threshold.
//                 if (scores[b, c, s] <= score_threshold):
//                     continue
//                 # Have enough the number of outputs.
//                 if (num_output_per_class >= max_output_per_class):
//                     continue
//                 # Removed, ignore.
//                 if removed_indices[s]:
//                     continue
//
//                 # Pick the bounding box with the largest score.
//                 selected_box_index = order[b, c, s]
//                 selected_box = boxes[b, selected_box_index, :]
//
//                 # Store the index of the picked box to the output.
//                 out_index = b * batch_size + c * max_output_per_class +
//                 num_output_per_class selected_indices[out_index] = [b, c,
//                 selected_box_index] num_output_per_class += 1
//
//                 # Remove boxes overlapped too much with the selected box,
//                 using # IOU. for o in range(spatial_size):
//                     other_box = boxes[b, o, :]
//                     iou = IOU(selected_box, other_box, center_point_box)
//                     if (not removed_indices[o]) and (iou > iou_threshold):
//                         removed_indices[o] = True
//                     else:
//                         removed_indices[o] = removed_indices[o]
//             effective_num_selected_indices += num_output_per_class
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
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0,
// 5]]).astype(np.int64) out = nms(boxes, scores, max_output_boxes_per_class,
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
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0,
// 5]]).astype(np.int64) out = nms(boxes, scores, max_output_boxes_per_class,
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
//     [[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
//     0.9]]]).astype(np.float32)
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
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0,
// 5]]).astype(np.int64) out = nms(boxes, scores, max_output_boxes_per_class,
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
