/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- GridSample.cpp - Lowering GridSample Op ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GridSample Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper functions for coordinate transformation
//===----------------------------------------------------------------------===//

// Transform normalized grid coordinate [-1,1] to input coordinate
// align_corners=0: [-1,1] -> [-0.5, size-0.5]
// align_corners=1: [-1,1] -> [0, size-1]
static Value transformCoordinate(MathBuilder &math, Location loc,
    Value gridCoord, Value inputSize, int64_t alignCorners) {
  Type floatType = gridCoord.getType();

  // Convert inputSize (index) to float
  Value inputSizeFloat = math.cast(floatType, inputSize);

  if (alignCorners == 1) {
    // x_input = ((x_grid + 1) * (size - 1)) / 2
    Value one = math.constant(floatType, 1.0);
    Value two = math.constant(floatType, 2.0);
    Value sizeMinus1 = math.sub(inputSizeFloat, one);
    Value gridPlus1 = math.add(gridCoord, one);
    Value scaled = math.mul(gridPlus1, sizeMinus1);
    return math.div(scaled, two);
  } else {
    // x_input = ((x_grid + 1) * size - 1) / 2
    Value one = math.constant(floatType, 1.0);
    Value two = math.constant(floatType, 2.0);
    Value gridPlus1 = math.add(gridCoord, one);
    Value scaled = math.mul(gridPlus1, inputSizeFloat);
    Value scaledMinus1 = math.sub(scaled, one);
    return math.div(scaledMinus1, two);
  }
}

// Check if coordinate is in bounds [0, size-1]
static Value isInBounds(
    MathBuilder &math, Location loc, Value coord, Value size) {
  Type floatType = coord.getType();
  Value zero = math.constant(floatType, 0.0);
  Value sizeFloat = math.cast(floatType, size);
  Value one = math.constant(floatType, 1.0);
  Value sizeMinus1 = math.sub(sizeFloat, one);

  Value geZero = math.sge(coord, zero);
  Value leSize = math.sle(coord, sizeMinus1);
  return math.andi(geZero, leSize);
}

// Helper to load value with padding mode handling (2D case)
// Returns zero for out-of-bounds (zeros mode) or clamped value (border mode)
static Value loadWithPadding(const KrnlBuilder &createKrnl, MathBuilder &math,
    Location loc, Value input, ArrayRef<Value> indices, Value yCoord,
    Value xCoord, Value H, Value W, StringRef paddingMode, Value zero) {
  Type floatType = yCoord.getType();

  if (paddingMode == "border") {
    // Clamp coordinates to [0, size-1]
    Value one = math.constant(floatType, 1.0);
    Value zeroFloat = math.constant(floatType, 0.0);
    Value hFloat = math.cast(floatType, H);
    Value wFloat = math.cast(floatType, W);
    Value hMax = math.sub(hFloat, one);
    Value wMax = math.sub(wFloat, one);

    Value xClamped = math.min(math.max(xCoord, zeroFloat), wMax);
    Value yClamped = math.min(math.max(yCoord, zeroFloat), hMax);
    Value xIdx = math.castToIndex(xClamped);
    Value yIdx = math.castToIndex(yClamped);

    // Build full index array
    SmallVector<Value, 4> fullIndices(indices.begin(), indices.end());
    fullIndices.push_back(yIdx);
    fullIndices.push_back(xIdx);
    return createKrnl.load(input, fullIndices);
  } else {
    // zeros padding: check bounds and return zero if out of bounds
    Value inBounds = math.andi(
        isInBounds(math, loc, xCoord, W), isInBounds(math, loc, yCoord, H));
    Value xIdx = math.castToIndex(xCoord);
    Value yIdx = math.castToIndex(yCoord);

    // Build full index array
    SmallVector<Value, 4> fullIndices(indices.begin(), indices.end());
    fullIndices.push_back(yIdx);
    fullIndices.push_back(xIdx);
    Value val = createKrnl.load(input, fullIndices);
    return math.select(inBounds, val, zero);
  }
}

// Helper to load value with padding mode handling (3D case)
// Returns zero for out-of-bounds (zeros mode) or clamped value (border mode)
static Value loadWithPadding3D(const KrnlBuilder &createKrnl, MathBuilder &math,
    Location loc, Value input, ArrayRef<Value> indices, Value zCoord,
    Value yCoord, Value xCoord, Value D, Value H, Value W,
    StringRef paddingMode, Value zero) {
  Type floatType = zCoord.getType();

  if (paddingMode == "border") {
    // Clamp coordinates to [0, size-1]
    Value one = math.constant(floatType, 1.0);
    Value zeroFloat = math.constant(floatType, 0.0);
    Value dFloat = math.cast(floatType, D);
    Value hFloat = math.cast(floatType, H);
    Value wFloat = math.cast(floatType, W);
    Value dMax = math.sub(dFloat, one);
    Value hMax = math.sub(hFloat, one);
    Value wMax = math.sub(wFloat, one);

    Value xClamped = math.min(math.max(xCoord, zeroFloat), wMax);
    Value yClamped = math.min(math.max(yCoord, zeroFloat), hMax);
    Value zClamped = math.min(math.max(zCoord, zeroFloat), dMax);
    Value xIdx = math.castToIndex(xClamped);
    Value yIdx = math.castToIndex(yClamped);
    Value zIdx = math.castToIndex(zClamped);

    // Build full index array
    SmallVector<Value, 5> fullIndices(indices.begin(), indices.end());
    fullIndices.push_back(zIdx);
    fullIndices.push_back(yIdx);
    fullIndices.push_back(xIdx);
    return createKrnl.load(input, fullIndices);
  } else {
    // zeros padding: check bounds and return zero if out of bounds
    Value inBounds = math.andi(math.andi(isInBounds(math, loc, xCoord, W),
                                   isInBounds(math, loc, yCoord, H)),
        isInBounds(math, loc, zCoord, D));
    Value xIdx = math.castToIndex(xCoord);
    Value yIdx = math.castToIndex(yCoord);
    Value zIdx = math.castToIndex(zCoord);

    // Build full index array
    SmallVector<Value, 5> fullIndices(indices.begin(), indices.end());
    fullIndices.push_back(zIdx);
    fullIndices.push_back(yIdx);
    fullIndices.push_back(xIdx);
    Value val = createKrnl.load(input, fullIndices);
    return math.select(inBounds, val, zero);
  }
}

// Compute cubic interpolation weight using Robert G. Keys method
// Reference: https://ieeexplore.ieee.org/document/1163711
static Value computeCubicWeight(
    MathBuilder &math, Location loc, Value s, double a) {
  Type floatType = s.getType();
  Value abs_s = math.abs(s);
  Value s2 = math.mul(abs_s, abs_s);
  Value s3 = math.mul(s2, abs_s);

  Value aVal = math.constant(floatType, a);
  Value one = math.constant(floatType, 1.0);
  Value two = math.constant(floatType, 2.0);
  Value three = math.constant(floatType, 3.0);
  Value four = math.constant(floatType, 4.0);
  Value five = math.constant(floatType, 5.0);
  Value eight = math.constant(floatType, 8.0);

  // For |s| <= 1: w(s) = (a+2)|s|³ - (a+3)|s|² + 1
  Value aPlusTwo = math.add(aVal, two);
  Value aPlusThree = math.add(aVal, three);
  Value term1 = math.mul(aPlusTwo, s3);
  Value term2 = math.mul(aPlusThree, s2);
  Value w1 = math.add(math.sub(term1, term2), one);

  // For 1 < |s| < 2: w(s) = a|s|³ - 5a|s|² + 8a|s| - 4a
  Value fiveA = math.mul(five, aVal);
  Value eightA = math.mul(eight, aVal);
  Value fourA = math.mul(four, aVal);
  Value t1 = math.mul(aVal, s3);
  Value t2 = math.mul(fiveA, s2);
  Value t3 = math.mul(eightA, abs_s);
  Value w2 = math.sub(math.add(math.sub(t1, t2), t3), fourA);

  // Select based on |s|
  Value isLessOne = math.sle(abs_s, one);
  Value isLessTwo = math.sle(abs_s, two);
  Value zero = math.constant(floatType, 0.0);
  Value result = math.select(isLessTwo, w2, zero);
  return math.select(isLessOne, w1, result);
}

//===----------------------------------------------------------------------===//
// 2D GridSample Lowering
//===----------------------------------------------------------------------===//

static LogicalResult lowerGridSample2D(ONNXGridSampleOp op,
    ONNXGridSampleOpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, ONNXGridSampleOpShapeHelper &shapeHelper,
    bool enableParallel) {

  Value input = adaptor.getX();
  Value grid = adaptor.getGrid();
  StringRef mode = op.getMode();
  int64_t alignCorners = op.getAlignCorners();
  StringRef paddingMode = op.getPaddingMode();

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
      MemRefBuilder>
      create(rewriter, loc);

  // Get dimensions
  MemRefType inputType = mlir::cast<MemRefType>(input.getType());
  Type elementType = inputType.getElementType();

  // Input shape: [N, C, H, W]
  DimsExpr inputDims;
  create.krnlIE.getShapeAsDims(input, inputDims);
  Value H = inputDims[2].getValue();
  Value W = inputDims[3].getValue();

  // Output shape: [N, C, H_out, W_out]
  int64_t outputRank = 4;
  ValueRange loopDef = create.krnl.defineLoops(outputRank);
  SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
  DimsExpr ubs = shapeHelper.getOutputDims();

  // Parallelize outer loops (N, C) for better performance
  // The two outer loops (batch and channel) are independent and can be
  // parallelized
  if (enableParallel) {
    SmallVector<Value, 2> outerLoops = {loopDef[0], loopDef[1]};
    tryCreateKrnlParallel(create.krnl, op, "gridsample 2d", outerLoops, lbs,
        ubs, 0, 2, {}, /*min iter for going parallel*/ 4);
  }

  // Constants
  Value zero = create.math.constant(elementType, 0.0);
  Value one = create.math.constant(elementType, 1.0);

  create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
      [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);

        Value n = loopInd[0];
        Value c = loopInd[1];
        Value h_out = loopInd[2];
        Value w_out = loopInd[3];

        // Load grid coordinates: grid[n, h_out, w_out, :]
        Value gridX = createKrnl.load(
            grid, {n, h_out, w_out, create.math.constantIndex(0)});
        Value gridY = createKrnl.load(
            grid, {n, h_out, w_out, create.math.constantIndex(1)});

        // Cast grid coordinates to input element type if needed
        Value gridXCast = create.math.cast(elementType, gridX);
        Value gridYCast = create.math.cast(elementType, gridY);

        // Transform coordinates from [-1,1] to input space
        Value x =
            transformCoordinate(create.math, loc, gridXCast, W, alignCorners);
        Value y =
            transformCoordinate(create.math, loc, gridYCast, H, alignCorners);

        Value result;

        if (mode == "nearest") {
          // Nearest neighbor interpolation
          // Use KrnlBuilder::roundEven which uses emulation internally to avoid
          // macOS linker issues with roundevenf
          Value x_nearest = createKrnl.roundEven(x);
          Value y_nearest = createKrnl.roundEven(y);
          result = loadWithPadding(createKrnl, create.math, loc, input, {n, c},
              y_nearest, x_nearest, H, W, paddingMode, zero);

        } else if (mode == "linear" || mode == "bilinear") {
          // Bilinear interpolation
          Value x0 = create.math.floor(x);
          Value x1 = create.math.add(x0, one);
          Value y0 = create.math.floor(y);
          Value y1 = create.math.add(y0, one);

          // Compute weights
          Value wx = create.math.sub(x, x0);
          Value wy = create.math.sub(y, y0);
          Value wx1 = create.math.sub(one, wx);
          Value wy1 = create.math.sub(one, wy);

          // Load 4 corner values with padding
          Value v00 = loadWithPadding(createKrnl, create.math, loc, input,
              {n, c}, y0, x0, H, W, paddingMode, zero);
          Value v01 = loadWithPadding(createKrnl, create.math, loc, input,
              {n, c}, y0, x1, H, W, paddingMode, zero);
          Value v10 = loadWithPadding(createKrnl, create.math, loc, input,
              {n, c}, y1, x0, H, W, paddingMode, zero);
          Value v11 = loadWithPadding(createKrnl, create.math, loc, input,
              {n, c}, y1, x1, H, W, paddingMode, zero);

          // Bilinear formula
          Value t0 = create.math.mul(create.math.mul(wy1, wx1), v00);
          Value t1 = create.math.mul(create.math.mul(wy1, wx), v01);
          Value t2 = create.math.mul(create.math.mul(wy, wx1), v10);
          Value t3 = create.math.mul(create.math.mul(wy, wx), v11);

          result =
              create.math.add(create.math.add(t0, t1), create.math.add(t2, t3));

        } else if (mode == "cubic") {
          // Bicubic interpolation using Robert G. Keys method
          Value x1 = create.math.floor(x);
          Value y1 = create.math.floor(y);

          // Get 16 neighboring points (4x4 grid)
          Value x0 = create.math.sub(x1, one);
          Value x2 = create.math.add(x1, one);
          Value x3 = create.math.add(x2, one);
          Value y0 = create.math.sub(y1, one);
          Value y2 = create.math.add(y1, one);
          Value y3 = create.math.add(y2, one);

          // Compute cubic weights
          Value dx = create.math.sub(x, x1);
          Value dy = create.math.sub(y, y1);

          double a = -0.75; // cubic_coeff_a
          Value wx0 =
              computeCubicWeight(create.math, loc, create.math.add(dx, one), a);
          Value wx1 = computeCubicWeight(create.math, loc, dx, a);
          Value wx2 =
              computeCubicWeight(create.math, loc, create.math.sub(one, dx), a);
          Value wx3 = computeCubicWeight(create.math, loc,
              create.math.sub(create.math.constant(elementType, 2.0), dx), a);

          Value wy0 =
              computeCubicWeight(create.math, loc, create.math.add(dy, one), a);
          Value wy1 = computeCubicWeight(create.math, loc, dy, a);
          Value wy2 =
              computeCubicWeight(create.math, loc, create.math.sub(one, dy), a);
          Value wy3 = computeCubicWeight(create.math, loc,
              create.math.sub(create.math.constant(elementType, 2.0), dy), a);

          // Compute bicubic interpolation
          Value sum = zero;
          SmallVector<Value, 4> xCoords = {x0, x1, x2, x3};
          SmallVector<Value, 4> yCoords = {y0, y1, y2, y3};
          SmallVector<Value, 4> wxWeights = {wx0, wx1, wx2, wx3};
          SmallVector<Value, 4> wyWeights = {wy0, wy1, wy2, wy3};

          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              Value val = loadWithPadding(createKrnl, create.math, loc, input,
                  {n, c}, yCoords[i], xCoords[j], H, W, paddingMode, zero);
              Value weight = create.math.mul(wyWeights[i], wxWeights[j]);
              sum = create.math.add(sum, create.math.mul(weight, val));
            }
          }

          result = sum;
        } else {
          // This should not be reached due to checks in matchAndRewrite
          result = zero;
        }

        // Store result
        createKrnl.store(result, alloc, loopInd);
      });

  return success();
}

//===----------------------------------------------------------------------===//
// 3D GridSample Lowering
//===----------------------------------------------------------------------===//

static LogicalResult lowerGridSample3D(ONNXGridSampleOp op,
    ONNXGridSampleOpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value alloc, ONNXGridSampleOpShapeHelper &shapeHelper,
    bool enableParallel) {

  Value input = adaptor.getX();
  Value grid = adaptor.getGrid();
  StringRef mode = op.getMode();
  int64_t alignCorners = op.getAlignCorners();
  StringRef paddingMode = op.getPaddingMode();

  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
      MemRefBuilder>
      create(rewriter, loc);

  // Get dimensions
  MemRefType inputType = mlir::cast<MemRefType>(input.getType());
  Type elementType = inputType.getElementType();

  // Input shape: [N, C, D, H, W]
  DimsExpr inputDims;
  create.krnlIE.getShapeAsDims(input, inputDims);
  Value D = inputDims[2].getValue();
  Value H = inputDims[3].getValue();
  Value W = inputDims[4].getValue();

  // Output shape: [N, C, D_out, H_out, W_out]
  int64_t outputRank = 5;
  ValueRange loopDef = create.krnl.defineLoops(outputRank);
  SmallVector<IndexExpr, 5> lbs(outputRank, LitIE(0));
  DimsExpr ubs = shapeHelper.getOutputDims();

  // Parallelize outer loops (N, C) for better performance
  // The two outer loops (batch and channel) are independent and can be
  // parallelized
  if (enableParallel) {
    SmallVector<Value, 2> outerLoops = {loopDef[0], loopDef[1]};
    tryCreateKrnlParallel(create.krnl, op, "gridsample 3d", outerLoops, lbs,
        ubs, 0, 2, {}, /*min iter for going parallel*/ 4);
  }

  // Constants
  Value zero = create.math.constant(elementType, 0.0);
  Value one = create.math.constant(elementType, 1.0);

  create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
      [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);

        Value n = loopInd[0];
        Value c = loopInd[1];
        Value d_out = loopInd[2];
        Value h_out = loopInd[3];
        Value w_out = loopInd[4];

        // Load grid coordinates: grid[n, d_out, h_out, w_out, :]
        Value gridX = createKrnl.load(
            grid, {n, d_out, h_out, w_out, create.math.constantIndex(0)});
        Value gridY = createKrnl.load(
            grid, {n, d_out, h_out, w_out, create.math.constantIndex(1)});
        Value gridZ = createKrnl.load(
            grid, {n, d_out, h_out, w_out, create.math.constantIndex(2)});

        // Cast grid coordinates to input element type if needed
        Value gridXCast = create.math.cast(elementType, gridX);
        Value gridYCast = create.math.cast(elementType, gridY);
        Value gridZCast = create.math.cast(elementType, gridZ);

        // Transform coordinates from [-1,1] to input space
        Value x =
            transformCoordinate(create.math, loc, gridXCast, W, alignCorners);
        Value y =
            transformCoordinate(create.math, loc, gridYCast, H, alignCorners);
        Value z =
            transformCoordinate(create.math, loc, gridZCast, D, alignCorners);

        Value result;

        if (mode == "nearest") {
          // Nearest neighbor interpolation
          // Use KrnlBuilder::roundEven which uses emulation internally to avoid
          // macOS linker issues with roundevenf
          Value x_nearest = createKrnl.roundEven(x);
          Value y_nearest = createKrnl.roundEven(y);
          Value z_nearest = createKrnl.roundEven(z);

          result =
              loadWithPadding3D(createKrnl, create.math, loc, input, {n, c},
                  z_nearest, y_nearest, x_nearest, D, H, W, paddingMode, zero);

        } else if (mode == "linear" || mode == "trilinear") {
          // Trilinear interpolation (8 corner points)
          Value x0 = create.math.floor(x);
          Value x1 = create.math.add(x0, one);
          Value y0 = create.math.floor(y);
          Value y1 = create.math.add(y0, one);
          Value z0 = create.math.floor(z);
          Value z1 = create.math.add(z0, one);

          // Compute weights
          Value wx = create.math.sub(x, x0);
          Value wy = create.math.sub(y, y0);
          Value wz = create.math.sub(z, z0);
          Value wx1 = create.math.sub(one, wx);
          Value wy1 = create.math.sub(one, wy);
          Value wz1 = create.math.sub(one, wz);

          // Load 8 corner values with padding
          Value v000 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z0, y0, x0, D, H, W, paddingMode, zero);
          Value v001 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z0, y0, x1, D, H, W, paddingMode, zero);
          Value v010 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z0, y1, x0, D, H, W, paddingMode, zero);
          Value v011 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z0, y1, x1, D, H, W, paddingMode, zero);
          Value v100 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z1, y0, x0, D, H, W, paddingMode, zero);
          Value v101 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z1, y0, x1, D, H, W, paddingMode, zero);
          Value v110 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z1, y1, x0, D, H, W, paddingMode, zero);
          Value v111 = loadWithPadding3D(createKrnl, create.math, loc, input,
              {n, c}, z1, y1, x1, D, H, W, paddingMode, zero);

          // Trilinear formula
          Value t0 = create.math.mul(
              create.math.mul(create.math.mul(wz1, wy1), wx1), v000);
          Value t1 = create.math.mul(
              create.math.mul(create.math.mul(wz1, wy1), wx), v001);
          Value t2 = create.math.mul(
              create.math.mul(create.math.mul(wz1, wy), wx1), v010);
          Value t3 = create.math.mul(
              create.math.mul(create.math.mul(wz1, wy), wx), v011);
          Value t4 = create.math.mul(
              create.math.mul(create.math.mul(wz, wy1), wx1), v100);
          Value t5 = create.math.mul(
              create.math.mul(create.math.mul(wz, wy1), wx), v101);
          Value t6 = create.math.mul(
              create.math.mul(create.math.mul(wz, wy), wx1), v110);
          Value t7 = create.math.mul(
              create.math.mul(create.math.mul(wz, wy), wx), v111);

          result = create.math.add(
              create.math.add(create.math.add(t0, t1), create.math.add(t2, t3)),
              create.math.add(
                  create.math.add(t4, t5), create.math.add(t6, t7)));
        } else {
          // This should not be reached due to checks in matchAndRewrite
          result = zero;
        }

        // Store result
        createKrnl.store(result, alloc, loopInd);
      });

  return success();
}

//===----------------------------------------------------------------------===//
// Main GridSample Lowering Pattern
//===----------------------------------------------------------------------===//

struct ONNXGridSampleOpLowering : public OpConversionPattern<ONNXGridSampleOp> {
  ONNXGridSampleOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXGridSampleOp::getOperationName());
  }

  bool enableParallel = false;

  LogicalResult matchAndRewrite(ONNXGridSampleOp gridSampleOp,
      ONNXGridSampleOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    Operation *op = gridSampleOp.getOperation();
    Location loc = ONNXLoc<ONNXGridSampleOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Check padding mode - zeros and border supported
    StringRef paddingMode = gridSampleOp.getPaddingMode();
    if (paddingMode != "zeros" && paddingMode != "border") {
      return emitError(
          loc, "Only zeros and border padding modes are currently supported");
    }

    // Check interpolation mode
    StringRef mode = gridSampleOp.getMode();
    if (mode != "nearest" && mode != "linear" && mode != "bilinear" &&
        mode != "cubic" && mode != "trilinear") {
      return emitError(loc, "Unsupported interpolation mode: ") << mode;
    }

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get shape using existing shape helper
    ONNXGridSampleOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert output type to MemRefType
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Allocate output
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Get input rank to determine 2D vs 3D
    Value input = adaptor.getX();
    int64_t inputRank = mlir::cast<MemRefType>(input.getType()).getRank();

    // Check mode compatibility with dimensionality
    if (inputRank == 4) {
      // 2D: supports nearest, linear/bilinear, cubic
      if (mode == "trilinear") {
        return emitError(
            loc, "trilinear mode is only supported for 3D (5D input)");
      }
    } else if (inputRank == 5) {
      // 3D: supports nearest, linear/trilinear
      if (mode == "cubic") {
        return emitError(loc, "cubic mode is not supported for 3D (5D input)");
      }
    }

    // Dispatch to 2D or 3D implementation
    LogicalResult result = success();
    if (inputRank == 4) {
      result = lowerGridSample2D(gridSampleOp, adaptor, rewriter, loc, alloc,
          shapeHelper, enableParallel);
    } else if (inputRank == 5) {
      result = lowerGridSample3D(gridSampleOp, adaptor, rewriter, loc, alloc,
          shapeHelper, enableParallel);
    } else {
      return emitError(
          loc, "GridSample only supports 2D (4D input) and 3D (5D input)");
    }

    if (failed(result))
      return failure();

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXGridSampleOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXGridSampleOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir

// Made with Bob
