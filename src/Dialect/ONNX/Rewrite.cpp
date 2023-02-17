/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include <math.h>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

// =============================================================================
// Helper functions for Rewrite.td and Rewrite.cpp files.
// =============================================================================

// If 'A' is NoneType, return -B. Otherwise return A-B.
Value subtractOrNeg(PatternRewriter &rewriter, Location loc, Value A, Value B) {
  if (A.getType().isa<NoneType>())
    return rewriter.create<ONNXNegOp>(loc, B);
  return rewriter.create<ONNXSubOp>(loc, A, B);
}

// Create an ArrayAttr of IntegerAttr(s) of values in [1, N].
ArrayAttr createArrayAttrOfOneToN(PatternRewriter &rewriter, int N) {
  SmallVector<int64_t, 4> vals;
  for (int i = 1; i <= N; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Create an ArrayAttr of IntegerAttr(s) of values in [N, M].
ArrayAttr createArrayAttrOfNToM(PatternRewriter &rewriter, int N, int M) {
  SmallVector<int64_t, 4> vals;
  for (int i = N; i <= M; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Get return type for a MatMulOp whose A's rank is N (>2) and B's rank is 2.
Type getReturnTypeForMatMulOpND2D(Value A, Value B) {
  ArrayRef<int64_t> aShape = A.getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> bShape = B.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t> resShape(aShape.begin(), aShape.end() - 1);
  resShape.emplace_back(bShape[bShape.size() - 1]);
  return RankedTensorType::get(
      resShape, A.getType().cast<ShapedType>().getElementType());
}

// Get the index of the axis value in the given permutation array.
IntegerAttr getIndexOfAxisInPerm(
    PatternRewriter &rewriter, ArrayAttr permAttr, IntegerAttr axis) {
  IntegerAttr result;
  for (uint64_t i = 0; i < permAttr.getValue().size(); ++i) {
    IntegerAttr attr = permAttr.getValue()[i].cast<IntegerAttr>();
    assert(attr && "Element in ArrayAttr is not IntegerAttr");
    if (attr.getValue().getSExtValue() == axis.getValue().getSExtValue())
      return rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), i);
  }
  return result;
}

// Transpose a variadic input using a permutation array.
SmallVector<Value, 4> transposeVariadicInput(PatternRewriter &rewriter,
    Location loc, ValueRange inputs, ArrayAttr permAttr) {
  SmallVector<Value, 4> transposedInputs;
  for (Value inp : inputs) {
    ShapedType inpType = inp.getType().cast<ShapedType>();
    assert(inpType && "Type is not ShapedType");
    ONNXTransposeOp transposeOp = rewriter.create<ONNXTransposeOp>(
        loc, UnrankedTensorType::get(inpType.getElementType()), inp, permAttr);
    (void)transposeOp.inferShapes([](Region &region) {});
    transposedInputs.emplace_back(transposeOp.getResult());
  }
  return transposedInputs;
}

// Check if all values are produced by ONNXTransposeOp.
bool areProducedByTransposeOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    if (v.isa<BlockArgument>())
      return false;
    return isa<ONNXTransposeOp>(v.getDefiningOp());
  });
}

// Create a DenseElementsAttr based on the shape of type.
DenseElementsAttr createDenseElementsAttrFromShape(PatternRewriter &rewriter,
    Value value, int64_t start = 0,
    llvm::Optional<int64_t> end = std::nullopt) {

  auto inType = value.getType().cast<ShapedType>();
  assert(inType.hasRank() && "inType must be ranked");
  auto shape = inType.getShape();
  int64_t rank = inType.getRank();

  int64_t endValue = end.has_value() ? end.value() : rank;

  SmallVector<int64_t, 1> dims = {endValue - start};
  SmallVector<int64_t, 4> values(
      shape.begin() + start, shape.begin() + endValue);
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

// Create a DenseElementsAttr from Shape Op
DenseElementsAttr createDenseElementsAttrFromShapeOp(
    PatternRewriter &rewriter, Operation *op) {
  ONNXShapeOp shapeOp = llvm::cast<ONNXShapeOp>(op);
  int64_t start, end;
  ONNXShapeOpShapeHelper::getStartEndValues(shapeOp, start, end);
  return createDenseElementsAttrFromShape(
      rewriter, shapeOp.getData(), start, end);
}

// Create ONNX Transpose op
// TODO: The same function in ONNXToZHighComon.cpp. Commonize them.
Value emitONNXTranspose(
    Location loc, PatternRewriter &rewriter, Value x, ArrayRef<int64_t> perms) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = x.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  Type transposedType;
  if (inputType.hasRank()) {
    assert((uint64_t)inputType.getRank() == perms.size() &&
           "Permutation array size is different from the input rank");
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t, 4> transposedShape;
    for (uint64_t i = 0; i < perms.size(); ++i)
      transposedShape.emplace_back(inputShape[perms[i]]);
    transposedType = RankedTensorType::get(transposedShape, elementType);
  } else {
    transposedType = UnrankedTensorType::get(elementType);
  }
  Value result =
      create.onnx.transpose(transposedType, x, rewriter.getI64ArrayAttr(perms));
  return result;
}

// Reverse all elements of the first or second dimension of `input`.
Value reverseAllElements(
    PatternRewriter &rewriter, Location loc, Value input, int64_t dimension) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t, 4> slens;
  assert((dimension == 0 or dimension == 1) &&
         "Reversed diemsnion need to be 0 or 1.");
  // Create `sequence_lengths`, `batch_axis` and `time_axis` to reverse all
  // elements. When reversing the first dim of input(d0 x d1), set `batch_axis`
  // = 1, and `time_axis` = 0 and create [d0, d0,...,d0] as `sequence_lengths`
  // whose the number of elements are d1.
  // Example:
  //   input(d0 x d1) = (4 x 3)) then, `sequence_lenghts` is [4, 4, 4].
  // When reverse the second dim of input(d0 x d1), set `batch_axis` = 0,
  // and `time_axis` = 1 and create [d1, d1,...,d1] as `sequence_lengths`
  // whose the number of elements are d0.
  // Example:
  // input(d0 x d1) = (4 x 3)) then, `sequence_lenghts` is [3, 3, 3, 3].
  int64_t batchAxis = dimension == 0 ? 1 : 0;
  int64_t timeAxis = dimension == 0 ? 0 : 1;
  for (int i = 0; i < inputShape[batchAxis]; ++i)
    slens.emplace_back(inputShape[timeAxis]);
  Value slensVal = create.onnx.constantInt64(slens);
  Type resultType = input.getType().cast<RankedTensorType>();
  Value result = create.onnx.reverseSequence(
      resultType, input, slensVal, batchAxis, timeAxis);
  return result;
}

// Reverse elements in weight tensor of ConvTranspose op. The reversed weight
// tensor are used as weight tensor of Conv op generated by rewriting.
// 1. Transpose weight tensor from NxCxD0xD1xD2x... to D0xD1xD2x ... xNxC to
//    reverse elements by using ReverseSequence op
//    The ReverseSequence op can reverse elements in the first and second
//    dimensions. So, spatial dimensions are moved using Transpose op.
// 2. Reverse The first two dimensions by two ReverseSequence ops
//    Reverse D0 by the first ReverseSequence op, then reverse D1 by the second
//    ReverseSequence op. Reverse D0 and D1 and move them to last
//    (D0xD1xD2xD3x... to D2xD3x...xD0xD1) to reverse D2 and D3. Continue this
//    to reverse all spatial dimensions.
// 3. Reverse the last spatial dimension (Dn) using single ReverseSequence if
//    rank is odd
// 4. Reverse non-spatial dimensions (N and C)
//    Transpose "N x C x D0 x D1 x D2 x ... x Dn" to "C x N x D0 x D1 x D2 x
//    ...x Dn".
Value reverseWeightTensor(
    PatternRewriter &rewriter, Location loc, Value input) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  assert(inputType.hasRank() && "Need rank to reverse weight tensor.");
  // 1. Transpose NxCxD0xD1xD2x... to D0xD1xD2x ... xNxC
  int64_t spatialOffset = 2; // for N and C
  int64_t spatialRank = inputType.getRank() - spatialOffset;
  SmallVector<int64_t, 4> permsval;
  for (int i = 0; i < spatialRank; ++i)
    permsval.emplace_back(spatialOffset + i);
  for (int i = 0; i < spatialOffset; ++i)
    permsval.emplace_back(i);
  ArrayRef<int64_t> perms(permsval);
  Value transposedInput = emitONNXTranspose(loc, rewriter, input, perms);
  // 2. Reverse the first and second dimensions
  for (int i = 0; i < spatialRank / 2; i += 2) {
    Value reverse0 =
        reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
    Value reverse1 =
        reverseAllElements(rewriter, loc, reverse0, /*dimension*/ 1);
    // Move two reversed dimensions to the last for next reverse.
    SmallVector<int64_t, 4> permsval0;
    for (int j = 0; j < inputType.getRank() - 2; ++j)
      permsval0.emplace_back(j + 2);
    for (int j = 0; j < 2; ++j)
      permsval0.emplace_back(j);
    ArrayRef<int64_t> perms(permsval0);
    transposedInput = emitONNXTranspose(loc, rewriter, reverse1, permsval0);
  }
  // 3. Reverse the rest of dimension if spatial rank is odd
  if (spatialRank % 2 != 0) {
    ShapedType tInType = transposedInput.getType().cast<ShapedType>();
    ArrayRef<int64_t> tInShape = tInType.getShape();
    Value reverse0;
    if (tInShape[1] == ShapedType::kDynamic) {
      // When N is unknown dim,
      // reshape "Dn x N x C x D0 xD1 x D2 x ... x Dn-1"
      // to "Dn x 1 x N x C x D0 xD1 x D2 x ... x Dn-1",
      // then, reshape back to original shape after reversed.
      int64_t tInRank = tInShape.size();
      Type tInShapeType =
          RankedTensorType::get({tInRank}, rewriter.getI64Type());
      Value tInShapeVals = create.onnx.shape(tInShapeType, transposedInput);
      SmallVector<int64_t, 6> reshapedShapeVec;
      reshapedShapeVec.emplace_back(tInShape[0]);
      reshapedShapeVec.emplace_back(1);
      for (int i = 1; i < tInType.getRank(); ++i)
        reshapedShapeVec.emplace_back(tInShape[i]);
      Type reshapedType = RankedTensorType::get(reshapedShapeVec, elementType);
      Type firstShapeType = RankedTensorType::get({1}, rewriter.getI64Type());
      Type otherShapeType =
          RankedTensorType::get({tInRank - 1}, rewriter.getI64Type());
      Value oneVal = create.onnx.constantInt64(ArrayRef<int64_t>({1}));
      Value firstShapeVal = create.onnx.slice(
          firstShapeType, tInShapeVals, /* starts */ 0, /* ends */ 1);
      Value otherShapeVals = create.onnx.slice(
          otherShapeType, tInShapeVals, /* starts */ 1, /* ends */ tInRank);
      Type reshapeShapeType =
          RankedTensorType::get({tInRank + 1}, rewriter.getI64Type());
      Value shape = create.onnx.concat(reshapeShapeType,
          ValueRange{firstShapeVal, oneVal, otherShapeVals}, 0);
      transposedInput =
          create.onnx.reshape(reshapedType, transposedInput, shape);
      reverse0 =
          reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
      reverse0 = create.onnx.reshape(tInType, reverse0, tInShapeVals);
    } else {
      reverse0 =
          reverseAllElements(rewriter, loc, transposedInput, /*dimension*/ 0);
    }

    // Move reversed one dimension to the last.
    SmallVector<int64_t, 4> permsval1;
    for (int j = 0; j < inputType.getRank() - 1; ++j)
      permsval1.emplace_back(j + 1);
    permsval1.emplace_back(0);
    ArrayRef<int64_t> perms(permsval1);
    transposedInput = emitONNXTranspose(loc, rewriter, reverse0, permsval1);
  }
  // 4. Reverse non-spatial dimensions
  SmallVector<int64_t, 4> permsval2;
  for (int i = 0; i < spatialOffset; ++i)
    permsval2.emplace_back(spatialOffset - 1 - i);
  for (int i = 0; i < spatialRank; ++i)
    permsval2.emplace_back(spatialOffset + i);
  ArrayRef<int64_t> perms2(permsval2);
  Value result = emitONNXTranspose(loc, rewriter, transposedInput, perms2);
  return result;
}

// Calculate padding size used in Conv op from pads for ConvTranspose op
ArrayAttr getPadsConvTranspose(PatternRewriter &rewriter, Location loc,
    ArrayAttr kernel, ArrayAttr pads, ArrayAttr dilation) {
  // Calculate pads in generated Conv op by rewriting ConvTranspose op
  // new_pads = kernel -  pads - 1
  // Reference: Dumoulin, Vincent, and Francesco Visin. "A guide to convolution
  // arithmetic for deep learning." arXiv preprint arXiv:1603.07285 (2016).
  SmallVector<int64_t, 4> newPads;
  SmallVector<int64_t, 2> newKernel;
  // If `dilations` is not default [1, 1], `kernel` is updated by inserting
  // spaces in kernel elements
  //   ex. kernel [2, 3] and dilation [2, 2], then new `kernel` is [3, 4]
  for (unsigned int i = 0; i < kernel.size(); ++i)
    newKernel.emplace_back(
        ArrayAttrIntVal(kernel, i) +
        (ArrayAttrIntVal(kernel, i) - 1) * (ArrayAttrIntVal(dilation, i) - 1));

  // Calculate new pads. `kernel` size is doubled for the calculation.
  for (unsigned int i = 0; i < kernel.size() * 2; ++i)
    newPads.emplace_back(
        newKernel[i % kernel.size()] - ArrayAttrIntVal(pads, i) - 1);
  return rewriter.getI64ArrayAttr(newPads);
}

// Check if strides is unit strides
bool hasUnitStrides(ArrayAttr strides) {
  SmallVector<int64_t, 3> vstrides;
  for (unsigned int i = 0; i < strides.size(); ++i)
    vstrides.emplace_back(ArrayAttrIntVal(strides, i));
  return llvm::all_of(vstrides, [](int64_t s) { return s == 1; });
}

ArrayAttr createUnitStrides(PatternRewriter &rewriter, ArrayAttr strides) {
  SmallVector<int64_t, 2> unitStrides;
  for (unsigned int i = 0; i < strides.size(); ++i)
    unitStrides.emplace_back(1);
  return rewriter.getI64ArrayAttr(unitStrides);
}

// Split on the specified axis. The length of each output is one.
ValueRange emitSplitAxisOutputLength1(
    PatternRewriter &rewriter, Location loc, Value input, int64_t axis) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Create `split` to split each output in `axis` into length 1.
  // Ex. inputShape[axis] = 3, then  onnx.Constant dense<1> : tensor<3xi64>
  SmallVector<int64_t, 1> values(inputShape[axis], 1);
  Value split = create.onnx.constantInt64(ArrayRef(values));
  SmallVector<int64_t> splitShape;
  for (int i = 0; i < inputType.getRank(); ++i) {
    if (i == axis)
      splitShape.emplace_back(1);
    else
      splitShape.emplace_back(inputShape[i]);
  }
  Type splitType = RankedTensorType::get(splitShape, elementType);
  SmallVector<Type, 4> splitTypes(inputShape[axis], splitType);
  ValueRange results =
      create.onnx.split(ArrayRef(splitTypes), input, split, axis);
  return results;
}

// Emit ONNXPadOp to add pads of `size` at end of the `axis`.
Value emitPadsAxisEnd(PatternRewriter &rewriter, Location loc, Value input,
    ArrayRef<int64_t> inputShape, int64_t axis, int64_t size) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  SmallVector<int64_t> resultShape;
  for (unsigned int i = 0; i < inputShape.size(); ++i) {
    if (i == axis)
      resultShape.emplace_back(inputShape[i] + size);
    else
      resultShape.emplace_back(inputShape[i]);
  }
  Type resultType = RankedTensorType::get(resultShape, elementType);
  // Specify padding at the end of each axis.
  SmallVector<int64_t, 1> values((int64_t)inputShape.size() * 2, 0);
  values[inputShape.size() + axis] = size;
  Value pads = create.onnx.constantInt64(ArrayRef(values));
  Value result = create.onnx.padZero(resultType, input, pads);
  return result;
}

Value emitConcat(
    PatternRewriter &rewriter, Location loc, ValueRange inputs, int64_t axis) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  ShapedType inputType = inputs[0].getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t concatAxisSize = 0;
  for (Value v : inputs) {
    ShapedType vType = v.getType().cast<ShapedType>();
    ArrayRef<int64_t> vShape = vType.getShape();
    concatAxisSize += vShape[axis];
  }
  SmallVector<int64_t> concatShape;
  for (unsigned int i = 0; i < inputShape.size(); ++i) {
    if (i == axis)
      concatShape.emplace_back(concatAxisSize);
    else
      concatShape.emplace_back(inputShape[i]);
  }
  Type concatType = RankedTensorType::get(concatShape, elementType);
  Value result = create.onnx.concat(concatType, inputs, axis);
  return result;
}

// Insert pads in specified axis.
Value insertPadAxis(PatternRewriter &rewriter, Location loc, Value input,
    int64_t axis, int64_t padSize) {
  // Split on the specified axis. The length of each output is one.
  ValueRange splitResults =
      emitSplitAxisOutputLength1(rewriter, loc, input, axis);
  // Add pad in split results except last one.
  Value splitLastResults = splitResults.back();
  ValueRange padInputs = splitResults.drop_back();
  SmallVector<Value, 4> padResults;
  for (Value v : padInputs) {
    ArrayRef<int64_t> vShape = v.getType().cast<ShapedType>().getShape();
    padResults.emplace_back(
        emitPadsAxisEnd(rewriter, loc, v, vShape, axis, padSize));
  }
  padResults.emplace_back(splitLastResults);
  // Concat padded results
  Value concatResult = emitConcat(rewriter, loc, ValueRange(padResults), axis);
  return concatResult;
}

// Insert pads between elements in input tensor in spatial dimensions.
// The padding size is strides - 1
Value insertPadsConvTransposeInput(
    PatternRewriter &rewriter, Location loc, Value input, ArrayAttr strides) {
  int64_t spatialOffset = 2;
  for (unsigned int i = 0; i < strides.size(); ++i) {
    input = insertPadAxis(rewriter, loc, input, /*axis*/ spatialOffset + i,
        /*padSize*/ ArrayAttrIntVal(strides, i) - 1);
  }
  return input;
}

// Insert additional padding to output of ConvOp in ConvTransposeOp
Value insertAdditionalPadsConvTranspose(PatternRewriter &rewriter, Location loc,
    ONNXConvOp op, Value input, ArrayAttr outputPaddingAttr,
    ArrayAttr outputShapeAttr) {
  ONNXConvOpShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  int inputRank = shapeHelper.getOutputDims().size();
  SmallVector<int64_t, 4> inputShape;
  for (int i = 0; i < inputRank; ++i) {
    int64_t d = shapeHelper.getOutputDims()[i].isLiteral()
                    ? shapeHelper.getOutputDims()[i].getLiteral()
                    : ShapedType::kDynamic;
    inputShape.emplace_back(d);
  }
  SmallVector<int64_t, 2> padSize;
  int64_t attrSize = ArrayAttrSize(outputShapeAttr);
  int64_t offset = inputRank - attrSize;
  for (int i = 0; i < attrSize; ++i) {
    IntegerAttr attr = outputShapeAttr.getValue()[i].cast<IntegerAttr>();
    int64_t size = attr.getValue().getSExtValue() - inputShape[offset + i];
    assert(size >= 0 && "Invalid output_shape attribute");
    padSize.emplace_back(size);
  }
  Value paddedInput = emitPadsAxisEnd(
      rewriter, loc, input, ArrayRef(inputShape), /*axis*/ 2, padSize[0]);
  for (int i = 1; i < attrSize; ++i) {
    ArrayRef<int64_t> paddedInputShape =
        paddedInput.getType().cast<ShapedType>().getShape();
    paddedInput = emitPadsAxisEnd(rewriter, loc, paddedInput, paddedInputShape,
        /*axis*/ 2 + i, padSize[i]);
  }
  return paddedInput;
}

} // namespace onnx_mlir

// =============================================================================
/// Include the patterns defined in the Declarative Rewrite framework.
// =============================================================================

#include "src/Dialect/ONNX/ONNXRewrite.inc"

// =============================================================================
// Rewrite pattern for loop (not handled in Rewrite.td).
// =============================================================================

// In some ONNX models, the maximum trip count for LoopOp is set to a big value,
// e.g. LONG_MAX and termination depends on the break condition inside the loop.
// In the current lowering of LoopOp, the maximum trip count is used to allocate
// a buffer for all intermediate loop results. Since the actual number of loop
// iterations may be much smaller than the maximum trip count, it is redundant
// and error-prone to allocate a large buffer. For example, we may get segfault
// if the maximum trip count is out of range.
//
// This pattern tries to derive a new maximum trip count for LoopOp by analyzing
// the break condition. It only handles a special case where the loop is like a
// for-loop with step, e.g. `for (i = LB, i < UB, i = i + Step)`.
//
// For example, the following loop which mimics LoopOp:
// ```
// max_trip_count=9223372036854775807
// LB = -100
// UB = 100
// Step = 1
//
// i = 0
// k = LB
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
//    keepGoing = (k < UB)
// }
// ```
//
// will be rewritten into:
//
// ```
// max_trip_count=200
// LB = -100
// UB = 100
//
// i = 0
// k = LB
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
// }
// ```
// where `max_trip_count` is replaced by an actual value derived from the loop.
//
class LoopOpRewriteMaxTripCountPattern : public OpRewritePattern<ONNXLoopOp> {
public:
  using OpRewritePattern<ONNXLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLoopOp onnxLoopOp, PatternRewriter &rewriter) const override {
    Location loc = onnxLoopOp.getLoc();
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // Match the following pattern:
    // ```
    // ubValue = ONNXConstantOp() {value = ...}
    // startValue = ONNXConstantOp() {value = ...}
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond_new = cond
    //     ONNXReturnOp (cond_new, ..., ubValue, ..., newCounterValue, ...)
    // ```
    bool matched;
    Value newMaxTripCountValue;
    std::tie(matched, newMaxTripCountValue) =
        matchOp(rewriter, loc, onnxLoopOp);
    if (!matched)
      return failure();

    // Rewrite
    loopOp->replaceUsesOfWith(maxTripCountValue, newMaxTripCountValue);
    // Modify the condition return
    Region &loopBody = onnxLoopOp.getBody();
    Operation *loopBodyTerminator = loopBody.front().getTerminator();
    loopBodyTerminator->setOperand(0, loopBody.front().getArgument(1));
    return success();
  }

private:
  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or not.
  bool isDefinedByIntegerConstantOp(Value v) const {
    if (v.isa<BlockArgument>())
      return false;
    Operation *definingOp = v.getDefiningOp();
    if (v.getType().cast<ShapedType>().getElementType().isa<IntegerType>() &&
        isa<ONNXConstantOp>(definingOp) &&
        cast<ONNXConstantOp>(definingOp)
            .getValueAttr()
            .isa<DenseElementsAttr>())
      return true;
    return false;
  }

  // A helper function to check whether an block argument is invariant to
  // iterations or not. By the definition of LoopOp, input block arguments are
  // shifted by 1 to the left in ReturnOp. If a block argument is unchanged when
  // being shifted in ReturnOp, then it is invariant to iterations.
  bool isInvariantBlockArg(Value v, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (v ==
               returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or an invariant block argument.
  bool isIntConstantOrInvariantBlockArg(Value v, Operation *returnOp) const {
    return ((v.isa<BlockArgument>() && isInvariantBlockArg(v, returnOp)) ||
            (!v.isa<BlockArgument>() && isDefinedByIntegerConstantOp(v)));
  }

  // A helper function to check whether an block argument is updated by a Value
  // inside the loop or not.
  bool isUpdatedArgByValue(Value v, Value newV, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (newV ==
               returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  // A helper function to get the value that is fed to an operation's argument.
  Value getFedValue(Value arg, Operation *op) const {
    return op->getOperands()[arg.cast<BlockArgument>().getArgNumber()];
  }

  // A helper function to get an integer constant from a value.
  int64_t getOneIntegerConstant(Value v) const {
    Operation *definingOp = v.getDefiningOp();
    DenseElementsAttr valueAttr = cast<ONNXConstantOp>(definingOp)
                                      .getValueAttr()
                                      .cast<DenseElementsAttr>();
    return (*valueAttr.getValues<APInt>().begin()).getSExtValue();
  }

  // A helper function to match the pattern of the given operation. It also
  // returns a constant value for the max trip count during the matching, which
  // is to avoid recomputing values in the rewriting phase.
  //
  // Pattern:
  // ```
  // ubValue = ONNXConstantOp() {value = ...}
  // startValue = ONNXConstantOp() {value = ...}
  // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
  //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
  //     stepValue = ONNXConstantOp() {value = ...}
  //     newCounterValue = ONNXAddOp(counterValue, stepValue).
  //     cond = LessOp(newCounterValue, ubValue)
  //     ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)
  // ```
  std::pair<bool, Value> matchOp(
      PatternRewriter &rewriter, Location loc, ONNXLoopOp onnxLoopOp) const {
    OnnxBuilder onnx(rewriter, loc);
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // The maximum trip count is a constant.
    if (!isDefinedByIntegerConstantOp(maxTripCountValue))
      return std::make_pair(false, maxTripCountValue);

    // Get the loop region.
    Region &loopBody = onnxLoopOp.getBody();
    // Make sure the region has only one block.
    if (!loopBody.hasOneBlock())
      return std::make_pair(false, maxTripCountValue);

    // Get ReturnOp of the body block.
    Block &bodyBlock = loopBody.front();
    Operation *returnOp = bodyBlock.getTerminator();
    if (!isa<ONNXReturnOp>(returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Analyze the break condition of the loop body to see if we can derive a
    // new maximum trip count or not.

    // The break condition is the first argument of ReturnOp.
    // `ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)`
    Value breakCond = returnOp->getOperands()[0];
    if (breakCond.isa<BlockArgument>())
      return std::make_pair(false, maxTripCountValue);
    Operation *breakCondOp = breakCond.getDefiningOp();

    // Only support LessOp as the op that defines the break condition at this
    // moment.
    // `cond = LessOp(newCounterValue, ubValue)`
    if (!isa<ONNXLessOp>(breakCondOp))
      return std::make_pair(false, maxTripCountValue);
    Value newCounterValue = breakCondOp->getOperands()[0];
    Value ubValue = breakCondOp->getOperands()[1];
    // Input type of Less must be integer.
    if (!newCounterValue.getType()
             .cast<ShapedType>()
             .getElementType()
             .isa<IntegerType>())
      return std::make_pair(false, maxTripCountValue);

    // Compute a trip count from the break condition, given that the upper bound
    // is fixed and the lower bound is increased by a constant step at each
    // iteration. So, the trip count will be `(upper_bound - lower_bound)/step`.

    // Only support ONNXAddOp at this moment.
    if (newCounterValue.isa<BlockArgument>() ||
        !isa<ONNXAddOp>(newCounterValue.getDefiningOp()))
      return std::make_pair(false, maxTripCountValue);
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond = LessOp(newCounterValue, ubValue)
    //     ONNXReturnOp (cond, ..., ubValue, ..., newCounterValue, ...)
    Operation *addOp = cast<ONNXAddOp>(newCounterValue.getDefiningOp());
    Value counterValue = addOp->getOperands()[0];
    Value stepValue = addOp->getOperands()[1];
    // Counter is a block argument and updated at each iteration.
    if (!isUpdatedArgByValue(counterValue, newCounterValue, returnOp))
      return std::make_pair(false, maxTripCountValue);
    // Step must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(stepValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Check the lower bound of the break condition.
    // LowerBound is the initial value of the counter.
    Value lbValue = getFedValue(counterValue, loopOp);

    // Check the upper bound of the break condition.
    // UpperBound must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(ubValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Get values for upper bound and step if they are invariant arguments.
    // Otherwise, clone them to location outside the loop.
    if (isInvariantBlockArg(ubValue, returnOp))
      ubValue = getFedValue(ubValue, loopOp);
    else
      ubValue = cast<ONNXConstantOp>(rewriter.clone(*ubValue.getDefiningOp()))
                    .getResult();
    if (isInvariantBlockArg(stepValue, returnOp))
      stepValue = getFedValue(stepValue, loopOp);
    else
      stepValue =
          cast<ONNXConstantOp>(rewriter.clone(*stepValue.getDefiningOp()))
              .getResult();

    // Case 1: the upper bound, lower bound and step are constants.
    // - Compute the new max trip count at the compile time.
    if (isDefinedByIntegerConstantOp(lbValue) &&
        isDefinedByIntegerConstantOp(ubValue) &&
        isDefinedByIntegerConstantOp(stepValue)) {
      int64_t lowerBound = getOneIntegerConstant(lbValue);
      int64_t upperBound = getOneIntegerConstant(ubValue);
      int64_t step = getOneIntegerConstant(stepValue);
      if ((step <= 0) || (upperBound <= lowerBound))
        return std::make_pair(false, maxTripCountValue);
      int64_t derivedTripCount =
          ceil((1.0 * (upperBound - lowerBound)) / (1.0 * step));
      int64_t maxTripCount = getOneIntegerConstant(maxTripCountValue);

      // Check that the new trip count is smaller than the original trip count.
      if (maxTripCount <= derivedTripCount)
        return std::make_pair(false, maxTripCountValue);

      SmallVector<int64_t, 1> values(1, derivedTripCount);
      DenseElementsAttr valueAttr = DenseElementsAttr::get(
          RankedTensorType::get({},
              maxTripCountValue.getType().cast<ShapedType>().getElementType()),
          ArrayRef(values));
      return std::make_pair(true, onnx.constant(valueAttr));
    }

    // Case 2: Not all of the lower bound, upper bound and step are constants,
    // emit code to compute the new max trip count.
    // - new_max_trip_count =
    //      min(old_max_trip_count, ceil(upper_bound - lower_bound)/step)
    TypeAttr tripCountType = TypeAttr::get(
        maxTripCountValue.getType().cast<ShapedType>().getElementType());

    // Cast the upper and lower bounds to the correct type.
    if (maxTripCountValue.getType().cast<ShapedType>().getElementType() !=
        ubValue.getType().cast<ShapedType>().getElementType())
      ubValue = onnx.cast(ubValue, tripCountType);
    if (maxTripCountValue.getType().cast<ShapedType>().getElementType() !=
        lbValue.getType().cast<ShapedType>().getElementType())
      lbValue = onnx.cast(lbValue, tripCountType);

    // Emit code to compute the max trip count.
    Value range = onnx.sub(ubValue, lbValue);
    Value rangeInFloat = onnx.cast(range, TypeAttr::get(rewriter.getF32Type()));
    Value stepInFloat =
        onnx.cast(stepValue, TypeAttr::get(rewriter.getF32Type()));
    Value tripCountInFloat = onnx.ceil(onnx.div(rangeInFloat, stepInFloat));
    Value newMaxTripCountValue = onnx.cast(tripCountInFloat, tripCountType);

    return std::make_pair(
        true, onnx.min(ValueRange({maxTripCountValue, newMaxTripCountValue})));
  }
};

namespace {
// RNNOpRewriteLayoutPattern helper functions and classes.

template <typename ONNXOp>
void inferShapes(ONNXOp op) {
  if (failed(op.inferShapes([](Region &region) {})))
    llvm_unreachable("unexpected inferShapes failure");
}

// To transpose between [batch_size, seq_length/num_directions, size]
//                  and [seq_length/num_directions, batch_size, size].
ArrayAttr perm3RNN(Builder &b) { return b.getI64ArrayAttr({1, 0, 2}); }

// To transpose from [seq_length, num_directions, batch_size, hidden_size]
//                to [batch_size, seq_length, num_directions, hidden_size].
ArrayAttr perm4RNN(Builder &b) { return b.getI64ArrayAttr({2, 0, 1, 3}); }

class InputOutputTransposer {
public:
  InputOutputTransposer(OpBuilder &b, Location loc) : create(b, loc) {}

  void transposeInput(MutableOperandRange operand, ArrayAttr perm) {
    assert(operand.size() == 1 && "should be called with singleton range");
    Value input = operand[0];
    if (!input.getType().isa<NoneType>()) {
      Value transposed = transpose(input, perm);
      operand.assign(transposed);
    }
  }

  void transposeOutput(Value output, ArrayAttr perm) {
    if (!output.getType().isa<NoneType>()) {
      Value transposed = transpose(output, perm);
      output.replaceAllUsesExcept(transposed, transposed.getDefiningOp());
    }
  }

private:
  // Helper to create an ONNX transposition, using
  // ONNXTransposeOp::inferShapes() to infer the output shape.
  Value transpose(Value input, ArrayAttr perm) {
    Type elType = onnx_mlir::getElementType(input.getType());
    Type unrankedType = UnrankedTensorType::get({elType}); // placeholder
    Value transposed = create.transpose(unrankedType, input, perm);
    auto transposeOp = llvm::cast<ONNXTransposeOp>(transposed.getDefiningOp());
    inferShapes(transposeOp); // sets transposed's shape
    return transposed;
  }

  onnx_mlir::OnnxBuilder create;
};
} // namespace

// Rewrites layout=1 to layout=0 by transposing inputs and outputs.
template <typename ONNXOp>
class RNNOpRewriteLayoutPattern : public OpRewritePattern<ONNXOp> {
public:
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp onnxOp, PatternRewriter &rewriter) const override {
    if (onnxOp.getLayout() == 0) {
      return success();
    }

    InputOutputTransposer transposer(rewriter, onnxOp.getLoc());
    ArrayAttr perm3 = perm3RNN(rewriter);

    // LSTM requires extra work for initial_c input and Y_c output.
    auto onnxLSTMOp = llvm::dyn_cast<ONNXLSTMOp>(*onnxOp);

    // Rewrite in-place because there are so many attributes, inputs, outputs.
    // Constructing a new op would be lengthy and hard to maintain.
    rewriter.updateRootInPlace(onnxOp, [&]() {
      // Transpose the X and initial_h inputs by inserting an ONNXTransposeOp
      // before each and replacing the each input with the transpose output.
      rewriter.setInsertionPoint(onnxOp); // insert before (redundant)
      transposer.transposeInput(onnxOp.getXMutable(), perm3);
      transposer.transposeInput(onnxOp.getInitialHMutable(), perm3);
      if (onnxLSTMOp)
        transposer.transposeInput(onnxLSTMOp.getInitialCMutable(), perm3);
      // Set layout to zero.
      onnxOp->setAttr(onnxOp.getLayoutAttrName(),
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, /*isSigned=*/true), 0));
      // Update the output shape. Since the onnxOp is reused, it potentially had
      // some shape inference for its output. But since the input changed, we
      // don't want these now-erroneous output shapes to influence the output of
      // the revised op (as current output shape is used to potentially refine
      // existing shape inference). Long story short, we must reset the output
      // shapes. The call below does that. It is then safe to call shape
      // inference with the revised inputs.
      resetTypesShapeToQuestionmarks(onnxOp);
      inferShapes(onnxOp);
    });
    // Transpose the Y and Y_h outputs by inserting an ONNXTransposeOp
    // after each and replace all uses of each with the transpose output.
    ValueRange results = onnxOp.getResults();
    if (results.size() > 0) {
      rewriter.setInsertionPointAfter(onnxOp);
      transposer.transposeOutput(onnxOp.getY(), perm4RNN(rewriter));
      transposer.transposeOutput(onnxOp.getYH(), perm3);
      if (onnxLSTMOp)
        transposer.transposeOutput(onnxLSTMOp.getYC(), perm3);
    }

    return success();
  }
};

// =============================================================================
/// Register optimization patterns as "canonicalization" patterns.
/// Add op to OpsWithCanonicalizer in gen_onnx_mlir.py to activate.
/// Please keep in alphabetical order.
// =============================================================================

/// on the ONNXBatchNormalizationInferenceModeOp.
void ONNXBatchNormalizationInferenceModeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FuseBatchNormInferenceModeConvPattern>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern1>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern2>(context);
}

/// on the ONNXAddOp.
void ONNXAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeAddPattern>(context);
  results.insert<MulAddToGemmOptPattern>(context);
  results.insert<FuseGemmFollowedByAddition>(context);
  results.insert<FuseAddConvPattern>(context);
  results.insert<FuseAddConvNullBiasPattern>(context);
}

/// on the ONNXCastOp.
void ONNXCastOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<CastEliminationPattern>(context);
  result.insert<FuseCastCastPattern>(context);
}

/// on the ONNXConstantOp.
void ONNXConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConstantOpNormalizationPattern1>(context);
  results.insert<ConstantOpNormalizationPattern2>(context);
  results.insert<ConstantOpNormalizationPattern3>(context);
  results.insert<ConstantOpNormalizationPattern4>(context);
  results.insert<ConstantOpNormalizationPattern5>(context);
  results.insert<ConstantOpNormalizationPattern6>(context);
}

/// on the ONNXConvTransposeOp.
void ONNXConvTransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConvTransposeOpPattern1>(context);
  results.insert<ConvTransposeOpPattern2>(context);
}

/// on the ONNXDepthToSpaceOp.
void ONNXDepthToSpaceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDepthToSpaceSpaceToDepthPattern>(context);
}

/// on the ONNXDropoutOp.
void ONNXDropoutOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<DropoutEliminationPattern>(context);
}

/// on the ONNXDimOp.
void ONNXDimOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DimOpToConstantPattern>(context);
}

/// on the ONNXGlobalAveragePoolOp.
void ONNXGlobalAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalAveragePoolPattern>(context);
}

/// on the ONNXGlobalMaxPoolOp.
void ONNXGlobalMaxPoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalMaxPoolPattern>(context);
}

/// on the ONNXGRUOp.
void ONNXGRUOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXGRUOp>>(context);
}

/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXLayoutTransformOp.
void ONNXLayoutTransformOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ONNXLayoutTransformEliminationPattern>(context);
}

/// on the ONNXLessOp.
void ONNXLessOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LessOpSameCastPattern>(context);
}

/// on the ONNXLoopOp.
void ONNXLoopOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LoopOpRewriteMaxTripCountPattern>(context);
}

/// on the ONNXLSTMOp.
void ONNXLSTMOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXLSTMOp>>(context);
}

/// on the ONNXMulOp.
void ONNXMulOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeMulPattern>(context);
  results.insert<FuseMulConvNullBiasPattern>(context);
}

/// on the ONNXReshapeOp.
void ONNXReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseReshapePattern>(context);
  result.insert<RemoveIdentityReshapePattern>(context);
  result.insert<SwapReshapeMatMulPattern>(context);
}

/// on the ONNXRNNOp.
void ONNXRNNOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXRNNOp>>(context);
}

/// on the ONNXShapeOp.
void ONNXShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ShapeToConstantPattern>(context);
}

/// on the ONNXSizeOp.
void ONNXSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SizeToConstantPattern>(context);
}

/// on the ONNXSoftmaxV11Op.
void ONNXSoftmaxV11Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SoftmaxV11ToLatestPattern>(context);
}

/// on the ONNXSpaceToDepthOp.
void ONNXSpaceToDepthOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveSpaceToDepthDepthToSpacePattern>(context);
}

/// on the ONNXSqueezeOp.
void ONNXSqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeUnsqueezePattern>(context);
  result.insert<RemoveSqueezeCastUnsqueezePattern>(context);
}

void ONNXSqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeV11UnsqueezeV11Pattern>(context);
  result.insert<RemoveSqueezeV11CastUnsqueezeV11Pattern>(context);
}

/// on the ONNXTransposeOp.
void ONNXTransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseTransposePattern>(context);
  result.insert<FuseTransposeAndAtanPattern>(context);
  result.insert<FuseTransposeAndCastPattern>(context);
  result.insert<FuseTransposeAndCeilPattern>(context);
  result.insert<FuseTransposeAndCosPattern>(context);
  result.insert<FuseTransposeAndCoshPattern>(context);
  result.insert<FuseTransposeAndEluPattern>(context);
  result.insert<FuseTransposeAndErfPattern>(context);
  result.insert<FuseTransposeAndAcosPattern>(context);
  result.insert<FuseTransposeAndAcoshPattern>(context);
  result.insert<FuseTransposeAndAsinPattern>(context);
  result.insert<FuseTransposeAndAsinhPattern>(context);
  result.insert<FuseTransposeAndAtanhPattern>(context);
  result.insert<FuseTransposeAndExpPattern>(context);
  result.insert<FuseTransposeAndFloorPattern>(context);
  result.insert<FuseTransposeAndHardSigmoidPattern>(context);
  result.insert<FuseTransposeAndIsNaNPattern>(context);
  result.insert<FuseTransposeAndLeakyReluPattern>(context);
  result.insert<FuseTransposeAndLogPattern>(context);
  result.insert<FuseTransposeAndNegPattern>(context);
  result.insert<FuseTransposeAndNotPattern>(context);
  result.insert<FuseTransposeAndReciprocalPattern>(context);
  result.insert<FuseTransposeAndReluPattern>(context);
  result.insert<FuseTransposeAndRoundPattern>(context);
  result.insert<FuseTransposeAndSeluPattern>(context);
  result.insert<FuseTransposeAndSigmoidPattern>(context);
  result.insert<FuseTransposeAndSignPattern>(context);
  result.insert<FuseTransposeAndSinPattern>(context);
  result.insert<FuseTransposeAndSinhPattern>(context);
  result.insert<FuseTransposeAndSoftplusPattern>(context);
  result.insert<FuseTransposeAndSoftsignPattern>(context);
  result.insert<FuseTransposeAndSqrtPattern>(context);
  result.insert<FuseTransposeAndTanPattern>(context);
  result.insert<FuseTransposeAndTanhPattern>(context);
  result.insert<RemoveIdentityTransposePattern>(context);
  result.insert<SwapTransposeConcatPattern>(context);
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
  result.insert<RemoveUnsqueezeCastSqueezePattern>(context);
}

void ONNXUnsqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeV11SqueezeV11Pattern>(context);
  result.insert<RemoveUnsqueezeV11CastSqueezeV11Pattern>(context);
}
