/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToKrnlCommon.cpp - ONNX dialects to Krnl lowering ---------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

Value OnnxToKrnlBuilder::reshape(
    const Value input, const ArrayRef<DimIndexExpr> shapeDims) const {
  assert(!shapeDims.empty() && "Shape dimensions should not be empty");

  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  MultiDialectBuilder<OnnxBuilder, MemRefBuilder, KrnlBuilder, MathBuilder>
      create(b(), loc());

  // If the output dimensions are all literals the 'onnx/Reshape' operation
  // can take the new shape via an 'onnx.Constant'.
  if (llvm::all_of(
          shapeDims, [](const DimIndexExpr &dim) { return dim.isLiteral(); })) {
    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : shapeDims)
      shape.push_back(dim.getLiteral());

    auto constantOp = createONNXConstantOpWithDenseAttr(
        b(), loc(), b().getI64TensorAttr(shape));

    Value reshapeRes = create.onnx.reshape(
        MemRefType::get(shape, elementType), input, constantOp);

    return reshapeRes;
  }

  // When the output dimensions aren't all literals we need to generate code
  // to compute the shape. Allocate a buffer and store the output dimension
  // into it.
  IndexType indexTy = b().getIndexType();
  int64_t length = shapeDims.size();
  memref::AllocOp alloc =
      create.mem.alignedAlloc(MemRefType::get({length}, indexTy), 16);

  for (int64_t i = 0; i < length; ++i) {
    Value index = create.math.constant(indexTy, i);
    Value data = shapeDims[i].getValue();
    create.krnl.store(data, alloc, index);
  }

  // Now create the 'onnx.Reshape' operation. Because the shape is not a
  // compile time constant it is effectively unknown.
  SmallVector<int64_t> shape(length, ShapedType::kDynamic);
  Value reshapeRes =
      create.onnx.reshape(MemRefType::get(shape, elementType), input, alloc);

  // The 'onnx.Reshape' operation yields a memref with unknown extents, so we
  // need to explicitly cast the result to the know size.
  SmallVector<int64_t, 6> castOutputShape;
  for (const IndexExpr &dim : shapeDims)
    castOutputShape.push_back(
        dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);

  Value castRes = create.mem.cast(create.onnx.toMemref(reshapeRes),
      MemRefType::get(castOutputShape, elementType));

  return castRes;
}

Value OnnxToKrnlBuilder::transpose(const Value input,
    const ArrayRef<int64_t> perm,
    const ArrayRef<DimIndexExpr> outputDims) const {
  assert(!outputDims.empty() && "Output dimensions should not be empty");
  assert(!perm.empty() && perm.size() == outputDims.size() &&
         "Expecting valid permutation array");
  MultiDialectBuilder<OnnxBuilder> create(b(), loc());

  // Compute the shape of the 'onnx.Transpose' result.
  SmallVector<int64_t, 6> shape;
  for (const IndexExpr &dim : outputDims)
    shape.push_back(dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);

  // Create the "onnx.Transpose" operation.
  ShapedType inputType = input.getType().cast<ShapedType>();
  Value transposeRes =
      create.onnx.transpose(MemRefType::get(shape, inputType.getElementType()),
          input, b().getI64ArrayAttr(perm));

  return transposeRes;
}

/// Check if all operands are scalar values at compile time.
bool hasAllScalarValues(ArrayRef<Value> values) {
  for (Value value : values) {
    if (value.getType().cast<ShapedType>().getRank() != 0)
      return false;
  }
  return true;
}

/// Check if the value is a KrnlGlobalOp with a dense attribute of non-negative
/// integer constants.
bool indicesAreNonNegativeConstants(Value indices) {
  DenseElementsAttr valueAttribute =
      krnl::getDenseElementAttributeFromKrnlValue(indices);
  if (!valueAttribute || !valueAttribute.getElementType().isa<IntegerType>())
    return false;

  return llvm::all_of(valueAttribute.getValues<IntegerAttr>(),
      [](const IntegerAttr &val) { return val.getInt() >= 0; });
}

// Create a mapping from result type's dimensions to input type's dimensions,
// given that the result type is the result of a reduction op over the input
// type.
std::map<int64_t, int64_t> getReductionMapping(
    MemRefType inputTy, ArrayRef<int64_t> axes, bool keepdims) {
  std::map<int64_t, int64_t> OutInDimMap;
  int64_t rank = inputTy.getRank();

  // Mark reduction axes.
  std::vector<bool> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  for (decltype(rank) inIndex = 0, outIndex = 0; inIndex < rank; ++inIndex) {
    // If it is a reduction axis, there is no relationship among dimensions.
    if (isReductionAxis[inIndex]) {
      if (keepdims)
        outIndex++;
    } else {
      OutInDimMap.insert(std::make_pair(outIndex, inIndex));
      outIndex++;
    }
  }

  return OutInDimMap;
}

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimension are supported.
void addDimensionToPack(ConversionPatternRewriter &rewriter, Location loc,
    krnl::KrnlIterateOperandPack &pack, Value operand, int index) {
  auto shape = operand.getType().cast<MemRefType>().getShape();
  assert(shape[index] != -1 && "expected kDynamic, not -1");
  if (shape[index] == ShapedType::kDynamic) {
    MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
    pack.pushConstantBound(0);
    pack.pushOperandBound(create.mem.dim(operand, index));
  } else {
    pack.pushConstantBound(0);
    pack.pushConstantBound(shape[index]);
  }
}

// Function that emits the definition of loops references.
void defineLoops(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, int64_t numLoops) {
  MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
  ValueRange loopsOp = create.krnl.defineLoops(numLoops);
  loops.reserve(numLoops);
  for (auto result : loopsOp)
    loops.push_back(result);
}

Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis, Type type) {
  MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
  ArrayRef<int64_t> shape = operand.getType().cast<ShapedType>().getShape();
  assert(shape[axis] != -1 && "expected kDynamic, not -1");
  return (shape[axis] == ShapedType::kDynamic)
             ? create.math.cast(type, create.mem.dim(operand, axis))
             : create.math.constant(type, shape[axis]);
}

namespace {
// Returns the DenseElementsAttr of input if it's a krnl.global constant or
// onnx.Constant, or if it's one step removed from a krnl/onnx constant by a
// builtin.unrealized_conversion_cast. Otherwise returns a nullptr attribute.
DenseElementsAttr getDenseElementAttrFromConstValue(mlir::Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(definingOp)) {
    if (castOp.getNumOperands() != 1)
      return nullptr;
    definingOp = castOp.getOperand(0).getDefiningOp();
  }
  if (auto globalOp = dyn_cast_or_null<KrnlGlobalOp>(definingOp)) {
    if (globalOp.getValue().has_value())
      return globalOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}
} // namespace

/// Emit an ONNXSqueezeV11Op. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXSqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr squeezedElements = inputElements.reshape(tensorType);
    Value constVal = create.onnx.constant(squeezedElements);
    return create.onnx.toMemref(constVal);
  } else {
    return create.onnx.toMemref(
        rewriter
            .create<ONNXSqueezeV11Op>(loc, tensorType,
                create.onnx.toTensor(input), rewriter.getI64ArrayAttr(axis))
            .getResult());
  }
}

/// Emit an ONNXUnsqueezeV11Op. If the input is constant, do const
/// propagation, and return a constant.
Value foldOrEmitONNXUnsqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr unsqueezedElements = inputElements.reshape(tensorType);
    Value constVal = create.onnx.constant(unsqueezedElements);
    return create.onnx.toMemref(constVal);
  } else {
    return create.onnx.toMemref(
        rewriter
            .create<ONNXUnsqueezeV11Op>(loc, tensorType,
                create.onnx.toTensor(input), rewriter.getI64ArrayAttr(axis))
            .getResult());
  }
}

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> foldOrEmitONNXSplitOp(ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Type> resultTypes, Value input, int64_t axis) {

  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  std::vector<Value> resVals;
  int outputNum = resultTypes.size();

  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    auto inputShape = inputElements.getType().getShape();
    assert(outputNum == 0 || inputShape[axis] % outputNum == 0);
    int64_t sizeOfEachSplit = outputNum != 0 ? inputShape[axis] / outputNum : 0;
    SmallVector<int64_t, 4> sizes(outputNum, sizeOfEachSplit);

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    std::vector<ElementsAttr> splits =
        elementsBuilder.split(inputElements, axis, sizes);
    for (ElementsAttr splitElements : splits) {
      // Avoid DisposableElementsAttr during conversion.
      DenseElementsAttr denseSplitElements =
          elementsBuilder.toDenseElementsAttr(splitElements);
      Value constVal = create.onnx.constant(denseSplitElements);
      resVals.emplace_back(create.onnx.toMemref(constVal));
    }
  } else {
    SmallVector<Type, 4> convertedTypes;
    for (auto t : resultTypes) {
      convertedTypes.emplace_back(create.onnx.toTensor(t));
    }
    ONNXSplitV11Op split = rewriter.create<ONNXSplitV11Op>(loc, convertedTypes,
        create.onnx.toTensor(input),
        /*axis=*/axis, nullptr);
    for (int i = 0; i < outputNum; ++i)
      resVals.emplace_back(create.onnx.toMemref(split.getOutputs()[i]));
  }
  return resVals;
}

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXTransposeOp(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, ArrayAttr permAttr) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    SmallVector<uint64_t, 4> perm;
    for (auto permVal : permAttr.getValue())
      perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    ElementsAttr transposedElements =
        elementsBuilder.transpose(inputElements, perm);
    // Avoid DisposableElementsAttr during conversion.
    DenseElementsAttr denseTransposedElements =
        elementsBuilder.toDenseElementsAttr(transposedElements);
    Value constVal = create.onnx.constant(denseTransposedElements);
    return create.onnx.toMemref(constVal);
  } else {
    return create.onnx.toMemref(
        rewriter
            .create<ONNXTransposeOp>(loc, create.onnx.toTensor(resultType),
                create.onnx.toTensor(input), permAttr)
            .getResult());
  }
}

/// Emit MemRef ReinterpretCastOp to create a new view for 'data'.
/// The new view is created using the given 'outputDims'.
Value emitMemRefReinterpretCastOp(ConversionPatternRewriter &rewriter,
    Location loc, Value data, SmallVectorImpl<IndexExpr> &outputDims,
    Type outputType) {
  MemRefBuilder createMemRef(rewriter, loc);
  Value newView = createMemRef.reinterpretCast(data, outputDims);
  // Set type to the output type to avoid unrealized_conversion_cast.
  // It's because the output type is sometimes better than the inferred type,
  // e.g. the output type has a static dim (e.g. set by users) that can be
  // dynamic in the inferred type.
  newView.setType(outputType);
  return newView;
}

/// Emit krnl iterate to compute argsort of a given MemRef along a given axis.
/// Output MemRef has the same shape as the input MemRef but is of IndexType.
/// By default, sort values in the descending order.
Value emitArgSort(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t axis, bool ascending) {
  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
      MemRefBuilder>
      create(rewriter, loc);
  IndexExprScope scope(create.krnl);

  MemRefType inputMemRefType = input.getType().cast<MemRefType>();
  Type indexType = rewriter.getIndexType();
  int64_t rank = inputMemRefType.getRank();
  assert(axis >= 0 && axis < rank && "axis is out of bound");
  LiteralIndexExpr zeroIE(0), oneIE(1);

  SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
  SmallVector<IndexExpr, 4> ubs;
  create.krnlIE.getShapeAsDims(input, ubs);

  // Create and initialize the result.
  MemRefType type = MemRefType::get(inputMemRefType.getShape(), indexType);
  Value order = create.mem.alignedAlloc(type, ubs);
  ValueRange initLoopDef = create.krnl.defineLoops(rank);
  create.krnl.iterateIE(initLoopDef, initLoopDef, lbs, ubs,
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        // order[axis_0, axis_1, ..., axis_k-1, k, axis_k+1, ....] = k
        createKrnl.store(loopInd[axis], order, loopInd);
      });

  // Do sorting in the specifed order of input and return their indices.
  if ((rank <= 6) && (axis == (rank - 1))) {
    // Emit krnl.Call to call omTensorSort API
    Type intType = rewriter.getIntegerType(64);
    Value valAxis = create.math.constant(intType, axis);
    Value valAscending = create.math.constant(intType, (int64_t)ascending);
    SmallVector<Value, 4> operands = {input, valAxis, valAscending};
    rewriter.create<KrnlCallOp>(loc, "omTensorSort", order, operands);
    return order;
  }
  // Do sorting in the descending order of input and return their indices.
  // Using bubble sort.
  SmallVector<IndexExpr, 4> outerUbs(ubs);
  outerUbs[axis] = ubs[axis] - oneIE;
  ValueRange loopDef = create.krnl.defineLoops(rank);
  create.krnl.iterateIE(loopDef, loopDef, lbs, outerUbs,
      [&](KrnlBuilder &createKrnl, ValueRange iLoopInd) {
        IndexExpr i1 = DimIndexExpr(iLoopInd[axis]) + oneIE;
        ValueRange swapLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(swapLoopDef, swapLoopDef, {i1}, {ubs[axis]},
            [&](KrnlBuilder &ck, ValueRange swapLoopInd) {
              MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                  ck);
              SmallVector<Value> kLoopInd(iLoopInd);
              kLoopInd[axis] = swapLoopInd[0];
              // Load current indices.
              Value iOrd = create.krnl.load(order, iLoopInd);
              Value kOrd = create.krnl.load(order, kLoopInd);
              // Load x.
              SmallVector<Value> xLoopInd(iLoopInd);
              xLoopInd[axis] = iOrd;
              Value x = create.krnl.load(input, xLoopInd);
              // Load y.
              SmallVector<Value> yLoopInd(iLoopInd);
              yLoopInd[axis] = kOrd;
              Value y = create.krnl.load(input, yLoopInd);
              // Compare values and swap indices.
              Value cond;
              if (ascending)
                cond = create.math.sgt(x, y);
              else
                cond = create.math.slt(x, y);
              create.scf.ifThenElse(cond, [&](SCFBuilder &createSCF) {
                KrnlBuilder createKrnl(createSCF);
                createKrnl.store(kOrd, order, iLoopInd);
                createKrnl.store(iOrd, order, kLoopInd);
              });
            });
      });

  return order;
}

/// This function returns a scalar of type 'dtype' from an optional value.
/// Optional value must be: NoneType, memref<1xdtype> or memref<dtype>.
/// Default value is used in case of NoneType.
Value getOptionalScalarValue(ConversionPatternRewriter &rewriter, Location loc,
    Value optionalScalar, Type elementType, double defaultValue) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  if (optionalScalar.getType().isa<NoneType>()) {
    return create.math.constant(elementType, defaultValue);
  } else if (optionalScalar.getType().cast<ShapedType>().getRank() == 0) {
    return create.krnl.load(optionalScalar, {});
  } else {
    Value zero = create.math.constantIndex(0);
    return create.krnl.load(optionalScalar, {zero});
  }
}

//===----------------------------------------------------------------------===//
// Support functions for help with custom layout.
//===----------------------------------------------------------------------===//

MemRefType convertTypeWithCustomONNXDataLayoutToMemRef(Type type) {
  // Get tensor rank, shape, and element type.
  RankedTensorType tensorType = type.dyn_cast<RankedTensorType>();
  assert(tensorType && "expected only ranked shapes");
  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t rank = shape.size();
  Type elementType = tensorType.getElementType();
  // Get encoding.
  mlir::ONNXTensorEncodingAttr encoding = getONNXTensorEncoding(type);
  assert(encoding && "expected ONNX tensor encoding");
  // Process encoding to generate an vector of affine expressions (dimExpr)
  // corresponding to the data layout.
  SmallVector<AffineExpr, 6> dimExpr;
  OpBuilder b(type.getContext());
  MLIRContext *context = b.getContext();
  if (encoding.getDataLayout() == ONNXTensorEncodingAttr::DataLayout::NCHWxC) {
    // perform the map for (N, C, H, W) -> (N, C/x, H, W, C%x) with C=Cin
    // tiled by x.
    int64_t N(0), C(1), H(2), W(3); // Indices for dims in affine expressions.
    int64_t xVal(encoding.getXFactor());
    assert(xVal > 0 && "expected strictly positive X factor");
    AffineExpr x = getAffineConstantExpr(xVal, context);
    AffineExpr newN = b.getAffineDimExpr(N);
    AffineExpr newC = b.getAffineDimExpr(C).floorDiv(x);
    AffineExpr newH = b.getAffineDimExpr(H);
    AffineExpr newW = b.getAffineDimExpr(W);
    AffineExpr newXC = b.getAffineDimExpr(C) % x;
    dimExpr.emplace_back(newN);
    dimExpr.emplace_back(newC);
    dimExpr.emplace_back(newH);
    dimExpr.emplace_back(newW);
    dimExpr.emplace_back(newXC);
  } else if (encoding.getDataLayout() ==
             ONNXTensorEncodingAttr::DataLayout::KCNMxCyK) {
    // perform the map for (K, C, N, M) -> (K/y, C/x, M, N, C%x, K%y) with
    // C=Cin and K=Cout tiled by x and y.
    int64_t K(0), C(1), N(2), M(3); // Indices for dims in affine expressions.
    int64_t xVal(encoding.getXFactor());
    int64_t yVal(encoding.getYFactor());
    assert(xVal > 0 && yVal > 0 && "expected strictly positive X & Y factors");
    AffineExpr x = getAffineConstantExpr(xVal, context);
    AffineExpr y = getAffineConstantExpr(yVal, context);
    AffineExpr newK = b.getAffineDimExpr(K).floorDiv(y);
    AffineExpr newC = b.getAffineDimExpr(C).floorDiv(x);
    AffineExpr newN = b.getAffineDimExpr(N);
    AffineExpr newM = b.getAffineDimExpr(M);
    AffineExpr newXC = b.getAffineDimExpr(C) % x;
    AffineExpr newYK = b.getAffineDimExpr(K) % y;
    dimExpr.emplace_back(newK);
    dimExpr.emplace_back(newC);
    dimExpr.emplace_back(newN);
    dimExpr.emplace_back(newM);
    dimExpr.emplace_back(newXC);
    dimExpr.emplace_back(newYK);
  } else if (encoding.getDataLayout() ==
             ONNXTensorEncodingAttr::DataLayout::STANDARD) {
    llvm_unreachable(
        "should not have an ONNX tensor encoding for standard data layout");
  } else {
    llvm_unreachable("unknown ONNX tensor encoding");
  }
  // Have our new dims affine expressions.
  AffineMap map =
      AffineMap::get(/*dims*/ rank, /*symbols*/ 0, dimExpr, context);
  MemRefType outType = MemRefType::get(shape, elementType);
  return MemRefType::Builder(outType).setLayout(AffineMapAttr::get(map));
}

//===----------------------------------------------------------------------===//
// Type conversion from Onnx types to Krnl types.
//===----------------------------------------------------------------------===//

KrnlTypeConverter::KrnlTypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addConversion([](ONNXStringType stringType) {
    return krnl::StringType::get(stringType.getContext());
  });

  addConversion([](TensorType tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    if (tensorType.getElementType().isa<ONNXStringType>()) {
      Type elementType = krnl::StringType::get(tensorType.getContext());
      return MemRefType::get(tensorType.getShape(), elementType);
    }
    // Accelerators may have special versions of TensorType. Call the
    // conversions of accelerators.
    for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
      MemRefType memRefType = accel->convertTensorTypeToMemRefType(tensorType);
      if (memRefType)
        return memRefType;
    }
    if (hasCustomONNXTensorDataLayout(tensorType))
      return convertTypeWithCustomONNXDataLayoutToMemRef(tensorType);
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  });

  addConversion([](SeqType seqType) {
    ShapedType seqElementType = seqType.getElementType();
    Type elementType = seqElementType.getElementType();
    Type seqElementConvertedType;
    if (seqElementType.hasRank()) {
      seqElementConvertedType =
          MemRefType::get(seqElementType.getShape(), elementType);
    } else {
      seqElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    }
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(seqType.getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    return MemRefType::get(shape, seqElementConvertedType);
  });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

int64_t KrnlTypeConverter::getDefaultAllocAlignment(Type type) {
  int64_t alignment = -1;
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    // Accelerators may have special versions of TensorType. Call the
    // conversions of accelerators.
    for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
      // The accelerator knows whether `tensorType` is its target or not to
      // decide the alignment.
      // -1 means the accelerator does not have a specific alignment.
      alignment = accel->getDefaultAllocAlignment(tensorType);
      if (alignment != -1)
        break;
    }
  }
  return alignment;
}

} // namespace onnx_mlir
