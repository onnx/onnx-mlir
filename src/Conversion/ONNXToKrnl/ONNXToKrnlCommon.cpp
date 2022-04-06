/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToKrnlCommon.cpp - ONNX dialects to Krnl lowering ---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

bool ONNXToKrnl_gEmitDealloc = false;

namespace onnx_mlir {

Value OnnxToKrnlBuilder::reshape(
    const Value input, const ArrayRef<DimIndexExpr> shapeDims) const {
  assert(!shapeDims.empty() && "Shape dimensions should not be empty");

  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();

  // If the output dimensions are all literals the 'onnx/Reshape' operation
  // can take the new shape via an 'onnx.Constant'.
  if (llvm::all_of(
          shapeDims, [](const DimIndexExpr &dim) { return dim.isLiteral(); })) {
    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : shapeDims)
      shape.push_back(dim.getLiteral());

    auto constantOp =
        createONNXConstantOpWithDenseAttr(b, loc, b.getI64TensorAttr(shape));

    Value reshapeRes = b.create<ONNXReshapeOp>(
        loc, MemRefType::get(shape, elementType), input, constantOp);

    return reshapeRes;
  }

  MemRefBuilder memRefBuilder(b, loc);
  KrnlBuilder krnlBuilder(memRefBuilder);

  // When the output dimensions aren't all literals we need to generate code
  // to compute the shape. Allocate a buffer and store the putput dimension
  // into it.
  IndexType indexTy = b.getIndexType();
  int64_t length = shapeDims.size();
  memref::AllocOp alloc =
      memRefBuilder.alignedAlloc(MemRefType::get({length}, indexTy), 16);

  for (int64_t i = 0; i < length; ++i) {
    Value index = emitConstantOp(b, loc, indexTy, i);
    Value data = shapeDims[i].getValue();
    krnlBuilder.store(data, alloc, index);
  }

  // Now create the 'onnx.Reshape' operation. Because the shape is not a
  // compile time constant it is effectively unknown.
  SmallVector<int64_t> shape(length, -1);
  Value reshapeRes = b.create<ONNXReshapeOp>(
      loc, MemRefType::get(shape, elementType), input, alloc);

  // The 'onnx.Reshape' operation yields a memref with unknown extents, so we
  // need to explicitly cast the result to the know size.
  SmallVector<int64_t, 6> castOutputShape;
  for (const IndexExpr &dim : shapeDims)
    castOutputShape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

  Value castRes = memRefBuilder.cast(
      reshapeRes, MemRefType::get(castOutputShape, elementType));

  return castRes;
}

Value OnnxToKrnlBuilder::transpose(const Value input,
    const ArrayRef<int64_t> perm,
    const ArrayRef<DimIndexExpr> outputDims) const {
  assert(!outputDims.empty() && "Output dimensions should not be empty");
  assert(!perm.empty() && perm.size() == outputDims.size() &&
         "Expecting valid permutation array");

  // Compute the shape of the 'onnx.Transpose' result.
  SmallVector<int64_t, 6> shape;
  for (const IndexExpr &dim : outputDims)
    shape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

  // Create the "onnx.Transpose" operation.
  ShapedType inputType = input.getType().cast<ShapedType>();
  Value transposeRes = b.create<ONNXTransposeOp>(loc,
      MemRefType::get(shape, inputType.getElementType()), input,
      b.getI64ArrayAttr(perm));

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

/// Get the corresponding MemRefType of a given TensorType/SeqType/MemRefType.
MemRefType convertToMemRefType(Type type) {
  // Convert the element type of the (tensor or memref) to a valid Krnl type.
  auto convertElemType = [](Type elemType) -> Type {
    if (elemType.isa<ONNXStringType>())
      return krnl::StringType::get(elemType.getContext());
    return elemType;
  };

  if (auto tensorType = type.dyn_cast_or_null<TensorType>()) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    MemRefType memRefType = MemRefType::get(
        tensorType.getShape(), convertElemType(tensorType.getElementType()));
    return memRefType;
  }

  if (auto seqType = type.dyn_cast_or_null<SeqType>()) {
    ShapedType seqElementType = seqType.getElementType();
    Type seqElementMemRefType =
        seqElementType.hasRank()
            ? (Type)convertToMemRefType(seqElementType)
            : (Type)UnrankedMemRefType::get(seqElementType.getElementType(), 0);
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(seqType.getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    MemRefType memRefType = MemRefType::get(shape, seqElementMemRefType);
    return memRefType;
  }

  assert(type.isa<MemRefType>() && "Expecting a MemRefType");
  auto memRefType = type.cast<MemRefType>();
  return MemRefType::get(
      memRefType.getShape(), convertElemType(memRefType.getElementType()));
}

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter &rewriter, bool insertDealloc, Value operand,
    int64_t alignment) {
  MemRefBuilder createMemRef(rewriter, loc);
  // Put together alloc operands for any dynamic dimensions of the memref.
  memref::AllocOp alloc;
  if (operand) {
    auto memRefShape = type.getShape();
    auto rank = memRefShape.size();

    SmallVector<Value, 4> allocOperands;
    for (unsigned int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0) {
        auto dim = createMemRef.dim(operand, i);
        allocOperands.push_back(dim);
      }
    alloc = createMemRef.alignedAlloc(type, allocOperands, alignment);
  } else {
    alloc = createMemRef.alignedAlloc(type, alignment);
  }

  if (!ONNXToKrnl_gEmitDealloc)
    return alloc;

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  if (insertDealloc) {
    auto dealloc = createMemRef.dealloc(alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  return alloc;
}

// Simple version of insert alloc and dealloc that does not handle alignment
// or additional operands (to be studied and added as needed). For unknown
// dimensions, it uses the index expressions to retrieve the corresponding
// values.
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, const SmallVectorImpl<IndexExpr> &outputDims,
    bool insertDealloc, int64_t alignment) {
  // Constant, use the normal insert with no additional operands or alignment.
  if (hasAllConstantDimensions(type))
    return insertAllocAndDealloc(
        type, loc, rewriter, insertDealloc, nullptr, alignment);
  // Otherwise, take the unkown operands from the output dim IndexExpressions
  SmallVector<Value, 2> allocOperands;
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();

  for (unsigned int i = 0; i < rank; ++i) {
    if (memRefShape[i] < 0) {
      // have dyn shape
      allocOperands.emplace_back(outputDims[i].getValue());
    }
  }
  MemRefBuilder createMemRef(rewriter, loc);
  memref::AllocOp allocOp =
      createMemRef.alignedAlloc(type, allocOperands, alignment);

  if (!ONNXToKrnl_gEmitDealloc)
    return allocOp;

  if (insertDealloc) {
    auto *parentBlock = allocOp.getOperation()->getBlock();
    auto dealloc = createMemRef.dealloc(allocOp);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return allocOp;
}

Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, const SmallVectorImpl<IndexExpr> &outputDims,
    int64_t alignment) {

  bool insertDealloc = checkInsertDealloc(op);

  return insertAllocAndDeallocSimple(
      rewriter, op, type, loc, outputDims, insertDealloc, alignment);
}

// Determine if current function returns the result value of the
// current op or the result value of reinterpret_cast op whose
// operand is the result value of current op. If it does then
// dealloc should not be inserted.
bool checkInsertDealloc(Operation *currentOp, int resultIndex) {
  if (ONNXToKrnl_gEmitDealloc == false)
    return false;

  auto parentBlock = currentOp->getBlock();
  bool insertDealloc = true;

  // Check if the result value of `currentOp` is an operand of
  // `ReinterpretCastOp`, and store the result value of `ReinterpretCastOp`.
  // Reshape, Squeeze, and Unsqueeze ops are checked because they are lowered to
  // `ReinterpretCastOp`.
  SmallVector<Value, 32> castOpResults;
  if (currentOp->getNumResults() > 0) {
    parentBlock->walk([currentOp, resultIndex, &castOpResults](Operation *op) {
      if (isa<memref::ReinterpretCastOp>(op) || isa<ONNXReshapeOp>(op) ||
          isa<ONNXSqueezeV11Op>(op) || isa<ONNXUnsqueezeV11Op>(op) ||
          isa<ONNXSqueezeOp>(op) || isa<ONNXUnsqueezeOp>(op)) {
        auto result = currentOp->getResult(resultIndex);
        for (const auto &operand : op->getOperands())
          if (operand == result)
            castOpResults.emplace_back(op->getResults()[0]);
      }
    });
  }
  // If there is at least one result to investigate.
  if (currentOp->getNumResults() > 0) {
    parentBlock->walk(
        [&insertDealloc, currentOp, resultIndex, &castOpResults](ReturnOp op) {
          auto result = currentOp->getResult(resultIndex);
          for (const auto &operand : op.getOperands()) {
            // Determine if current function returns the result value of the
            // current op.
            if (operand == result)
              insertDealloc = false;
            // Determin if the result value of reinterpret_cast op whose operand
            // is the result value of current op
            for (const auto &castOpResult : castOpResults)
              if (operand == castOpResult)
                insertDealloc = false;
          }
        });
  }
  return insertDealloc;
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
  if (shape[index] < 0) {
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
  return (shape[axis] < 0)
             ? create.math.cast(type, create.mem.dim(operand, axis))
             : create.math.constant(type, shape[axis]);
}

/// Emit an ONNXSqueezeV11Op. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXSqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  if (krnl::isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
    char *inputBuffer = createArrayFromDenseElementsAttr(
        input.getDefiningOp()
            ->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>());

    Value constVal = createDenseONNXConstantOp(
        rewriter, loc, resultType.cast<ShapedType>(), inputBuffer)
                         .getResult();
    free(inputBuffer);
    return constVal;
  } else {
    return rewriter
        .create<ONNXSqueezeV11Op>(
            loc, resultType, input, rewriter.getI64ArrayAttr(axis))
        .getResult();
  }
}

/// Emit an ONNXUnsqueezeV11Op. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXUnsqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  if (krnl::isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
    char *inputBuffer = createArrayFromDenseElementsAttr(
        input.getDefiningOp()
            ->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>());

    Value constVal = createDenseONNXConstantOp(
        rewriter, loc, resultType.cast<ShapedType>(), inputBuffer)
                         .getResult();
    free(inputBuffer);
    return constVal;
  } else {
    return rewriter
        .create<ONNXUnsqueezeV11Op>(
            loc, resultType, input, rewriter.getI64ArrayAttr(axis))
        .getResult();
  }
}

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> foldOrEmitONNXSplitOp(ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Type> resultTypes, Value input, int64_t axis) {
  std::vector<Value> resVals;

  int outputNum = resultTypes.size();
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  Type elementType = inputType.getElementType();

  // Compute split offsets.
  SmallVector<int64_t, 4> splitOffsets;
  int64_t offset = 0;
  for (int i = 0; i < outputNum; ++i) {
    splitOffsets.emplace_back(offset);
    offset += inputShape[axis] / outputNum;
  }

  if (krnl::isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
    char *inputBuffer = createArrayFromDenseElementsAttr(
        input.getDefiningOp()
            ->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>());

    std::vector<char *> resBuffers;
    ConstPropSplitImpl(elementType, inputBuffer, inputShape,
        /*splitAxis=*/axis, /*splitOffsets=*/splitOffsets, resultTypes,
        resBuffers);

    for (int i = 0; i < outputNum; ++i) {
      Value constVal = createDenseONNXConstantOp(
          rewriter, loc, resultTypes[i].cast<ShapedType>(), resBuffers[i])
                           .getResult();
      resVals.emplace_back(constVal);
      free(resBuffers[i]);
    }
    free(inputBuffer);
  } else {
    ONNXSplitV11Op split =
        rewriter.create<ONNXSplitV11Op>(loc, resultTypes, input,
            /*axis=*/axis, nullptr);
    for (int i = 0; i < outputNum; ++i)
      resVals.emplace_back(split.outputs()[i]);
  }
  return resVals;
}

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation, and
/// return a constant.
Value foldOrEmitONNXTransposeOp(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, ArrayAttr permAttr) {
  auto inputType = input.getType().cast<ShapedType>();
  auto inputShape = inputType.getShape();
  auto resultShape = resultType.cast<ShapedType>().getShape();
  Type elementType = inputType.getElementType();

  // Get perm attribute.
  SmallVector<uint64_t, 4> perm;
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

  if (krnl::isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
    char *inputBuffer = createArrayFromDenseElementsAttr(
        input.getDefiningOp()
            ->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>());

    char *resBuffer = allocateBufferFor(resultType, /*useMaxSize=*/true);
    ConstPropTransposeImpl(
        elementType, inputBuffer, inputShape, perm, resultShape, resBuffer);
    Value constVal = createDenseONNXConstantOp(
        rewriter, loc, resultType.cast<ShapedType>(), resBuffer)
                         .getResult();
    free(resBuffer);
    free(inputBuffer);
    return constVal;
  } else
    return rewriter.create<ONNXTransposeOp>(loc, resultType, input, permAttr)
        .getResult();
}

/// Emit MemRef ReinterpretCastOp to create a new view for 'data'.
/// The new view is created using the given 'memRefType' and 'outputDims'.
Value emitMemRefReinterpretCastOp(ConversionPatternRewriter &rewriter,
    Location loc, Value data, const MemRefType &memRefType,
    SmallVectorImpl<IndexExpr> &outputDims) {
  MemRefBuilder createMemRef(rewriter, loc);
  return createMemRef.reinterpretCast(data, outputDims);
}

/// Emit krnl iterate to compute argsort of a given MemRef along a given axis.
/// Output MemRef has the same shape as the input MemRef but is of IndexType.
/// By default, sort values in the descending order.
Value emitArgSort(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t axis, bool ascending) {
  KrnlBuilder createKrnl(rewriter, loc);
  IndexExprScope scope(createKrnl);

  MemRefType inputMemRefType = input.getType().cast<MemRefType>();
  Type indexType = rewriter.getIndexType();
  int64_t rank = inputMemRefType.getRank();
  assert(axis >= 0 && axis < rank && "axis is out of bound");
  LiteralIndexExpr zeroIE(0), oneIE(1);

  MemRefBoundsIndexCapture inputBounds(input);
  SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
  SmallVector<IndexExpr, 4> ubs;
  inputBounds.getDimList(ubs);

  // Create and initialize the result.
  Value order = insertAllocAndDeallocSimple(rewriter, nullptr,
      MemRefType::get(inputMemRefType.getShape(), indexType), loc, ubs,
      /*insertDealloc=*/true);
  ValueRange initLoopDef = createKrnl.defineLoops(rank);
  createKrnl.iterateIE(initLoopDef, initLoopDef, lbs, ubs,
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        // order[axis_0, axis_1, ..., axis_k-1, k, axis_k+1, ....] = k
        createKrnl.store(loopInd[axis], order, loopInd);
      });

  // Do sorting in the descending order of input and return their indices.
  // Using bubble sort.
  SmallVector<IndexExpr, 4> outerUbs(ubs);
  outerUbs[axis] = ubs[axis] - oneIE;
  ValueRange loopDef = createKrnl.defineLoops(rank);
  createKrnl.iterateIE(loopDef, loopDef, lbs, outerUbs,
      [&](KrnlBuilder &createKrnl, ValueRange iLoopInd) {
        IndexExpr i1 = DimIndexExpr(iLoopInd[axis]) + oneIE;
        ValueRange swapLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(swapLoopDef, swapLoopDef, {i1}, {ubs[axis]},
            [&](KrnlBuilder &createKrnl, ValueRange swapLoopInd) {
              MultiDialectBuilder<KrnlBuilder, MathBuilder, SCFBuilder> create(
                  createKrnl);
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

/// Return a DenseElementAttr of a KrnlGlobalOp or ONNXConstantOp.
/// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr
/// lambda type, using ONNX and Krnl operations.
DenseElementsAttr getDenseElementAttributeFromConstantValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<mlir::KrnlGlobalOp>(definingOp)) {
    if (globalOp.value().hasValue())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  } else if (auto globalOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (globalOp.value().hasValue())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

/// This function returns a scalar of type 'dtype' from an optional value.
/// Optional value must be: NoneType, memref<1xdtype> or memref<dtype>. Default
/// value is used in case of NoneType.
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
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

} // namespace onnx_mlir
