/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToKrnlCommon.cpp - ONNX dialects to Krnl lowering ---------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

bool gEmitDealloc = true;

/// Check if all operands are scalar values at compile time.
bool hasAllScalarValues(ArrayRef<Value> values) {
  for (Value value : values) {
    if (value.getType().cast<ShapedType>().getRank() != 0)
      return false;
  }
  return true;
}

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
MemRefType convertToMemRefType(Type type) {
  MemRefType memRefType;
  auto tensorType = type.dyn_cast<TensorType>();
  if (tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    memRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else {
    memRefType = type.dyn_cast<MemRefType>();
  }
  return memRefType;
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

  if (!gEmitDealloc)
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
    MemRefType type, Location loc, SmallVectorImpl<IndexExpr> &outputDims,
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

  if (!gEmitDealloc)
    return allocOp;

  if (insertDealloc) {
    auto *parentBlock = allocOp.getOperation()->getBlock();
    auto dealloc = createMemRef.dealloc(allocOp);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return allocOp;
}

Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, SmallVectorImpl<IndexExpr> &outputDims,
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
  if (gEmitDealloc == false)
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
    KrnlIterateOperandPack &pack, Value operand, int index) {
  auto shape = operand.getType().cast<MemRefType>().getShape();
  if (shape[index] < 0) {
    pack.pushConstantBound(0);
    pack.pushOperandBound(
        rewriter.create<memref::DimOp>(loc, operand, index).getResult());
  } else {
    pack.pushConstantBound(0);
    pack.pushConstantBound(shape[index]);
  }
}

// Function that emits the definition of loops references.
void defineLoops(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, int64_t numLoops) {
  auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, numLoops);
  loops.reserve(numLoops);
  for (auto result : loopsOp.getResults())
    loops.push_back(result);
}

Value emitPositiveInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  Attribute constantAttr;

  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        // 0x7C00
        float value = std::numeric_limits<float>::infinity();
        constantAttr = rewriter.getF16FloatAttr(value);
      })
      .Case<Float32Type>([&](Type) {
        // 0x7F800000
        float value = std::numeric_limits<float>::infinity();
        constantAttr = rewriter.getF32FloatAttr(value);
      })
      .Case<Float64Type>([&](Type) {
        // 0x7FF0000000000000
        double value = std::numeric_limits<double>::infinity();
        constantAttr = rewriter.getF64FloatAttr(value);
      })
      .Case<IntegerType>([&](Type) {
        auto width = type.cast<IntegerType>().getWidth();
        // The latest llvm-project includes a patch which allows getting the
        // sign of IntegerType:
        // https://github.com/llvm/llvm-project/commit/35b685270b410f6a1351c2a527021f22330c25b9
        // as follows:
        //   auto isSigned = type.cast<IntegerType>().isSigned();
        // TODO (tungld): update the following statement once our llvm-project
        // is upgraded to include the patch.
        auto isSigned = true;
        if (width == 8) {
          if (isSigned) {
            int8_t value = std::numeric_limits<int8_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint8_t value = std::numeric_limits<uint8_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 16) {
          if (isSigned) {
            int16_t value = std::numeric_limits<int16_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint16_t value = std::numeric_limits<uint16_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 32) {
          if (isSigned) {
            int32_t value = std::numeric_limits<int32_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint32_t value = std::numeric_limits<uint32_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 64) {
          if (isSigned) {
            int64_t value = std::numeric_limits<int64_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint64_t value = std::numeric_limits<uint64_t>::max();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else {
          llvm_unreachable("unsupported element type");
        }
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  return rewriter.create<arith::ConstantOp>(loc, constantAttr);
}

Value emitNegativeInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  Attribute constantAttr;

  TypeSwitch<Type>(type)
      .Case<Float16Type>([&](Type) {
        // 0xFC00
        float value = -std::numeric_limits<float>::infinity();
        constantAttr = rewriter.getF16FloatAttr(value);
      })
      .Case<Float32Type>([&](Type) {
        // 0xFF800000
        float value = -std::numeric_limits<float>::infinity();
        constantAttr = rewriter.getF32FloatAttr(value);
      })
      .Case<Float64Type>([&](Type) {
        // 0xFFF0000000000000
        double value = -std::numeric_limits<double>::infinity();
        constantAttr = rewriter.getF64FloatAttr(value);
      })
      .Case<IntegerType>([&](Type) {
        auto width = type.cast<IntegerType>().getWidth();
        // The latest llvm-project includes a patch which allows getting the
        // sign of IntegerType:
        // https://github.com/llvm/llvm-project/commit/35b685270b410f6a1351c2a527021f22330c25b9
        // as follows:
        //   auto isSigned = type.cast<IntegerType>().isSigned();
        // TODO (tungld): update the following statement once our llvm-project
        // is upgraded to include the patch.
        auto isSigned = true;
        if (width == 8) {
          if (isSigned) {
            int8_t value = std::numeric_limits<int8_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint8_t value = std::numeric_limits<uint8_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 16) {
          if (isSigned) {
            int16_t value = std::numeric_limits<int16_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint16_t value = std::numeric_limits<uint16_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 32) {
          if (isSigned) {
            int32_t value = std::numeric_limits<int32_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint32_t value = std::numeric_limits<uint32_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else if (width == 64) {
          if (isSigned) {
            int64_t value = std::numeric_limits<int64_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          } else {
            uint64_t value = std::numeric_limits<uint64_t>::min();
            constantAttr = rewriter.getIntegerAttr(type, APInt(width, value));
          }
        } else {
          llvm_unreachable("unsupported element type");
        }
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });

  return rewriter.create<arith::ConstantOp>(loc, constantAttr);
}

Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis, Type type) {
  ArrayRef<int64_t> shape = operand.getType().cast<ShapedType>().getShape();
  Value dimVal;
  if (shape[axis] < 0) {
    MemRefBuilder createMemRef(rewriter, loc);
    Value dim = createMemRef.dim(operand, axis);
    if (type.isa<IndexType>())
      dimVal = dim;
    else
      dimVal = rewriter.create<arith::IndexCastOp>(loc, dim, type);
  } else {
    dimVal = emitConstantOp(rewriter, loc, type, shape[axis]);
  }
  return dimVal;
}

/// Emit an ONNXSqueezeV11Op. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXSqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  if (isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
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
  if (isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
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

  if (isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
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

  if (isKrnlGlobalConstant(input) || isDenseONNXConstant(input)) {
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
    Location loc, Value data, MemRefType memRefType,
    SmallVectorImpl<IndexExpr> &outputDims) {
  int64_t rank = memRefType.getRank();

  // Compute new sizes and strides.
  SmallVector<IndexExpr, 4> sizesIE, stridesIE;
  sizesIE.resize(rank);
  stridesIE.resize(rank);
  IndexExpr strideIE = LiteralIndexExpr(1);
  for (int i = rank - 1; i >= 0; --i) {
    sizesIE[i] = outputDims[i];
    stridesIE[i] = strideIE;
    if (i > 0)
      strideIE = strideIE * sizesIE[i];
  }

  SmallVector<OpFoldResult, 4> sizes, strides;
  sizes.resize(rank);
  strides.resize(rank);
  for (int i = rank - 1; i >= 0; --i) {
    if (sizesIE[i].isLiteral())
      sizes[i] = rewriter.getIndexAttr(sizesIE[i].getLiteral());
    else
      sizes[i] = sizesIE[i].getValue();
    if (stridesIE[i].isLiteral())
      strides[i] = rewriter.getIndexAttr(stridesIE[i].getLiteral());
    else
      strides[i] = stridesIE[i].getValue();
  }

  // Emit ReinterpretCastOp.
  Value newView =
      rewriter.create<memref::ReinterpretCastOp>(loc, memRefType, data,
          /*offset=*/rewriter.getIndexAttr(0), sizes, strides);
  return newView;
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
/// Optional value can be: NoneType, memref<1xdtype> or memref<dtype>. Default
/// value is used in case of NoneType.
Value getOptionalScalarValue(ConversionPatternRewriter &rewriter, Location loc,
    Value optionalScalar, Type elementType, double defaultValue) {
  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);

  if (optionalScalar.getType().isa<NoneType>()) {
    return createMath.constant(elementType, defaultValue);
  } else if (optionalScalar.getType().cast<ShapedType>().getRank() == 0) {
    return createKrnl.load(optionalScalar, {});
  } else {
    Value zero = createMath.constantIndex(0);
    return createKrnl.load(optionalScalar, {zero});
  }
}
