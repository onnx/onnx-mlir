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

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType type) {
  auto memRefShape = type.getShape();
  for (int i = 0; i < memRefShape.size(); ++i)
    if (memRefShape[i] < 0)
      return false;
  return true;
}

/// Check is all operands are scalar values at compile time.
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

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(Operation *op) {
  Operation *parentFuncOp = op->getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter &rewriter, bool insertDealloc, ArrayRef<Value> operands,
    int64_t alignment) {
  // Put together alloc operands for any dynamic dimensions of the memref.
  AllocOp alloc;
  if (!operands.empty()) {
    auto memRefShape = type.getShape();
    auto rank = memRefShape.size();

    std::map<int, Value> fromOperands;
    for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
      int memRefDimIdx = rank - 1 - reversedIdx;
      if (memRefShape[memRefDimIdx] < 0) { // unknown dimension
        Value maxDim = nullptr;
        for (int i = 0; i < operands.size(); i++) {
          auto operandShape =
              operands[i].getType().cast<MemRefType>().getShape();
          int operandDimIdx = operandShape.size() - 1 - reversedIdx;

          if (operandDimIdx < 0)
            continue;

          // In case of operations with broadcasting, the dimension of the
          // alloc result is the maximum size along each dimension of the
          // operands.
          auto operandDim =
              rewriter.create<DimOp>(loc, operands[i], operandDimIdx);
          if (maxDim) {
            auto maxCondition = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::sgt, operandDim, maxDim);
            maxDim = rewriter.create<SelectOp>(
                loc, maxCondition, operandDim, maxDim);
          } else {
            maxDim = operandDim;
          }
        }
        fromOperands.insert(std::make_pair(memRefDimIdx, maxDim));
      }
    }

    SmallVector<Value, 4> allocOperands;
    for (int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0)
        allocOperands.push_back(fromOperands[i]);
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc =
          rewriter.create<AllocOp>(loc, type, allocOperands, constAlignAttr);
    } else {
      alloc = rewriter.create<AllocOp>(loc, type, allocOperands);
    }
  } else {
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      SmallVector<Value, 4> allocOperandsEmpty;
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc = rewriter.create<AllocOp>(
          loc, type, allocOperandsEmpty, constAlignAttr);
    } else {
      alloc = rewriter.create<AllocOp>(loc, type);
    }
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  if (insertDealloc) {
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  return alloc;
}

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
bool checkInsertDealloc(Operation *currentOp, int resultIndex) {
  auto parentBlock = currentOp->getBlock();

  bool insertDealloc = true;
  parentBlock->walk([&insertDealloc, currentOp, resultIndex](ReturnOp op) {
    // If there is at least one result to investigate.
    if (currentOp->getNumResults() > 0) {
      auto result = currentOp->getResult(resultIndex);
      for (const auto &operand : op.getOperands())
        if (operand == result)
          insertDealloc = false;
    }
  });

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
// Dynamic dimenions are supported.
void addDimensionToPack(ConversionPatternRewriter &rewriter, Location loc,
    KrnlIterateOperandPack &pack, Value operand, int index) {
  auto shape = operand.getType().cast<MemRefType>().getShape();
  if (shape[index] < 0) {
    pack.pushConstantBound(0);
    pack.pushOperandBound(
        rewriter.create<DimOp>(loc, operand, index).getResult());
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

unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Get run-time dimension information for unknown dimensions used for
// broadcasting.
std::map<int, std::map<int, Value>> getBroadcastedDimInfo(Location loc,
    ConversionPatternRewriter &rewriter, MemRefType memRefType,
    ArrayRef<Value> operands) {
  auto memRefShape = memRefType.getShape();
  int64_t rank = memRefShape.size();
  // For unknown dimensions, we need to get dimension values at runtime in
  // order to do broadcasting.
  std::map<int, std::map<int, Value>> DimInfo;
  // For each result dimension, compute the number of sharing operands.
  // Sharing operands are operands sharing the same index (counting from the
  // rightmost to the leftmost) for a given dimension.
  std::map<int, int> sharedDimCount;
  for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    int dimIdx = rank - 1 - reversedIdx;
    sharedDimCount[dimIdx] = 0;
    for (int i = 0; i < operands.size(); ++i) {
      auto shape = operands[i].getType().cast<MemRefType>().getShape();
      if (reversedIdx <= shape.size() - 1)
        sharedDimCount[dimIdx]++;
    }
  }
  // An unknown dimension can have a value of 1 or N (N > 1).
  // If its value is 1, it is broadcasted dimension.
  // Otherwise, non-broadcasted dimension.
  // We only care about unknown dimensions whose number of sharing operands is
  // more than one, since they are potentially broadcasted dimensions.
  for (int i = 0; i < operands.size(); ++i) {
    std::map<int, Value> broadcastedDims;
    auto shape = operands[i].getType().cast<MemRefType>().getShape();
    int size = shape.size();
    for (int j = 0; j < shape.size(); ++j) {
      if (shape[j] < 0 && sharedDimCount[rank - size + j] > 1) {
        auto dim = rewriter.create<DimOp>(loc, operands[i], j).getResult();
        auto one = rewriter.create<ConstantIndexOp>(loc, 1);
        auto isBroadcasted =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dim, one);
        broadcastedDims.insert(std::make_pair(j, isBroadcasted));
      }
    }
    DimInfo.insert(std::make_pair(i, broadcastedDims));
  }
  return DimInfo;
}

// Extract induction variables that are used for broadcasting values of a
// given operand.
std::vector<Value> getLoopIVsForBroadcasting(Location loc,
    ConversionPatternRewriter &rewriter, ArrayRef<Value> loopIVs, Value operand,
    std::map<int, Value> broadcastedDims) {
  // `operand` must has a ranked type. This should have been checked by the
  // shape inference pass.
  auto operandShape = operand.getType().cast<MemRefType>().getShape();
  auto rank = operandShape.size();
  auto loopCount = loopIVs.size();

  std::vector<Value> newLoopIVs;
  for (unsigned reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    auto dimIdx = rank - 1 - reversedIdx;
    auto loopIdx = loopCount - 1 - reversedIdx;
    if (operandShape[dimIdx] == 1) {
      // Broadcasted dimension
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      newLoopIVs.insert(newLoopIVs.begin(), zero);
    } else if ((operandShape[dimIdx] == -1) &&
               (broadcastedDims.find(dimIdx) != broadcastedDims.end())) {
      // Unknown dimension, it can have a value of 1 or N (N > 1).
      // If its value is 1, it is broadcasted dimension.
      // Otherwise, non-broadcasted dimension.
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      auto idx = rewriter.create<SelectOp>(
          loc, broadcastedDims[dimIdx], zero, loopIVs[loopIdx]);
      newLoopIVs.insert(newLoopIVs.begin(), idx);
    } else {
      // Non-broadcasted dimension
      newLoopIVs.insert(newLoopIVs.begin(), loopIVs[loopIdx]);
    }
  }
  return newLoopIVs;
}

Value emitConstantOp(
    PatternRewriter &rewriter, Location loc, Type type, double value) {
  Attribute constantAttr;
  auto typeKind = type.getKind();
  if (typeKind == StandardTypes::F16) {
    constantAttr = rewriter.getF16FloatAttr((float)value);
  } else if (typeKind == StandardTypes::F32) {
    constantAttr = rewriter.getF32FloatAttr((float)value);
  } else if (typeKind == StandardTypes::F64) {
    constantAttr = rewriter.getF64FloatAttr(value);
  } else if (typeKind == StandardTypes::Integer) {
    auto width = type.cast<IntegerType>().getWidth();
    if (width == 1) {
      constantAttr = rewriter.getBoolAttr(false);
    } else {
      constantAttr =
          rewriter.getIntegerAttr(type, APInt(width, (int64_t)value));
    }
  } else if (typeKind == StandardTypes::Index) {
    constantAttr = rewriter.getIntegerAttr(type, (int64_t)value);
  } else {
    llvm_unreachable("unsupported element type");
  }

  return rewriter.create<ConstantOp>(loc, constantAttr);
}

Value emitPositiveInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  Attribute constantAttr;
  auto typeKind = type.getKind();
  if (typeKind == StandardTypes::F16) {
    // 0x7C00
    float value = std::numeric_limits<float>::infinity();
    constantAttr = rewriter.getF16FloatAttr(value);
  } else if (typeKind == StandardTypes::F32) {
    // 0x7F800000
    float value = std::numeric_limits<float>::infinity();
    constantAttr = rewriter.getF32FloatAttr(value);
  } else if (typeKind == StandardTypes::F64) {
    // 0x7FF0000000000000
    double value = std::numeric_limits<double>::infinity();
    constantAttr = rewriter.getF64FloatAttr(value);
  } else if (typeKind == StandardTypes::Integer) {
    auto width = type.cast<IntegerType>().getWidth();
    // The latest llvm-project includes a patch which allows getting the sign of
    // IntegerType:
    // https://github.com/llvm/llvm-project/commit/35b685270b410f6a1351c2a527021f22330c25b9
    // as follows:
    //   auto isSigned = type.cast<IntegerType>().isSigned();
    // TODO (tungld): update the following statement once our llvm-project is
    // upgraded to include the patch.
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
  } else {
    llvm_unreachable("unsupported element type");
  }

  return rewriter.create<ConstantOp>(loc, constantAttr);
}

Value emitNegativeInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  Attribute constantAttr;
  auto typeKind = type.getKind();
  if (typeKind == StandardTypes::F16) {
    // 0xFC00
    float value = -std::numeric_limits<float>::infinity();
    constantAttr = rewriter.getF16FloatAttr(value);
  } else if (typeKind == StandardTypes::F32) {
    // 0xFF800000
    float value = -std::numeric_limits<float>::infinity();
    constantAttr = rewriter.getF32FloatAttr(value);
  } else if (typeKind == StandardTypes::F64) {
    // 0xFFF0000000000000
    double value = -std::numeric_limits<double>::infinity();
    constantAttr = rewriter.getF64FloatAttr(value);
  } else if (typeKind == StandardTypes::Integer) {
    auto width = type.cast<IntegerType>().getWidth();
    // The latest llvm-project includes a patch which allows getting the sign of
    // IntegerType:
    // https://github.com/llvm/llvm-project/commit/35b685270b410f6a1351c2a527021f22330c25b9
    // as follows:
    //   auto isSigned = type.cast<IntegerType>().isSigned();
    // TODO (tungld): update the following statement once our llvm-project is
    // upgraded to include the patch.
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
  } else {
    llvm_unreachable("unsupported element type");
  }

  return rewriter.create<ConstantOp>(loc, constantAttr);
}

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

bool checkOpResultIsUsedByGetRef(AllocOp *allocOp) {
  FuncOp function = getContainingFunction(allocOp->getOperation());

  bool opIsUsedInGetRef = false;
  function.walk([&opIsUsedInGetRef, allocOp](KrnlGetRefOp op) {
    auto result = allocOp->getResult();
    for (const auto &operand : op.getOperands())
      if (operand == result)
        opIsUsedInGetRef = true;
  });

  return opIsUsedInGetRef;
}

// TODO: support dynamic sizes.
int64_t getMemRefSizeInBytes(Value val) {
  auto memRefType = convertToMemRefType(val.getType());
  auto memRefShape = memRefType.getShape();
  int64_t size = 1;
  for (int i = 0; i < memRefShape.size(); i++)
    size *= memRefShape[i];
  size *= getMemRefEltSizeInBytes(memRefType);
  return size;
}

Value getDynamicMemRefSizeInBytes(
    MemRefType type, Location loc, PatternRewriter &rewriter, AllocOp allocOp) {
  // Initialize the size variable with the size in bytes of the type.
  int64_t typeSize = getMemRefEltSizeInBytes(type);
  Value result =
      emitConstantOp(rewriter, loc, rewriter.getIndexType(), typeSize);

  // Multiply all dimensions (constant and dynamic).
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();
  int dynDimIdx = 0;
  for (int idx = 0; idx < rank; ++idx) {
    if (memRefShape[idx] < 0) {
      // Dyanmic size.
      auto dynamicDim = allocOp.getOperands()[dynDimIdx];
      dynDimIdx++;
      result = rewriter.create<MulIOp>(loc, result, dynamicDim);
    } else {
      // Static size.
      auto staticDim = emitConstantOp(
          rewriter, loc, rewriter.getIndexType(), memRefShape[idx]);
      result = rewriter.create<MulIOp>(loc, result, staticDim);
    }
  }

  return result;
}
