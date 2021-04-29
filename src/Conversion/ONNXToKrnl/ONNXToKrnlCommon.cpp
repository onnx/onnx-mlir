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
  // Put together alloc operands for any dynamic dimensions of the memref.
  memref::AllocOp alloc;
  if (operand) {
    auto memRefShape = type.getShape();
    auto rank = memRefShape.size();

    SmallVector<Value, 4> allocOperands;
    for (int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0) {
        auto dim = rewriter.create<memref::DimOp>(loc, operand, i);
        allocOperands.push_back(dim);
      }
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc =
          rewriter.create<memref::AllocOp>(loc, type, allocOperands, constAlignAttr);
    } else {
      alloc = rewriter.create<memref::AllocOp>(loc, type, allocOperands);
    }
  } else {
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      SmallVector<Value, 4> allocOperandsEmpty;
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc = rewriter.create<memref::AllocOp>(
          loc, type, allocOperandsEmpty, constAlignAttr);
    } else {
      alloc = rewriter.create<memref::AllocOp>(loc, type);
    }
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  if (insertDealloc) {
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  return alloc;
}

// Simple version of insert alloc and dealloc that does not handle alignment
// or additional operands (to be studied and added as needed). For unknown
// dimensions, it uses the index expressions to retrieve the corresponding
// values.
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, SmallVectorImpl<IndexExpr> &outputDims) {

  bool insertDealloc = checkInsertDealloc(op);

  // Constant, use the normal insert with no additional operands or alignment.
  if (hasAllConstantDimensions(type))
    return insertAllocAndDealloc(type, loc, rewriter, insertDealloc);
  // Otherwise, take the unkown operands from the output dim IndexExpressions
  SmallVector<Value, 2> allocOperands;
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();

  for (int i = 0; i < rank; ++i) {
    if (memRefShape[i] < 0) {
      // have dyn shape
      allocOperands.emplace_back(outputDims[i].getValue());
    }
  }
  memref::AllocOp allocOp = rewriter.create<memref::AllocOp>(loc, type, allocOperands);
  if (insertDealloc) {
    auto *parentBlock = allocOp.getOperation()->getBlock();
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, allocOp);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return allocOp;
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
  return rewriter.create<ConstantOp>(loc, constantAttr);
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

  return rewriter.create<ConstantOp>(loc, constantAttr);
}

Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis, Type type) {
  ArrayRef<int64_t> shape = operand.getType().cast<ShapedType>().getShape();
  Value dimVal;
  if (shape[axis] < 0) {
    Value dim = rewriter.create<memref::DimOp>(loc, operand, axis);
    if (type.isa<IndexType>())
      dimVal = dim;
    else
      dimVal = rewriter.create<IndexCastOp>(loc, dim, type);
  } else {
    dimVal = emitConstantOp(rewriter, loc, type, shape[axis]);
  }
  return dimVal;
}
