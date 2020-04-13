//====------ ONNXToKrnlCommon.hpp - ONNX dialects to Krnl lowering --------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to KRNL.
//===----------------------------------------------------------------------===//

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType type);

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
MemRefType convertToMemRefType(Type type);

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter,
                                   bool insertDealloc,
                                   ArrayRef<Value> operands = {});

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
bool checkInsertDealloc(Operation *currentOp);

// Create a mapping from result type's dimensions to input type's dimensions,
// given that the result type is the result of a reduction op over the input
// type.
std::map<int64_t, int64_t>
getReductionMapping(MemRefType inputTy, ArrayRef<int64_t> axes, bool keepdims);

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimenions are supported.
void addDimensionToPack(ConversionPatternRewriter &rewriter,
                               Location loc, KrnlIterateOperandPack &pack,
                               Value operand, int index);

// Function that defines the KRNL dialect loops and their respective
// optimized version.
KrnlOptimizeLoopsOp
emitOptimizedLoops(ConversionPatternRewriter &rewriter, Location loc,
                   std::vector<Value> &loops,
                   std::vector<Value> &optimizedLoops, int64_t numLoops);

// Function that emits the loops and their optimized version.
// The function returns a reference to the inner optimization block.
Block *defineLoops(ConversionPatternRewriter &rewriter, Location loc,
                          std::vector<Value> &loops,
                          std::vector<Value> &optimizedLoops,
                          int64_t numLoops);

// Function which emits a basic set of loops and optimized loops
// for a given operation argument. A reference to the loop optimization
// block is returned in the last argument of the function.
void emitKrnlLoopsAndIterationForOperand(
    ConversionPatternRewriter &rewriter, Location loc, Value operand,
    std::vector<Value> &originalLoops, KrnlOptimizeLoopsOp &optimizedLoopsOp,
    KrnlIterateOp &iterateOp);

unsigned getMemRefEltSizeInBytes(MemRefType memRefType);

// Get run-time dimension information for unknown dimensions used for
// broadcasting.
std::map<int, std::map<int, Value>>
getBroadcastedDimInfo(Location loc, ConversionPatternRewriter &rewriter,
                      MemRefType memRefType, ArrayRef<Value> operands);

// Extract induction variables that are used for broadcasting values of a
// given operand.
std::vector<Value>
getLoopIVsForBroadcasting(Location loc, ConversionPatternRewriter &rewriter,
                          ArrayRef<Value> loopIVs, Value operand,
                          std::map<int, Value> broadcastedDims);

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
Value emitConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type, double value);

// Emit a positive infinity constant of a specific type.
// Supported types: F16, F32, F64, Int8, Int16, Int32, Int64.
// In case of Integer, emit the maximum value.
Value emitPositiveInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type);

// Emit a negative infinity constant of a specific type.
// Supported types: F16, F32, F64, Int8, Int16, Int32, Int64.
// In case of Float, emit the negative of the positive infinity.
// In case of Integer, emit the minimum value.
Value emitNegativeInfinityConstantOp(
    ConversionPatternRewriter &rewriter, Location loc, Type type);

int64_t ArrayAttrIntVal(ArrayAttr a, int i);

//===----------------------------------------------------------------------===//
// This is to get a scalar operation of a given type for a specific operation.
//===----------------------------------------------------------------------===//
template <typename Op>
struct ScalarOp {
  using FOp = void;
  using IOp = void;
};

template <typename FOp>
using ScalarFOp = typename ScalarOp<FOp>::FOp;
template <typename IOp>
using ScalarIOp = typename ScalarOp<IOp>::IOp;

// Get the identity element of an operation.
// Return NULL if the function does not have identity.
// Specialize this for a new Op.
template <typename Op>
Value getIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type type) {
  return nullptr;
}

//===----------------------------------------------------------------------===//
// This is used in the innermost loop of a KrnlIterateOp to insert computation
// composed of one or many scalar ops.
// Use template specialization for each of different ONNX operations.
//===----------------------------------------------------------------------===//
template <typename Op>
Value emitScalarOpFor(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Type elementType, ArrayRef<Value> scalarOperands) {
  if (elementType.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<Op>>(
        loc, elementType, scalarOperands, mlir::None);
  } else if (elementType.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<Op>>(
        loc, elementType, scalarOperands, mlir::None);
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  TensorTypeConverter() {
    addConversion(convertType);
  }

  static LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) {
    if (auto type = convertToMemRefType(t)) {
      results.push_back(type);
      return success();
    }

    results.push_back(t);
    return success();
  }

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(FunctionType funcType) {
    return llvm::all_of(funcType.getInputs(),
                        [this](Type type) { return isLegal(type); });
  }
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// `Math` directory methods:

void populateLoweringONNXElementwiseOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXGemmOpPattern(OwningRewritePatternList &patterns,
                                       MLIRContext *ctx);

void populateLoweringONNXMatMulOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXReductionOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXSoftmaxOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

// `NN` directory methods:

void populateLoweringONNXConvOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXNormalizationOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXPoolingOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

// `Tensor` directory methods:

void populateLoweringONNXUnsqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXTransposeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXPadConstantValuePadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXPadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXIdentityOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXConstantOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);
