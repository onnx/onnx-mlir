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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to KRNL.
//===----------------------------------------------------------------------===//

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType type);

/// Check is all operands are scalar values at compile time.
bool hasAllScalarValues(ArrayRef<Value> values);

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
MemRefType convertToMemRefType(Type type);

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(Operation *op);

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter &rewriter, bool insertDealloc,
    ArrayRef<Value> operands = {}, int64_t alignment = -1);

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
bool checkInsertDealloc(Operation *currentOp, int resultIndex = 0);

// Create a mapping from result type's dimensions to input type's dimensions,
// given that the result type is the result of a reduction op over the input
// type.
std::map<int64_t, int64_t> getReductionMapping(
    MemRefType inputTy, ArrayRef<int64_t> axes, bool keepdims);

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimenions are supported.
void addDimensionToPack(ConversionPatternRewriter &rewriter, Location loc,
    KrnlIterateOperandPack &pack, Value operand, int index);

// Function that emits the define_loop operation to define `numLoops`
// number of krnl loops, and fill `loop` with the newly defined loops.
void defineLoops(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, int64_t numLoops);

unsigned getMemRefEltSizeInBytes(MemRefType memRefType);

// Get run-time dimension information for unknown dimensions used for
// broadcasting.
std::map<int, std::map<int, Value>> getBroadcastedDimInfo(Location loc,
    ConversionPatternRewriter &rewriter, MemRefType memRefType,
    ArrayRef<Value> operands);

// Extract induction variables that are used for broadcasting values of a
// given operand.
std::vector<Value> getLoopIVsForBroadcasting(Location loc,
    ConversionPatternRewriter &rewriter, ArrayRef<Value> loopIVs, Value operand,
    std::map<int, Value> broadcastedDims);

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
Value emitConstantOp(
    PatternRewriter &rewriter, Location loc, Type type, double value);

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
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  TensorTypeConverter() { addConversion(convertType); }

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
    return llvm::all_of(
        llvm::concat<const Type>(funcType.getInputs(), funcType.getResults()),
        [this](Type type) { return isLegal(type); });
  }
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// `Math` directory methods:

void populateLoweringONNXElementwiseOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXGemmOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

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

// `RNN` directory methods:
void populateLoweringONNXGRUOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

// `Tensor` directory methods:

void populateLoweringONNXUnsqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXTransposeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXGatherOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXPadConstantValuePadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXPadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXIdentityOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXConstantOfShapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXConstantOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXConcatOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXSqueezeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXSplitOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

void populateLoweringONNXSizeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

bool checkOpResultIsUsedByGetRef(AllocOp *allocOp);

int64_t getMemRefSizeInBytes(Value val);

Value getDynamicMemRefSizeInBytes(
    MemRefType type, Location loc, PatternRewriter &rewriter, AllocOp allocOp);

/// This function returns the index in the list of alloc arguments of the
/// dynamic dimension corresponding to `index` in the MemRef shape.
/// As an example:
///
/// alloc(%d0, %d1, %d2) : memref<10x?x?x20x?x30xf32>
///
/// In the above alloc the list of alloc arguments is being represented by
/// %d0, %d1 and %d2. Their indices 0, 1, 2 correspond to `index` values
/// 1, 2 and 4 in the MemRef shape respectively
int64_t getAllocArgIndex(AllocOp allocOp, int64_t index);
