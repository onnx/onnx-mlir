/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

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

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter &rewriter, bool insertDealloc, Value operand = nullptr,
    int64_t alignment = -1);

// Insert an allocation and deallocation for the given MemRefType, handling
// compile time relying on the above function, and extracting the runtime
// definitions from the index expressions otherwise.
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, SmallVectorImpl<IndexExpr> &outputDims,
    int64_t alignment = -1);
// Same where boolean to assert if dealloc is to be gen or not is specified
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, SmallVectorImpl<IndexExpr> &outputDims,
    bool insertDealloc, int64_t alignment = -1);

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
// Dynamic dimensions are supported.
void addDimensionToPack(ConversionPatternRewriter &rewriter, Location loc,
    KrnlIterateOperandPack &pack, Value operand, int index);

// Function that emits the define_loop operation to define `numLoops`
// number of krnl loops, and fill `loop` with the newly defined loops.
void defineLoops(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, int64_t numLoops);

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

/// Get a dimension value from a memref. Emit a constant if the dimension is
/// constant. Otherwise, emit a dim op.
/// If the return type is different from IndexType, emit a cast op to cast the
/// output of the dim op.
Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis, Type type);

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

  /// Return true if the operands/results of call have a legal type.
  bool isSignatureLegal(mlir::CallOp call) {
    auto f = [this](Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// `ControlFlow` directory methods:

void populateLoweringONNXLoopOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXScanOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

// `Math` directory methods:

void populateLoweringONNXClipOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXElementwiseOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXGemmOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXLRNOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXMatMulOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXReductionOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXSoftmaxOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

// `NN` directory methods:

void populateLoweringONNXConvOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXNormalizationOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXPoolingOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

// `RNN` directory methods:
void populateLoweringONNXGRUOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXLSTMOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);
void populateLoweringONNXRNNOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

// `Tensor` directory methods:
void populateLoweringONNXArgMaxOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXUnsqueezeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXTransposeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXGatherOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXPadConstantValuePadOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXPadOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXReshapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXIdentityOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXConstantOfShapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXConstantOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXConcatOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXShapeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXSliceOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXSqueezeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXSplitOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXSizeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXTileOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringONNXFlattenOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);

bool checkOpResultIsUsedByGetRef(memref::AllocOp *allocOp);

/// This function returns the index in the list of alloc arguments of the
/// dynamic dimension corresponding to `index` in the MemRef shape.
/// As an example:
///
/// alloc(%d0, %d1, %d2) : memref<10x?x?x20x?x30xf32>
///
/// In the above alloc the list of alloc arguments is being represented by
/// %d0, %d1 and %d2. Their indices 0, 1, 2 correspond to `index` values
/// 1, 2 and 4 in the MemRef shape respectively
int64_t getAllocArgIndex(memref::AllocOp allocOp, int64_t index);

/// This function returns a location with the corresponding ONNX operator name
/// inside. This is useful when tracing what expanded MLIR instructions
/// correspond to what ONNX operator.
///
///
template <typename OP_TYPE>
Location ONNXLoc(Operation *op) {
  return NameLoc::get(
      Identifier::get(OP_TYPE::getOperationName(), op->getContext()),
      op->getLoc());
}
