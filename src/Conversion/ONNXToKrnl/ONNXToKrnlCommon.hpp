/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToKrnlCommon.hpp - ONNX dialects to Krnl lowering --------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

// A global variable to indicate whether this pass will emit dealloc for
// allocated memrefs or not during the conversion of ONNX to Krnl.
extern bool ONNXToKrnl_gEmitDealloc;

//===----------------------------------------------------------------------===//
// Extends OnnxBuilder with member functions that might generate Krnl dialect
// operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct OnnxToKrnlBuilder : public OnnxBuilder {
  OnnxToKrnlBuilder(OpBuilder &b, Location loc) : OnnxBuilder(b, loc) {}
  OnnxToKrnlBuilder(DialectBuilder &db) : OnnxBuilder(db) {}

  // Generate an 'onnx.reshape' operation on the 'input' tensor, the new shape
  // is provided by 'shapeDims'.
  Value reshape(
      const Value input, const ArrayRef<DimIndexExpr> shapeDims) const;

  // Generate a 'onnx.Transpose' operation on the 'input' tensor given the
  // permutation array 'perm' and the operator output dimensions 'outputDims'.
  Value transpose(const Value input, const ArrayRef<int64_t> perm,
      const ArrayRef<DimIndexExpr> outputDims) const;
};

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to KRNL.
//===----------------------------------------------------------------------===//

/// Check if all operands are scalar values at compile time.
bool hasAllScalarValues(ArrayRef<Value> values);

/// Check if the value is a KrnlGlobalOp with a dense attribute of non-negative
/// integer constants.
bool indicesAreNonNegativeConstants(Value indices);

/// Insert an allocation and deallocation for the given MemRefType.
Value insertAllocAndDealloc(MemRefType type, Location loc,
    PatternRewriter &rewriter, bool insertDealloc, Value operand = nullptr,
    int64_t alignment = -1);

// Insert an allocation and deallocation for the given MemRefType, handling
// compile time relying on the above function, and extracting the runtime
// definitions from the index expressions otherwise.
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, const SmallVectorImpl<IndexExpr> &outputDims,
    int64_t alignment = -1);
// Same where boolean to assert if dealloc is to be gen or not is specified
Value insertAllocAndDeallocSimple(PatternRewriter &rewriter, Operation *op,
    MemRefType type, Location loc, const SmallVectorImpl<IndexExpr> &outputDims,
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
    krnl::KrnlIterateOperandPack &pack, Value operand, int index);

// Function that emits the define_loop operation to define `numLoops`
// number of krnl loops, and fill `loop` with the newly defined loops.
void defineLoops(ConversionPatternRewriter &rewriter, Location loc,
    std::vector<Value> &loops, int64_t numLoops);

/// Get a dimension value from a memref. Emit a constant if the dimension is
/// constant. Otherwise, emit a dim op.
/// If the return type is different from IndexType, emit a cast op to cast the
/// output of the dim op.
Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis, Type type);

/// Emit an ONNXSqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
Value foldOrEmitONNXSqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis);

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
Value foldOrEmitONNXUnsqueezeV11Op(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis);

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> foldOrEmitONNXSplitOp(ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Type> resultTypes, Value input, int64_t axis);

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation, and
/// return a constant.
Value foldOrEmitONNXTransposeOp(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, ArrayAttr permAttr);

/// Emit MemRef ReinterpretCastOp to create a new view for 'data'.
/// The new view is created using the given 'outputDims'.
Value emitMemRefReinterpretCastOp(ConversionPatternRewriter &rewriter,
    Location loc, Value data, SmallVectorImpl<IndexExpr> &outputDims);

/// Emit krnl iterate to compute argsort of a given MemRef along a given axis.
/// Output MemRef has the same shape as the input MemRef but is of IndexType.
Value emitArgSort(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t axis, bool ascending = false);

/// Return a DenseElementAttr of a KrnlGlobalOp or ONNXConstantOp.
/// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr
/// lambda type, using ONNX and Krnl operations.
DenseElementsAttr getDenseElementAttributeFromConstantValue(Value value);

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
// Type conversion from Onnx types to Krnl types:
//   - from Tensor type to the Standard dialect MemRef type
//   - from onnx.StringType to krnl.StringType
//===----------------------------------------------------------------------===//

class KrnlTypeConverter : public TypeConverter {
public:
  using TypeConverter::TypeConverter;

  KrnlTypeConverter();

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
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// For all ONNX operations.
void populateONNXToKrnlConversionPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *, bool enableTiling);

// `ControlFlow` directory methods:
void populateLoweringONNXLoopOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXScanOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `Math` directory methods:
void populateLoweringONNXClipOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXCumSumOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXElementwiseOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXGemmOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *, bool enableTiling);
void populateLoweringONNXHardmaxOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXLRNOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXMatMulOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *, bool enableTiling);
void populateLoweringONNXRandomNormalOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXRandomNormalLikeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXReductionOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSoftmaxOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXTopKOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `ML` directory methods:
void populateLoweringONNXCategoryMapperOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `NN` directory methods:
void populateLoweringONNXConvOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXNormalizationOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXPoolingOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `ObjectDetection` directory methods:
void populateLoweringONNXNonMaxSuppressionOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `RNN` directory methods:
void populateLoweringONNXGRUOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXLSTMOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXRNNOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `Sequence` directory methods:
void populateLoweringONNXSequenceAtOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSequenceEmptyOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSequenceEraseOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSequenceInsertOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSequenceLengthOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

// `Tensor` directory methods:
void populateLoweringONNXArgMaxOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXUnsqueezeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXUnsqueezeV11OpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXTransposeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXGatherOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXGatherElementsOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXGatherNDOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXPadConstantValuePadOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXPadOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXRangeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXReshapeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXIdentityOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXConstantOfShapeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXConstantOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXConcatOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXDepthToSpaceOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSpaceToDepthOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXScatterElementsOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXScatterNDOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXShapeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSliceOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSqueezeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSqueezeV11OpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSplitOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSplitV11OpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXSizeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXTileOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXFlattenOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXResizeOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXNonZeroOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXReverseSequenceOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXExpandOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXOneHotOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
void populateLoweringONNXCompressOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

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
      StringAttr::get(op->getContext(), OP_TYPE::getOperationName()),
      op->getLoc());
}

/// This function returns a scalar of type 'dtype' from an optional value.
/// Optional value must be: NoneType, memref<1xdtype> or memref<dtype>.
/// Default value is used in case of NoneType.
Value getOptionalScalarValue(ConversionPatternRewriter &rewriter, Location loc,
    Value optionalScalar, Type elementType, double defaultValue);

} // namespace onnx_mlir
