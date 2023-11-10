/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToKrnlCommon.hpp - ONNX dialects to Krnl lowering --------===//
//
// Copyright 2019-2023 The IBM Research Authors.
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
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

//===----------------------------------------------------------------------===//
// Extends OnnxBuilder with member functions that might generate Krnl dialect
// operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct OnnxToKrnlBuilder : public OnnxBuilder {
  OnnxToKrnlBuilder(mlir::Location loc) : OnnxBuilder(loc) {}
  OnnxToKrnlBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : OnnxBuilder(b, loc) {}
  OnnxToKrnlBuilder(const DialectBuilder &db) : OnnxBuilder(db) {}
  virtual ~OnnxToKrnlBuilder() {}

  // Generate an 'onnx.reshape' operation on the 'input' tensor, the new shape
  // is provided by 'shapeDims'.
  mlir::Value reshape(const mlir::Value input,
      const llvm::ArrayRef<DimIndexExpr> shapeDims) const;

  // Generate a 'onnx.Transpose' operation on the 'input' tensor given the
  // permutation array 'perm' and the operator output dimensions 'outputDims'.
  mlir::Value transpose(const mlir::Value input,
      const llvm::ArrayRef<int64_t> perm,
      const llvm::ArrayRef<DimIndexExpr> outputDims) const;
};

// Recursive class specialized for ONNXtoKrnlBuilder refereed to as krnlOnnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxToKrnlBuilder, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnlOnnx(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnlOnnx(db) {}
  OnnxToKrnlBuilder krnlOnnx;
};

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to KRNL.
//===----------------------------------------------------------------------===//

/// Check if one/all operands are scalar values at compile time.
bool isScalarValue(mlir::Value value);
bool hasAllScalarValues(mlir::ValueRange values);
// HasOneElement returns true for scalars as well as tensors that contain only
// one elements, such as 1xf32 or 1x1x1xf32.
bool hasOneElement(mlir::Value value);
// Same as hasOneElement, but check only from the innerDims innermost
// dimensions.
bool hasOneElementInInnermostDims(mlir::Value value, int64_t innerDims);

/// Check if the value is a KrnlGlobalOp with a dense attribute of non-negative
/// integer constants.
bool indicesAreNonNegativeConstants(mlir::Value indices);

// Create a mapping from result type's dimensions to input type's dimensions,
// given that the result type is the result of a reduction op over the input
// type.
std::map<int64_t, int64_t> getReductionMapping(
    mlir::MemRefType inputTy, llvm::ArrayRef<int64_t> axes, bool keepdims);

// Add bounds associated with the op operand to the KRNL iteration pack.
// Dynamic dimensions are supported.
void addDimensionToPack(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, krnl::KrnlIterateOperandPack &pack, mlir::Value operand,
    int index);

// Function that emits the define_loop operation to define `numLoops`
// number of krnl loops, and fill `loop` with the newly defined loops.
void defineLoops(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    std::vector<mlir::Value> &loops, int64_t numLoops);

/// Get a dimension value from a memref. Emit a constant if the dimension is
/// constant. Otherwise, emit a dim op.
/// If the return type is different from IndexType, emit a cast op to cast the
/// output of the dim op.
mlir::Value getDimOrConstant(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value operand, int64_t axis, mlir::Type type);

/// Check whether this op should be lowered to Krnl.Call according to option
/// opsToCall. The op name is used for matching
bool checkOpToCall(mlir::Operation *op, std::string opsForCall);

//===----------------------------------------------------------------------===//
// Fold and emit support.
//===----------------------------------------------------------------------===//

/// Emit an ONNXSqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXSqueezeV11Op(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXUnsqueezeV11Op(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<mlir::Value> foldOrEmitONNXSplitOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::ArrayRef<mlir::Type> resultTypes, mlir::Value input, int64_t axis);

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXTransposeOp(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Type resultType, mlir::Value input,
    mlir::ArrayAttr permAttr);

/// Emit MemRef ReinterpretCastOp to create a new view for 'data'.
/// The new view is created using the given 'outputDims'.
mlir::Value emitMemRefReinterpretCastOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value data, llvm::SmallVectorImpl<IndexExpr> &outputDims,
    mlir::Type outputType);

/// Emit krnl iterate to compute argsort of a given MemRef along a given axis.
/// Output MemRef has the same shape as the input MemRef but is of IndexType.
mlir::Value emitArgSort(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value input, int64_t axis,
    bool ascending = false);

//===----------------------------------------------------------------------===//
// This is to get a scalar operation of a given type for a specific operation.
//===----------------------------------------------------------------------===//

// Definition for easier readability
using NotSuportedScalarOp = void; // Unsupported, e.g. integer version of cos.
using CustomScalarOp = void *;    // Custom support, e.g. float version of cosh.

template <typename Op>
struct ScalarOp {
  using FOp = NotSuportedScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <typename FOp>
using ScalarFOp = typename ScalarOp<FOp>::FOp;
template <typename IOp>
using ScalarIOp = typename ScalarOp<IOp>::IOp;

// Get the identity element of an operation.
// Return NULL if the function does not have identity.
// Specialize this for a new Op.
template <typename Op>
mlir::Value getIdentityValue(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Type type) {
  return nullptr;
}

//===----------------------------------------------------------------------===//
// emitScalarOpFor
//===----------------------------------------------------------------------===//
//
// This is used in the innermost loop of a KrnlIterateOp to insert computation
// composed of one or many scalar ops.
// Use template specialization for each of different ONNX operations.
//
// Note that all values passed in scalarOperands are already loaded in memory.
// *  If they are scalar, then a scalar is loaded. If used in SIMD mode, that
//    vector was splatted to the right shape.
// *  If they have a non value, then that non-value is simply passed on.
// *  If they are a variable with a rank>0, then that the loaded value has been
//    loaded with the right loop indices in it.
//
// So there should be no "loading" of any values inside the emitScalarOpFor
// functions
//===----------------------------------------------------------------------===//

template <typename Op>
mlir::Value emitScalarOpFor(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Operation *op, mlir::Type elementType,
    llvm::ArrayRef<mlir::Value> scalarOperands) {
  // Find the actual element type, regardless of whether we have a vector or
  // scalar elementary type. For some operations, the output in a different type
  // than its input(s), e.g. isNan where inputs are float and output is boolean
  // int. Thus we look at the type the first input argument, and not the output
  // elementType.
  mlir::Type actualElementType =
      MathBuilder::elementTypeWithVector(scalarOperands[0].getType());
  // Perform int or float operation depending on the actual elementary type.
  if (actualElementType.isa<mlir::IntegerType>()) {
    // Generate the integer code only if the scalar integer op is non-void
    // (unsupported) and non-int (supported by custom sequence of ops).
    if constexpr (!(std::is_same<ScalarIOp<Op>, NotSuportedScalarOp>::value) &&
                  !(std::is_same<ScalarIOp<Op>, CustomScalarOp>::value))
      return rewriter.create<ScalarIOp<Op>>(
          loc, elementType, scalarOperands, std::nullopt);
    llvm_unreachable("unsupported integer operation");
  } else if (actualElementType.isa<mlir::FloatType>()) {
    // Generate the floating point code only if the scalar integer op is
    // non-void (unsupported) and non-int (supported by custom sequence of ops).
    if constexpr (!(std::is_same<ScalarFOp<Op>, NotSuportedScalarOp>::value) &&
                  !(std::is_same<ScalarFOp<Op>, CustomScalarOp>::value))
      return rewriter.create<ScalarFOp<Op>>(
          loc, elementType, scalarOperands, std::nullopt);
    llvm_unreachable("unsupported float operation");
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Type conversion from Onnx types to Krnl types:
//   - from Tensor type to the Standard dialect MemRef type
//   - from onnx.StringType to krnl.StringType
//===----------------------------------------------------------------------===//

class KrnlTypeConverter : public mlir::TypeConverter {
public:
  KrnlTypeConverter();

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(
                            funcType.getInputs(), funcType.getResults()),
        [this](mlir::Type type) { return isLegal(type); });
  }

  /// Return true if the operands/results of call have a legal type.
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }

  // Return the default alignment value used when allocating a MemRef buffer for
  // the given type. E.g. some special types for accelerators requires
  // 4K-aligned buffers.
  static int64_t getDefaultAllocAlignment(mlir::Type type);
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// For all ONNX operations.
void populateONNXToKrnlConversionPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableTiling,
    bool enableParallel);

// `ControlFlow` directory methods:
void populateLoweringONNXIfOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXLoopOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXScanOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `Math` directory methods:
void populateLoweringONNXClipOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXCumSumOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXElementwiseOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, DimAnalysis *, bool enableSIMD,
    bool enableParallel);
void populateLoweringONNXGemmOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableTiling,
    bool enableSIMD, bool enableParallel);
void populateLoweringONNXHardmaxOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXLRNOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXMatMulOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, DimAnalysis *,
    bool enableTiling, bool enableSIMD, bool enableParallel);
void populateLoweringONNXMatMulIntegerOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXRandomNormalOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXRandomNormalLikeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReductionOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableSIMD,
    bool enableParallel);
void populateLoweringONNXSoftmaxOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableParallel);
void populateLoweringONNXTopKOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXTriluOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `ML` directory methods:
void populateLoweringONNXCategoryMapperOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `NN` directory methods:
void populateLoweringONNXConvOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableParallel,
    std::string opsForCall);
mlir::LogicalResult generateONNXLayerNormalizationOpONNXCode(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::ONNXLayerNormalizationOp lnOp);
void populateLoweringONNXNormalizationOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, DimAnalysis *, bool enableSIMD);
void populateLoweringONNXPoolingOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `ObjectDetection` directory methods:
void populateLoweringONNXNonMaxSuppressionOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `Quantization` directory methods:
void populateLoweringONNXDynamicQuantizeLinearOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXQuantizeLinearOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `RNN` directory methods:
void populateLoweringONNXGRUOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXLSTMOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXRNNOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `Sequence` directory methods:
void populateLoweringONNXSequenceAtOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSequenceEmptyOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSequenceEraseOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSequenceInsertOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSequenceLengthOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `Tensor` directory methods:
void populateLoweringONNXArgMinMaxOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXDimOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXUnsqueezeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXUnsqueezeV11OpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXTransposeOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableParallel);
void populateLoweringONNXGatherOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXGatherElementsOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXGatherNDOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXPadConstantValuePadOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXPadOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXRangeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReshapeOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, DimAnalysis *);
void populateLoweringONNXIdentityOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConstantOfShapeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConstantOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConcatOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableParallel);
void populateLoweringONNXConcatShapeTransposeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXDepthToSpaceOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSpaceToDepthOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXScatterElementsOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXScatterNDOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXShapeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSliceOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSqueezeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSqueezeV11OpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSplitOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSplitV11OpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSizeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXTileOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXFlattenOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXResizeOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXNonZeroOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReverseSequenceOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXExpandOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXOneHotOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXCompressOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXPrintSignaturePattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXLayoutTransformOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXUniqueOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// `Additional` directory methods:
void populateLoweringONNXShapeTransformOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

void populateLoweringONNXCustomOpPattern(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

// Utilities for generating krnl.call for ONNX Ops

// Create allocate based on COMPUTED shapeHelper.
// The generic computed shapehelper avoids the specific type of shape helper
// for each op, and shape helper may be used in krnl loop generation, too.
// Use template in case some op has special allocation. For the generic cases,
// the typename is only used in location, which is not absolutely needed
template <typename OP_TYPE>
std::vector<mlir::Value> allocForONNXOp(mlir::Operation *op,
    mlir::ConversionPatternRewriter &rewriter,
    const mlir::TypeConverter *const typeConverter,
    ONNXOpShapeHelper &shapeHelper) {
  mlir::Location loc = ONNXLoc<OP_TYPE>(op);

  // Get shape.
  MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
      rewriter, loc);

  std::vector<mlir::Value> allocs;
  for (uint64_t i = 0; i < op->getResults().size(); i++) {
    mlir::Value output = op->getResults()[i];
    // Convert the output type to MemRefType.
    mlir::Type convertedType = typeConverter->convertType(output.getType());
    assert(convertedType && convertedType.isa<mlir::MemRefType>() &&
           "Failed to convert type to MemRefType");
    mlir::MemRefType memRefType = convertedType.cast<mlir::MemRefType>();

    // Insert an allocation and deallocation for the result of this operation.
    mlir::Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims(i));
    allocs.emplace_back(alloc);
  }
  return allocs;
}

// Template to create ONNXOp to Call pattern
template <typename OP_TYPE, typename SHAPEHELPER_TYPE>
struct ONNXGenericOpToCall : public mlir::OpConversionPattern<OP_TYPE> {
  using ADAPTOR_TYPE = typename OP_TYPE::Adaptor;
  ONNXGenericOpToCall(mlir::TypeConverter &typeConverter,
      mlir::MLIRContext *ctx, std::string opsForCall)
      : mlir::OpConversionPattern<OP_TYPE>(
            typeConverter, ctx, /*benefit higher than default*/ 10),
        opsForCall(opsForCall) {}
  std::string opsForCall;

  mlir::LogicalResult match(OP_TYPE onnxOp) const final {
    mlir::Operation *op = onnxOp.getOperation();
    if (!checkOpToCall(op, opsForCall))
      return mlir::failure();

    // Additional checks

    return mlir::success();
  }
  void rewrite(OP_TYPE onnxOp, ADAPTOR_TYPE adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Operation *op = onnxOp.getOperation();
    mlir::Location loc = onnx_mlir::ONNXLoc<OP_TYPE>(op);
    mlir::ValueRange operands = adaptor.getOperands();

    // Get shape.
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);

    SHAPEHELPER_TYPE shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    // Insert an allocation and deallocation for the result of this operation.
    std::vector<mlir::Value> allocs = allocForONNXOp<OP_TYPE>(
        onnxOp, rewriter, this->typeConverter, shapeHelper);

    // Create krnl.call here.
    // You may customize the krnl.call according to your library
    // Use Op name in ONNX as the fuction name. Remove the leading "onnx."
    std::string funcName = op->getName().getStringRef().str().substr(5);
    rewriter.create<mlir::KrnlCallOp>(loc, funcName, allocs, op, operands,
        /*keep all attributes*/ true);
    rewriter.replaceOp(op, allocs);
  }
};

using ONNXConvOpToCall =
    ONNXGenericOpToCall<mlir::ONNXConvOp, ONNXConvOpShapeHelper>;

/// This function returns the index in the list of alloc arguments of the
/// dynamic dimension corresponding to `index` in the MemRef shape.
/// As an example:
///
/// alloc(%d0, %d1, %d2) : memref<10x?x?x20x?x30xf32>
///
/// In the above alloc the list of alloc arguments is being represented by
/// %d0, %d1 and %d2. Their indices 0, 1, 2 correspond to `index` values
/// 1, 2 and 4 in the MemRef shape respectively
int64_t getAllocArgIndex(mlir::memref::AllocOp allocOp, int64_t index);

/// This function returns a scalar of type 'dtype' from an optional value.
/// Optional value must be: NoneType, memref<1xdtype> or memref<dtype>.
/// Default value is used in case of NoneType.
mlir::Value getOptionalScalarValue(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value optionalScalar, mlir::Type elementType,
    double defaultValue);

//===----------------------------------------------------------------------===//
// Support functions for help with custom layout.
//===----------------------------------------------------------------------===//

mlir::MemRefType convertTypeWithCustomONNXDataLayoutToMemRef(mlir::Type type);

// Determine if the MemRef val has a custom layout (i.e. non-identity).
bool hasNonIdentityLayout(mlir::Value val);
// Determine if one or more operands have custom layouts. Return false when
// every layout is an identity layout.
bool hasNonIdentityLayout(mlir::ValueRange operands);

//===----------------------------------------------------------------------===//
// Support functions for reporting.
//===----------------------------------------------------------------------===//

// Populated by configureOnnxToKrnlLoweringPass().

struct OnnxToKrnlLoweringConfiguration {
  static int reportOnParallel;
  static std::string defaultParallelComment;
  static int reportOnSimd;
  static std::string defaultSimdComment;
};

namespace impl {
void onnxToKrnlSimdReport(mlir::Operation *op, bool successful,
    int64_t vectorLength, int64_t simdLoopTripCount,
    const std::string &comment);
void onnxToKrnlParallelReport(mlir::Operation *op, bool successful,
    int64_t loopLevel, int64_t parallelLoopTripCount,
    const std::string &comment);
} // namespace impl

// When reporting is enabled (--opt-report=Parallel), report on if/how are
// the ONNX operation parallelized.
//
// Loop level: -1: none; 0: outermost; 1: next to outermost...
// Parallel loop trip count; 0: none; -1: runtime only; >0: min number known at
// compile time.
// Comment: explanation of how parallelism was achieved / or failed. Comments
// cannot have ',' in them.
inline void onnxToKrnlParallelReport(mlir::Operation *op,
    bool successful = false, int64_t loopLevel = -1,
    int64_t parallelLoopTripCount = 0, const std::string &comment = "") {
  if (OnnxToKrnlLoweringConfiguration::reportOnParallel)
    impl::onnxToKrnlParallelReport(
        op, successful, loopLevel, parallelLoopTripCount, comment);
}

// When reporting is enabled (--opt-report=Simd), report on if/how are
// the ONNX operation simdized.
//
// Vector Length: 0: none; -1: runtime only; >0 min number known at compile
// time.
// Simd loop trip count; 0: none; -1: runtime only; >0: min number known at
// compile time.
// Comment: explanation of how SIMD was achieved / or failed. Comments cannot
// have ',' in them. Use the following comment templates. If SIMD is not
// supported, comments should be "unsupported". If SIMD is supported but fails,
// comment should be "no simd [in <specific place>] because <reason>." When simd
// succeeds, comment indicates what type of pattern is used.
inline void onnxToKrnlSimdReport(mlir::Operation *op, bool successful = false,
    int64_t vectorLength = 0, int64_t simdLoopTripCount = 0,
    const std::string &comment = "") {
  if (OnnxToKrnlLoweringConfiguration::reportOnSimd)
    impl::onnxToKrnlSimdReport(
        op, successful, vectorLength, simdLoopTripCount, comment);
}

} // namespace onnx_mlir
