/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NewShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility>

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

using DimsExpr = llvm::SmallVector<IndexExpr, 4>;

//===----------------------------------------------------------------------===//
// Top shape helper class
//===----------------------------------------------------------------------===//

struct NewONNXOpShapeHelper {
  /* Constructor for shape inference.

   This class and its specialized subclasses are used in one of two situation:
   1) For shape analysis (where no code is generated) during shape inference.
   2) For shape generation (where runtime shapes are computed using generated
   code) during lowering.

   @param op Operation to be analyzed.

   @param operands Operands of the operation to be analyzed. When passing an
   empty list, the operands are taken from the operations. When passing an
   explicit, non-empty list, these operands are used instead of the ones from
   the operation.

   The former option (empty list) is used during shape inference.

   The later option (explicit list) is used during lowering, as typically there
   are two sets of operands, the ones already lowered (explicit list) and the
   original ones (contained within the op). Using shapes during lowering
   typically deals with already-lowered operands.

   However, during lowering, it may be sometime advantageous to perform the
   analysis of the index expressions in the "old" dialect, e.g. in ONNX instead
   of the destination dialect. To enable this, just pass `{}` as operands and
   the original operands associated with the unmodified operation will be used.

   @param ieBuilder Class that scans the operands to gather IndexExpr from them.
   Typically used to gather shape and constant values.

   During shape inference, we typically use IndexExprBuilderForAnalysis
   (src/Dialect/Mlir/DialectBuilder.hpp), which uses questionmark for values
   unkown at compile time. This builder is default when no ieBuilder is given.

   During lowering, we typically use and Index Expr Builder that generates code
   for values unknown at compile time. Example of such subclasses are
   IndexExprBuilderForKrnl (generates Krnl ops, in
   src/Dialect/Krnl/DialectBuilder.hpp, ) or IndexExprBuilderForMhlo (generates
   Shape/MHLO ops, in src/Conversion/ONNXToMhlo/DialectBuilder.hpp).

   @param scope Index expression scope to be used. If none is provided, a new
   scope is created and stored internally. This scope will then be destructed
   when the current object is destructed.

   Passing a scope is critically important when, to evaluate the shape of a
   given operation, we must also analysis the shape of an other operation. Both
   shape helper MUST share the same scope as otherwise there will be references
   to "deleted" index expressions (as all index expressions are deleted when its
   directly enclosing scope vanishes).
   */

  NewONNXOpShapeHelper(mlir::Operation *op, /* Op to be analyzed. */
      mlir::ArrayRef<mlir::Value> operands, /* If empty, use operands from op.*/
      IndexExprBuilder *ieBuilder, /* Use IndexExprBuilderForAnalysis if null.*/
      IndexExprScope *scope);      /* Install local scope if null. */
  virtual ~NewONNXOpShapeHelper();

  // Every leaf class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  virtual mlir::LogicalResult computeShape() = 0;
  // Compute shape and assert on failure.
  void computeShapeOrAssert();

  // Invoke the virtual computeShape, and on success, update the types of the
  // original operation. First call is used for operations with one result,
  // second for operations with one or more results.
  mlir::LogicalResult computeShapeAndUpdateType(mlir::Type elementType);
  mlir::LogicalResult computeShapeAndUpdateTypes(mlir::TypeRange elementTypes);

  // Get/set output dims for the N-th output dimension as Index Expressions.
  DimsExpr &getOutputDims(int n = 0) { return privateOutputsDims[n]; }
  void setOutputDims(DimsExpr inferredDims, int n = 0);

  // Obtain the n-th output result as value.
  mlir::Value getOutput(int n = 0) { return op->getResult(n); }

  // Get index expression scope and operation.
  IndexExprScope *getScope() { return scope; }
  mlir::Operation *getOp() { return op; }

protected:
  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor.
  mlir::Operation *op;
  mlir::ArrayRef<mlir::Value> operands;
  IndexExprBuilder *createIE;
  IndexExprScope *scope;

private:
  //  outputsDims is computed by the child's struct `computeShape` function. It
  //  can be set using setOutputDims and retrieved using getOutputDims.
  llvm::SmallVector<DimsExpr, 1> privateOutputsDims;
  // Used to cache the operation's operands (shape inference only).
  llvm::SmallVector<mlir::Value> privateOperandsCache;
  bool ownScope, ownBuilder;
};

//===----------------------------------------------------------------------===//
// Unary Ops
//===----------------------------------------------------------------------===//

/// Compute an output shape for a unary element-wise operation. The output and
/// input of an unary element-wise operation have the same shape.
struct NewONNXUnaryOpShapeHelper : public NewONNXOpShapeHelper {
  NewONNXUnaryOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : NewONNXOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~NewONNXUnaryOpShapeHelper() {}

  mlir::LogicalResult computeShape() final;
};

//===----------------------------------------------------------------------===//
// Broadcast Ops
//===----------------------------------------------------------------------===//

// Compute a broadcasted shape from the shapes of given operands. Operands must
// be ranked in advance.
struct NewONNXBroadcastOpShapeHelper : public NewONNXOpShapeHelper {
  NewONNXBroadcastOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr,
      bool hasUniBroadcasting = false, bool hasNoBroadcasting = false)
      : NewONNXOpShapeHelper(op, operands, ieBuilder, scope), inputsDims(),
        outputRank(0), hasUniBroadcasting(hasUniBroadcasting),
        hasNoBroadcasting(hasNoBroadcasting) {}
  virtual ~NewONNXBroadcastOpShapeHelper() {}

  // Custom shape compute which takes additional parameters.
  mlir::LogicalResult customComputeShape(
      mlir::ArrayRef<mlir::Value> initialOperands, DimsExpr *additionalOperand);

  // Default shape compute (every operands of the operation and no additional
  // parameters).
  mlir::LogicalResult computeShape() override {
    return customComputeShape(operands, nullptr);
  }

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - operandIndex: index of the operand in 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  mlir::LogicalResult getAccessExprs(mlir::Value operand, uint64_t i,
      const llvm::SmallVectorImpl<IndexExpr> &outputAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank. Instantiated during ComputeShape.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // A vector of IndexExprs representing the output shape.
  // in upper DimsExpr outputDims;
  uint64_t outputRank;

protected:
  // When unidirectional broadcasting is true, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool hasUniBroadcasting;
  // When isNoBroadcasting is true, the shape of all input is assumed to be
  // same. This flag is used to test dynamic shape. There is no impact on static
  // shape.
  bool hasNoBroadcasting;
};

// Helper for ExpandOp
struct NewONNXExpandOpShapeHelper : public NewONNXBroadcastOpShapeHelper {
  NewONNXExpandOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : NewONNXBroadcastOpShapeHelper(op, operands, ieBuilder, scope) {}
  virtual ~NewONNXExpandOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
};

//===----------------------------------------------------------------------===//
// Shape op
//===----------------------------------------------------------------------===//

struct NewONNXShapeOpShapeHelper : public NewONNXOpShapeHelper {
  NewONNXShapeOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands,
      IndexExprBuilder *ieBuilder = nullptr, IndexExprScope *scope = nullptr)
      : NewONNXOpShapeHelper(op, operands, ieBuilder, scope), start(-1),
        end(-1) {}
  virtual ~NewONNXShapeOpShapeHelper() {}
  mlir::LogicalResult computeShape() final;
  // Compute the shape values of input data for dimensions between start and
  // end.
  void computeSelectedDataShape(DimsExpr &selectedDataShape);
  // Compute start & end value without calls to computeShape.
  static void getStartEndValues(
      mlir::ONNXShapeOp shapeOp, int64_t &startVal, int64_t &endVal);
  // Additional data for ShapeOp.
  int64_t start, end; // Start and end properly normalized (-1 is undef).
};

//===----------------------------------------------------------------------===//
// Pooling Ops
//===----------------------------------------------------------------------===//

// Generic pool shape helper, further refined by specific pooling ops.
struct NewONNXPoolOpShapeHelper : public NewONNXOpShapeHelper {
  NewONNXPoolOpShapeHelper(mlir::Operation *op,
      mlir::ArrayRef<mlir::Value> operands, IndexExprBuilder *ieBuilder,
      bool hasFilter, bool ceilMode, IndexExprScope *scope)
      : NewONNXOpShapeHelper(op, operands, ieBuilder, scope),
        hasFilter(hasFilter), ceilMode(ceilMode) {}
  virtual ~NewONNXPoolOpShapeHelper() {}
  mlir::LogicalResult customComputeShape(mlir::Value X /* image */,
      mlir::Value W /* filter */,
      mlir::Optional<mlir::ArrayAttr> kernelShapeOpt, llvm::StringRef autoPad,
      mlir::Optional<mlir::ArrayAttr> padOpt,
      mlir::Optional<mlir::ArrayAttr> strideOpt,
      mlir::Optional<mlir::ArrayAttr> dilationOpt);

  // Additional data for pool operations.
  bool hasFilter; // If has filter, it also has CO and optional kernel.
  bool ceilMode;  // Use ceil or floor for auto_pad=NOTSET policy.
  // Values set by customComputeShape.
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
};

#define DECLARE_SHAPE_HELPER(SHAPE_HELPER)                                     \
  class SHAPE_HELPER : public NewONNXPoolOpShapeHelper {                       \
  public:                                                                      \
    SHAPE_HELPER(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,    \
        IndexExprBuilder *ieBuilder = nullptr,                                 \
        IndexExprScope *scope = nullptr);                                      \
    virtual ~SHAPE_HELPER() {}                                                 \
    mlir::LogicalResult computeShape() final;                                  \
  };
DECLARE_SHAPE_HELPER(NewONNXAveragePoolOpShapeHelper)
DECLARE_SHAPE_HELPER(NewONNXConvOpShapeHelper)
DECLARE_SHAPE_HELPER(NewONNXMaxPoolSingleOutOpShapeHelper)
#undef DECLARE_SHAPE_HELPER

} // namespace onnx_mlir