/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ShapeHelperOpInterface.hpp - Definition for ShapeHelper -----===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeHelperInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SHAPE_HELPER_INFERENCE_H
#define ONNX_MLIR_SHAPE_HELPER_INFERENCE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

struct ONNXOpShapeHelper {
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
   src/Dialect/Krnl/DialectBuilder.hpp, ) or IndexExprBuilderForStableHhlo
   (generates Shape/Stablehlo ops, in
   src/Conversion/ONNXToStablehlo/DialectBuilder.hpp).

   @param scope Index expression scope to be used. If none is provided, a new
   scope is created and stored internally. This scope will then be destructed
   when the current object is destructed.

   Passing a scope is critically important when, to evaluate the shape of a
   given operation, we must also analysis the shape of an other operation. Both
   shape helper MUST share the same scope as otherwise there will be references
   to "deleted" index expressions (as all index expressions are deleted when its
   directly enclosing scope vanishes).
   */

  ONNXOpShapeHelper(mlir::Operation *op, /* Op to be analyzed. */
      mlir::ValueRange operands,         /* If empty, use operands from op.*/
      IndexExprBuilder *ieBuilder, /* Use IndexExprBuilderForAnalysis if null.*/
      IndexExprScope *scope);      /* Install local scope if null. */
  virtual ~ONNXOpShapeHelper();

  // Return true if implemented.
  virtual bool isImplemented() { return true; }

  // Every leaf class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  // Unimplemented operations return success, as these operations may be
  // transformed later in a sequence of operations with implemented shape
  // inference. To ensure an implementation, check the `isImplemented` function.
  // This is used, for example, in dynamic analysis, where unimplemented shape
  // inferences are simply ignored (and conservatively assume no knowledge about
  // that operation's transfer function).
  virtual mlir::LogicalResult computeShape() = 0;

  // Compute shape and assert on failure.
  void computeShapeAndAssertOnFailure();

  // Invoke the virtual computeShape, and on success, update the types of the
  // original operation. First call is used for operations where all the results
  // share the same output type, second for operations where all results have
  // their own output types.
  mlir::LogicalResult computeShapeAndUpdateType(
      mlir::Type elementType, mlir::Attribute encoding = nullptr);
  // If encoding list can be empty or have one entry per type.
  mlir::LogicalResult computeShapeAndUpdateTypes(
      mlir::TypeRange elementTypeRange,
      mlir::ArrayRef<mlir::Attribute> encodingList = {});

  // Get output dims for the N-th output dimension as Index Expressions.
  // Scalar may have a DimsExpr that is empty. Requires an implementation.
  DimsExpr &getOutputDims(int n = 0) {
    if (!isImplemented()) {
      llvm::errs() << "Implementation of shape helper for op " << op->getName()
                   << "is not currently available; please open an issue on "
                   << "\"https://github.com/onnx/onnx-mlir/\" and/or consider "
                   << "contributing code if this op is required.\n";
      llvm_unreachable("missing implementation for shape inference");
    }
    return privateOutputsDims[n];
  }
  // Set output dims, merging the dims associated with the current type with
  // inferred dims provided here, as appropriate.
  void setOutputDims(
      const DimsExpr &inferredDims, int n = 0, bool refineShape = true);

  // Obtain the n-th output result as value.
  mlir::Value getOutput(int n = 0) { return op->getResult(n); }

  // Get index expression scope and operation.
  IndexExprScope *getScope() { return scope; }
  mlir::Operation *getOp() { return op; }

  // Set the operands with a vector of Value
  void setOperands(mlir::ValueRange);

protected:
  // Helper for ops for which the output (n'th) is the same as the type of a
  // given input operand's type.
  mlir::LogicalResult setOutputDimsFromOperand(
      mlir::Value operand, int n = 0, bool refineShape = true);
  // Helper for ops for which the output (n'th) is a constant shape. Value
  // ShapedType::kDynamic indicates runtime dim.
  mlir::LogicalResult setOutputDimsFromLiterals(
      llvm::SmallVector<int64_t, 4> shape, int n = 0, bool refineShape = true);
  // Helper for ops for which the output (n'th) is defined by the shape of
  // another type. Type must have constant shape (all values !=
  // ShapedType::kDynamic).
  mlir::LogicalResult setOutputDimsFromTypeWithConstantShape(
      mlir::Type type, int n = 0, bool refineShape = true);

  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor.
  mlir::Operation *op;
  mlir::ValueRange operands;
  IndexExprBuilder *createIE;
  IndexExprScope *scope;

private:
  // OutputsDims is computed by the child's struct `computeShape` function. It
  // can be set using setOutputDims and retrieved using getOutputDims.
  llvm::SmallVector<DimsExpr, 1> privateOutputsDims;
  // Used to cache the operation's operands (shape inference only).
  llvm::SmallVector<mlir::Value> privateOperandsCache;
  bool ownScope, ownBuilder;
};

} // namespace onnx_mlir

/// Include the auto-generated declarations.
#include "src/Interface/ShapeHelperOpInterface.hpp.inc"
#endif
