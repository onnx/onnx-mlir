/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToLinalgCommon.hpp - ONNX dialects to Krnl lowering --------===//
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
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

// A global variable to indicate whether this pass will emit dealloc for
// allocated memrefs or not during the conversion of ONNX to Krnl.
extern bool ONNXToKrnl_gEmitDealloc;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Reuse the type converter for OnnxToKrnl now
// Type conversion from Onnx types to Krnl types:
//   - from Tensor type to the Standard dialect MemRef type
//   - from onnx.StringType to krnl.StringType
//===----------------------------------------------------------------------===//

class LinalgTypeConverter : public mlir::TypeConverter {
public:
  LinalgTypeConverter();

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
};

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

// For all ONNX operations.
void populateONNXToLinalgConversionPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableTiling);

// `Math` directory methods:
void populateLoweringONNXMatMulOpPattern(mlir::RewritePatternSet &,
    mlir::TypeConverter &, mlir::MLIRContext *, bool enableTiling);


} // namespace onnx_mlir
