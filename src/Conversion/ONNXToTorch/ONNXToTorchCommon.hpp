/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- ONNXToTorchCommon.hpp - ONNX dialects to Torch lowering -===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ========================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===-----------------------------------------------------------------===//

#pragma once

#include <map>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
#include "llvm/Transforms/Scalar/LICM.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// A global variable to indicate whether this pass will emit dealloc for
// allocated memrefs or not during the conversion of ONNX to Krnl.
// extern bool ONNXToKrnl_gEmitDealloc;

//===-----------------------------------------------------------------===//
// Type conversion from Onnx types to Krnl types:
//   - from Tensor type to the Standard dialect MemRef type
//   - from onnx.StringType to krnl.StringType
//===-----------------------------------------------------------------===//

class TorchTypeConverter : public TypeConverter {
public:
  using TypeConverter::TypeConverter;

  TorchTypeConverter();

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

//===-----------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===-----------------------------------------------------------------===//

// `NN` directory methods:
void populateLoweringONNXToTorchConvOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchConstOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchLeakyReluOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchMaxPoolSingleOutOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchConstantPadNdOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchReluOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchGlobalAveragePoolOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchReduceMeanOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);

void populateLoweringONNXToTorchGemmOpPattern(
    RewritePatternSet &, TypeConverter &, MLIRContext *);
