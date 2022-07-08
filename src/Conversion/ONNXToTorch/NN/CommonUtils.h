/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- CommonUtils.h -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

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

typedef struct dim_pads {
  /// Padding along each spatial dimension. Usually this will refer to with
  /// and height. (pad_dimN_start, pad_dimN_end, ..., pad_dim1_start,
  /// pad_dim1_end)
  std::vector<Value> padding;
  /// Symmetric padding only has needs to spatial dimension and therefore
  /// assumes padding size of two.
  bool isSymmetric;
} dim_pads;

dim_pads createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty, Location loc,
    ConversionPatternRewriter &rewriter);
std::vector<Value> createArrayAttribute(::mlir::ArrayAttr onnxArrayAttr,
    Type ty, Location loc, ConversionPatternRewriter &rewriter,
    int default_val = 0);
Torch::ValueTensorType toTorchType(mlir::MLIRContext *ctx, Type t);
mlir::Value getTorchTensor(Value operand, ConversionPatternRewriter &rewriter,
    mlir::MLIRContext *context, Location loc);
Value getIntValue(int val, ConversionPatternRewriter &rewriter,
    mlir::MLIRContext *context, Location loc);
std::vector<int> toVector(mlir::ArrayAttr axesAttr);
mlir::FloatAttr convertToIEEEDouble(mlir::FloatAttr attr);