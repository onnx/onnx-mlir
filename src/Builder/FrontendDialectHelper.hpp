/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.hpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <numeric>
#include <regex>
#include <tuple>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "onnx/onnx_pb.h"
#include "src/Builder/SymbolTable.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

namespace onnx_mlir {

struct InitializedTensorMapping : SymbolMapping<onnx::TensorProto> {
  mlir::Value EmitInitializerForInputTensor(
      mlir::Location loc, mlir::OpBuilder &builder, const std::string &name);

  // Get initialized tensor.
  onnx::TensorProto GetInitializedTensor(std::string name) {
    return GetTensorByOnnxName(name);
  }
};

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(
    mlir::OpBuilder &builder, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
