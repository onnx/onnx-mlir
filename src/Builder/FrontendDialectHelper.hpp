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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "onnx/onnx_pb.h"
#include "src/Builder/SymbolTable.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

namespace onnx_mlir {

class ExternalDataReader {
public:
  ExternalDataReader(const std::string &externalDataDir);
  ~ExternalDataReader();
  llvm::StringRef read(
      const std::string &fileName, size_t offset, size_t length);

private:
  const std::string externalDataDir;
  std::unordered_map<std::string, std::unique_ptr<llvm::MemoryBuffer>> files;
};

mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
    mlir::OpBuilder &builder, ExternalDataReader &dataReader,
    const onnx::TensorProto &initializer);

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    ExternalDataReader &dataReader, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
