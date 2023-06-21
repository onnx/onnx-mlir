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

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include "onnx/onnx_pb.h"

#include <memory>

namespace onnx_mlir {

struct ExternalDataFileSlice {
  std::shared_ptr<llvm::MemoryBuffer> file;
  uint64_t offset;
  uint64_t length;
  llvm::StringRef getBufferSlice() const {
    return file->getBuffer().substr(offset, length);
  }
};

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::MLIRContext *ctx,
    const onnx::TensorProto &initializer,
    const ExternalDataFileSlice *externalDataFileSlice = nullptr);

} // namespace onnx_mlir
