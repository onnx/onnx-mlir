/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.hpp ----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FRONTEND_HELPER_H
#define ONNX_MLIR_FRONTEND_HELPER_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "onnx/onnx_pb.h"

#include <string>

namespace onnx_mlir {

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::MLIRContext *ctx,
    const std::string &externalDataDir, const onnx::TensorProto &initializer);

} // namespace onnx_mlir
#endif
