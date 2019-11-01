//===- frontend_dialect_transformer.hpp - MLIR Operations -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
}  // namespace mlir

//===----------------------------------------------------------------------===//
// Import a model into one of ONNF's frontend models.
//===----------------------------------------------------------------------===//

namespace onnf {
/*!
 *  Import an ONNX model into ONNF's ONNX Dialect.
 *  @param model onnx model.
 *  @return MLIR::module generated for the ONNX model.
 */
mlir::OwningModuleRef ImportFrontendModel(onnx::ModelProto model);

/*!
 *  Import an ONNX model file into ONNF's ONNX Dialect.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @return MLIR::module generated for the ONNX model.
 */
mlir::OwningModuleRef ImportFrontendModelFile(std::string model_fname);

/*!
 *  TODO: Import models into other extension dialects that cover the
 *  operations specific to other frameworks such as Tensorflow or Pytorch.
 */
}  // namespace onnf
