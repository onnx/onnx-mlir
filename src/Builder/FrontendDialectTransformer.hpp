/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- FrontendDialectTransformer.hpp - MLIR Operations -----------===//
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

#include "src/Builder/FrontendDialectHelper.hpp"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Import a model into the ONNX MLIR dialect.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

/*!
 * Options to control the translation of an ONNX model to ONNX-MLIR.
 */
struct ImportOptions {
  // Use types/shapes in the input-model for translation (for intermediate
  // variables)
  bool useOnnxModelTypes = false;
  bool invokeOnnxVersionConverter = false;
};

/*!
 *  Import an ONNX model file into the ONNX Dialect.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @return MLIR::module generated for the ONNX model.
 */
void ImportFrontendModelFile(std::string model_fname,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module,
    ImportOptions options = ImportOptions());

/*!
 *  Import an ONNX model proto into the ONNX Dialect.
 *  @param model the onnx model protobuf.
 *  @return MLIR::module generated for the ONNX model.
 */
void ImportFrontendModel(const onnx::ModelProto &model,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module,
    ImportOptions options = ImportOptions());

/*!
 *  TODO: Import models into other extension dialects that cover the
 *  operations specific to other frameworks such as Tensorflow or Pytorch.
 */
} // namespace onnx_mlir
