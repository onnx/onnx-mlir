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
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Import a model into the ONNX-MLIR dialect.
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
  // Custom shape information for the graph inputs.
  // Its format is 'input_id:dim,dim,dim|input_id:dim,dim,dim'
  // E.g. An ONNX model has two dynamic inputs
  //   - (arg0: tensor<?x?x?xf32>, arg1: tensor<?x5xf32>)
  // If we want to compile the model for static dimensions, we can use:
  //   - shapeInformation = '0:3,4,5|1:10,5'
  // to obtain a model with two staic inputs:
  //   - (arg0: tensor<3x4x5xf32>, arg1: tensor<10x5xf32>)
  //
  std::string shapeInformation = "";
};

/*!
 *  Import an ONNX model array into the ONNX Dialect.
 *  @param onnxBuffer buffer containing onnx model protobuf.
 *  @param bufferSize size of buffer containing onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @return 0 on success, error number of failure.
 */
int ImportFrontendModelArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *errorMessage, ImportOptions options = ImportOptions());

/*!
 *  Import an ONNX model file into the ONNX Dialect.
 *  @param model_fname file name pointing to the onnx model protobuf.
 *  @param MLIR::module generated for the ONNX model.
 *  @return 0 on success, error number of failure.
 */
int ImportFrontendModelFile(std::string model_fname, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *errorMessage,
    ImportOptions options = ImportOptions());

/*!
 *  Import an ONNX model proto into the ONNX Dialect.
 *  @param model the onnx model protobuf.
 *  @return MLIR::module generated for the ONNX model.
 */
void ImportFrontendModel(const onnx::ModelProto &model,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    ImportOptions options = ImportOptions());

/*!
 *  TODO: Import models into other extension dialects that cover the
 *  operations specific to other frameworks such as Tensorflow or Pytorch.
 */
} // namespace onnx_mlir
