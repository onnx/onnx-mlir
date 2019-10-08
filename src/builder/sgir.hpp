//===----------------------------------------------------------------------===//
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

namespace onnf {
/*!
 *  Import an ONNX Model into SGIR.
 *  @param model onnx model.
 *  @return MLIR::module generated for the ONNX model.
 */
mlir::OwningModuleRef SGIRImportModel(onnx::ModelProto model);

/*!
 * Import an ONNX Model file into SGIR.
 * @param model_fname file name pointing to the onnx model protobuf.
 * @return MLIR::module generated for the ONNX model.
 */
mlir::OwningModuleRef SGIRImportModelFile(std::string model_fname);
}  // namespace onnf
