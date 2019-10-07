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
} // namespace mlir

namespace onnf {
  /*! 
   *  Import an ONNX Model into SGIR
   *  @param model onnx model.
   *  @return MLIR::module generated for the ONNX model
   */
  mlir::OwningModuleRef SGIRImportModel(onnx::ModelProto model);

} //namespace onnf

