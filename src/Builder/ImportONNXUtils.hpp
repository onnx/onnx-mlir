/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for importing and cleaning of onnx models.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_IMPORT_UTILS_H
#define ONNX_MLIR_IMPORT_UTILS_H

#include "onnx/onnx_pb.h"

bool IsTopologicallySorted(const onnx::GraphProto &graph);

bool SortGraph(onnx::GraphProto *graph);
#endif
