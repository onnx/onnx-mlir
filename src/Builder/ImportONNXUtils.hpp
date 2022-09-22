/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for import and clean up onnx files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "onnx/onnx_pb.h"

bool TopologicallySorted(const onnx::GraphProto &graph);

bool SortGraph(onnx::GraphProto *graph);
