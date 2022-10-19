/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for importing and cleaning of onnx models.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "onnx/onnx_pb.h"

bool IsTopologicallySorted(const onnx::GraphProto &graph);

bool SortGraph(onnx::GraphProto *graph);
