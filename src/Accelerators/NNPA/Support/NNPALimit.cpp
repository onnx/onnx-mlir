/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- NNPALimit.cpp --------------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// The NNPA constant values.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <assert.h>
#include <string>

//===----------------------------------------------------------------------===//
// Compatibility checks

/// Convert the input NNPA level, ie. "z16", to a integer value representing the
/// level, ie. "16". When unkown / out of bounds, returns 0.
int64_t convertNNPALevel(std::string inputNNPALevel) {
  if (inputNNPALevel.size() != 3 || inputNNPALevel[0] != 'z')
    return 0;
  if (inputNNPALevel[1] == '1') {
    if (inputNNPALevel[2] == '6')
      return 16;
  }
  return 0;
}

/// A function to check whether the input NNPA level, ie. "z16", is compatible
/// with the current NNPA level.
bool isCompatibleWithNNPALevel(std::string inputNNPALevel) {
  int64_t inLevel = convertNNPALevel(inputNNPALevel);
  int64_t mcpuLevel = convertNNPALevel(onnx_mlir::mcpu);
  if (inLevel == 0 && mcpuLevel == 0)
    return false;
  return inLevel <= mcpuLevel;
}

//===----------------------------------------------------------------------===//
// Max dimension checks

// The NNPA maximum supported dimension index size value by using
// zdnn_get_nnpa_max_dim_idx_size() This value depends on HW.
static constexpr int64_t NNPA_Z16_MAXIMUM_DIMENSION_INDEX_SIZE = 32768;

int64_t NNPAGetMaxForDim(int64_t dim, int64_t rank) {
  assert(rank >= 0 && "expected positive rank");
  assert(dim >= 0 && dim < rank && "dim outside range [0..rank)");
  if (rank > 4)
    return 0;
  if (isCompatibleWithNNPALevel(NNPA_Z16))
    return NNPA_Z16_MAXIMUM_DIMENSION_INDEX_SIZE;
  return 0;
}
