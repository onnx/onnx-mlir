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

using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Scan mcpu and march flags into NNPALevel

static NNPALevel getNNPAFromTargetFlag(std::string str) {
  // Coded it efficiently as it is called over and over again.
  if (str.size() == 3) {
    if (str[0] == 'z') {
      if (str[1] == '1') {
        if (str[2] == '6')
          return NNPALevel::M14;
      }
    }
  } else if (str.size() == 6) {
    if (str[0] == 'a' && str[1] == 'r' && str[2] == 'c' && str[3] == 'h') {
      if (str[4] == '1') {
        if (str[5] == '4')
          return NNPALevel::M14;
        if (str[5] == '5')
          return NNPALevel::M15;
      }
    }
  }
  return NNPALevel::NONE;
}

// Read march flag, and if undefined, then read mcpu.
NNPALevel getNNPAFromFlags() {
  NNPALevel level = getNNPAFromTargetFlag(march);
  if (level == NNPALevel::NONE)
    level = getNNPAFromTargetFlag(mcpu);
  return level;
}

//===----------------------------------------------------------------------===//
// Print NNPALevel as a string (depending on which option was given)

// Print level using mcpu, march, or both depending on the options that were
// given to the compiler. Favor the zYY names below over the archXX names.
std::string getNNPAString(NNPALevel level) {
  std::string val;
  if (!mcpu.empty()) {
    // The mcpu compiler option is defined, give an answer
    if (level == NNPALevel::M14)
      val = "--mcpu=z16"; // Note: --mcpu is deprecated.
    else if (level == NNPALevel::M15)
      val = "--mcpu=arch15"; // Note: --mcpu is deprecated.
    else
      assert(level == NNPALevel::NONE && "unknown mcpu option");
  }
  if (!march.empty()) {
    if (!val.empty() && level != NNPALevel::NONE)
      val = val.append(" ");
    // The march compiler option is defined, give an answer
    if (level == NNPALevel::M14)
      val = val.append("--march=z16");
    else if (level == NNPALevel::M15)
      val = val.append("--march=arch15");
    else
      assert(level == NNPALevel::NONE && "unknown march option");
  }
  return val;
}

/// A function to check whether the input NNPA level, ie. "z16", is compatible
/// with the current NNPA level.
bool isCompatibleWithNNPALevel(NNPALevel level) {
  NNPALevel flagLevel = getNNPAFromFlags();
  if (level == NNPALevel::NONE && flagLevel == NNPALevel::NONE)
    return false;
  return level <= flagLevel;
}

/// A function to check whether the current --march, ie. "z16", is less than or
/// equal to the given NNPA level.
bool isLessEqualNNPALevel(NNPALevel level) {
  NNPALevel flagLevel = getNNPAFromFlags();
  if (level == NNPALevel::NONE && flagLevel == NNPALevel::NONE)
    return false;
  return flagLevel <= level;
}

//===----------------------------------------------------------------------===//
// Max dimension checks

// The NNPA maximum supported dimension index size value by using
// zdnn_get_nnpa_max_dim_idx_size() This value depends on HW.
static constexpr int64_t NNPA_ARCH14_MAXIMUM_DIMENSION_INDEX_SIZE = 32768;

/*
  ARCH15 sizes are dimension dependent:
        for(int i=1; i<=4; ++i) {
                uint32_t maxDimSize = zdnn_get_max_for_dim((uint8_t) i);
                printf("  max size for dim e%i: %i\n", i, (int) maxDimSize);
        }

  max size for dim e1: 2097152
  max size for dim e2: 1048576
  max size for dim e3: 32768
  max size for dim e4: 32768
*/
static constexpr int64_t NNPA_ARCH15_MAXIMUM_DIMENSION_INDEX_SIZES[] = {
    /*e1*/ 2097152, /*e2*/ 1048576, /*e3*/ 32768, /*e4*/ 32768};

int64_t NNPAGetMaxForDim(int64_t dim, int64_t rank) {
  assert(rank >= 0 && "expected positive rank");
  assert(dim >= 0 && dim < rank && "dim outside range [0..rank)");
  if (rank > 4)
    return 0;
  // rank 4: (index from memref = 0, 1, 2, 3) -> e (4, 3, 2, 1)
  // rank 3: (index from memref = 0, 1, 2) -> e (3, 2, 1)
  // rank 2: (index from memref = 0, 1) -> e (2, 1)
  // rank 1: (index from memref = 0) -> e (1)
  int64_t e = rank - dim;

  // List from newest NNPA to oldest, to select the most recent compatible
  // one.
  if (isCompatibleWithNNPALevel(NNPALevel::M15))
    return NNPA_ARCH15_MAXIMUM_DIMENSION_INDEX_SIZES[e - 1];

  if (isCompatibleWithNNPALevel(NNPALevel::M14))
    return NNPA_ARCH14_MAXIMUM_DIMENSION_INDEX_SIZE;

  return 0;
}
