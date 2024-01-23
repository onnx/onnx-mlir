/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zDNNExtension.h -----------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Sets of extensions to the zdnn library.
//
//===----------------------------------------------------------------------===//

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define USE_PTHREAD 1

// AIU parameters getting from zdnn_private.h.
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

// Default chunk size used when spliting a big tensor.
// Must be divisible by AIU_STICKS_PER_PAGE.
#define DEFAULT_ZTENSOR_SPLIT_SIZE 4096
#define DEFAULT_ZTENSOR_SPLIT_ENABLED 0
#define DEFAULT_ZTENSOR_SPLIT_DEBUG 0

// -----------------------------------------------------------------------------
// Misc Macros
// -----------------------------------------------------------------------------

#define CEIL(a, b) (uint64_t)((a + b - 1) / b) // positive numbers only

// -----------------------------------------------------------------------------
// Common structures
// -----------------------------------------------------------------------------

typedef struct zTensorShape {
  uint32_t dim6;
  uint32_t dim5;
  uint32_t dim4;
  uint32_t dim3;
  uint32_t dim2;
  uint32_t dim1;
} zTensorShape;

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

uint32_t ZTensorSplitSizeFromEnv();
bool ZTensorSplitEnabledFromEnv();
bool ZTensorSplitDebugFromEnv();

void getZTensorShape(const zdnn_ztensor *t, zTensorShape *shape);
zdnn_status allocZTensorInDim2(
    const zdnn_ztensor *input, uint32_t chunkSize, zdnn_ztensor *output);
zdnn_status freeZTensorChunk(zdnn_ztensor *t);
void copyZTensorInDim2(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t offset, bool fromChunk);
void copyZTensorInDim2Scalar(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t offset, bool fromChunk);

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

zdnn_status zdnn_matmul_bcast_op_ext(const zdnn_ztensor *inputA,
    const zdnn_ztensor *inputB, const zdnn_ztensor *inputC, int opType,
    zdnn_ztensor *output);

#ifdef __cplusplus
}
#endif
