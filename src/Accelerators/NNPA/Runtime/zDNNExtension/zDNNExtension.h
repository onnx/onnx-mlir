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

// AIU parameters getting from zdnn_private.h.
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_STICKS_PER_PAGE 32

// Chunk size used when spliting a big tensor.
// Must be divisible by AIU_STICKS_PER_PAGE.
#define CHUNK_SIZE 4096

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

void getZTensorShape(const zdnn_ztensor *t, zTensorShape *shape);
void createZTensorInDim2(
    const zdnn_ztensor *input, uint32_t pos, bool isLast, zdnn_ztensor *output);
zdnn_status freeZTensorChunk(zdnn_ztensor *t, bool freeBuffer);
void copyZTensorInDim2(
    const zdnn_ztensor *input, uint32_t pos, bool isLast, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// Extension Functions
// Same name as zdnn functions but with the `_ext` postfix.
// -----------------------------------------------------------------------------

zdnn_status zdnn_matmul_op_ext(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, zdnn_ztensor *output);

#ifdef __cplusplus
}
#endif
