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
#define AIU_BYTES_PER_STICK 128
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

// Default chunk size used when spliting a big zTensor.
// Must be divisible by AIU_STICKS_PER_PAGE.
#define DEFAULT_ZTENSOR_SPLIT_SIZE 1024
// zTensor splitting is off by default.
#define DEFAULT_ZTENSOR_SPLIT_ENABLED 0
// zTensor splitting debug is off by default.
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

typedef struct ChunkInfo {
  uint32_t size;
} ChunkInfo;

typedef struct SplitInfo {
  // Axis to split the tensor. Used to refer to an axis in (e4, e3, e2, e1)
  uint32_t axis;
  // Size of the dimension at axis
  uint32_t totalSize;
  // Size of each chunk. The last chunk may be smaller
  uint32_t chunkSize;
  // Size of each chunk in the stickifified tensor. The last chunk may be
  // smaller
  uint32_t chunkSizeInStick;
  // The number of chunks
  uint32_t numOfChunks;
  // Information for each chunk
  ChunkInfo *chunks;
  // Sub zTensors
  zdnn_ztensor *tensors;
} SplitInfo;

// -----------------------------------------------------------------------------
// Query variable environments
// -----------------------------------------------------------------------------

/**
 * \brief Get a value set by the variable environment OM_ZTENSOR_SPLIT_ENABLED.
 */
bool ZTensorSplitEnabledFromEnv();
/**
 * \brief Get a value set by the variable environment OM_ZTENSOR_SPLIT_SIZE.
 */
uint32_t ZTensorSplitSizeFromEnv();
/**
 * \brief Get a value set by the variable environment OM_ZTENSOR_SPLIT_DEBUG.
 */
bool ZTensorSplitDebugFromEnv();

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/**
 * \brief Initialize a SplitInfo struct.
 *
 * @param input input ztensor to split.
 * @param axis axis to split the input
 * @param chunkSize size of each chunk
 * @param splitInfo information of all chunks
 * @return true if the ztensor is splitable. Otherwise, false.
 */
bool initSplitInfo(const zdnn_ztensor *input, uint32_t axis, uint32_t chunkSize,
    SplitInfo *splitInfo);

/**
 * \brief Free buffers related to a SplitInfo struct.
 *
 * This does not free the SplitInfo itself.
 *
 * @param splitInfo information of all chunks
 */
void freeSplitInfoBuffer(SplitInfo *splitInfo);

/**
 * \brief Split a ztensor into multiple chunks.
 *
 * @param input a ztensor to split
 * @param splitInfo information of all chunks
 * @param copyData whether or not copy data from ztensor to each chunk
 */
void splitZTensor(
    const zdnn_ztensor *input, SplitInfo *splitInfo, bool copyData);
/**
 * \brief Merge chunks into a ztensor.
 *
 * @param splitInfo information of all chunks
 * @param output a ztensor obtained by merging the chunks
 */
void mergeZTensors(const SplitInfo *splitInfo, zdnn_ztensor *output);

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
