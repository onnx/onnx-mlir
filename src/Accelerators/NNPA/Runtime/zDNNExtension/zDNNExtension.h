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

#ifndef ONNX_MLIR_ZDNNEXTENSION_H
#define ONNX_MLIR_ZDNNEXTENSION_H

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

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

extern bool OMZTensorSplitEnabled;
extern bool OMZTensorSplitDebug;
extern uint32_t OMZTensorSplitSize;

// -----------------------------------------------------------------------------
// Misc Macros
// -----------------------------------------------------------------------------

#define CEIL(a, b) (uint64_t)((a + b - 1) / b) // positive numbers only

// -----------------------------------------------------------------------------
// Common structures
// -----------------------------------------------------------------------------

typedef struct OrigShape {
  uint32_t e4;
  uint32_t e3;
  uint32_t e2;
  uint32_t e1;
} OrigShape;

typedef struct zTensorShape {
  uint32_t dim6;
  uint32_t dim5;
  uint32_t dim4;
  uint32_t dim3;
  uint32_t dim2;
  uint32_t dim1;
} zTensorShape;

typedef struct ChunkInfo {
  // Dim size for this chunk along the original axis.
  uint32_t dimSize;
  // Offset of the split point of this chunk in the stickified axis.
  uint32_t offsetInStick;
  // ztensor of this chunk.
  zdnn_ztensor *ztensor;
} ChunkInfo;

typedef struct SplitInfo {
  // Original ztensor.
  const zdnn_ztensor *origZTensor;
  // Axis to split the tensor. Used to refer to an axis in (e4, e3, e2, e1).
  uint32_t axis;
  // Size of the dimension at axis.
  uint32_t totalSize;
  // Size of each chunk. The last chunk may be smaller.
  uint32_t chunkSize;
  // The number of chunks.
  uint32_t numOfChunks;
  // If reuse the original ztensor's buffer or not.
  // If yes, there is no need to allocate buffers for chunks, and chunk ztensors
  // will use the original ztensor's buffer.
  bool reuseOrigBuffer;
  // Information for each chunk.
  ChunkInfo *chunks;
} SplitInfo;

// -----------------------------------------------------------------------------
// Initialiation for zDNN extension
// -----------------------------------------------------------------------------

/**
 * \brief Initialization for zDNN extension.
 */
void zDNNExtensionInit();

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/**
 * \brief Get the original shape of ztensor.
 *
 * @param input input ztensor
 * @param shape shape information
 */
void getOrigShape(const zdnn_ztensor *t, OrigShape *shape);

/**
 * \brief Initialize a SplitInfo struct.
 *
 * @param splitInfo information for splitting
 * @return true if the ztensor is splitable. Otherwise, false
 */
bool initSplitInfo(SplitInfo *splitInfo);

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
 * @param splitInfo information for splitting
 * @param copyData whether or not copy data from ztensor to each chunk
 */
void splitZTensor(const SplitInfo *splitInfo, bool copyData);
/**
 * \brief Merge chunks into a ztensor.
 *
 * @param splitInfo information for splitting
 * @param output a ztensor obtained by merging the chunks
 */
void mergeZTensors(const SplitInfo *splitInfo);

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

// Elementwise Operations
zdnn_status zdnn_add_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_sub_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_mul_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_div_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_min_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_max_ext(const zdnn_ztensor *inputA, const zdnn_ztensor *inputB,
    zdnn_ztensor *output);
zdnn_status zdnn_exp_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_log_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_relu_ext(
    const zdnn_ztensor *input, const void *clippingValue, zdnn_ztensor *output);
zdnn_status zdnn_sigmoid_ext(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_softmax_ext(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output);
zdnn_status zdnn_tanh_ext(const zdnn_ztensor *input, zdnn_ztensor *output);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_ZDNNEXTENSION_H
