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

// Default values used when spliting a big zTensor.
// Must be divisible by AIU_2BYTE_CELLS_PER_STICK.
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

typedef enum CopyDirection {
  FULL_TO_TILES,
  TILES_TO_FULL,
} CopyDirection;

// (e4, e3, e2, e1) in ztensor.transformed_desc.
typedef enum SplitAxis {
  E4 = 0,
  E3 = 1,
  E2 = 2,
  E1 = 3,
} SplitAxis;

// Unmapped shape is 4-dimension (e4, e3, e2, e1) in ztensor.transformed_desc.
typedef struct UnmappedShape {
  uint32_t e4;
  uint32_t e3;
  uint32_t e2;
  uint32_t e1;
} UnmappedShape;

// clang-format off
// Mapped shape is 6-dimension (d6, d5, d4, d3, d2, d1) in the stickified tensor.
// Mapping: (e4, e3, e2, e1) -> (d6 = e4, d5 = e1/64, d4 = e3, d3 = e2/32, d2 = 32, d1 = 64)
// clang-format on
typedef struct MappedShape {
  uint32_t d6;
  uint32_t d5;
  uint32_t d4;
  uint32_t d3;
  uint32_t d2;
  uint32_t d1;
} MappedShape;

typedef struct SplitInfo {
  // The full ztensor that will be splitted.
  const zdnn_ztensor *fullZTensor;
  // Axis to split fullZTensor. It refers to axes (e4, e3, e2, e1) in
  // fullZTensor.transformed_desc.
  SplitAxis axis;
  // Value (the number of elements) is used to split the axis equally.
  uint32_t numOfElemsPerTile;
  // Indicate whether tiles reuse fullZTensor->buffer or not.
  // If "reuseOrigBuffer=true", each tile->buffer points to a different part in
  // fullZTensor->buffer using offsets without overlap and there is no data
  // copy. Each tile stills has its own descriptors.
  bool reuseFullBuffer;
  // Indicate whether tiles reuse fullZTensor or not.
  // "reuseFullZTensor = true" only when there is only one tile. This is used to
  // simplify iteration over tiles.
  bool reuseFullZTensor;
  // The number of tile ztensors.
  uint32_t numOfTiles;
  // Tile ztensors.
  // When "reuseFullZTensor = true", this points to fullZTensor.
  zdnn_ztensor *tiles;
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

inline void omUnreachable() {
// Uses compiler specific extensions if possible.
// Even if no extension is used, undefined behavior is still raised by
// an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
  __assume(false);
#else // GCC, Clang
  __builtin_unreachable();
#endif
}

/**
 * \brief Get the unmapped shape (4D) of ztensor.
 *
 * Unmapped shape is 4-dimension in ztensor.tranformed_desc.
 *
 * @param input input ztensor
 * @param shape shape information
 */
void getUnmappedShape(const zdnn_ztensor *t, UnmappedShape *shape);

/**
 * \brief Initialize a SplitInfo struct.
 *
 * This will initialize values in SplitInfo and allocate buffers.
 *
 * Before calling this function, splitInfo requires fullZTensor, axis, and
 * numOfElemsPerTile to be defined as they are effectively input parameters of
 * this function.
 *
 * Make sure to call FreeSplitInfoData to free buffers.
 *
 * @param splitInfo information for splitting
 * @param initTiles whether initialize ztensors for tiles or not.
 * @param tag a string to use when printing debug info
 * @return true if the ztensor is splitable. Otherwise, false
 */
bool initSplitInfo(SplitInfo *splitInfo, bool initTile, const char *tag);

/**
 * \brief Initialize a SplitInfo struct.
 *
 * This will initialize a ztensor for a specific tile.
 *
 * @param splitInfo information for splitting
 * @param tileID the id of a tile in the range of [0, numOfTiles - 1]
 * @return zdnn_status
 */
zdnn_status initTileWithAlloc(const SplitInfo *splitInfo, uint32_t tileID);

/**
 * \brief Free ztensor tile data.
 *
 * This will free descriptors and buffer in a ztensor for a specific tile.
 *
 * @param splitInfo information for splitting
 * @param tileID the id of a tile in the range of [0, numOfTiles - 1]
 */
void freeTileData(const SplitInfo *splitInfo, uint32_t tileID);

/**
 * \brief Free buffers related to a SplitInfo struct.
 *
 * This does not free the SplitInfo itself.
 *
 * @param splitInfo split information
 */
void FreeSplitInfoData(SplitInfo *splitInfo);

/**
 * \brief Print SplitInfo.
 *
 * @param tag a string to use when printing debug info
 */
void printSplitInfo(const SplitInfo *splitInfo, const char *tag);

/**
 * \brief Copy data between the full ztensor and its tiles.
 *
 * @param splitInfo information for splitting
 * @param direction whether copy data from the full ztensor to tiles or vice
 * versa.
 */
void copyData(const SplitInfo *splitInfo, CopyDirection direction);

/**
 * \brief Copy data between the full ztensor and a specific tile.
 *
 * @param splitInfo information for splitting
 * @param tileID the id of a tile in the range of [0, numOfTiles - 1]
 * @param direction whether copy data from the full ztensor to tiles or vice
 * versa.
 */
void copyDataForTile(
    const SplitInfo *splitInfo, uint32_t tileID, CopyDirection direction);

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
