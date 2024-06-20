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

// z/OS specific includes
#ifdef __MVS__
// z/OS needs <time.h> in addition to <sys/time.h>
#include <time.h>
#endif

#include <stdlib.h>
#include <sys/time.h>

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
// zDNN status message is off by default.
#define DEFAULT_STATUS_MESSAGES_ENABLED 0

extern bool OMZTensorSplitEnabled;
extern bool OMZTensorSplitDebug;
extern uint32_t OMZTensorSplitSize;
// We want to enable zdnn status messages when a user
// manually specifies the environment variable.
extern bool OMStatusMessagesEnabled;

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
#if defined(__GNUC__) || defined(__clang__) // GCC, Clang
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#else
  ((void)0);
#endif
}

/**
 * \brief Check zdnn status
 *
 * Check if the zdnn status is not a zdnn_ok and print out the
 * status message along with the error
 *
 * @param status zdnn status
 * @param zdnn_name name of the zdnn api
 */
void checkStatus(zdnn_status status, const char *zdnn_name);

#define CHECK_ZDNN_STATUS(status, zdnn_name) checkStatus(status, zdnn_name)

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
 * \brief Get the NNPA maximum dimension index size.
 *
 * @return the NNPA maximum dimension index size.
 */
uint32_t getMDIS();

/**
 * \brief Initialize a SplitInfo struct.
 *
 * This will initialize values in SplitInfo and allocate buffers.
 *
 * Before calling this function, splitInfo requires fullZTensor, axis, and
 * numOfElemsPerTile to be defined as they are effectively input parameters of
 * this function.
 *
 * Make sure to call freeSplitInfoData to free buffers.
 *
 * @param splitInfo information for splitting
 * @param splitInfo the full ztensor that will be splitted
 * @param axis dimension to split fullZTensor
 * @param numOfElemsPerTile value is used to split the axis equally
 * @param allocTileBuffers whether alloc buffers for the ztensor tiles or not
 * @param tag a string to use when printing debug info
 * @return true if the full ztensor is splitable. Otherwise, false
 */
bool initSplitInfo(SplitInfo *splitInfo, const zdnn_ztensor *fullZTensor,
    SplitAxis axis, uint32_t numOfElemsPerTile, bool allocTileBuffers,
    const char *tag);

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
void freeSplitInfoData(SplitInfo *splitInfo);

/**
 * \brief Print SplitInfo.
 *
 * @param tag a string to use when printing debug info
 */
void printSplitInfo(const SplitInfo *splitInfo, const char *tag);

/**
 * \brief Allocate memory for the tile buffer.
 *
 * @param tile tile->buffer will point to the allocated buffer
 * @return zdnn_status
 */
zdnn_status allocTileBuffer(zdnn_ztensor *tile);

/**
 * \brief Free memory for the tile buffer.
 *
 * @param tile ztensor tile
 */
void freeTileBuffer(zdnn_ztensor *tile);

/**
 * \brief Get a pointer pointing to the tile buffer.
 *
 * @param tile ztensor of the tile
 * @return a pointer to the tile buffer
 */
void *getTileBuffer(zdnn_ztensor *tile);

/**
 * \brief Set the tile buffer pointing to the given buffer.
 *
 * @param tile ztensor of the tile.
 */
void setTileBuffer(zdnn_ztensor *tile, void *buffer);

/**
 * \brief Get a pointer pointing to a tile.
 *
 * @param splitInfo information for splitting
 * @param tileID the id of a tile in the range of [0, numOfTiles - 1]
 * @return a pointer to the tile.
 */
zdnn_ztensor *getTile(const SplitInfo *splitInfo, uint32_t tileID);

/**
 * \brief Get the number of tiles.
 *
 * @param splitInfo information for splitting
 * @return the number of tiles.
 */
uint32_t getNumOfTiles(const SplitInfo *splitInfo);

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

// -----------------------------------------------------------------------------
// Misc Utility Functions
// -----------------------------------------------------------------------------
float GetElapseTime(const struct timeval start_t, const struct timeval end_t);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_ZDNNEXTENSION_H
