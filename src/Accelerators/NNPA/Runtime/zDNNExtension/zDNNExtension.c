/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zDNNExtension.c -----------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Sets of extensions to the zdnn library.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zDNNExtension.h"

#ifdef __cplusplus
extern "C" {
#endif

bool OMZTensorSplitEnabled = DEFAULT_ZTENSOR_SPLIT_ENABLED;
bool OMZTensorSplitDebug = DEFAULT_ZTENSOR_SPLIT_DEBUG;
uint32_t OMZTensorSplitSize = DEFAULT_ZTENSOR_SPLIT_SIZE;
bool OMStatusMessagesEnabled = DEFAULT_STATUS_MESSAGES_ENABLED;

static uint32_t ZTensorSplitSizeFromEnv() {
  uint32_t cs = DEFAULT_ZTENSOR_SPLIT_SIZE;
  const char *s = getenv("OM_ZTENSOR_SPLIT_SIZE");
  if (s) {
    uint32_t userSize = atoi(s);
    if ((userSize % AIU_2BYTE_CELLS_PER_STICK) != 0)
      printf("OM_ZTENSOR_SPLIT_SIZE is not multiple of %d, use the default "
             "value %d\n",
          AIU_2BYTE_CELLS_PER_STICK, cs);
    else
      cs = userSize;
  }
  return cs;
}

static bool ZTensorSplitEnabledFromEnv() {
  int enabled = DEFAULT_ZTENSOR_SPLIT_ENABLED;
  const char *s = getenv("OM_ZTENSOR_SPLIT_ENABLED");
  if (s)
    enabled = atoi(s);
  return enabled;
}

static bool ZTensorSplitDebugFromEnv() {
  int enabled = DEFAULT_ZTENSOR_SPLIT_DEBUG;
  const char *s = getenv("OM_ZTENSOR_SPLIT_DEBUG");
  if (s)
    enabled = atoi(s);
  return enabled;
}

static bool StatusMessagesEnabledEnv() {
  int enabled = DEFAULT_STATUS_MESSAGES_ENABLED;
  const char *s = getenv("OM_STATUS_MESSAGES_ENABLED");
  if (s)
    enabled = atoi(s);
  return enabled;
}

// malloc_aligned_4k is from zdnn.
static void *malloc_aligned_4k(size_t size) {
  // Request one more page + size of a pointer from the OS.
  unsigned short extra_allocation =
      (AIU_PAGESIZE_IN_BYTES - 1) + sizeof(void *);

  // Make sure size is reasonable.
  if (!size || size > SIZE_MAX) {
    return NULL;
  }

  void *ptr = malloc(size + extra_allocation);
  if (!ptr) {
    perror("Error during malloc");
    fprintf(stderr, "errno = %d\n", errno);
    return ptr;
  }

  // Find the 4k boundary after ptr.
  void *aligned_ptr = (void *)(((uintptr_t)ptr + extra_allocation) &
                               ~(AIU_PAGESIZE_IN_BYTES - 1));
  // Put the original malloc'd address right before aligned_ptr.
  ((void **)aligned_ptr)[-1] = ptr;

  return aligned_ptr;
}

void zDNNExtensionInit() {
  OMZTensorSplitEnabled = ZTensorSplitEnabledFromEnv();
  OMZTensorSplitDebug = ZTensorSplitDebugFromEnv();
  OMZTensorSplitSize = ZTensorSplitSizeFromEnv();
  OMStatusMessagesEnabled = StatusMessagesEnabledEnv();
  if (OMZTensorSplitDebug) {
    printf("OM_ZTENSOR_SPLIT_ENABLED: %d\n", OMZTensorSplitEnabled);
    printf("OM_ZTENSOR_SPLIT_SIZE: %d\n", OMZTensorSplitSize);
  }
  if (OMStatusMessagesEnabled) {
    printf("OM_STATUS_MESSAGES_ENABLED: %d\n", OMStatusMessagesEnabled);
  }
}

void checkStatus(zdnn_status status, const char *zdnn_name) {
  if (OMStatusMessagesEnabled && status != ZDNN_OK) {
    fprintf(stdout, "[zdnnx] %s : %s\n", zdnn_name,
        zdnn_get_status_message(status));
  }
}

#define CHECK_ZDNN_STATUS(status, zdnn_name) checkStatus(status, zdnn_name)

void getUnmappedShape(const zdnn_ztensor *t, UnmappedShape *shape) {
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape->e4 = desc->dim4;
  shape->e3 = desc->dim3;
  shape->e2 = desc->dim2;
  shape->e1 = desc->dim1;
}

static uint32_t getUnmappedDim(const zdnn_ztensor *t, SplitAxis axis) {
  if (axis == E4)
    return t->transformed_desc->dim4;
  if (axis == E3)
    return t->transformed_desc->dim3;
  if (axis == E2)
    return t->transformed_desc->dim2;
  if (axis == E1)
    return t->transformed_desc->dim1;
  return 0;
}

static void getMappedShape(const zdnn_ztensor *t, MappedShape *shape) {
  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape->d6 = desc->dim4;
  shape->d5 = CEIL(desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
  shape->d4 = desc->dim3;
  shape->d3 = CEIL(desc->dim2, AIU_STICKS_PER_PAGE);
  shape->d2 = AIU_STICKS_PER_PAGE;
  shape->d1 = AIU_2BYTE_CELLS_PER_STICK;
  // Tensor size is large, use uint64_t.
  uint64_t sizeFromDim = (uint64_t)shape->d6 * (uint64_t)shape->d5 *
                         (uint64_t)shape->d4 * (uint64_t)shape->d3 *
                         (uint64_t)shape->d2 * (uint64_t)shape->d1 *
                         (uint64_t)AIU_2BYTE_CELL_SIZE;
  uint64_t sizeFromBuffer = t->buffer_size;
  if (sizeFromDim != sizeFromBuffer)
    assert(false && "buffer size mismatched");
}

static uint32_t getMappedNumOfElemsPerTile(const SplitInfo *splitInfo) {
  // Mapping: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  switch (splitInfo->axis) {
  case (E4):
    return splitInfo->numOfElemsPerTile;
  case (E3):
    return splitInfo->numOfElemsPerTile;
  case (E2):
    return CEIL(splitInfo->numOfElemsPerTile, AIU_STICKS_PER_PAGE);
  case (E1):
    return CEIL(splitInfo->numOfElemsPerTile, AIU_2BYTE_CELLS_PER_STICK);
  }
  omUnreachable();
  return 0;
}

uint32_t getMDIS() { return zdnn_get_nnpa_max_dim_idx_size(); }

zdnn_ztensor *getTile(const SplitInfo *splitInfo, uint32_t tileID) {
  return splitInfo->tiles + tileID;
}

uint32_t getNumOfTiles(const SplitInfo *splitInfo) {
  return splitInfo->numOfTiles;
}

zdnn_status allocTileBuffer(zdnn_ztensor *tile) {
  if (!(tile->buffer = malloc_aligned_4k(tile->buffer_size)))
    return ZDNN_ALLOCATION_FAILURE;
  return ZDNN_OK;
}

void freeTileBuffer(zdnn_ztensor *tile) {
  if (tile->buffer)
    zdnn_free_ztensor_buffer(tile);
}

void *getTileBuffer(zdnn_ztensor *tile) { return tile->buffer; }

void setTileBuffer(zdnn_ztensor *tile, void *buffer) { tile->buffer = buffer; }

zdnn_status initTile(
    const SplitInfo *splitInfo, uint32_t tileID, bool allocBuffer) {
  const zdnn_ztensor *fullZTensor = splitInfo->fullZTensor;

  SplitAxis axis = splitInfo->axis;
  zdnn_ztensor *tile = splitInfo->tiles + tileID;

  // Adjust the number of elements per tile for the last tile.
  uint32_t numOfElemsPerTile = splitInfo->numOfElemsPerTile;
  if (tileID == splitInfo->numOfTiles - 1) {
    // The last tile.
    numOfElemsPerTile = getUnmappedDim(fullZTensor, axis) -
                        tileID * splitInfo->numOfElemsPerTile;
  }

  // Allocate one buffer for two descriptors.
  zdnn_tensor_desc *descriptors = malloc(2 * sizeof(struct zdnn_tensor_desc));
  if (!descriptors)
    return ZDNN_ALLOCATION_FAILURE;
  zdnn_tensor_desc *preTransDesc = descriptors;
  zdnn_tensor_desc *transDesc = descriptors + 1;

  // Copy pre_transform_desc from the fullZTensor but adjust the dimension size
  // at the split axis.
  // Because the split axis is the one in transform_desc, we map it back to the
  // one in pre_transform_desc.
  // See zDNN/src/zdnn/zdnn/tensor_desc.c for the mapping between dimensions
  // in pre_transform_desc and transform_desc. Here we use the inverse mapping.
  *preTransDesc = *fullZTensor->pre_transformed_desc;
  switch (preTransDesc->layout) {
  case (ZDNN_1D):
    // shape (a) <- dims4-1 (1, 1, 1, a)
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_2D):
    // shape (a, b) -> dims4-1 (1, 1, a, b)
    if (axis == E2)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_2DS):
    // shape (a, b) -> dims4-1 (a, 1, 1, b)
    if (axis == E4)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_3D):
    // shape (a, b, c) -> dims4-1 (1, a, b, c)
    if (axis == E3)
      preTransDesc->dim3 = numOfElemsPerTile;
    if (axis == E2)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_3DS):
    // shape (a, b, c) -> dims4-1 (a, 1, b, c)
    if (axis == E4)
      preTransDesc->dim3 = numOfElemsPerTile;
    if (axis == E2)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_4D):
  case (ZDNN_NHWC):
  case (ZDNN_HWCK):
    // shape (a, b, c, d) -> dims4-1 (a, b, c, d)
    // shape (n, h, w, c) -> dims4-1 (n, h, w, c)
    if (axis == E4)
      preTransDesc->dim4 = numOfElemsPerTile;
    if (axis == E3)
      preTransDesc->dim3 = numOfElemsPerTile;
    if (axis == E2)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim1 = numOfElemsPerTile;
    break;
  case (ZDNN_NCHW):
    // shape (n, c, h, w) -> dims4-1 (n, h, w, c)
    if (axis == E4)
      preTransDesc->dim4 = numOfElemsPerTile;
    if (axis == E3)
      preTransDesc->dim2 = numOfElemsPerTile;
    if (axis == E2)
      preTransDesc->dim1 = numOfElemsPerTile;
    if (axis == E1)
      preTransDesc->dim3 = numOfElemsPerTile;
    break;
  default:
    omUnreachable();
  }

  // Generate a transformed desc.
  zdnn_status status = zdnn_generate_transformed_desc(preTransDesc, transDesc);
  if (status != ZDNN_OK)
    return status;

  // Initialize the tile.
  zdnn_init_ztensor(preTransDesc, transDesc, tile);

  // The tile is already transformed.
  tile->is_transformed = true;

  // Set a buffer size for the tile.
  tile->buffer_size = zdnn_getsize_ztensor(transDesc);
  if (splitInfo->reuseFullBuffer) {
    // No need to alloc buffers if reuseFullZTensor.
    // Set a buffer for the tile.
    // All tiles except the last one have the same buffer size.
    // The offset for the last tile is simple "totalSize - lastSize".
    uint64_t reuseBufferOffset =
        (tileID == splitInfo->numOfTiles - 1)
            ? (fullZTensor->buffer_size - tile->buffer_size)
            : (tile->buffer_size * tileID);
    tile->buffer = fullZTensor->buffer + reuseBufferOffset;
    // Make sure the tile buffer is within the full buffer.
    assert(
        ((reuseBufferOffset + tile->buffer_size) <= fullZTensor->buffer_size) &&
        "Tile buffer is outside the original buffer");
    return ZDNN_OK;
  }

  if (allocBuffer)
    return allocTileBuffer(tile);

  return ZDNN_OK;
}

void freeTileData(const SplitInfo *splitInfo, uint32_t tileID) {
  zdnn_ztensor *tile = splitInfo->tiles + tileID;
  // Free the tile buffer if it has its own buffer.
  if (!splitInfo->reuseFullBuffer)
    freeTileBuffer(tile);
  // Free the tile descriptors it has its own ztensor.
  if (!splitInfo->reuseFullZTensor) {
    // We allocated one buffer for both two descriptors, so just one free is
    // enought.
    if (tile->pre_transformed_desc)
      free(tile->pre_transformed_desc);
  }
}

/// Copy data between the full ztensor and the i-th tile.
/// Each tile will read/write to different part of the full ztensor buffer.
/// There is no data conflict if calling this function for different tiles at
/// the same time.
void copyDataForTile(
    const SplitInfo *splitInfo, uint32_t tileID, CopyDirection direction) {
  // No data copy if reuseFullBuffer.
  if (splitInfo->reuseFullBuffer)
    return;

  zdnn_ztensor *tile = splitInfo->tiles + tileID;
  const zdnn_ztensor *full = splitInfo->fullZTensor;
  uint32_t mappedNumOfElemsPerTile = getMappedNumOfElemsPerTile(splitInfo);
  uint32_t offset = tileID * mappedNumOfElemsPerTile;
  bool fullToTile = (direction == FULL_TO_TILES);

  // Buffer pointers.
  void *src = (fullToTile) ? full->buffer : tile->buffer;
  void *dst = (fullToTile) ? tile->buffer : full->buffer;
  assert(src && "Source buffer is NULL");
  assert(dst && "Destination buffer is NULL");

  // Shape information.
  MappedShape shapeOfFull;
  getMappedShape(splitInfo->fullZTensor, &shapeOfFull);
  MappedShape shapeOfTile;
  getMappedShape(tile, &shapeOfTile);
  assert(shapeOfFull.d6 == shapeOfTile.d6);
  if (splitInfo->axis != E1)
    assert(shapeOfFull.d5 == shapeOfTile.d5);
  assert(shapeOfFull.d4 == shapeOfTile.d4);
  if (splitInfo->axis != E2)
    assert(shapeOfFull.d3 == shapeOfTile.d3);
  assert(shapeOfFull.d2 == shapeOfTile.d2);
  assert(shapeOfFull.d1 == shapeOfTile.d1);
  // Ensure that each element is 2 bytes.
  assert(splitInfo->fullZTensor->transformed_desc->type == ZDNN_DLFLOAT16);

  uint64_t D6 = shapeOfTile.d6;
  uint64_t D5 = shapeOfTile.d5;
  uint64_t D4 = shapeOfTile.d4;
  uint64_t D3 = shapeOfTile.d3;

  // Splitting e1.
  if (splitInfo->axis == E1) {
    uint64_t SD5 = (fullToTile ? shapeOfFull.d5 : shapeOfTile.d5);
    uint64_t TD5 = (fullToTile ? shapeOfTile.d5 : shapeOfFull.d5);
    for (uint64_t d6 = 0; d6 < D6; ++d6) {
      for (uint64_t d5 = 0; d5 < D5; ++d5) {
        uint64_t sd5 = (fullToTile ? (offset + d5) : d5);
        uint64_t td5 = (fullToTile ? d5 : (offset + d5));
        uint64_t SD5Offset = sd5 + SD5 * d6;
        uint64_t TD5Offset = td5 + TD5 * d6;
        uint64_t copyInBytes = D4 * D3 * AIU_PAGESIZE_IN_BYTES;
        uint64_t offsetSrc = copyInBytes * SD5Offset;
        uint64_t offsetDst = copyInBytes * TD5Offset;
        memcpy(dst + offsetDst, src + offsetSrc, copyInBytes);
      }
    }
    return;
  }

  // Splitting e2.
  if (splitInfo->axis == E2) {
    uint64_t SD3 = (fullToTile ? shapeOfFull.d3 : shapeOfTile.d3);
    uint64_t TD3 = (fullToTile ? shapeOfTile.d3 : shapeOfFull.d3);
    for (uint64_t d6 = 0; d6 < D6; ++d6) {
      for (uint64_t d5 = 0; d5 < D5; ++d5) {
        for (uint64_t d4 = 0; d4 < D4; ++d4) {
          uint64_t SD4Offset = d4 + D4 * (d5 + D5 * d6);
          uint64_t TD4Offset = d4 + D4 * (d5 + D5 * d6);
          if (splitInfo->axis != 2)
            continue;
          for (uint64_t d3 = 0; d3 < D3; ++d3) {
            uint64_t sd3 = (fullToTile ? (offset + d3) : d3);
            uint64_t td3 = (fullToTile ? d3 : (offset + d3));
            uint64_t SD3Offset = sd3 + SD3 * SD4Offset;
            uint64_t TD3Offset = td3 + TD3 * TD4Offset;
            // Copy one page at a time.
            uint64_t offsetSrc = AIU_PAGESIZE_IN_BYTES * SD3Offset;
            uint64_t offsetDst = AIU_PAGESIZE_IN_BYTES * TD3Offset;
            memcpy(dst + offsetDst, src + offsetSrc, AIU_PAGESIZE_IN_BYTES);
          }
        }
      }
    }
    return;
  }

  // Splitting e3.
  if (splitInfo->axis == E3) {
    uint64_t SD4 = (fullToTile ? shapeOfFull.d4 : shapeOfTile.d4);
    uint64_t TD4 = (fullToTile ? shapeOfTile.d4 : shapeOfFull.d4);
    for (uint64_t d6 = 0; d6 < D6; ++d6) {
      for (uint64_t d5 = 0; d5 < D5; ++d5) {
        for (uint64_t d4 = 0; d4 < D4; ++d4) {
          uint64_t sd4 = (fullToTile ? (offset + d4) : d4);
          uint64_t td4 = (fullToTile ? d4 : (offset + d4));
          uint64_t SD4Offset = sd4 + SD4 * (d5 + D5 * d6);
          uint64_t TD4Offset = td4 + TD4 * (d5 + D5 * d6);
          uint64_t copyInBytes = D3 * AIU_PAGESIZE_IN_BYTES;
          uint64_t offsetSrc = copyInBytes * SD4Offset;
          uint64_t offsetDst = copyInBytes * TD4Offset;
          memcpy(dst + offsetDst, src + offsetSrc, copyInBytes);
        }
      }
    }
    return;
  }

  // Splitting e4.
  if (splitInfo->axis == E4) {
    for (uint64_t d6 = 0; d6 < D6; ++d6) {
      uint64_t sd6 = (fullToTile ? (offset + d6) : d6);
      uint64_t td6 = (fullToTile ? d6 : (offset + d6));
      uint64_t copyInBytes = D5 * D4 * D3 * AIU_PAGESIZE_IN_BYTES;
      uint64_t offsetSrc = sd6 * copyInBytes;
      uint64_t offsetDst = td6 * copyInBytes;
      memcpy(dst + offsetDst, src + offsetSrc, copyInBytes);
    }
    return;
  }
}

/// This function does the same data copy as copyDataForTile but copies elements
/// one-by-one. It is just used to check the correctness of copyDataForTile or
/// for debugging purpose.
static void copyDataForTileScalar(
    const SplitInfo *splitInfo, uint32_t tileID, CopyDirection direction) {
  if (splitInfo->axis != E2) {
    printf("Only support E2 dimension at this moment.");
    return;
  }

  // No data copy if reuseFullBuffer.
  if (splitInfo->reuseFullBuffer)
    return;

  zdnn_ztensor *tile = splitInfo->tiles + tileID;
  const zdnn_ztensor *full = splitInfo->fullZTensor;
  uint32_t mappedNumOfElemsPerTile = getMappedNumOfElemsPerTile(splitInfo);
  uint32_t offset = tileID * mappedNumOfElemsPerTile;
  bool fullToTile = (direction == FULL_TO_TILES);

  // Buffers pointers.
  uint16_t *src =
      (fullToTile) ? (uint16_t *)full->buffer : (uint16_t *)tile->buffer;
  uint16_t *dst =
      (fullToTile) ? (uint16_t *)tile->buffer : (uint16_t *)full->buffer;
  assert(src && "Source buffer is NULL");
  assert(dst && "Destination buffer is NULL");

  // Shape information.
  MappedShape shapeOfFull;
  getMappedShape(splitInfo->fullZTensor, &shapeOfFull);
  MappedShape shapeOfTile;
  getMappedShape(tile, &shapeOfTile);
  assert(shapeOfFull.d6 == shapeOfTile.d6);
  assert(shapeOfFull.d5 == shapeOfTile.d5);
  assert(shapeOfFull.d4 == shapeOfTile.d4);
  assert(shapeOfFull.d2 == shapeOfTile.d2);
  assert(shapeOfFull.d1 == shapeOfTile.d1);
  // Ensure that each element is 2 bytes.
  assert(splitInfo->fullZTensor->transformed_desc->type == ZDNN_DLFLOAT16);

  uint64_t D6 = shapeOfTile.d6;
  uint64_t D5 = shapeOfTile.d5;
  uint64_t D4 = shapeOfTile.d4;
  uint64_t D3 = shapeOfTile.d3;
  uint64_t D2 = shapeOfTile.d2;
  uint64_t D1 = shapeOfTile.d1;

  uint64_t SD3 = (fullToTile ? shapeOfFull.d3 : shapeOfTile.d3);
  uint64_t TD3 = (fullToTile ? shapeOfTile.d3 : shapeOfFull.d3);

  for (uint64_t d6 = 0; d6 < D6; ++d6) {
    for (uint64_t d5 = 0; d5 < D5; ++d5) {
      for (uint64_t d4 = 0; d4 < D4; ++d4) {
        for (uint64_t d3 = 0; d3 < D3; ++d3) {
          uint64_t sd3 = (fullToTile ? (d3 + offset) : d3);
          uint64_t td3 = (fullToTile ? d3 : (d3 + offset));
          uint64_t SD3Offset = sd3 + SD3 * (d4 + D4 * (d5 + D5 * d6));
          uint64_t TD3Offset = td3 + TD3 * (d4 + D4 * (d5 + D5 * d6));
          for (uint64_t d2 = 0; d2 < D2; ++d2) {
            for (uint64_t d1 = 0; d1 < D1; ++d1) {
              uint64_t offsetSrc = d1 + D1 * (d2 + D2 * SD3Offset);
              uint64_t offsetDst = d1 + D1 * (d2 + D2 * TD3Offset);
              *(dst + offsetDst) = *(src + offsetSrc);
            }
          }
        }
      }
    }
  }
  return;
}

bool initSplitInfo(SplitInfo *splitInfo, const zdnn_ztensor *fullZTensor,
    SplitAxis axis, uint32_t numOfElemsPerTile, bool allocTileBuffers,
    const char *tag) {
  splitInfo->axis = axis;
  splitInfo->fullZTensor = fullZTensor;
  splitInfo->numOfElemsPerTile = numOfElemsPerTile;

  // Splitting has not yet been supported for the following cases, so redirect
  // to the original zdnn function by setting splitInfo->numOfTiles = 1.
  zdnn_data_layouts layout = fullZTensor->transformed_desc->layout;
  bool isNotSupported = (layout == ZDNN_FICO) || (layout == ZDNN_BIDIR_ZRH) ||
                        (layout == ZDNN_BIDIR_FICO) || (layout == ZDNN_ZRH) ||
                        (layout == ZDNN_4DS);

  // numOfTiles.
  if (!OMZTensorSplitEnabled || isNotSupported)
    splitInfo->numOfTiles = 1;
  else {
    uint32_t totalNumOfElems = getUnmappedDim(fullZTensor, splitInfo->axis);
    splitInfo->numOfTiles = CEIL(totalNumOfElems, numOfElemsPerTile);
  }

  // reuseFullZTensor.
  if (splitInfo->numOfTiles == 1) {
    // No split benefit.
    splitInfo->reuseFullZTensor = true;
    splitInfo->reuseFullBuffer = true;
    splitInfo->tiles = (zdnn_ztensor *)fullZTensor;
    if (OMZTensorSplitDebug)
      printSplitInfo(splitInfo, tag);
    return false;
  }
  splitInfo->reuseFullZTensor = false;

  // reuseFullBuffer.
  // (e4, e3, e2, e1) -> (d6=e4, d5=e1/64, d4=e3, d3=e2/32, d2=32, d1=64)
  splitInfo->reuseFullBuffer = false;
  if (axis == E4) {
    // Always reuse if splitting on e4 (batchsize).
    splitInfo->reuseFullBuffer = true;
  } else {
    // Reuse if the outer loops' bounds are one.
    MappedShape shapeOfFull;
    getMappedShape(splitInfo->fullZTensor, &shapeOfFull);
    if (shapeOfFull.d6 == 1) {
      if (axis == E1) {
        splitInfo->reuseFullBuffer = true;
      } else {
        if (shapeOfFull.d5 == 1) {
          if (axis == E3) {
            splitInfo->reuseFullBuffer = true;
          } else {
            if (shapeOfFull.d4 == 1) {
              if (axis == E2)
                splitInfo->reuseFullBuffer = true;
            }
          }
        }
      }
    }
  }

  // Allocate tile ztensors.
  splitInfo->tiles = malloc(splitInfo->numOfTiles * sizeof(zdnn_ztensor));
  assert(splitInfo->tiles && "Failed to allocate tile ztensors");

  for (uint32_t i = 0; i < splitInfo->numOfTiles; ++i) {
    zdnn_status status = initTile(splitInfo, i, allocTileBuffers);
    if (status != ZDNN_OK)
      assert(false && "Failed to initialize a tile");
  }

  if (OMZTensorSplitDebug)
    printSplitInfo(splitInfo, tag);

  return true;
}

void freeSplitInfoData(SplitInfo *splitInfo) {
  if (splitInfo->reuseFullZTensor)
    return;

  // Free the ztensor buffer and descriptors.
  for (uint32_t i = 0; i < splitInfo->numOfTiles; ++i)
    freeTileData(splitInfo, i);

  // Free tiles.
  if (splitInfo->tiles)
    free(splitInfo->tiles);
}

void copyData(const SplitInfo *splitInfo, CopyDirection direction) {
  for (uint32_t i = 0; i < splitInfo->numOfTiles; ++i) {
    // Copy data between the full ztensor and the i-th tile.
    // Each tile will read/write to a distinct part of the full ztensor buffer.
    copyDataForTile(splitInfo, i, direction);
  }
}

void printSplitInfo(const SplitInfo *splitInfo, const char *tag) {
  UnmappedShape unmappedShapeOfFull;
  getUnmappedShape(splitInfo->fullZTensor, &unmappedShapeOfFull);
  printf("[%s] Full zTensor shape:  e4 = %d, e3 = %d, e2 = %d, e1 = %d.\n",
      tag ? tag : "", unmappedShapeOfFull.e4, unmappedShapeOfFull.e3,
      unmappedShapeOfFull.e2, unmappedShapeOfFull.e1);
  printf("[%s] Split the full ztensor along e%d into %d tiles of %d "
         "elements. ReuseFullZTensor: %d, ReuseFullBuffer: %d \n",
      tag ? tag : "", (4 - splitInfo->axis), splitInfo->numOfTiles,
      splitInfo->numOfElemsPerTile, splitInfo->reuseFullZTensor,
      splitInfo->reuseFullBuffer);
}

float GetElapseTime(const struct timeval start_t, const struct timeval end_t) {
  return (((end_t.tv_sec * 1000000.) + end_t.tv_usec) -
             ((start_t.tv_sec * 1000000) + start_t.tv_usec)) /
         1000;
}

#ifdef __cplusplus
}
#endif
