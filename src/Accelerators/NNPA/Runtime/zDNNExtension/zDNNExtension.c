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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zDNNExtension.h"

#ifdef __cplusplus
extern "C" {
#endif

bool OMZTensorSplitEnabled;
bool OMZTensorSplitDebug;
uint32_t OMZTensorSplitSize;

static uint32_t ZTensorSplitSizeFromEnv() {
  uint32_t cs = DEFAULT_ZTENSOR_SPLIT_SIZE;
  const char *s = getenv("OM_ZTENSOR_SPLIT_SIZE");
  if (s) {
    uint32_t userSize = atoi(s);
    if (userSize % AIU_STICKS_PER_PAGE != 0)
      printf("OM_ZTENSOR_SPLIT_SIZE is not multiple of %d, use the default "
             "value %d\n",
          AIU_STICKS_PER_PAGE, cs);
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

void zDNNExtensionInit() {
  OMZTensorSplitEnabled = ZTensorSplitEnabledFromEnv();
  OMZTensorSplitDebug = ZTensorSplitDebugFromEnv();
  OMZTensorSplitSize = ZTensorSplitSizeFromEnv();
  if (OMZTensorSplitDebug) {
    printf("OM_ZTENSOR_SPLIT_ENABLED: %d\n", OMZTensorSplitEnabled);
    printf("OM_ZTENSOR_SPLIT_SIZE: %d\n", OMZTensorSplitSize);
  }
}

void getOrigShape(const zdnn_ztensor *t, OrigShape *shape) {
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape->e4 = desc->dim4;
  shape->e3 = desc->dim3;
  shape->e2 = desc->dim2;
  shape->e1 = desc->dim1;
}

static void getZTensorShape(const zdnn_ztensor *t, zTensorShape *shape) {
  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape->dim6 = desc->dim4;
  shape->dim5 = CEIL(desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
  shape->dim4 = desc->dim3;
  shape->dim3 = CEIL(desc->dim2, AIU_STICKS_PER_PAGE);
  shape->dim2 = AIU_STICKS_PER_PAGE;
  shape->dim1 = AIU_2BYTE_CELLS_PER_STICK;
  // Tensor size is large, use uint64_t.
  uint64_t sizeFromDim = (uint64_t)shape->dim6 * (uint64_t)shape->dim5 *
                         (uint64_t)shape->dim4 * (uint64_t)shape->dim3 *
                         (uint64_t)shape->dim2 * (uint64_t)shape->dim1 *
                         (uint64_t)AIU_2BYTE_CELL_SIZE;
  uint64_t sizeFromBuffer = t->buffer_size;
  assert(sizeFromDim == sizeFromBuffer && "buffer size mismatched");
}

static zdnn_status allocZTensorChunk(const zdnn_ztensor *input, uint32_t axis,
    uint32_t chunkSize, zdnn_ztensor *output) {
  zdnn_tensor_desc *descriptors = malloc(2 * sizeof(zdnn_tensor_desc));
  if (!descriptors)
    return ZDNN_ALLOCATION_FAILURE;
  zdnn_tensor_desc *preTransDesc = descriptors;
  zdnn_tensor_desc *transDesc = descriptors + 1;
  // Copy pre_transform_desc from the input.
  preTransDesc->layout = input->pre_transformed_desc->layout;
  preTransDesc->format = input->pre_transformed_desc->format;
  preTransDesc->type = input->pre_transformed_desc->type;
  preTransDesc->dim4 =
      (axis == 0) ? chunkSize : input->pre_transformed_desc->dim4;
  preTransDesc->dim3 =
      (axis == 1) ? chunkSize : input->pre_transformed_desc->dim3;
  preTransDesc->dim2 =
      (axis == 2) ? chunkSize : input->pre_transformed_desc->dim2;
  preTransDesc->dim1 =
      (axis == 3) ? chunkSize : input->pre_transformed_desc->dim1;
  // Copy a transformed desc.
  zdnn_status status = zdnn_generate_transformed_desc(preTransDesc, transDesc);
  if (status != ZDNN_OK)
    return status;
  // Init a zTensor with malloc.
  return zdnn_init_ztensor_with_malloc(preTransDesc, transDesc, output);
}

static void freeZTensorChunk(zdnn_ztensor *t) {
  zdnn_free_ztensor_buffer(t);
  // We allocated one buffer for both two descriptors, so just one free is
  // enought.
  if (t->pre_transformed_desc)
    free(t->pre_transformed_desc);
}

static void copyZTensorChunk(
    const SplitInfo *splitInfo, uint32_t chunkID, bool fromChunk) {
  // Only support the second innermost axis in the CPU tensor at this moment.
  // axis = 2 in the CPU tensor corresponds to dim3 in zTensor.
  if (splitInfo->axis != 2) {
    printf("Only support the second innermost dimension at this moment.");
    return;
  }

  ChunkInfo *chunk = splitInfo->chunks + chunkID;
  uint32_t offset = chunk->offsetInStick;

  // Buffers pointers.
  void *src, *dst;
  if (fromChunk) {
    src = chunk->ztensor->buffer;
    dst = splitInfo->origZTensor->buffer;
  } else {
    src = splitInfo->origZTensor->buffer;
    dst = chunk->ztensor->buffer;
  }

  // Shape information.
  zTensorShape origShape;
  getZTensorShape(splitInfo->origZTensor, &origShape);
  zTensorShape chunkShape;
  getZTensorShape(chunk->ztensor, &chunkShape);
  assert(origShape.dim6 == chunkShape.dim6);
  assert(origShape.dim5 == chunkShape.dim5);
  assert(origShape.dim4 == chunkShape.dim4);
  assert(origShape.dim2 == chunkShape.dim2);
  assert(origShape.dim1 == chunkShape.dim1);
  // Ensure that each element is 2 bytes.
  assert(splitInfo->origZTensor->transformed_desc->type == ZDNN_DLFLOAT16);

  uint64_t D6 = chunkShape.dim6;
  uint64_t D5 = chunkShape.dim5;
  uint64_t D4 = chunkShape.dim4;
  uint64_t D3 = chunkShape.dim3;
  uint64_t SD3 = (fromChunk ? chunkShape.dim3 : origShape.dim3);
  uint64_t TD3 = (fromChunk ? origShape.dim3 : chunkShape.dim3);

  for (uint64_t d6 = 0; d6 < D6; ++d6) {
    for (uint64_t d5 = 0; d5 < D5; ++d5) {
      for (uint64_t d4 = 0; d4 < D4; ++d4) {
        uint64_t SD4Offset = d4 + D4 * (d5 + D5 * d6);
        uint64_t TD4Offset = d4 + D4 * (d5 + D5 * d6);
        for (uint64_t d3 = 0; d3 < D3; ++d3) {
          uint64_t sd3 = (fromChunk ? d3 : (offset + d3));
          uint64_t td3 = (fromChunk ? (offset + d3) : d3);
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

static void copyZTensorChunkScalar(
    const SplitInfo *splitInfo, uint32_t chunkID, bool fromChunk) {
  // Only support the second innermost axis in the CPU tensor at this moment.
  // axis = 2 in the CPU tensor corresponds to dim3 in zTensor.
  if (splitInfo->axis != 2) {
    printf("Only support the second innermost dimension at this moment.");
    return;
  }

  ChunkInfo *chunk = splitInfo->chunks + chunkID;
  uint32_t offset = chunk->offsetInStick;

  // Buffers pointers.
  void *src, *dst;
  if (fromChunk) {
    src = chunk->ztensor->buffer;
    dst = splitInfo->origZTensor->buffer;
  } else {
    src = splitInfo->origZTensor->buffer;
    dst = chunk->ztensor->buffer;
  }

  // Shape information.
  zTensorShape origShape;
  getZTensorShape(splitInfo->origZTensor, &origShape);
  zTensorShape chunkShape;
  getZTensorShape(chunk->ztensor, &chunkShape);
  assert(origShape.dim6 == chunkShape.dim6);
  assert(origShape.dim5 == chunkShape.dim5);
  assert(origShape.dim4 == chunkShape.dim4);
  assert(origShape.dim2 == chunkShape.dim2);
  assert(origShape.dim1 == chunkShape.dim1);
  // Ensure that each element is 2 bytes.
  assert(splitInfo->origZTensor->transformed_desc->type == ZDNN_DLFLOAT16);

  uint64_t D6 = chunkShape.dim6;
  uint64_t D5 = chunkShape.dim5;
  uint64_t D4 = chunkShape.dim4;
  uint64_t D3 = chunkShape.dim3;
  uint64_t D2 = chunkShape.dim2;
  uint64_t D1 = chunkShape.dim1;

  uint64_t SD3 = (fromChunk ? chunkShape.dim3 : origShape.dim3);
  uint64_t TD3 = (fromChunk ? origShape.dim3 : chunkShape.dim3);

  for (uint64_t d6 = 0; d6 < D6; ++d6) {
    for (uint64_t d5 = 0; d5 < D5; ++d5) {
      for (uint64_t d4 = 0; d4 < D4; ++d4) {
        for (uint64_t d3 = 0; d3 < D3; ++d3) {
          uint64_t sd3 = (fromChunk ? d3 : (d3 + offset));
          uint64_t td3 = (fromChunk ? (d3 + offset) : d3);
          uint64_t SD3Offset = sd3 + SD3 * (d4 + D4 * (d5 + D5 * d6));
          uint64_t TD3Offset = td3 + TD3 * (d4 + D4 * (d5 + D5 * d6));
          for (uint64_t d2 = 0; d2 < D2; ++d2) {
            for (uint64_t d1 = 0; d1 < D1; ++d1) {
              // Copy 2 bytes at a time.
              uint64_t offsetSrc =
                  AIU_2BYTE_CELL_SIZE * (d1 + D1 * (d2 + D2 * SD3Offset));
              uint64_t offsetDst =
                  AIU_2BYTE_CELL_SIZE * (d1 + D1 * (d2 + D2 * TD3Offset));
              memcpy(dst + offsetDst, src + offsetSrc, AIU_2BYTE_CELL_SIZE);
            }
          }
        }
      }
    }
  }
  return;
}

bool initSplitInfo(SplitInfo *splitInfo) {
  // Only support the second innermost dimension at this moment.
  if (splitInfo->axis != 2)
    return false;

  const zdnn_ztensor *input = splitInfo->origZTensor;
  splitInfo->totalSize = input->transformed_desc->dim2;
  splitInfo->numOfChunks = CEIL(splitInfo->totalSize, splitInfo->chunkSize);

  if (splitInfo->numOfChunks == 1)
    return false;

  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  uint32_t chunkSizeInStick;
  if (splitInfo->axis == 0) // e4
    chunkSizeInStick = splitInfo->chunkSize;
  else if (splitInfo->axis == 1) // e3
    chunkSizeInStick = splitInfo->chunkSize;
  else if (splitInfo->axis == 2) // e2
    chunkSizeInStick = CEIL(splitInfo->chunkSize, AIU_STICKS_PER_PAGE);
  else if (splitInfo->axis == 3) // e1
    chunkSizeInStick = CEIL(splitInfo->chunkSize, AIU_2BYTE_CELLS_PER_STICK);
  else
    return false;

  splitInfo->chunks = malloc(splitInfo->numOfChunks * sizeof(ChunkInfo));
  assert(splitInfo->chunks && "Failed to allocate ChunkInfo struct");
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    ChunkInfo *chunkInfo = splitInfo->chunks + i;
    if (i == splitInfo->numOfChunks - 1)
      chunkInfo->dimSize = splitInfo->totalSize - i * splitInfo->chunkSize;
    else
      chunkInfo->dimSize = splitInfo->chunkSize;
    chunkInfo->offsetInStick = i * chunkSizeInStick;
  }
  return true;
}

void freeSplitInfoBuffer(SplitInfo *splitInfo) {
  // Free the sub tensors.
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    zdnn_ztensor *t = (splitInfo->chunks + i)->ztensor;
    // Free the ztensor buffer and descriptors.
    freeZTensorChunk(t);
    // Free ztensor struct.
    free(t);
  }
  // Free chunk info.
  if (splitInfo->chunks)
    free(splitInfo->chunks);
}

void splitZTensor(const SplitInfo *splitInfo, bool copyData) {
  const zdnn_ztensor *input = splitInfo->origZTensor;
  uint32_t axis = splitInfo->axis;
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    ChunkInfo *chunkInfo = splitInfo->chunks + i;
    chunkInfo->ztensor = malloc(sizeof(zdnn_ztensor));
    assert(chunkInfo->ztensor && "Failed to allocate zTensor struct");
    zdnn_ztensor *chunk = chunkInfo->ztensor;
    // Allocate ztensor struct for the chunk.
    zdnn_status status =
        allocZTensorChunk(input, /*axis=*/axis, chunkInfo->dimSize, chunk);
    assert(status == ZDNN_OK && "Failed to allocate zTensor chunk");
    if (copyData) {
      // Copy data from the input to the chunk.
      copyZTensorChunk(splitInfo, i, /*fromChunk=*/false);
    }
  }
}

void mergeZTensors(const SplitInfo *splitInfo) {
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i)
    copyZTensorChunk(splitInfo, i, /*fromChunk=*/true);
}

#ifdef __cplusplus
}
#endif
