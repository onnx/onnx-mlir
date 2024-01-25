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

#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t ZTensorSplitSizeFromEnv() {
  uint32_t cs = DEFAULT_ZTENSOR_SPLIT_SIZE;
  const char *s = getenv("OM_ZTENSOR_SPLIT_SIZE");
  if (s)
    cs = atoi(s);
  assert(cs % AIU_STICKS_PER_PAGE == 0);
  if (ZTensorSplitDebugFromEnv())
    printf("OM_ZTENSOR_SPLIT_SIZE: %d\n", cs);
  return cs;
}

bool ZTensorSplitEnabledFromEnv() {
  int enabled = DEFAULT_ZTENSOR_SPLIT_ENABLED;
  const char *s = getenv("OM_ZTENSOR_SPLIT_ENABLED");
  if (s)
    enabled = atoi(s);
  if (ZTensorSplitDebugFromEnv())
    printf("OM_ZTENSOR_SPLIT_ENABLED: %d\n", enabled);
  return (enabled != 0);
}

bool ZTensorSplitDebugFromEnv() {
  int enabled = DEFAULT_ZTENSOR_SPLIT_DEBUG;
  const char *s = getenv("OM_ZTENSOR_SPLIT_DEBUG");
  if (s)
    enabled = atoi(s);
  return (enabled != 0);
}

void getZTensorShape(const zdnn_ztensor *t, zTensorShape *shape) {
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

zdnn_status allocZTensorChunk(const zdnn_ztensor *input, uint32_t axis,
    uint32_t chunkSize, zdnn_ztensor *output) {
  zdnn_status status;
  zdnn_tensor_desc *preTransDesc = malloc(sizeof(zdnn_tensor_desc));
  zdnn_tensor_desc *transDesc = malloc(sizeof(zdnn_tensor_desc));
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
  status = zdnn_generate_transformed_desc(preTransDesc, transDesc);
  assert(status == ZDNN_OK);
  // Init a zTensor with malloc.
  status = zdnn_init_ztensor_with_malloc(preTransDesc, transDesc, output);
  return status;
}

zdnn_status freeZTensorChunk(zdnn_ztensor *t) {
  if (t->pre_transformed_desc)
    free(t->pre_transformed_desc);
  if (t->transformed_desc)
    free(t->transformed_desc);
  zdnn_free_ztensor_buffer(t);
  return ZDNN_OK;
}

void copyZTensorChunk(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t axis, uint32_t offset, bool fromChunk) {
  // Only support the second innermost axis in the CPU tensor at this moment.
  // axis = 2 in the CPU tensor corresponds to dim3 in zTensor.
  if (axis != 2) {
    printf("Only support the second innermost dimension at this moment.");
    return;
  }

  zTensorShape inShape, outShape;
  getZTensorShape(input, &inShape);
  getZTensorShape(output, &outShape);
  zTensorShape origShape = fromChunk ? outShape : inShape;
  zTensorShape chunkShape = fromChunk ? inShape : outShape;
  assert(origShape.dim6 == chunkShape.dim6);
  assert(origShape.dim5 == chunkShape.dim5);
  assert(origShape.dim4 == chunkShape.dim4);
  assert(origShape.dim2 == chunkShape.dim2);
  assert(origShape.dim1 == chunkShape.dim1);
  // Ensure that each element is 2 bytes.
  assert(input->transformed_desc->type == ZDNN_DLFLOAT16);

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
          uint64_t offsetDest = AIU_PAGESIZE_IN_BYTES * TD3Offset;
          memcpy(output->buffer + offsetDest, input->buffer + offsetSrc,
              AIU_PAGESIZE_IN_BYTES);
        }
      }
    }
  }
  return;
}

void copyZTensorChunkScalar(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t axis, uint32_t offset, bool fromChunk) {
  // Only support the second innermost axis in the CPU tensor at this moment.
  // axis = 2 in the CPU tensor corresponds to dim3 in zTensor.
  if (axis != 2) {
    printf("Only support the second innermost dimension at this moment.");
    return;
  }

  zTensorShape inShape, outShape;
  getZTensorShape(input, &inShape);
  getZTensorShape(output, &outShape);
  zTensorShape origShape = fromChunk ? outShape : inShape;
  zTensorShape chunkShape = fromChunk ? inShape : outShape;
  assert(origShape.dim6 == chunkShape.dim6);
  assert(origShape.dim5 == chunkShape.dim5);
  assert(origShape.dim4 == chunkShape.dim4);
  assert(origShape.dim2 == chunkShape.dim2);
  assert(origShape.dim1 == chunkShape.dim1);

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
              uint64_t offsetDest =
                  AIU_2BYTE_CELL_SIZE * (d1 + D1 * (d2 + D2 * TD3Offset));
              memcpy(output->buffer + offsetDest, input->buffer + offsetSrc,
                  AIU_2BYTE_CELL_SIZE);
            }
          }
        }
      }
    }
  }
  return;
}

bool initSplitInfo(const zdnn_ztensor *input, uint32_t axis, uint32_t chunkSize,
    SplitInfo *splitInfo) {
  // Only support the second innermost dimension at this moment.
  if (axis != 2)
    return false;

  splitInfo->axis = axis;
  splitInfo->chunkSize = chunkSize;
  splitInfo->totalSize = input->transformed_desc->dim2;
  splitInfo->chunkSizeInStick = CEIL(chunkSize, AIU_STICKS_PER_PAGE);
  splitInfo->numOfChunks = CEIL(splitInfo->totalSize, chunkSize);

  if (splitInfo->numOfChunks == 1)
    return false;

  splitInfo->chunks = malloc(splitInfo->numOfChunks * sizeof(ChunkInfo));
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    ChunkInfo *chunkInfo = splitInfo->chunks + i;
    if (i == splitInfo->numOfChunks - 1)
      chunkInfo->size = splitInfo->totalSize - i * splitInfo->chunkSize;
    else
      chunkInfo->size = splitInfo->chunkSize;
  }
  return true;
}

void freeSplitInfoBuffer(SplitInfo *splitInfo) {
  // Free chunk info.
  if (splitInfo->chunks)
    free(splitInfo->chunks);
  // Free the sub tensors.
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i)
    freeZTensorChunk(splitInfo->tensors + i);
  if (splitInfo->tensors)
    free(splitInfo->tensors);
}

void splitZTensor(
    const zdnn_ztensor *input, SplitInfo *splitInfo, bool copyData) {
  splitInfo->tensors =
      malloc(splitInfo->numOfChunks * sizeof(struct zdnn_ztensor));
  assert(splitInfo->tensors && "Failed to allocate a buffer");
  uint32_t axis = splitInfo->axis;
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    zdnn_ztensor *chunk = splitInfo->tensors + i;
    ChunkInfo *chunkInfo = splitInfo->chunks + i;
    // Allocate ztensor struct for the chunk.
    allocZTensorChunk(input, /*axis=*/axis, chunkInfo->size, chunk);
    if (copyData) {
      // Copy data from the input to the chunk.
      uint32_t offset = i * splitInfo->chunkSizeInStick;
      copyZTensorChunk(chunk, input, axis, offset, /*fromChunk=*/false);
    }
  }
}

void mergeZTensors(const SplitInfo *splitInfo, zdnn_ztensor *output) {
  for (uint32_t i = 0; i < splitInfo->numOfChunks; ++i) {
    uint32_t offset = i * splitInfo->chunkSizeInStick;
    copyZTensorChunk(output, splitInfo->tensors + i, splitInfo->axis, offset,
        /*fromChunk=*/true);
  }
}

#ifdef __cplusplus
}
#endif
