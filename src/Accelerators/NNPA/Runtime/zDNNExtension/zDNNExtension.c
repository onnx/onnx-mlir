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
  if (ZTensorSplitDebugFromEnv()) {
    printf("shape: [%d, %d, %d, %d, %d, %d]\n", shape->dim6, shape->dim5,
        shape->dim4, shape->dim3, shape->dim2, shape->dim1);
    printf("sizeFromDim %lu\n", sizeFromDim);
    printf("sizeFromBuffer %lu\n", sizeFromBuffer);
  }
  assert(sizeFromDim == sizeFromBuffer && "buffer size mismatched");
}

zdnn_status allocZTensorInDim2(
    const zdnn_ztensor *input, uint32_t chunkSize, zdnn_ztensor *output) {
  zdnn_status status;
  zdnn_tensor_desc *preTransDesc = malloc(sizeof(zdnn_tensor_desc));
  zdnn_tensor_desc *transDesc = malloc(sizeof(zdnn_tensor_desc));
  // Copy pre_transform_desc from the input.
  preTransDesc->layout = input->pre_transformed_desc->layout;
  preTransDesc->format = input->pre_transformed_desc->format;
  preTransDesc->type = input->pre_transformed_desc->type;
  preTransDesc->dim4 = input->pre_transformed_desc->dim4;
  preTransDesc->dim3 = input->pre_transformed_desc->dim3;
  preTransDesc->dim2 = chunkSize;
  preTransDesc->dim1 = input->pre_transformed_desc->dim1;
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

void copyZTensorInDim2(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t offset, bool fromChunk) {
  zTensorShape inShape, outShape;
  getZTensorShape(input, &inShape);
  getZTensorShape(output, &outShape);
  zTensorShape origShape = fromChunk ? outShape : inShape;
  zTensorShape chunkShape = fromChunk ? inShape : outShape;
  assert(origShape.dim6 == chunkShape.dim6);
  assert(origShape.dim5 == chunkShape.dim5);
  assert(origShape.dim4 == chunkShape.dim4);

  uint64_t SD5 = chunkShape.dim5;
  uint64_t SD4 = chunkShape.dim4;
  uint64_t SD3 = (fromChunk ? chunkShape.dim3 : origShape.dim3);
  uint64_t TD5 = chunkShape.dim5;
  uint64_t TD4 = chunkShape.dim4;
  uint64_t TD3 = (fromChunk ? origShape.dim3 : chunkShape.dim3);

  for (uint64_t d6 = 0; d6 < chunkShape.dim6; ++d6) {
    for (uint64_t d5 = 0; d5 < chunkShape.dim5; ++d5) {
      for (uint64_t d4 = 0; d4 < chunkShape.dim4; ++d4) {
        for (uint64_t d3 = 0; d3 < chunkShape.dim3; ++d3) {
          uint64_t SD3Size = d4 + SD4 * (d5 + SD5 * d6);
          uint64_t TD3Size = d4 + TD4 * (d5 + TD5 * d6);
          // Copy one page at a time.
          uint64_t sd3 = (fromChunk ? d3 : offset);
          uint64_t td3 = (fromChunk ? offset : d3);
          uint64_t offsetSrc = AIU_PAGESIZE_IN_BYTES * (sd3 + SD3 * SD3Size);
          uint64_t offsetDest = AIU_PAGESIZE_IN_BYTES * (td3 + TD3 * TD3Size);
          memcpy(output->buffer + offsetDest, input->buffer + offsetSrc,
              AIU_PAGESIZE_IN_BYTES);
        }
      }
    }
  }
  return;
}

void copyZTensorInDim2Scalar(zdnn_ztensor *output, const zdnn_ztensor *input,
    uint32_t offset, bool fromChunk) {
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
  for (uint32_t d6 = 0; d6 < chunkShape.dim6; ++d6) {
    for (uint32_t d5 = 0; d5 < chunkShape.dim5; ++d5) {
      for (uint32_t d4 = 0; d4 < chunkShape.dim4; ++d4) {
        for (uint32_t d3 = 0; d3 < chunkShape.dim3; ++d3) {
          for (uint32_t d2 = 0; d2 < chunkShape.dim2; ++d2) {
            for (uint32_t d1 = 0; d1 < chunkShape.dim1; ++d1) {
              uint32_t sd6 = d6;
              uint32_t sd5 = d5;
              uint32_t sd4 = d4;
              uint32_t sd3 = (fromChunk ? d3 : (d3 + offset));
              uint32_t sd2 = d2;
              uint32_t sd1 = d1;

              uint32_t SD5 = chunkShape.dim5;
              uint32_t SD4 = chunkShape.dim4;
              uint32_t SD3 = (fromChunk ? chunkShape.dim3 : origShape.dim3);
              uint32_t SD2 = chunkShape.dim2;
              uint32_t SD1 = chunkShape.dim1;

              uint32_t td6 = d6;
              uint32_t td5 = d5;
              uint32_t td4 = d4;
              uint32_t td3 = (fromChunk ? (d3 + offset) : d3);
              uint32_t td2 = d2;
              uint32_t td1 = d1;

              uint32_t TD5 = chunkShape.dim5;
              uint32_t TD4 = chunkShape.dim4;
              uint32_t TD3 = (fromChunk ? origShape.dim3 : chunkShape.dim3);
              uint32_t TD2 = chunkShape.dim2;
              uint32_t TD1 = chunkShape.dim1;

              uint64_t offsetSrc =
                  sd1 +
                  SD1 *
                      (sd2 +
                          SD2 * (sd3 + SD3 * (sd4 + SD4 * (sd5 + SD5 * sd6))));
              uint64_t offsetDest =
                  td1 +
                  TD1 *
                      (td2 +
                          TD2 * (td3 + TD3 * (td4 + TD4 * (td5 + TD5 * td6))));
              memcpy(output->buffer + 2 * offsetDest,
                  input->buffer + 2 * offsetSrc, 2);
            }
          }
        }
      }
    }
  }
  return;
}

#ifdef __cplusplus
}
#endif
