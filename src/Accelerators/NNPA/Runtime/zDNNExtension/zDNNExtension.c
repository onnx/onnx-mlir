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
#include <string.h>

#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"

#ifdef __cplusplus
extern "C" {
#endif

void getZTensorShape(const zdnn_ztensor *t, zTensorShape *shape) {
  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape->dim6 = desc->dim4;
  shape->dim5 = CEIL(desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
  shape->dim4 = desc->dim3;
  shape->dim3 = CEIL(desc->dim2, AIU_STICKS_PER_PAGE);
  shape->dim2 = AIU_STICKS_PER_PAGE;
  shape->dim1 = AIU_2BYTE_CELLS_PER_STICK;
  if (DEBUG) {
    printf("zTensor shape: [%d, %d, %d, %d, %d, %d]\n", shape->dim6,
        shape->dim5, shape->dim4, shape->dim3, shape->dim2, shape->dim1);
    printf("zTensor buffer size: %lu\n", t->buffer_size);
  }
  assert(shape->dim6 * shape->dim5 * shape->dim4 * shape->dim3 * shape->dim2 *
                 shape->dim1 * AIU_2BYTE_CELL_SIZE ==
             t->buffer_size &&
         "buffer size mismatched");
}

void createZTensorInDim2(const zdnn_ztensor *input, uint32_t pos, bool isLast,
    zdnn_ztensor *output) {
  zdnn_status status;
  zdnn_tensor_desc *preTransDesc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));
  zdnn_tensor_desc *transDesc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));
  // Copy pre_transform_desc from the input.
  preTransDesc->layout = input->pre_transformed_desc->layout;
  preTransDesc->format = input->pre_transformed_desc->format;
  preTransDesc->type = input->pre_transformed_desc->type;
  preTransDesc->dim4 = input->pre_transformed_desc->dim4;
  preTransDesc->dim3 = input->pre_transformed_desc->dim3;
  if (isLast)
    preTransDesc->dim2 = input->pre_transformed_desc->dim2 - pos * CHUNK_SIZE;
  else
    preTransDesc->dim2 = CHUNK_SIZE;
  preTransDesc->dim1 = input->pre_transformed_desc->dim1;
  // Copy a transformed desc.
  status = zdnn_generate_transformed_desc(preTransDesc, transDesc);
  assert(status == ZDNN_OK);
  // Init a zTensor with malloc.
  if (pos == 0) {
    // Directly read from the big input tensor if this is the first chunk.
    zdnn_init_ztensor(preTransDesc, transDesc, output);
    uint64_t bufferSize = zdnn_getsize_ztensor(transDesc);
    output->buffer_size = bufferSize;
    output->buffer = input->buffer;
    status = ZDNN_OK;
  } else {
    status = zdnn_init_ztensor_with_malloc(preTransDesc, transDesc, output);
  }
}

zdnn_status freeZTensorChunk(zdnn_ztensor *t, bool freeBuffer) {
  if (t->pre_transformed_desc)
    free(t->pre_transformed_desc);
  if (t->transformed_desc)
    free(t->transformed_desc);
  if (freeBuffer)
    zdnn_free_ztensor_buffer(t);
  return ZDNN_OK;
}

void copyZTensorInDim2(zdnn_ztensor *output, zdnn_ztensor *input, uint32_t pos,
    bool isLast, bool reversed) {
  zTensorShape inShape, outShape;
  getZTensorShape(input, &inShape);
  getZTensorShape(output, &outShape);
  uint32_t startD3 = CEIL(pos * CHUNK_SIZE, AIU_STICKS_PER_PAGE);
  for (uint32_t d6 = 0; d6 < outShape.dim6; ++d6) {
    for (uint32_t d5 = 0; d5 < outShape.dim5; ++d5) {
      for (uint32_t d4 = 0; d4 < outShape.dim4; ++d4) {
        for (uint32_t d3 = 0; d3 < outShape.dim3; ++d3) {
          // Copy 4K bytes.
          void *source =
              input->buffer +
              AIU_PAGESIZE_IN_BYTES *
                  ((startD3 + d3) +
                      inShape.dim3 *
                          (d4 + outShape.dim4 * (d5 + outShape.dim5 * d6)));
          void *target =
              output->buffer +
              AIU_PAGESIZE_IN_BYTES *
                  (d3 + outShape.dim3 *
                            (d4 + outShape.dim4 * (d5 + outShape.dim5 * d6)));
          if (!reversed)
            memcpy(target, source, AIU_PAGESIZE_IN_BYTES);
          else
            memcpy(source, target, AIU_PAGESIZE_IN_BYTES);
        }
      }
    }
  }
  return;
}

#ifdef __cplusplus
}
#endif
