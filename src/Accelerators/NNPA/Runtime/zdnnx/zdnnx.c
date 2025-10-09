/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zdnnx.c -------------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
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

#include "zdnnx.h"
#include "zdnnx_private.h"

bool zdnnx_is_telum_1 = false;
bool zdnnx_status_message_enabled = 0;

static void get_ztensor_shape(const zdnn_tensor_desc *desc, uint32_t *shape) {
  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  shape[D6] = desc->dim4;
  shape[D5] = CEIL(desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
  shape[D4] = desc->dim3;
  shape[D3] = CEIL(desc->dim2, AIU_STICKS_PER_PAGE);
  shape[D2] = AIU_STICKS_PER_PAGE;
  shape[D1] = AIU_2BYTE_CELLS_PER_STICK;
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

/**
 * Copy data between the full ztenstor and a tile.
 *
 * A tile is determined by its indices, i.e. (t4, t3, t2, t1).
 * If tile_to_full is true then copy data from the tile to the full ztensor.
 * If tile_to_full is false then copy data from the full ztensor to the tile.
 *
 * Copying is no-op if there is no splitting or the full buffer is reused.
 *
 * This function can be called for different tiles in parallel.
 */
static inline void copy_data_for_tile(
    zdnnx_tile *tile, bool block_copy, bool tile_to_full) {
  // No-op if the full buffer is reused.
  if (zdnnx_is_full_buffer_reuse(tile->split_info))
    return;

  zdnnx_split_info *split_info = tile->split_info;
  uint32_t ie4 = tile->indices[E4];
  uint32_t ie3 = tile->indices[E3];
  uint32_t ie2 = tile->indices[E2];
  uint32_t ie1 = tile->indices[E1];

  const zdnn_ztensor *full_tensor = split_info->full_ztensor;
  const zdnn_ztensor *tile_tensor = &(tile->data);
  assert(full_tensor->transformed_desc->type != ZDNN_BINARY_INT8 &&
         "Does not support copying int8 data");

  // Get pointers to the full and tile buffers.
  void *src = full_tensor->buffer;
  void *dst = tile_tensor->buffer;
  if (tile_to_full) {
    src = tile_tensor->buffer;
    dst = full_tensor->buffer;
  }
  assert(src && "Source buffer is NULL");
  assert(dst && "Destination buffer is NULL");

  // Get shapes of the full and tile tensors.
  uint32_t shape_of_full[6];
  get_ztensor_shape(full_tensor->transformed_desc, shape_of_full);
  uint32_t shape_of_tile[6];
  get_ztensor_shape(tile_tensor->transformed_desc, shape_of_tile);

  // Determine shapes of the source and target buffer depending on the copy
  // direction.
  uint64_t src_d5_dim = shape_of_full[D5];
  uint64_t dst_d5_dim = shape_of_tile[D5];
  uint64_t src_d4_dim = shape_of_full[D4];
  uint64_t dst_d4_dim = shape_of_tile[D4];
  uint64_t src_d3_dim = shape_of_full[D3];
  uint64_t dst_d3_dim = shape_of_tile[D3];
  if (tile_to_full) {
    src_d5_dim = shape_of_tile[D5];
    dst_d5_dim = shape_of_full[D5];
    src_d4_dim = shape_of_tile[D4];
    dst_d4_dim = shape_of_full[D4];
    src_d3_dim = shape_of_tile[D3];
    dst_d3_dim = shape_of_full[D3];
  }

  // Compute tile offsets along mapped dimensions d6, d5, d4, and d3 in the
  // full stickified buffer.
  // Use the max tile's shape to compute the offsets.
  uint32_t max_mapped_shape[6];
  get_ztensor_shape(&(split_info->tile_transformed_desc), max_mapped_shape);
  uint32_t d6_offset = ie4 * max_mapped_shape[D6];
  uint32_t d5_offset = ie1 * max_mapped_shape[D5];
  uint32_t d4_offset = ie3 * max_mapped_shape[D4];
  uint32_t d3_offset = ie2 * max_mapped_shape[D3];

  // Loops iterate along the tile dimensions.
  uint64_t tile_d6_dim = shape_of_tile[D6];
  uint64_t tile_d5_dim = shape_of_tile[D5];
  uint64_t tile_d4_dim = shape_of_tile[D4];
  uint64_t tile_d3_dim = shape_of_tile[D3];
  uint64_t tile_d2_dim = shape_of_tile[D2];
  uint64_t tile_d1_dim = shape_of_tile[D1];

  if (block_copy) {
    // Copy data with 4K-block unit.
    if (tile_to_full) {
      // Tile to full.
      for (uint64_t d6 = 0; d6 < tile_d6_dim; ++d6) {
        uint64_t src_d6_offset = d6;
        uint64_t dst_d6_offset = (d6_offset + d6);
        for (uint64_t d5 = 0; d5 < tile_d5_dim; ++d5) {
          uint64_t src_d5_offset = d5 + src_d5_dim * src_d6_offset;
          uint64_t dst_d5_offset = d5 + d5_offset + dst_d5_dim * dst_d6_offset;
          for (uint64_t d4 = 0; d4 < tile_d4_dim; ++d4) {
            uint64_t src_d4_offset = d4 + src_d4_dim * src_d5_offset;
            uint64_t dst_d4_offset =
                d4 + d4_offset + dst_d4_dim * dst_d5_offset;
            for (uint64_t d3 = 0; d3 < tile_d3_dim; ++d3) {
              uint64_t src_d3_offset = d3 + src_d3_dim * src_d4_offset;
              uint64_t dst_d3_offset =
                  d3 + d3_offset + dst_d3_dim * dst_d4_offset;
              // Copy one page at a time.
              uint64_t src_offset = AIU_PAGESIZE_IN_BYTES * src_d3_offset;
              uint64_t dst_offset = AIU_PAGESIZE_IN_BYTES * dst_d3_offset;
              memcpy(dst + dst_offset, src + src_offset, AIU_PAGESIZE_IN_BYTES);
            }
          }
        }
      }
    } else {
      // Full to tile.
      for (uint64_t d6 = 0; d6 < tile_d6_dim; ++d6) {
        uint64_t src_d6_offset = (d6_offset + d6);
        uint64_t dst_d6_offset = d6;
        for (uint64_t d5 = 0; d5 < tile_d5_dim; ++d5) {
          uint64_t src_d5_offset = d5 + d5_offset + src_d5_dim * src_d6_offset;
          uint64_t dst_d5_offset = d5 + dst_d5_dim * dst_d6_offset;
          for (uint64_t d4 = 0; d4 < tile_d4_dim; ++d4) {
            uint64_t src_d4_offset =
                d4 + d4_offset + src_d4_dim * src_d5_offset;
            uint64_t dst_d4_offset = d4 + dst_d4_dim * dst_d5_offset;
            for (uint64_t d3 = 0; d3 < tile_d3_dim; ++d3) {
              uint64_t src_d3_offset =
                  d3 + d3_offset + src_d3_dim * src_d4_offset;
              uint64_t dst_d3_offset = d3 + dst_d3_dim * dst_d4_offset;
              // Copy one page at a time.
              uint64_t src_offset = AIU_PAGESIZE_IN_BYTES * src_d3_offset;
              uint64_t dst_offset = AIU_PAGESIZE_IN_BYTES * dst_d3_offset;
              memcpy(dst + dst_offset, src + src_offset, AIU_PAGESIZE_IN_BYTES);
            }
          }
        }
      }
    }
  } else {
    // Copy data with 2-byte unit.
    // Use for debugging only, e.g. checking the correctness.
    uint16_t *src_i16 = (uint16_t *)src;
    uint16_t *dst_i16 = (uint16_t *)dst;
    if (tile_to_full) {
      // Tile to full.
      for (uint64_t d6 = 0; d6 < tile_d6_dim; ++d6) {
        uint64_t src_d6_offset = d6;
        uint64_t dst_d6_offset = (d6_offset + d6);
        for (uint64_t d5 = 0; d5 < tile_d5_dim; ++d5) {
          uint64_t src_d5_offset = d5 + src_d5_dim * src_d6_offset;
          uint64_t dst_d5_offset = d5 + d5_offset + dst_d5_dim * dst_d6_offset;
          for (uint64_t d4 = 0; d4 < tile_d4_dim; ++d4) {
            uint64_t src_d4_offset = d4 + src_d4_dim * src_d5_offset;
            uint64_t dst_d4_offset =
                d4 + d4_offset + dst_d4_dim * dst_d5_offset;
            for (uint64_t d3 = 0; d3 < tile_d3_dim; ++d3) {
              uint64_t src_d3_offset = d3 + src_d3_dim * src_d4_offset;
              uint64_t dst_d3_offset =
                  d3 + d3_offset + dst_d3_dim * dst_d4_offset;
              for (uint64_t d2 = 0; d2 < tile_d2_dim; ++d2) {
                for (uint64_t d1 = 0; d1 < tile_d1_dim; ++d1) {
                  uint64_t src_offset =
                      d1 + tile_d1_dim * (d2 + tile_d2_dim * src_d3_offset);
                  uint64_t dst_offset =
                      d1 + tile_d1_dim * (d2 + tile_d2_dim * dst_d3_offset);
                  *(dst_i16 + dst_offset) = *(src_i16 + src_offset);
                }
              }
            }
          }
        }
      }
    } else {
      // Full to tile.
      for (uint64_t d6 = 0; d6 < tile_d6_dim; ++d6) {
        uint64_t src_d6_offset = (d6_offset + d6);
        uint64_t dst_d6_offset = d6;
        for (uint64_t d5 = 0; d5 < tile_d5_dim; ++d5) {
          uint64_t src_d5_offset = d5 + d5_offset + src_d5_dim * src_d6_offset;
          uint64_t dst_d5_offset = d5 + dst_d5_dim * dst_d6_offset;
          for (uint64_t d4 = 0; d4 < tile_d4_dim; ++d4) {
            uint64_t src_d4_offset =
                d4 + d4_offset + src_d4_dim * src_d5_offset;
            uint64_t dst_d4_offset = d4 + dst_d4_dim * dst_d5_offset;
            for (uint64_t d3 = 0; d3 < tile_d3_dim; ++d3) {
              uint64_t src_d3_offset =
                  d3 + d3_offset + src_d3_dim * src_d4_offset;
              uint64_t dst_d3_offset = d3 + dst_d3_dim * dst_d4_offset;
              for (uint64_t d2 = 0; d2 < tile_d2_dim; ++d2) {
                for (uint64_t d1 = 0; d1 < tile_d1_dim; ++d1) {
                  uint64_t src_offset =
                      d1 + tile_d1_dim * (d2 + tile_d2_dim * src_d3_offset);
                  uint64_t dst_offset =
                      d1 + tile_d1_dim * (d2 + tile_d2_dim * dst_d3_offset);
                  *(dst_i16 + dst_offset) = *(src_i16 + src_offset);
                }
              }
            }
          }
        }
      }
    }
  }
}

static void prepare_tile_desc(uint64_t *bufferSize,
    zdnn_tensor_desc *pre_transformed_desc, zdnn_tensor_desc *transformed_desc,
    const zdnn_ztensor *full_ztensor, const uint32_t *tile_shape) {
  // Generate a pre-transformed desc.
  // Copy pre_transform_desc from the full_ztensor but adjust the dimension size
  // at the split axis.
  // Because the split axis is the one in transform_desc, we map it back to the
  // one in pre_transform_desc.
  // See zDNN/src/zdnn/zdnn/tensor_desc.c for the mapping between dimensions
  // in pre_transform_desc and transform_desc. Here we use the inverse mapping.
  *pre_transformed_desc = *full_ztensor->pre_transformed_desc;
  pre_transformed_desc->dim4 = 1;
  pre_transformed_desc->dim3 = 1;
  pre_transformed_desc->dim2 = 1;
  pre_transformed_desc->dim1 = 1;
  switch (pre_transformed_desc->layout) {
  case (ZDNN_1D):
    // shape (a) <- dims4-1 (1, 1, 1, a)
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_2D):
    // shape (a, b) -> dims4-1 (1, 1, a, b)
    pre_transformed_desc->dim2 = tile_shape[E2];
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_2DS):
    // shape (a, b) -> dims4-1 (a, 1, 1, b)
    pre_transformed_desc->dim2 = tile_shape[E4];
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_3D):
    // shape (a, b, c) -> dims4-1 (1, a, b, c)
    pre_transformed_desc->dim3 = tile_shape[E3];
    pre_transformed_desc->dim2 = tile_shape[E2];
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_3DS):
    // shape (a, b, c) -> dims4-1 (a, 1, b, c)
    pre_transformed_desc->dim3 = tile_shape[E4];
    pre_transformed_desc->dim2 = tile_shape[E2];
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_4D):
  case (ZDNN_NHWC):
  case (ZDNN_HWCK):
    // shape (a, b, c, d) -> dims4-1 (a, b, c, d)
    // shape (n, h, w, c) -> dims4-1 (n, h, w, c)
    pre_transformed_desc->dim4 = tile_shape[E4];
    pre_transformed_desc->dim3 = tile_shape[E3];
    pre_transformed_desc->dim2 = tile_shape[E2];
    pre_transformed_desc->dim1 = tile_shape[E1];
    break;
  case (ZDNN_NCHW):
    // shape (n, c, h, w) -> dims4-1 (n, h, w, c)
    pre_transformed_desc->dim4 = tile_shape[E4];
    pre_transformed_desc->dim3 = tile_shape[E1];
    pre_transformed_desc->dim2 = tile_shape[E3];
    pre_transformed_desc->dim1 = tile_shape[E2];
    break;
  default:
    zdnnx_unreachable();
  }

  // Generate a transformed desc.
  zdnn_status status;
  if (full_ztensor->transformed_desc->type == ZDNN_BINARY_INT8) {
    zdnn_quantized_transform_types transform_type;
    if (full_ztensor->transformed_desc->format == ZDNN_FORMAT_4DFEATURE)
      transform_type = QUANTIZED_INT8;
    else
      transform_type = QUANTIZED_WEIGHTS_INT8;
    status = zdnn_generate_quantized_transformed_desc(
        pre_transformed_desc, transform_type, transformed_desc);
  } else {
    status =
        zdnn_generate_transformed_desc(pre_transformed_desc, transformed_desc);
  }
  assert((status == ZDNN_OK) && "Failed to generate a transformed desc.");
  (void)status; // Prevent unused warning when assert is disabled.

  // Set buffer size.
  *bufferSize = zdnn_getsize_ztensor(transformed_desc);
}

// -----------------------------------------------------------------------------
// Init/shutdown Functions
// -----------------------------------------------------------------------------
void zdnnx_init() {
  zdnn_init();
  zdnnx_is_telum_1 =
      zdnn_is_nnpa_installed() &&
      (zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) == false);
  // Get message status setting by users.
  const char *s = getenv("OM_STATUS_MESSAGES_ENABLED");
  if (s)
    zdnnx_status_message_enabled = atoi(s);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

void zdnnx_get_transformed_shape(const zdnn_ztensor *t, uint32_t *shape) {
  const zdnn_tensor_desc *desc = t->transformed_desc;
  shape[E4] = desc->dim4;
  shape[E3] = desc->dim3;
  shape[E2] = desc->dim2;
  shape[E1] = desc->dim1;
}

uint32_t zdnnx_get_transformed_dim(const zdnn_ztensor *t, zdnnx_axis axis) {
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

void *zdnnx_alloc_buffer(zdnnx_split_info *split_info) {
  void *buffer = malloc_aligned_4k(split_info->tile_buffer_size);
  return buffer;
}

void zdnnx_free_buffer(void *aligned_ptr) {
  // TODO: call free_aligned_4k in the zdnn library instead.
  if (aligned_ptr) {
    // Get the original malloc'd address from where we put it and free it
    void *original_ptr = ((void **)aligned_ptr)[-1];
    free(original_ptr);
  }
}

uint32_t zdnnx_get_transformed_dim_per_tile(
    const zdnn_ztensor *input, uint32_t num_tiles, zdnnx_axis axis) {
  uint32_t transformed_shape[4];
  zdnnx_get_transformed_shape(input, transformed_shape);
  switch (axis) {
  case (E4):
    return CEIL(transformed_shape[E4], num_tiles);
  case (E3):
    return CEIL(transformed_shape[E3], num_tiles);
  case (E2):
    return CEIL(CEIL(transformed_shape[E2], num_tiles), AIU_STICKS_PER_PAGE) *
           AIU_STICKS_PER_PAGE;
  case (E1):
    return CEIL(CEIL(transformed_shape[E1], num_tiles),
               AIU_2BYTE_CELLS_PER_STICK) *
           AIU_2BYTE_CELLS_PER_STICK;
  }
  zdnnx_unreachable();
  return 0;
}

void zdnnx_create_view(const zdnn_ztensor *input, zdnn_ztensor *input_view,
    uint32_t *view_shape, zdnn_data_layouts view_layout) {
  // Initialize the view with the original info from the input.
  *input_view = *input;

  // Update dim sizes in the pre_transformed_desc.
  input_view->pre_transformed_desc->dim4 = view_shape[E4];
  input_view->pre_transformed_desc->dim3 = view_shape[E3];
  input_view->pre_transformed_desc->dim2 = view_shape[E2];
  input_view->pre_transformed_desc->dim1 = view_shape[E1];
  // Update layout in the pre_transformed_desc.
  input_view->pre_transformed_desc->layout = view_layout;

  // Update the view's transformed desc for the updated pre_transformed_desc.
  zdnn_status status = zdnn_generate_transformed_desc(
      input_view->pre_transformed_desc, input_view->transformed_desc);
  assert((status == ZDNN_OK) && "Failed to generate a transformed desc.");
  (void)status; // Prevent unused warning when assert is disabled.

  // Verify the buffer size.
  uint64_t bufferSize = zdnn_getsize_ztensor(input_view->transformed_desc);
  assert(bufferSize == input->buffer_size && "Invalid ztensor view");
  (void)bufferSize; // Prevent unused warning when assert is disabled.
}

// -----------------------------------------------------------------------------
// Functions to work with spliting information
// -----------------------------------------------------------------------------

bool zdnnx_has_one_tile(zdnnx_split_info *split_info) {
  return (split_info->flags & NO_SPLIT);
}

bool zdnnx_has_no_buffer_reuse(zdnnx_split_info *split_info) {
  return ((split_info->flags & REUSE_FULL_BUFFER) == 0);
}

bool zdnnx_prepare_split_info(zdnnx_split_info *split_info,
    const zdnn_ztensor *full_ztensor, uint32_t tile_size_e4,
    uint32_t tile_size_e3, uint32_t tile_size_e2, uint32_t tile_size_e1,
    const char *debug_msg) {
  if (full_ztensor == NULL) {
    // TODO (tung): revise this.
    printf("Null full_ztensor when calling zdnnx_prepare_split_info");
    return false;
  }

  // Not yet supported cases.
  zdnn_data_layouts layout = full_ztensor->transformed_desc->layout;
  bool notSupported = (layout == ZDNN_FICO) || (layout == ZDNN_BIDIR_ZRH) ||
                      (layout == ZDNN_BIDIR_FICO) || (layout == ZDNN_ZRH) ||
                      (layout == ZDNN_4DS);
  // d2 and d1 in the mapped shape are 32 and 64. Make sure that each
  // element is 2 bytes, so that can do 4K-block copying if tiling.
  //
  // At this moment, ZDNN_BINARY_INT8 is supported only when reusing the full
  // buffer. Will check this at the end of this function.
  if ((full_ztensor->transformed_desc->type != ZDNN_DLFLOAT16) &&
      (full_ztensor->transformed_desc->type != ZDNN_BINARY_INT8))
    notSupported = true;

  // Set full_ztensor.
  split_info->full_ztensor = full_ztensor;

  // Initialize flags. Clear all bits.
  split_info->flags = 0;

  // Set num_tiles, tile_shape and last_tile_shape.
  uint32_t *tile_shape = split_info->tile_shape;
  uint32_t *last_tile_shape = split_info->last_tile_shape;
  uint32_t *num_tiles = split_info->num_tiles;

  if (notSupported) {
    // No split. Prepare to quit now.
    num_tiles[E4] = 1;
    num_tiles[E3] = 1;
    num_tiles[E2] = 1;
    num_tiles[E1] = 1;
    split_info->flags |= NO_SPLIT;
    // No splitting is also considered as reusing the full buffer but not vice
    // versa.
    split_info->flags |= REUSE_FULL_BUFFER;
#ifdef ZDNNX_DEBUG
    zdnnx_print_split_info(split_info, debug_msg);
#endif
    return true;
  }

  // Set tile_shape.
  uint32_t full_shape[4];
  zdnnx_get_transformed_shape(full_ztensor, full_shape);
  tile_shape[E4] = (tile_size_e4 == 0) ? full_shape[E4] : tile_size_e4;
  tile_shape[E3] = (tile_size_e3 == 0) ? full_shape[E3] : tile_size_e3;
  if (tile_size_e2 == 0) {
    tile_shape[E2] = full_shape[E2];
  } else {
    if ((tile_size_e2 != full_shape[E2]) &&
        (tile_size_e2 % AIU_STICKS_PER_PAGE != 0)) {
      tile_shape[E2] =
          CEIL(tile_size_e2, AIU_STICKS_PER_PAGE) * AIU_STICKS_PER_PAGE;
      printf("Warning: TileSize for E2 (%d) is not multiple of %d. Adjust to "
             "%d.\n",
          tile_size_e2, AIU_STICKS_PER_PAGE, tile_shape[E2]);
    } else {
      tile_shape[E2] = tile_size_e2;
    }
  }
  if (tile_size_e1 == 0) {
    tile_shape[E1] = full_shape[E1];
  } else {
    if ((tile_size_e1 != full_shape[E1]) &&
        (tile_size_e1 % AIU_2BYTE_CELLS_PER_STICK != 0)) {
      tile_shape[E1] = CEIL(tile_size_e1, AIU_2BYTE_CELLS_PER_STICK) *
                       AIU_2BYTE_CELLS_PER_STICK;
      printf("Warning: TileSize for E1 (%d) is not multiple of %d. Adjust to "
             "%d.\n",
          tile_size_e1, AIU_2BYTE_CELLS_PER_STICK, tile_shape[E1]);
    } else {
      tile_shape[E1] = tile_size_e1;
    }
  }

  // Set num_tiles.
  num_tiles[E4] = CEIL(full_shape[E4], tile_shape[E4]);
  num_tiles[E3] = CEIL(full_shape[E3], tile_shape[E3]);
  num_tiles[E2] = CEIL(full_shape[E2], tile_shape[E2]);
  num_tiles[E1] = CEIL(full_shape[E1], tile_shape[E1]);
  if (num_tiles[E4] * num_tiles[E3] * num_tiles[E2] * num_tiles[E1] == 1) {
    // Only one tile. No split. Quit now.
    split_info->flags |= NO_SPLIT;
    // No splitting is also considered as reusing the full buffer but not vice
    // versa.
    split_info->flags |= REUSE_FULL_BUFFER;
#ifdef ZDNNX_DEBUG
    zdnnx_print_split_info(split_info, debug_msg);
#endif
    return true;
  }

  // Set last_tile_shape.
  last_tile_shape[E4] = full_shape[E4] - tile_shape[E4] * (num_tiles[E4] - 1);
  last_tile_shape[E3] = full_shape[E3] - tile_shape[E3] * (num_tiles[E3] - 1);
  last_tile_shape[E2] = full_shape[E2] - tile_shape[E2] * (num_tiles[E2] - 1);
  last_tile_shape[E1] = full_shape[E1] - tile_shape[E1] * (num_tiles[E1] - 1);
  if (last_tile_shape[E4] == tile_shape[E4])
    split_info->flags |= EQUAL_SPLIT_E4;
  if (last_tile_shape[E3] == tile_shape[E3])
    split_info->flags |= EQUAL_SPLIT_E3;
  if (last_tile_shape[E2] == tile_shape[E2])
    split_info->flags |= EQUAL_SPLIT_E2;
  if (last_tile_shape[E1] == tile_shape[E1])
    split_info->flags |= EQUAL_SPLIT_E1;

  // Set tileDesc.
  prepare_tile_desc(&(split_info->tile_buffer_size),
      &(split_info->tile_pre_transformed_desc),
      &(split_info->tile_transformed_desc), full_ztensor, tile_shape);

  // Set flag.
  // (e4, e3, e2, e1) -> (d6=e4, d5=e1/64, d4=e3, d3=e2/32, d2=32, d1=64)
  // The full buffer is reused only if, for values d6, d5, d4 and d3 of a tile:
  // - Only one value (say d5) is smaller than full dim size, and
  // - values on the left-hand side of d5 must be 1, and
  // - values on the right-hand side of d5 must be full dim size.
  uint32_t shape_of_full_ztensor[6], shape_of_tile_ztensor[6];
  get_ztensor_shape(full_ztensor->transformed_desc, shape_of_full_ztensor);
  get_ztensor_shape(
      &(split_info->tile_transformed_desc), shape_of_tile_ztensor);
  bool reuse_full_buffer = false;
  if (shape_of_tile_ztensor[D6] == 1) {
    // values on the left-hand side of d5 are 1.
    if (shape_of_tile_ztensor[D5] == 1) {
      // values on the left-hand side of d4 are 1.
      if (shape_of_tile_ztensor[D4] == 1) {
        // values on the left-hand side of d3 are 1.
        reuse_full_buffer = true;
        split_info->flags |= REUSE_FULL_BUFFER_D3;
      } else {
        // values on the right-hand side of d4 must be full dim size.
        if (shape_of_tile_ztensor[D3] == shape_of_full_ztensor[D3]) {
          reuse_full_buffer = true;
          split_info->flags |= REUSE_FULL_BUFFER_D4;
        }
      }
    } else {
      // values on the right-hand side of d5 must be full dim size.
      if ((shape_of_tile_ztensor[D4] == shape_of_full_ztensor[D4]) &&
          (shape_of_tile_ztensor[D3] == shape_of_full_ztensor[D3])) {
        reuse_full_buffer = true;
        split_info->flags |= REUSE_FULL_BUFFER_D5;
      }
    }
  } else {
    // values on the right-hand side of d6 must be full dim size.
    if ((shape_of_tile_ztensor[D5] == shape_of_full_ztensor[D5]) &&
        (shape_of_tile_ztensor[D4] == shape_of_full_ztensor[D4]) &&
        (shape_of_tile_ztensor[D3] == shape_of_full_ztensor[D3])) {
      reuse_full_buffer = true;
      split_info->flags |= REUSE_FULL_BUFFER_D6;
    }
  }

  if (reuse_full_buffer)
    split_info->flags |= REUSE_FULL_BUFFER;

  // No split since copying int8 is not supported yet.
  if (full_ztensor->transformed_desc->type != ZDNN_BINARY_INT8 &&
      !reuse_full_buffer) {
    // Prepare to quit now.
    num_tiles[E4] = 1;
    num_tiles[E3] = 1;
    num_tiles[E2] = 1;
    num_tiles[E1] = 1;
    split_info->flags |= NO_SPLIT;
    // No splitting is also considered as reusing the full buffer but not vice
    // versa.
    split_info->flags |= REUSE_FULL_BUFFER;
#ifdef ZDNNX_DEBUG
    zdnnx_print_split_info(split_info, debug_msg);
#endif
    return true;
  }

#ifdef ZDNNX_DEBUG
  zdnnx_print_split_info(split_info, debug_msg);
#endif

  return true;
}

void zdnnx_print_split_info(
    zdnnx_split_info *split_info, const char *debug_msg) {
  uint32_t transformed_shape_of_full[4];
  zdnnx_get_transformed_shape(
      split_info->full_ztensor, transformed_shape_of_full);

  const char *debugInfo = (debug_msg == NULL) ? "" : debug_msg;
  printf("[%s] Full Tensor shape [e4, e3, e2, e1]:  [%d, %d, %d, %d].\n",
      debugInfo, transformed_shape_of_full[E4], transformed_shape_of_full[E3],
      transformed_shape_of_full[E2], transformed_shape_of_full[E1]);
  if (zdnnx_has_one_tile(split_info)) {
    printf("[%s] No splitting.\n", debugInfo);
  } else {
    printf("[%s] Tile shape [e4, e3, e2, e1]:  [%d, %d, %d, %d].\n", debugInfo,
        split_info->tile_shape[E4], split_info->tile_shape[E3],
        split_info->tile_shape[E2], split_info->tile_shape[E1]);
    printf("[%s] Last tile shape [e4, e3, e2, e1]:  [%d, %d, %d, %d].\n",
        debugInfo, split_info->last_tile_shape[E4],
        split_info->last_tile_shape[E3], split_info->last_tile_shape[E2],
        split_info->last_tile_shape[E1]);
    printf(
        "[%s] The number of tiles along [e4, e3, e2, e1]: [%d, %d, %d, %d].\n",
        debugInfo, split_info->num_tiles[E4], split_info->num_tiles[E3],
        split_info->num_tiles[E2], split_info->num_tiles[E1]);
    printf("[%s] Reuse the full buffer: %s. Equal split: %s\n", debugInfo,
        (zdnnx_is_full_buffer_reuse(split_info)) ? "YES" : "NO",
        zdnnx_is_equal_split(split_info) ? "YES" : "NO");
  }
}

// -----------------------------------------------------------------------------
// Functions to work with tiles
// -----------------------------------------------------------------------------

void zdnnx_set_tile(zdnnx_split_info *split_info, zdnnx_tile *tile,
    char *external_buffer, uint32_t ie4, uint32_t ie3, uint32_t ie2,
    uint32_t ie1) {
  tile->split_info = split_info;
  tile->indices[E4] = ie4;
  tile->indices[E3] = ie3;
  tile->indices[E2] = ie2;
  tile->indices[E1] = ie1;

  // Initialize flags. Clear all bits.
  tile->flags = 0;

  zdnn_ztensor *tile_ztensor = &(tile->data);
  if (zdnnx_has_one_tile(split_info)) {
    // No splitting, so the tile is the full ztensor.
    *tile_ztensor = *split_info->full_ztensor;
    return;
  }

  const zdnn_ztensor *full_ztensor = split_info->full_ztensor;
  uint32_t *num_tiles = split_info->num_tiles;
  uint32_t *tile_shape = split_info->tile_shape;
  uint32_t *last_tile_shape = split_info->last_tile_shape;

  // Set buffer_size, pre_transform_desc, and transformed_desc.
  if (zdnnx_is_full_tile(tile)) {
    // Reuse descriptors in split_info.
    tile_ztensor->buffer_size = split_info->tile_buffer_size;
    tile_ztensor->pre_transformed_desc =
        &(split_info->tile_pre_transformed_desc);
    tile_ztensor->transformed_desc = &(split_info->tile_transformed_desc);
  } else {
    // Construct shape of the last tile.
    uint32_t shape[4];
    shape[E4] =
        (ie4 == num_tiles[E4] - 1) ? last_tile_shape[E4] : tile_shape[E4];
    shape[E3] =
        (ie3 == num_tiles[E3] - 1) ? last_tile_shape[E3] : tile_shape[E3];
    shape[E2] =
        (ie2 == num_tiles[E2] - 1) ? last_tile_shape[E2] : tile_shape[E2];
    shape[E1] =
        (ie1 == num_tiles[E1] - 1) ? last_tile_shape[E1] : tile_shape[E1];
    // Prepare descriptors for the last tile.
    tile_ztensor->pre_transformed_desc = &(tile->pre_transformed_desc);
    tile_ztensor->transformed_desc = &(tile->transformed_desc);
    prepare_tile_desc(&tile_ztensor->buffer_size,
        tile_ztensor->pre_transformed_desc, tile_ztensor->transformed_desc,
        full_ztensor, shape);
  }

  // Set buffer.
  // Check if we can reuse the full buffer first.
  // If we cannot reuse the full buffer, then use the given buffer if any or
  // allocate a new buffer.
  if (zdnnx_is_full_buffer_reuse(split_info)) {
    // Reuse the full buffer, so compute an offset for the tile buffer to
    // point to.
    // (d6 = e4, d5 = e1/64, d4 = e3, d3 = e2/32, d2 = 32, d1 = 64)
    if (split_info->flags & REUSE_FULL_BUFFER_D6) {
      uint64_t reuseBufferOffset = split_info->tile_buffer_size * ie4;
      tile_ztensor->buffer = full_ztensor->buffer + reuseBufferOffset;
    }
    if (split_info->flags & REUSE_FULL_BUFFER_D5) {
      uint64_t d6BuffSize = CEIL(full_ztensor->buffer_size, num_tiles[E4]);
      uint64_t d6Offset = d6BuffSize * ie4;
      uint64_t reuseBufferOffset =
          d6Offset + split_info->tile_buffer_size * ie1;
      tile_ztensor->buffer = full_ztensor->buffer + reuseBufferOffset;
    }
    if (split_info->flags & REUSE_FULL_BUFFER_D4) {
      uint64_t d6BuffSize = CEIL(full_ztensor->buffer_size, num_tiles[E4]);
      uint64_t d6Offset = d6BuffSize * ie4;
      uint64_t d5BuffSize = CEIL(d6BuffSize, num_tiles[E1]);
      uint64_t d5Offset = d6Offset + d5BuffSize * ie1;
      uint64_t reuseBufferOffset =
          d5Offset + split_info->tile_buffer_size * ie3;
      tile_ztensor->buffer = full_ztensor->buffer + reuseBufferOffset;
    }
    if (split_info->flags & REUSE_FULL_BUFFER_D3) {
      uint64_t d6BuffSize = CEIL(full_ztensor->buffer_size, num_tiles[E4]);
      uint64_t d6Offset = d6BuffSize * ie4;
      uint64_t d5BuffSize = CEIL(d6BuffSize, num_tiles[E1]);
      uint64_t d5Offset = d6Offset + d5BuffSize * ie1;
      uint64_t d4BuffSize = CEIL(d5BuffSize, num_tiles[E3]);
      uint64_t d4Offset = d5Offset + d4BuffSize * ie3;
      uint64_t reuseBufferOffset =
          d4Offset + split_info->tile_buffer_size * ie2;
      tile_ztensor->buffer = full_ztensor->buffer + reuseBufferOffset;
    }
  } else {
    if (external_buffer) {
      // Users gave a buffer, use it.
      tile_ztensor->buffer = external_buffer;
      // Tile uses the external buffer.
      tile->flags |= TILE_USE_EXTERNAL_BUFFER;
    } else {
      // Otherwise, allocate a buffer for the tile.
      tile_ztensor->buffer = malloc_aligned_4k(tile_ztensor->buffer_size);
    }
  }

  // The tile is already transformed.
  tile_ztensor->is_transformed = true;

  // Set reserved.
  memset(&tile_ztensor->reserved, 0, sizeof(tile_ztensor->reserved));

  // Set rec_scale and offset.
  tile_ztensor->rec_scale = full_ztensor->rec_scale;
  tile_ztensor->offset = full_ztensor->offset;
}

void zdnnx_free_tile_buffer(zdnnx_tile *tile) {
  if (tile->flags & TILE_USE_EXTERNAL_BUFFER)
    return;

  if (zdnnx_is_full_buffer_reuse(tile->split_info))
    return;

  // Tile has its own buffer. Free the buffer.
  zdnn_ztensor *ztensor = &(tile->data);
  if (ztensor->buffer) {
    zdnn_free_ztensor_buffer(ztensor);
    ztensor->buffer = NULL;
  }
}

void zdnnx_copy_data_to_full(zdnnx_tile *tile) {
  copy_data_for_tile(tile, /*block_copy=*/true, /*tile_to_full=*/true);
}

void zdnnx_copy_data_to_tile(zdnnx_tile *tile) {
  copy_data_for_tile(tile, /*block_copy=*/true, /*tile_to_full=*/false);
}
