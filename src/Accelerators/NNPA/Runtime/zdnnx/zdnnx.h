/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zdnnx.h -------------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sets of primitive operations for splitting a ztensor into tiles and copying
// data between the ztensor and the tiles.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_ZDNNX_H
#define ZDNNX_ZDNNX_H

#include "zdnn.h"

// -----------------------------------------------------------------------------
// Common structures
// -----------------------------------------------------------------------------

extern bool zdnnx_is_telum_1;
// We want to enable zdnn status messages when a user manually specifies the
// environment variable.
extern bool zdnnx_status_message_enabled;

// (e4, e3, e2, e1) in ztensor.transformed_desc.
typedef enum zdnnx_axis {
  E4 = 0,
  E3 = 1,
  E2 = 2,
  E1 = 3,
} zdnnx_axis;

// Input info about full tensor and split.
typedef struct zdnnx_split_info {
  /* public: */

  /* The full ztensor that will be splitted. */
  const zdnn_ztensor *full_ztensor;
  /* The number of partitions along each dimension. */
  /* Four values for E4, E3, E2, and E1. */
  /* Used as the upperbounds when iterating tiles. */
  uint32_t num_tiles[4];

  /* private: */

  /* The following info for quickly creating a tile. */
  /* Shape of the largest tiles. */
  uint32_t tile_shape[4];
  /* Shape of the last tiles. */
  uint32_t last_tile_shape[4];
  /* Buffer size for largest tiles*/
  uint64_t tile_buffer_size;
  /* Pre-transformed descriptor for largest tiles*/
  zdnn_tensor_desc tile_pre_transformed_desc;
  /* Transformed descriptor for largest tiles*/
  zdnn_tensor_desc tile_transformed_desc;
  /* Flags for special cases such as no splitting, reuse the full
   * buffer, reuse descriptors. See zdnnx_split_flags for more information.
   */
  uint64_t flags;
} zdnnx_split_info;

typedef struct zdnnx_tile {
  /* zdnnx_split_info used to create this tile. */
  zdnnx_split_info *split_info;
  /* Indices of this tile in zdnnx_split_info. */
  uint32_t indices[4];
  /* A zdnn ztensor to hold the data of this tile.
   * - if NO_SPLIT or REUSE_FULL_BUFFER, data->buffer would point to the full
   * buffer. Otherwise,
   *    - a new buffer is allocated, or
   *    - data->buffer points to a buffer given by users and the flag
   *    TILE_USE_EXTERNAL_BUFFER is set.
   * - If all tiles have the same size, data->pre_transformed_desc would
   * point to split_info->tile_pre_transformed_desc, same for
   * data->tranformed_desc. Otherwise, last tiles are smaller and
   * pre_transformed_desc and transformed_desc of this tile would be used.
   */
  zdnn_ztensor data;
  /* Pre-transformed descriptor for this tile when the tile is smaller than the
   * others.
   */
  zdnn_tensor_desc pre_transformed_desc;
  /* Transformed descriptor for this tile when the tile is smaller than the
   * others.
   */
  zdnn_tensor_desc transformed_desc;
  /* Flags for special cases such as using external buffer. */
  uint64_t flags;
} zdnnx_tile;

// -----------------------------------------------------------------------------
// Init/shutdown Functions
// -----------------------------------------------------------------------------
void zdnnx_init();

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/**
 * \brief Get the transformed shape (4D) of ztensor.
 *
 * Shape is 4-dimension in ztensor.tranformed_desc.
 *
 * @param input input ztensor
 * @param shape shape information that must be enough to hold 4 elements.
 */
void zdnnx_get_transformed_shape(const zdnn_ztensor *t, uint32_t *shape);

/**
 * \brief Get the dim along an axis in the transformed shape of ztensor.
 *
 * @param input input ztensor
 * @param axis axis E1, E2, E3, or E4
 * @return the dimension size.
 */
uint32_t zdnnx_get_transformed_dim(const zdnn_ztensor *t, zdnnx_axis axis);

/**
 * \brief Get the dimension size of the given axis in each tile if
 * splitting the input ztensor into num_tiles.
 *
 * @param input input ztensor
 * @param num_tiles the number of tiles along axis
 * @param axis axis E1, E2, E3, or E4
 * @return the dimension size.
 */
uint32_t zdnnx_get_transformed_dim_per_tile(
    const zdnn_ztensor *input, uint32_t num_tiles, zdnnx_axis axis);

/**
 * \brief Create a view of a ztensor
 *
 * Create a view of a ztensor by using a new shape and layout.
 *
 * @param input input ztensor
 * @param input_view view ztensor
 * @param view_shape 4D shape of the view ztensor
 * @param view_layout zdnn layout of the view ztensor
 *
 */
void zdnnx_create_view(const zdnn_ztensor *input, zdnn_ztensor *input_view,
    uint32_t *view_shape, zdnn_data_layouts view_layout);

/**
 * \brief Allocate a 4K-aligned buffer large enough to be used as a ztensor
 * buffer for the largest tiles in zdnnx_split_info.
 *
 * There is no actual link creating between the allocated buffer and
 * zdnnx_split_info, so users must manage the buffer by themselves, e.g. free
 * the buffer when not using it.
 *
 * @param split_info information for splitting
 * @return a pointer to the allocated buffer.
 */
void *zdnnx_alloc_buffer(zdnnx_split_info *split_info);

/**
 * \brief free() what was allocated via zdnnx_alloc_buffer()
 *
 * @param aligned_ptr Pointer returned by zdnnx_alloc_buffer()
 */
void zdnnx_free_buffer(void *aligned_ptr);

// -----------------------------------------------------------------------------
// Functions to work with spliting information
// -----------------------------------------------------------------------------

/**
 * \brief Get the number of tiles along an axis.
 *
 * @param split_info information for splitting
 * @param axis axis E1, E2, E3, or E4
 * @return the number of tiles.
 * @note For clang compilers, this function needs to be static inline if the
 *implementation is in a header file. Otherwise, it will cause errors when
 *linking multiple object files that include this header file. This is a common
 *issue with inline functions defined in header files.
 **/
static inline uint32_t zdnnx_get_num_tiles(
    zdnnx_split_info *split_info, zdnnx_axis axis) {
  return split_info->num_tiles[axis];
}
/**
 * \brief Check if there is only one tile in split_info.
 *
 * If there is only one tile in split_info, it means there is no splitting at
 * all. The only tile is the same as the full tensor.
 *
 * @param split_info information for splitting
 * @return true if there is only one tile. Otherwise, there are at least two
 * tiles.
 */
bool zdnnx_has_one_tile(zdnnx_split_info *split_info);

/**
 * \brief Check if tiles do not reuse the full buffer.
 *
 * If tiles do not reuse the full buffer, they will allocate their own buffers.
 *
 * @param split_info information for splitting
 * @return true if tiles do not reuse the full buffer. Otherwise, tiles reuse
 * the full buffer by using offset to access.
 */
bool zdnnx_has_no_buffer_reuse(zdnnx_split_info *split_info);

/**
 * \brief Initialize a zdnnx_split_info struct.
 *
 * This will initialize zdnnx_split_info that contains information to split a
 * full ztensor into tiles.
 *
 * @param split_info information for splitting
 * @param full_ztensor the full ztensor that will be splitted
 * @param tileSizeE4 the max tile size along dimension E4. 0 means using the
 * same size in the full ztensor.
 * @param tileSizeE3 the max tile size along dimension E3. 0 means using the
 * same size in the full ztensor.
 * @param tileSizeE2 the max tile size along dimension E2. 0 means using the
 * same size in the full ztensor.
 * @param tileSizeE1 the max tile size along dimension E1. 0 means using the
 * same size in the full ztensor.
 * @param debug_msg a string to use when printing debug info. This is optional
 * and can be NULL.
 * @return true if the preparation is done. Otherwise false.
 */
bool zdnnx_prepare_split_info(zdnnx_split_info *split_info,
    const zdnn_ztensor *full_ztensor, uint32_t tileSizeE4, uint32_t tileSizeE3,
    uint32_t tileSizeE2, uint32_t tileSizeE1, const char *debug_msg);

/**
 * \brief Print zdnnx_split_info.
 *
 * @param split_info information for splitting
 * @param debug_msg a string to use when printing debug info. This is optional
 * and can be NULL.
 */
void zdnnx_print_split_info(
    zdnnx_split_info *split_info, const char *debug_msg);

// -----------------------------------------------------------------------------
// Functions to work with tiles
// -----------------------------------------------------------------------------

/**
 * \brief Initialize a tile
 *
 * The tile is identified by its indices, i.e. (ie4, ie3, ie2, ie1).
 *
 * This function prepares all necessary information for a tile so that it is
 * ready to be copied data to or used in zdnn functions.
 *
 * Depending on split_info, this function will allocate a buffer for the tile or
 * reuse of existing buffers.
 *
 * If NO_SPLIT or REUSE_FULL_BUFFER bit is set in split_info->flags, the tile
 * reused the full buffer. No new buffer is allocated. Otherwise, there are two
 * cases depending on whether an external buffer is passed in by users or not:
 * - If the buffer is NULL, a new buffer is allocated and the tile owns this
 * buffer.
 * - If the buffer is not NULL, the tile uses this buffer without allocating any
 * buffer. The tile does not free the external buffer.
 *
 * As a rule of thumb, always call zdnnx_free_tile_buffer(tile) when the
 * external buffer passing to zdnnx_set_tile() is NULL. It's because when the
 * external buffer is NULL, the tile potentially creates its own buffer.
 *
 * @param split_info information for splitting
 * @param tile a pointer to a tile.
 * @param external_buffer a buffer for the tile data.
 * @param ie4 the tile index along dimension E4
 * @param ie3 the tile index along dimension E3
 * @param ie2 the tile index along dimension E2
 * @param ie1 the tile index along dimension E1
 */
void zdnnx_set_tile(zdnnx_split_info *split_info, zdnnx_tile *tile,
    char *external_buffer, uint32_t ie4, uint32_t ie3, uint32_t ie2,
    uint32_t ie1);

/**
 * \brief Free a tile buffer.
 *
 * This will free the buffer in the ztensor of a specific tile.
 *
 * When the tile is reusing the full buffer or the tile uses an external
 * buffer, this function does nothing.
 *
 * @param tile a pointer to a tile.
 */
void zdnnx_free_tile_buffer(zdnnx_tile *tile);

/**
 * \brief Copy data from the full ztensor to a specific tile.
 *
 * Copying is no-op if there is no splitting or the full buffer is reused.
 * This function can be called for different tiles in parallel.
 *
 * @param tile a pointer to a tile.
 */
void zdnnx_copy_data_to_tile(zdnnx_tile *tile);

/**
 * \brief Copy data from a specific tile to the full ztensor.
 *
 * Copying is no-op if there is no splitting or the full buffer is reused.
 * This function can be called for different tiles in parallel.
 *
 * @param tile a pointer to a tile.
 */
void zdnnx_copy_data_to_full(zdnnx_tile *tile);

#endif // ZDNNX_ZDNNX_H
