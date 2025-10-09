/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ zdnnx_private.h -----------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sets of private functions in the zdnn extension library.
//
//===----------------------------------------------------------------------===//

#ifndef ZDNNX_ZDNNX_PRIVATE_H
#define ZDNNX_ZDNNX_PRIVATE_H

// AIU parameters getting from zdnn_private.h.
#define AIU_BYTES_PER_STICK 128
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

// -----------------------------------------------------------------------------
// Misc Macros
// -----------------------------------------------------------------------------

#define CEIL(a, b) (uint64_t)((a + b - 1) / b) // positive numbers only

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

// (d6, d5, d4, d3, d2, d1) for the ztensor buffer's shape.
typedef enum zdnnx_axis_6d {
  D6 = 0,
  D5 = 1,
  D4 = 2,
  D3 = 3,
  D2 = 4,
  D1 = 5,
} zdnnx_axis_6d;

// -----------------------------------------------------------------------------
// Flags for struct split_info
// -----------------------------------------------------------------------------

// clang-format off
#define  NO_SPLIT              (1 << 0)    // There is no splitting at all.
#define  REUSE_FULL_BUFFER     (1 << 1)    // Reuse the full buffer for all tiles. No splitting is also considered as reusing, but not vice versa.
#define  REUSE_FULL_BUFFER_D6  (1 << 2)    // Reuse the full buffer for all tiles and and D6 is the splitting axis.
#define  REUSE_FULL_BUFFER_D5  (1 << 3)    // Reuse the full buffer for all tiles and and D5 is the splitting axis.
#define  REUSE_FULL_BUFFER_D4  (1 << 4)    // Reuse the full buffer for all tiles and and D4 is the splitting axis.
#define  REUSE_FULL_BUFFER_D3  (1 << 5)    // Reuse the full buffer for all tiles and and D3 is the splitting axis.
#define  EQUAL_SPLIT_E4        (1 << 10)   // Equal split along E4, meaning all tiles has the same number of elements along E4.
#define  EQUAL_SPLIT_E3        (1 << 11)   // Equal split along E3, meaning all tiles has the same number of elements along E3.
#define  EQUAL_SPLIT_E2        (1 << 12)   // Equal split along E2, meaning all tiles has the same number of elements along E2.
#define  EQUAL_SPLIT_E1        (1 << 13)   // Equal split along E1, meaning all tiles has the same number of elements along E1.
// clang-format on

// -----------------------------------------------------------------------------
// Flags for struct zdnnx_tile
// -----------------------------------------------------------------------------

// clang-format off
#define TILE_USE_EXTERNAL_BUFFER  (1 << 1)    // This tile uses an external buffer passed in by its users.
// clang-format on

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

bool zdnnx_is_equal_split(zdnnx_split_info *split_info);
bool zdnnx_is_full_tile(zdnnx_tile *tile);
bool zdnnx_is_full_buffer_reuse(zdnnx_split_info *split_info);
void zdnnx_unreachable();

#endif // ZDNNX_ZDNNX_PRIVATE_H
