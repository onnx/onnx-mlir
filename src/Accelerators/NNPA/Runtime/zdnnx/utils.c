/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ utils.c -------------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sets of utils functions in the zdnn extension library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

#include "zdnnx.h"
#include "zdnnx_private.h"

bool zdnnx_is_equal_split(zdnnx_split_info *split_info) {
  bool equalSplit =
      ((split_info->flags & (EQUAL_SPLIT_E4 | EQUAL_SPLIT_E3 | EQUAL_SPLIT_E2 |
                                EQUAL_SPLIT_E1)) ==
          (EQUAL_SPLIT_E4 | EQUAL_SPLIT_E3 | EQUAL_SPLIT_E2 | EQUAL_SPLIT_E1));
  return equalSplit;
}

bool zdnnx_is_full_buffer_reuse(zdnnx_split_info *split_info) {
  return (split_info->flags & REUSE_FULL_BUFFER);
}

bool zdnnx_is_full_tile(zdnnx_tile *tile) {
  zdnnx_split_info *split_info = tile->split_info;
  uint32_t *num_tiles = split_info->num_tiles;
  // Most of the tiles except last ones are full tiles.
  bool full_tile = true;
  // Check last tiles.
  if (tile->indices[E4] == num_tiles[E4] - 1)
    full_tile &= ((split_info->flags & EQUAL_SPLIT_E4) == 0);
  if (tile->indices[E3] == num_tiles[E3] - 1)
    full_tile &= ((split_info->flags & EQUAL_SPLIT_E3) == 0);
  if (tile->indices[E2] == num_tiles[E2] - 1)
    full_tile &= ((split_info->flags & EQUAL_SPLIT_E2) == 0);
  if (tile->indices[E1] == num_tiles[E1] - 1)
    full_tile &= ((split_info->flags & EQUAL_SPLIT_E1) == 0);
  return full_tile;
}

void zdnnx_unreachable() {
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
