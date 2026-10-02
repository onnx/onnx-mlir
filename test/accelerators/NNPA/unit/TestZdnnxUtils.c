/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- TestZdnnxUtils.c - Unit tests for zdnnx_is_full_tile (f032) -------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Regression tests for f032: zdnnx_is_full_tile() EQUAL_SPLIT flag inversion.
//
// The original code tested "(split_info->flags & EQUAL_SPLIT_Ex) == 0" to
// decide whether a last tile is full.  This inverted the semantics: an
// equal-split (full) last tile was reported as non-full, and an undersized
// last tile was reported as full, causing zdnnx_set_tile to hand the wrong
// descriptor to NNPA (OOB write on hardware).
//
// These tests exercise zdnnx_is_full_tile() directly against hand-crafted
// zdnnx_split_info / zdnnx_tile structs without needing NNPA hardware.
//
//===----------------------------------------------------------------------===//

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/* zdnnx.h includes zdnn.h (resolved via NNPA_INCLUDE_PATH) and defines the
 * public structs (zdnnx_split_info, zdnnx_tile, zdnnx_axis, bool etc.).
 * It must be included before zdnnx_private.h which uses those types. */
#include "zdnnx.h"
/* zdnnx_private.h defines EQUAL_SPLIT_Ex flags and zdnnx_is_full_tile(). */
#include "zdnnx_private.h"

/* Forward declaration: zdnnx_is_full_tile is declared in zdnnx_private.h */

/* Build a minimal split_info with the given flags and num_tiles. */
static zdnnx_split_info make_split_info(
    uint64_t flags, uint32_t nE4, uint32_t nE3, uint32_t nE2, uint32_t nE1) {
  zdnnx_split_info si;
  memset(&si, 0, sizeof(si));
  si.flags = flags;
  si.num_tiles[E4] = nE4;
  si.num_tiles[E3] = nE3;
  si.num_tiles[E2] = nE2;
  si.num_tiles[E1] = nE1;
  return si;
}

/* Build a tile pointing at split_info with the given indices. */
static zdnnx_tile make_tile(zdnnx_split_info *si, uint32_t iE4, uint32_t iE3,
    uint32_t iE2, uint32_t iE1) {
  zdnnx_tile t;
  memset(&t, 0, sizeof(t));
  t.split_info = si;
  t.indices[E4] = iE4;
  t.indices[E3] = iE3;
  t.indices[E2] = iE2;
  t.indices[E1] = iE1;
  return t;
}

/* Returns 0 on pass, 1 on fail. */
static int testInteriorTileAlwaysFull() {
  /* Interior tile: none of its indices is the last in its axis. */
  zdnnx_split_info si = make_split_info(0, 4, 4, 4, 4);
  zdnnx_tile t = make_tile(&si, 1, 1, 1, 1); /* not at the last index */
  if (!zdnnx_is_full_tile(&t)) {
    fprintf(
        stderr, "FAIL: interior tile (1,1,1,1) of (4,4,4,4) should be full\n");
    return 1;
  }
  printf("PASS: interior tile correctly reported as full\n");
  return 0;
}

/* Returns 0 on pass, 1 on fail. */
static int testLastTileEqualSplitIsFull() {
  /* Equal split along all axes: EQUAL_SPLIT_Ex flags are ALL SET.
   * The last tile in every axis should be reported as full. */
  uint64_t flags =
      EQUAL_SPLIT_E4 | EQUAL_SPLIT_E3 | EQUAL_SPLIT_E2 | EQUAL_SPLIT_E1;
  zdnnx_split_info si = make_split_info(flags, 3, 3, 3, 3);
  zdnnx_tile t = make_tile(&si, 2, 2, 2, 2); /* last index in every axis */
  if (!zdnnx_is_full_tile(&t)) {
    fprintf(stderr, "FAIL: last tile of equal-split should be full "
                    "(EQUAL_SPLIT flags set) — f032 regression\n");
    return 1;
  }
  printf("PASS: last tile of equal-split correctly reported as full\n");
  return 0;
}

/* Returns 0 on pass, 1 on fail. */
static int testLastTileUnequalSplitIsNotFull() {
  /* Unequal split along E4: EQUAL_SPLIT_E4 is CLEAR (last E4 tile is smaller).
   * The other axes are equal-split so their flags are set. */
  uint64_t flags = EQUAL_SPLIT_E3 | EQUAL_SPLIT_E2 | EQUAL_SPLIT_E1;
  /* EQUAL_SPLIT_E4 deliberately omitted */
  zdnnx_split_info si = make_split_info(flags, 3, 3, 3, 3);
  zdnnx_tile t = make_tile(&si, 2, 2, 2, 2); /* last index in every axis */
  if (zdnnx_is_full_tile(&t)) {
    fprintf(stderr, "FAIL: last tile of unequal E4 split should NOT be full "
                    "(EQUAL_SPLIT_E4 clear) — f032 regression\n");
    return 1;
  }
  printf("PASS: last tile of unequal-split correctly reported as not full\n");
  return 0;
}

/* Returns 0 on pass, 1 on fail. */
static int testLastTileE1UnequalSplitIsNotFull() {
  /* Unequal split along E1 only. */
  uint64_t flags = EQUAL_SPLIT_E4 | EQUAL_SPLIT_E3 | EQUAL_SPLIT_E2;
  zdnnx_split_info si = make_split_info(flags, 2, 2, 2, 2);
  zdnnx_tile t = make_tile(&si, 1, 1, 1, 1); /* last index in every axis */
  if (zdnnx_is_full_tile(&t)) {
    fprintf(stderr, "FAIL: last tile of unequal E1 split should NOT be full "
                    "(EQUAL_SPLIT_E1 clear) — f032 regression\n");
    return 1;
  }
  printf(
      "PASS: last tile of unequal E1 split correctly reported as not full\n");
  return 0;
}

int main(void) {
  int rc = 0;
  rc += testInteriorTileAlwaysFull();
  rc += testLastTileEqualSplitIsFull();
  rc += testLastTileUnequalSplitIsNotFull();
  rc += testLastTileE1UnequalSplitIsNotFull();
  return rc;
}
