/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- TestZdnnxCreateView.c - Unit tests for zdnnx_create_view (f036) ---===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Regression tests for f036: zdnnx_create_view() shallow-copy descriptor alias.
//
// The original implementation did "*input_view = *input" and then modified
// input_view->pre_transformed_desc in place.  Because zdnn_ztensor stores
// pre_transformed_desc and transformed_desc as *pointers* (not inline structs),
// the plain struct copy aliased the pointees: subsequent dim writes through
// input_view->pre_transformed_desc silently overwrote the original tensor's
// shape metadata.
//
// The fix redirects input_view's descriptor pointers to caller-supplied storage
// and deep-copies the original descriptor before mutating it.  These tests
// verify that after zdnnx_create_view() returns, the original tensor's
// descriptors are unchanged.
//
// ZHigh→ZLow lowering path note
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// zdnnx_create_view is NOT emitted by the ZLow→LLVM lowering pass.  The
// ZLowToLLVMCommon.cpp ApiSpec table only references zdnnx_add, zdnnx_sub,
// zdnnx_matmul_op etc. — none of which call zdnnx_create_view.
// zdnnx_create_view is called exclusively from the OMP runtime helpers in
// omp_ops.c (zdnnx_omp_unary_elementwise and zdnnx_omp_binary_elementwise).
// No lowering pass is impacted by this change.
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "zdnnx.h"
#include "zdnnx_private.h"

/* zdnnx_create_view requires a fully-initialized zdnn_ztensor with valid
 * descriptors.  We use zdnn_init_pre_transformed_desc /
 * zdnn_generate_transformed_desc from the zdnn library.  Both are available
 * on all platforms via the libzdnn stub that ships with the build. */

/* Build a minimal 4D ztensor with shape (e4, e3, e2, e1) using ZDNN_4D layout.
 * pre_desc and tfrmd_desc are the caller-supplied descriptor storage.
 * The ztensor's buffer is set to NULL (we do not exercise the buffer itself).
 */
static void make_ztensor_4d(zdnn_ztensor *zt, zdnn_tensor_desc *pre,
    zdnn_tensor_desc *tfrmd, uint32_t e4, uint32_t e3, uint32_t e2,
    uint32_t e1) {
  memset(zt, 0, sizeof(*zt));
  memset(pre, 0, sizeof(*pre));
  memset(tfrmd, 0, sizeof(*tfrmd));

  zdnn_init_pre_transformed_desc(ZDNN_4D, FP32, pre, e4, e3, e2, e1);
  zdnn_generate_transformed_desc(pre, tfrmd);

  zt->pre_transformed_desc = pre;
  zt->transformed_desc = tfrmd;
  zt->buffer = NULL;
  zt->buffer_size = zdnn_getsize_ztensor(tfrmd);
}

/* Returns 0 on pass, 1 on fail.
 *
 * Creates a view that collapses E3/E2/E1 into E4 (mimicking omp_ops.c) and
 * verifies that the original tensor's pre_transformed_desc is unchanged. */
static int testOriginalDescUnmodifiedAfterCreateView() {
  zdnn_ztensor input;
  zdnn_tensor_desc input_pre, input_tfrmd;

  /* Shape: (2, 1, 32, 64) -- E2=32 and E1=64 are stride-aligned. */
  make_ztensor_4d(&input, &input_pre, &input_tfrmd, 2, 1, 32, 64);

  /* Snapshot the original pre-transformed dimensions. */
  uint32_t orig_dim4 = input_pre.dim4;
  uint32_t orig_dim3 = input_pre.dim3;
  uint32_t orig_dim2 = input_pre.dim2;
  uint32_t orig_dim1 = input_pre.dim1;

  /* Create the view: collapse into (e4_flat, 1, 32, 64). */
  zdnn_ztensor view;
  zdnn_tensor_desc view_pre, view_tfrmd;
  uint32_t view_shape[4] = {2 * 1 * (32 / 32) * (64 / 64), 1, 32, 64};
  zdnnx_create_view(&input, &view, &view_pre, &view_tfrmd, view_shape, ZDNN_4D);

  /* The original pre_transformed_desc must be unchanged. */
  int fail = 0;
  if (input_pre.dim4 != orig_dim4) {
    fprintf(stderr,
        "FAIL: original dim4 mutated: was %" PRIu32 " now %" PRIu32
        " — f036 regression\n",
        orig_dim4, input_pre.dim4);
    fail = 1;
  }
  if (input_pre.dim3 != orig_dim3) {
    fprintf(stderr,
        "FAIL: original dim3 mutated: was %" PRIu32 " now %" PRIu32
        " — f036 regression\n",
        orig_dim3, input_pre.dim3);
    fail = 1;
  }
  if (input_pre.dim2 != orig_dim2) {
    fprintf(stderr,
        "FAIL: original dim2 mutated: was %" PRIu32 " now %" PRIu32
        " — f036 regression\n",
        orig_dim2, input_pre.dim2);
    fail = 1;
  }
  if (input_pre.dim1 != orig_dim1) {
    fprintf(stderr,
        "FAIL: original dim1 mutated: was %" PRIu32 " now %" PRIu32
        " — f036 regression\n",
        orig_dim1, input_pre.dim1);
    fail = 1;
  }
  if (!fail)
    printf(
        "PASS: original tensor descriptor unchanged after zdnnx_create_view\n");

  /* Verify view has the correct shape written into the caller-supplied storage.
   */
  if (view_pre.dim4 != view_shape[0]) {
    fprintf(stderr, "FAIL: view_pre.dim4 %" PRIu32 " != expected %" PRIu32 "\n",
        view_pre.dim4, view_shape[0]);
    fail = 1;
  }
  if (!fail)
    printf("PASS: view descriptor has correct collapsed dim4\n");

  return fail;
}

/* Returns 0 on pass, 1 on fail.
 *
 * Verifies that the view's descriptor pointers point to the caller-supplied
 * storage, not to the original tensor's descriptors. */
static int testViewDescriptorPointersAreIndependent() {
  zdnn_ztensor input;
  zdnn_tensor_desc input_pre, input_tfrmd;
  make_ztensor_4d(&input, &input_pre, &input_tfrmd, 4, 1, 32, 64);

  zdnn_ztensor view;
  zdnn_tensor_desc view_pre, view_tfrmd;
  uint32_t view_shape[4] = {4, 1, 32, 64}; /* same shape — trivial view */
  zdnnx_create_view(&input, &view, &view_pre, &view_tfrmd, view_shape, ZDNN_4D);

  int fail = 0;
  if (view.pre_transformed_desc != &view_pre) {
    fprintf(stderr,
        "FAIL: view.pre_transformed_desc (%p) should point to caller-supplied "
        "view_pre (%p) — f036 regression\n",
        (void *)view.pre_transformed_desc, (void *)&view_pre);
    fail = 1;
  }
  if (view.transformed_desc != &view_tfrmd) {
    fprintf(stderr,
        "FAIL: view.transformed_desc (%p) should point to caller-supplied "
        "view_tfrmd (%p) — f036 regression\n",
        (void *)view.transformed_desc, (void *)&view_tfrmd);
    fail = 1;
  }
  if (view.pre_transformed_desc == input.pre_transformed_desc) {
    fprintf(stderr,
        "FAIL: view.pre_transformed_desc aliases input.pre_transformed_desc "
        "— f036 regression\n");
    fail = 1;
  }
  if (!fail)
    printf(
        "PASS: view descriptor pointers are independent of original tensor\n");
  return fail;
}

int main(void) {
  int rc = 0;
  rc += testOriginalDescUnmodifiedAfterCreateView();
  rc += testViewDescriptorPointersAreIndependent();
  return rc;
}
