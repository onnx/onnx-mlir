/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- omp_ops.c ---------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Parallel operations that split ztensors into tiles and use OpenMP to run
// tiles on multiple zAIUs.
//
//===----------------------------------------------------------------------===//

#ifdef ZDNNX_WITH_OMP

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "omp_ops.h"
#include "zdnnx.h"
#include "zdnnx_ops.h"

// Keep these values to avoid calling omp functions multiple times.
static uint32_t zdnnx_num_zaiu_threads = 0;
static uint32_t zdnnx_min_num_threads_per_zaiu = 0;

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

uint32_t zdnnx_get_num_zaiu_threads() {
  if (zdnnx_num_zaiu_threads > 0)
    return zdnnx_num_zaiu_threads;

  // If OM_NUM_ZAIU_THREADS is set, use it.
  // Otherwise, use OMP_NUM_THREADS.
  const char *env_num_zaiu_threads = getenv("OM_NUM_ZAIU_THREADS");
  if (env_num_zaiu_threads != NULL) {
    zdnnx_num_zaiu_threads = atoi(env_num_zaiu_threads);
  } else {
    const char *env = getenv("OMP_NUM_THREADS");
    zdnnx_num_zaiu_threads = (env != NULL) ? atoi(env) : 1;
  }

#ifdef ZDNNX_DEBUG
  printf("[zdnnx] num_zaiu_threads: %d\n", zdnnx_num_zaiu_threads);
#endif
  return zdnnx_num_zaiu_threads;
}

static uint32_t zdnnx_get_min_num_threads_per_zaiu() {
  if (zdnnx_min_num_threads_per_zaiu > 0)
    return zdnnx_min_num_threads_per_zaiu;
  // Fix to 1 at this moment.
  zdnnx_min_num_threads_per_zaiu = 1;
#ifdef ZDNNX_DEBUG
  printf(
      "[zdnnx] min_num_threads_per_zaiu: %d\n", zdnnx_min_num_threads_per_zaiu);
#endif
  return zdnnx_min_num_threads_per_zaiu;
}

// -----------------------------------------------------------------------------
// Matrix multiplication
// -----------------------------------------------------------------------------

static inline zdnn_status compute_tile_sizes_for_matmul(
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    const zdnn_ztensor *input_c, uint32_t *ts_e4, uint32_t *ts_e2,
    uint32_t *ts_e1, bool *split_bs_only, bool *split_m, bool *split_n) {
  // For a MatMul of A(BS,M,K)*B(K,N)+C(N),
  // - BS is e4 in (e4, e3, e2, e1), M is e2 and N is e1.
  //
  // An ideal case is that only batch size is tiled by the number of zaius.
  // Otherwise, we should tile other dimensions such as M and/or N.
  // When tiling M is needed, we will tile the batch size by unit 1 to minimize
  // data copy.
  //
  // A is tiled only if M exceeds MDIS because splitting A is expensive. In such
  // a case, we split A into X small tiles each of 8192 elements along M and
  // create X threads to copy the tiles. Once a tile is available we can start
  // the computation immediately.
  //
  // B is tiled based on the number of zAIUs. Splitting B is no-data-copy if E4
  // == 1.
  // - If A does not exceed MDIS, there is no A-based parallelism and the full A
  // is shared among multiple zAIUs.
  // - If A exceeds MDIS, it is tiled. In this case, threads for the outter loop
  // for A tiles are spread over multiple zAIUs and all threads for the inner
  // loop for B tiles share the same zAIU.
  //
  // In all cases, use `OMP_PLACES` to specify zAIUs so that threads are spread
  // over multiple zAIUs.
  uint32_t mdis_e1 = zdnnx_get_nnpa_max_dim_size(E1);
  uint32_t mdis_e2 = zdnnx_get_nnpa_max_dim_size(E2);
  uint32_t num_zaiu_threads = zdnnx_get_num_zaiu_threads();
  uint32_t num_threads_per_zaiu = zdnnx_get_min_num_threads_per_zaiu();
  uint32_t BS = zdnnx_get_transformed_dim(input_a, E4);
  uint32_t M = zdnnx_get_transformed_dim(input_a, E2);
  uint32_t K = zdnnx_get_transformed_dim(input_a, E1);
  uint32_t N = zdnnx_get_transformed_dim(input_b, E1);

  if (K > mdis_e1) {
    printf("[zdnnx] matmul: K (%d) exceeds MDIS (%d)\n", K, mdis_e1);
    return ZDNN_EXCEEDS_MDIS;
  }

  bool enoughBS = (num_zaiu_threads == 1 && BS >= num_threads_per_zaiu) ||
                  (num_zaiu_threads > 1 && BS >= num_zaiu_threads);
  bool exceedM = M > mdis_e2;
  bool exceedN = N > mdis_e1;

  // Set tile_size for BS.
  if (enoughBS && !exceedM && !exceedN) {
    // Only splitting BS is enough.
    uint32_t num_tiles_per_e4 = num_zaiu_threads;
    if (num_zaiu_threads == 1)
      num_tiles_per_e4 = num_threads_per_zaiu;
    *ts_e4 = zdnnx_get_transformed_dim_per_tile(input_a, num_tiles_per_e4, E4);
    *split_bs_only = true;
  } else if (BS != 1) {
    // Set tile_size to 1 to minize data copy.
    *ts_e4 = 1;
  }

  // Set tile_size for M.
  if (exceedM) {
    // Only split M when M exceeds MDIS.
    *ts_e2 = 8192;
    *split_m = true;
  }

  // Set tile_size for N.
  if (exceedN) {
    *ts_e1 = 4096;
    *split_n = true;
  } else if (!*split_bs_only) {
    uint32_t num_tiles_per_e1 = 1;
    if (*split_m) {
      // Tiles over M are parallelized over multiple zaius.
      // Multiple threads per zaiu along N is to minimize data copy for the
      // output.
      num_tiles_per_e1 = num_threads_per_zaiu;
    } else {
      // Two threads along N to minimize warmup overhead of zdnn calls
      // if possible.
      num_tiles_per_e1 =
          num_zaiu_threads * ((num_threads_per_zaiu < 2) ? 1 : 2);
    }
    *ts_e1 = zdnnx_get_transformed_dim_per_tile(input_b, num_tiles_per_e1, E1);
    while (*ts_e1 > mdis_e1) {
      num_tiles_per_e1 *= 2;
      *ts_e1 =
          zdnnx_get_transformed_dim_per_tile(input_b, num_tiles_per_e1, E1);
    }
    *split_n = (num_tiles_per_e1 != 1);
  }
  return ZDNN_OK;
}

static inline zdnn_status call_zdnn_matmul_op(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output, bool is_bcast) {
  if (is_bcast)
    return zdnn_matmul_bcast_op(
        input_a, input_b, input_c, (zdnn_matmul_bcast_ops)op_type, output);
  return zdnn_matmul_op(
      input_a, input_b, input_c, (zdnn_matmul_ops)op_type, output);
}

zdnn_status zdnnx_omp_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output, bool is_bcast) {
  // MatMul types in zdnn:
  // - unstacked: A (2D),  B (2D),  C (1D),  Y (2D)
  // - stacked  : A (3DS), B (3DS), C (2DS), Y (3DS)
  // - bcast    : A (3DS), B (2D),  C (1D),  Y (3DS)
  zdnn_data_layouts a_layout = input_a->pre_transformed_desc->layout;
  zdnn_data_layouts b_layout = input_b->pre_transformed_desc->layout;
  zdnn_data_layouts c_layout = input_c->pre_transformed_desc->layout;
  bool is_stacked =
      (a_layout == ZDNN_3DS && b_layout == ZDNN_3DS && c_layout == ZDNN_2DS);

  // Compute tile sizes.
  uint32_t ts_e4 = 0, ts_e2 = 0, ts_e1 = 0;
  bool split_bs_only = false, split_m = false, split_n = false;
  zdnn_status ts_status = compute_tile_sizes_for_matmul(input_a, input_b,
      input_c, &ts_e4, &ts_e2, &ts_e1, &split_bs_only, &split_m, &split_n);
  if (ts_status != ZDNN_OK)
    return ts_status;

  zdnnx_split_info si_a, si_b, si_c, si_y;
  zdnnx_prepare_split_info(&si_a, input_a, ts_e4, 0, ts_e2, 0, "MatMul A");
  zdnnx_prepare_split_info(&si_b, input_b, ts_e4, 0, 0, ts_e1, "MatMul B");
  zdnnx_prepare_split_info(&si_c, input_c, ts_e4, 0, 0, ts_e1, "MatMul C");
  zdnnx_prepare_split_info(&si_y, output, ts_e4, 0, ts_e2, ts_e1, "MatMul Y");

  // No splitting, call the zdnn matmul without any changes.
  if (zdnnx_has_one_tile(&si_a) && zdnnx_has_one_tile(&si_b)) {
#ifdef ZDNNX_DEBUG
    printf("[MatMul] calling the original zdnn matmul.\n");
#endif
    zdnn_status status = call_zdnn_matmul_op(
        input_a, input_b, input_c, op_type, output, is_bcast);
    return status;
  }

  // Call zdnn_matmul_op on each tile.
  // For each output tile at index (m, n): Y(m, n) = A(m, K) * B(K, n).
  uint32_t BT = zdnnx_get_num_tiles(&si_a, E4);
  uint32_t MT = zdnnx_get_num_tiles(&si_a, E2);
  uint32_t NT = zdnnx_get_num_tiles(&si_b, E1);
  // Iterate over the tiles along the first dim of A.
  // Only enable parallelism at this level if there are enough work.

#pragma omp parallel for num_threads(BT) if (split_bs_only)
  for (uint32_t bs = 0; bs < BT; ++bs) {
#pragma omp parallel for num_threads(MT) if (split_m)
    for (uint32_t m = 0; m < MT; ++m) {
      /* Prepare and set an A tile at index (0, 0, m, 0). */
      zdnnx_tile ta;
      zdnnx_set_tile(&si_a, &ta, NULL, bs, 0, m, 0);
      /* Copy if reuse is off. */
      zdnnx_copy_data_to_tile(&ta);

      // Iterate over the tiles along the second dim of B.
#pragma omp parallel for shared(m, ta, is_bcast) num_threads(NT) if (split_n)
      for (uint32_t n = 0; n < NT; ++n) {
        /* Prepare and set B and C tiles at index (0, 0, 0, n). */
        zdnnx_tile tb, tc;
        zdnnx_set_tile(&si_b, &tb, NULL, (is_stacked) ? bs : 0, 0, 0, n);
        zdnnx_set_tile(&si_c, &tc, NULL, (is_stacked) ? bs : 0, 0, 0, n);
        /* Copy if reuse is off. */
        zdnnx_copy_data_to_tile(&tb);
        zdnnx_copy_data_to_tile(&tc);

        /* Prepare and set an output tile at index (0, 0, m, n). */
        zdnnx_tile ty;
        zdnnx_set_tile(&si_y, &ty, NULL, bs, 0, m, n);

        /* Operation */
        zdnn_status status = call_zdnn_matmul_op(
            &ta.data, &tb.data, &tc.data, op_type, &ty.data, is_bcast);
        assert(status == ZDNN_OK);

        /* Copy the output tile at (0, 0, m, n) to the full output. */
        zdnnx_copy_data_to_full(&ty);

        /* Free the B, C and Y tile buffers. */
        zdnnx_free_tile_buffer(&tb);
        zdnnx_free_tile_buffer(&tc);
        zdnnx_free_tile_buffer(&ty);
      }

      // Free the A tile buffer.
      zdnnx_free_tile_buffer(&ta);
    }
  }

  return ZDNN_OK;
}

zdnn_status zdnnx_omp_quantized_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    const bool disable_clipping, const bool dequantize, const bool pre_computed,
    void *work_area, zdnn_ztensor *output) {
  // MatMul types in zdnn:
  // - unstacked: A (2D),  B (2D),  C (1D),  Y (2D)
  // - stacked  : A (3DS), B (3DS), C (2DS), Y (3DS)
  // - bcast    : A (3DS), B (2D),  C (1D),  Y (3DS)
  zdnn_data_layouts a_layout = input_a->pre_transformed_desc->layout;
  zdnn_data_layouts b_layout = input_b->pre_transformed_desc->layout;
  zdnn_data_layouts c_layout = input_c->pre_transformed_desc->layout;
  bool is_stacked =
      (a_layout == ZDNN_3DS && b_layout == ZDNN_3DS && c_layout == ZDNN_2DS);

  // Compute tile sizes.
  uint32_t ts_e4 = 0, ts_e2 = 0, ts_e1 = 0;
  bool split_bs_only = false, split_m = false, split_n = false;
  zdnn_status ts_status = compute_tile_sizes_for_matmul(input_a, input_b,
      input_c, &ts_e4, &ts_e2, &ts_e1, &split_bs_only, &split_m, &split_n);
  if (ts_status != ZDNN_OK)
    return ts_status;

  zdnnx_split_info si_a, si_b, si_c, si_y;
  zdnnx_prepare_split_info(
      &si_a, input_a, ts_e4, 0, ts_e2, 0, "Quantized MatMul A");
  zdnnx_prepare_split_info(
      &si_b, input_b, ts_e4, 0, 0, ts_e1, "Quantized MatMul B");
  zdnnx_prepare_split_info(
      &si_c, input_c, ts_e4, 0, 0, ts_e1, "Quantized MatMul C");
  zdnnx_prepare_split_info(
      &si_y, output, ts_e4, 0, ts_e2, ts_e1, "Quantized MatMul Y");

  // No splitting, call the zdnn matmul without any changes.
  if (zdnnx_has_one_tile(&si_a) && zdnnx_has_one_tile(&si_b)) {
#ifdef ZDNNX_DEBUG
    printf("[Quantized MatMul] calling the original zdnn quantized matmul.\n");
#endif
    zdnn_status status = zdnn_quantized_matmul_op(input_a, input_b, input_c,
        op_type, clip_min, clip_max, disable_clipping, dequantize, pre_computed,
        work_area, output);
    return status;
  }

  // Call zdnn_matmul_op on each tile.
  // For each output tile at index (m, n): Y(m, n) = A(m, K) * B(K, n).
  uint32_t BT = zdnnx_get_num_tiles(&si_a, E4);
  uint32_t MT = zdnnx_get_num_tiles(&si_a, E2);
  uint32_t NT = zdnnx_get_num_tiles(&si_b, E1);
  // Iterate over the tiles along the first dim of A.
  // Only enable parallelism at this level if there are enough work.

#pragma omp parallel for num_threads(BT) if (split_bs_only)
  for (uint32_t bs = 0; bs < BT; ++bs) {
#pragma omp parallel for num_threads(MT) if (split_m)
    for (uint32_t m = 0; m < MT; ++m) {
      /* Prepare and set an A tile at index (0, 0, m, 0). */
      zdnnx_tile ta;
      zdnnx_set_tile(&si_a, &ta, NULL, bs, 0, m, 0);
      /* Copy if reuse is off. */
      zdnnx_copy_data_to_tile(&ta);

      // Iterate over the tiles along the second dim of B.
#pragma omp parallel for shared(m, ta) num_threads(NT) if (split_n)
      for (uint32_t n = 0; n < NT; ++n) {
        /* Prepare and set B and C tiles at index (0, 0, 0, n). */
        zdnnx_tile tb, tc;
        zdnnx_set_tile(&si_b, &tb, NULL, (is_stacked) ? bs : 0, 0, 0, n);
        zdnnx_set_tile(&si_c, &tc, NULL, (is_stacked) ? bs : 0, 0, 0, n);
        /* Copy if reuse is off. */
        zdnnx_copy_data_to_tile(&tb);
        zdnnx_copy_data_to_tile(&tc);

        /* Prepare and set an output tile at index (0, 0, m, n). */
        zdnnx_tile ty;
        zdnnx_set_tile(&si_y, &ty, NULL, bs, 0, m, n);

        /* Operation */
        // TODO: could we reuse work_area in the parallel scenario?
        zdnn_status status = zdnn_quantized_matmul_op(&ta.data, &tb.data,
            &tc.data, op_type, clip_min, clip_max, disable_clipping, dequantize,
            pre_computed, NULL, &ty.data);
        assert(status == ZDNN_OK);

        /* Copy the output tile at (0, 0, m, n) to the full output. */
        zdnnx_copy_data_to_full(&ty);

        /* Free the B, C and Y tile buffers. */
        zdnnx_free_tile_buffer(&tb);
        zdnnx_free_tile_buffer(&tc);
        zdnnx_free_tile_buffer(&ty);
      }

      // Free the A tile buffer.
      zdnnx_free_tile_buffer(&ta);
    }
  }

  return ZDNN_OK;
}

zdnn_status zdnnx_omp_unary_elementwise(const zdnn_ztensor *input,
    const void *scalar_input, zdnn_ztensor *output, ElemementwiseOp op_type) {
#ifdef ZDNNX_DEBUG
  printf("[OMP UnaryElementwise op_type %d]\n", op_type);
#endif

  uint32_t input_shape[4];
  zdnnx_get_transformed_shape(input, input_shape);
  // Tensor size is at least 3MB (on z17).
  bool isBigTensor = input->buffer_size >= 3 * 1024 * 1024;

  // Reshape the input tensor by collapsing all dimensions into E4, so that we
  // have enough parallel works in any case and reuse is always possible.
  zdnn_ztensor input_view, output_view;
  if (isBigTensor && input_shape[E1] % 64 == 0 && input_shape[E2] % 32 == 0) {
    // Only collapse when E1 is a multiple of 64  and E2 is a multiple of 32 to
    // avoid accessing padding values. Otherwise zdnn will warn range violation.
    // To remove this constraint, the compiler needs to memset a newly allocated
    // ztensor with in-range values.
    zdnn_data_layouts view_layout = ZDNN_4D;
    uint32_t view_shape[4];
    uint32_t e4 = input_shape[E4] * input_shape[E3] * (input_shape[E2] / 32) *
                  (input_shape[E1] / 64);
    view_shape[0] = e4;
    view_shape[1] = 1;
    view_shape[2] = 32;
    view_shape[3] = 64;
    zdnnx_create_view(input, &input_view, view_shape, view_layout);
    zdnnx_create_view(output, &output_view, view_shape, view_layout);
  } else {
    // View is exactly same as the original tensor.
    input_view = *input;
    output_view = *output;
  }

  // Select suitable tile sizes.
  uint32_t ts_e4 = 0, ts_e3 = 0, ts_e2 = 0, ts_e1 = 0;
  uint32_t num_threads_per_zaiu = zdnnx_get_min_num_threads_per_zaiu();
  uint32_t num_tiles = zdnnx_get_num_zaiu_threads() *
                       ((num_threads_per_zaiu > 2) ? 2 : num_threads_per_zaiu);
  if (isBigTensor) {
    ts_e4 = zdnnx_get_transformed_dim_per_tile(&input_view, num_tiles, E4);
    uint32_t mdis_e4 = zdnnx_get_nnpa_max_dim_size(E4);
    while (ts_e4 > mdis_e4) {
      ts_e4 /= 2;
    }
  }

  // Prepare split information.
  zdnnx_split_info si_x, si_y;
  zdnnx_prepare_split_info(
      &si_x, &input_view, ts_e4, ts_e3, ts_e2, ts_e1, "UnaryElementwise X");
  zdnnx_prepare_split_info(
      &si_y, &output_view, ts_e4, ts_e3, ts_e2, ts_e1, "UnaryElementwise Y");

  // No splitting, call the zdnn op without any changes.
  if (zdnnx_has_one_tile(&si_x)) {
    zdnn_status status;
    if (op_type == ZDNNX_EXP_OP)
      status = zdnn_exp(input, output);
    else if (op_type == ZDNNX_GELU_OP)
      status = zdnn_gelu(input, output);
    else if (op_type == ZDNNX_INVSQRT_OP)
      status = zdnn_invsqrt(input, *(const float *)scalar_input, output);
    else if (op_type == ZDNNX_LOG_OP)
      status = zdnn_log(input, output);
    else if (op_type == ZDNNX_RELU_OP)
      status = zdnn_relu(input, scalar_input, output);
    else if (op_type == ZDNNX_SIGMOID_OP)
      status = zdnn_sigmoid(input, output);
    else if (op_type == ZDNNX_SQRT_OP)
      status = zdnn_sqrt(input, output);
    else if (op_type == ZDNNX_TANH_OP)
      status = zdnn_tanh(input, output);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
    return status;
  }

  // Call zdnn op on each tile.
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_x, E4);
  uint32_t num_tiles_e3 = zdnnx_get_num_tiles(&si_x, E3);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_x, E2);
  uint32_t num_tiles_e1 = zdnnx_get_num_tiles(&si_x, E1);
#pragma omp parallel for num_threads(num_tiles) collapse(4)
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e3 = 0; e3 < num_tiles_e3; ++e3) {
      for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
        for (uint32_t e1 = 0; e1 < num_tiles_e1; ++e1) {
          // Prepare tile structs.
          zdnnx_tile tx, ty;
          zdnnx_set_tile(&si_x, &tx, NULL, e4, e3, e2, e1);
          zdnnx_set_tile(&si_y, &ty, NULL, e4, e3, e2, e1);
          zdnnx_copy_data_to_tile(&tx);

          if (op_type == ZDNNX_EXP_OP)
            zdnn_exp(&tx.data, &ty.data);
          else if (op_type == ZDNNX_GELU_OP)
            zdnn_gelu(&tx.data, &ty.data);
          else if (op_type == ZDNNX_INVSQRT_OP)
            zdnn_invsqrt(&tx.data, *(const float *)scalar_input, &ty.data);
          else if (op_type == ZDNNX_LOG_OP)
            zdnn_log(&tx.data, &ty.data);
          else if (op_type == ZDNNX_RELU_OP)
            zdnn_relu(&tx.data, scalar_input, &ty.data);
          else if (op_type == ZDNNX_SIGMOID_OP)
            zdnn_sigmoid(&tx.data, &ty.data);
          else if (op_type == ZDNNX_SQRT_OP)
            zdnn_sqrt(&tx.data, &ty.data);
          else if (op_type == ZDNNX_TANH_OP)
            zdnn_tanh(&tx.data, &ty.data);
          else
            printf("[zdnnx] Not found the zdnn function type %d\n", op_type);

          zdnnx_copy_data_to_full(&ty);

          // Free buffers.
          zdnnx_free_tile_buffer(&tx);
          zdnnx_free_tile_buffer(&ty);
        }
      }
    }
  }

  return ZDNN_OK;
}

zdnn_status zdnnx_omp_binary_elementwise(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, zdnn_ztensor *output,
    ElemementwiseOp op_type) {
#ifdef ZDNNX_DEBUG
  printf("[OMP BinaryElementwise op_type %d]\n", op_type);
#endif

  uint32_t input_shape[4];
  zdnnx_get_transformed_shape(input_a, input_shape);
  // Tensor size is at least 3MB (on z17).
  bool isBigTensor = input_a->buffer_size >= 3 * 1024 * 1024;

  // Reshape the input tensor by collapsing all dimensions into E4, so that we
  // have enough parallel works in any case and reuse is always possible.
  zdnn_ztensor input_a_view, input_b_view, output_view;
  if (isBigTensor && input_shape[E1] % 64 == 0 && input_shape[E2] % 32 == 0) {
    // Only collapse when E1 is a multiple of 64  and E2 is a multiple of 32 to
    // avoid accessing padding values. Otherwise zdnn will warn range violation.
    // To remove this constraint, the compiler needs to memset a newly allocated
    // ztensor with in-range values.
    zdnn_data_layouts view_layout = ZDNN_4D;
    uint32_t view_shape[4];
    uint32_t e4 = input_shape[E4] * input_shape[E3] * (input_shape[E2] / 32) *
                  (input_shape[E1] / 64);
    view_shape[0] = e4;
    view_shape[1] = 1;
    view_shape[2] = 32;
    view_shape[3] = 64;
    zdnnx_create_view(input_a, &input_a_view, view_shape, view_layout);
    zdnnx_create_view(input_b, &input_b_view, view_shape, view_layout);
    zdnnx_create_view(output, &output_view, view_shape, view_layout);
  } else {
    // View is exactly same as the original tensor.
    input_a_view = *input_a;
    input_b_view = *input_b;
    output_view = *output;
  }

  // Select suitable tile sizes.
  uint32_t ts_e4 = 0, ts_e3 = 0, ts_e2 = 0, ts_e1 = 0;
  uint32_t num_threads_per_zaiu = zdnnx_get_min_num_threads_per_zaiu();
  uint32_t num_tiles = zdnnx_get_num_zaiu_threads() *
                       ((num_threads_per_zaiu > 2) ? 2 : num_threads_per_zaiu);
  if (isBigTensor) {
    ts_e4 = zdnnx_get_transformed_dim_per_tile(&input_a_view, num_tiles, E4);
    uint32_t mdis_e4 = zdnnx_get_nnpa_max_dim_size(E4);
    while (ts_e4 > mdis_e4) {
      ts_e4 /= 2;
    }
  }

  // Prepare split information.
  zdnnx_split_info si_a, si_b, si_y;
  zdnnx_prepare_split_info(
      &si_a, &input_a_view, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise A");
  zdnnx_prepare_split_info(
      &si_b, &input_b_view, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise B");
  zdnnx_prepare_split_info(
      &si_y, &output_view, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise Y");

  // No splitting, call the zdnn op without any changes.
  if (zdnnx_has_one_tile(&si_a)) {
    zdnn_status status;
    if (op_type == ZDNNX_ADD_OP)
      status = zdnn_add(input_a, input_b, output);
    else if (op_type == ZDNNX_SUB_OP)
      status = zdnn_sub(input_a, input_b, output);
    else if (op_type == ZDNNX_MUL_OP)
      status = zdnn_mul(input_a, input_b, output);
    else if (op_type == ZDNNX_DIV_OP)
      status = zdnn_div(input_a, input_b, output);
    else if (op_type == ZDNNX_MAX_OP)
      status = zdnn_max(input_a, input_b, output);
    else if (op_type == ZDNNX_MIN_OP)
      status = zdnn_min(input_a, input_b, output);
    else
      status = ZDNN_UNAVAILABLE_FUNCTION;
    return status;
  }

  // Call zdnn op on each tile.
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_a, E4);
  uint32_t num_tiles_e3 = zdnnx_get_num_tiles(&si_a, E3);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_a, E2);
  uint32_t num_tiles_e1 = zdnnx_get_num_tiles(&si_a, E1);
#pragma omp parallel for num_threads(num_tiles) collapse(4)
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e3 = 0; e3 < num_tiles_e3; ++e3) {
      for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
        for (uint32_t e1 = 0; e1 < num_tiles_e1; ++e1) {
          // Prepare tile structs.
          zdnnx_tile ta, tb, ty;

          zdnnx_set_tile(&si_a, &ta, NULL, e4, e3, e2, e1);
          zdnnx_set_tile(&si_b, &tb, NULL, e4, e3, e2, e1);
          zdnnx_set_tile(&si_y, &ty, NULL, e4, e3, e2, e1);

          zdnnx_copy_data_to_tile(&ta);
          zdnnx_copy_data_to_tile(&tb);

          if (op_type == ZDNNX_ADD_OP)
            zdnn_add(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_SUB_OP)
            zdnn_sub(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MUL_OP)
            zdnn_mul(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_DIV_OP)
            zdnn_div(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MAX_OP)
            zdnn_max(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MIN_OP)
            zdnn_min(&ta.data, &tb.data, &ty.data);
          else
            printf("[zdnnx] Not found the zdnn function type %d\n", op_type);

          zdnnx_copy_data_to_full(&ty);

          // Free buffers.
          zdnnx_free_tile_buffer(&ta);
          zdnnx_free_tile_buffer(&tb);
          zdnnx_free_tile_buffer(&ty);
        }
      }
    }
  }

  return ZDNN_OK;
}

zdnn_status zdnnx_omp_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
#ifdef ZDNNX_DEBUG
  printf("[OMP Softmax]\n");
#endif

  uint32_t shape[4];
  zdnnx_get_transformed_shape(input, shape);
  uint32_t mdis_e2 = zdnnx_get_nnpa_max_dim_size(E2);
  uint32_t mdis_e4 = zdnnx_get_nnpa_max_dim_size(E4);
  uint64_t mts = zdnnx_get_nnpa_max_tensor_size();
  uint32_t num_threads_per_zaiu = zdnnx_get_min_num_threads_per_zaiu();
  uint32_t num_tiles = zdnnx_get_num_zaiu_threads() *
                       ((num_threads_per_zaiu > 2) ? 2 : num_threads_per_zaiu);

  // Select suitable tile sizes.
  // For softmax, do not split E1 since it affects accuracy of the final result.
  // Softmax uses 3DS layout, so E3 = 1 and no splitting.
  // Best is to split E4 only. Otherwise, split E2 also.
  uint32_t ts_e4 = 0, ts_e2 = 0;
  ts_e4 = zdnnx_get_transformed_dim_per_tile(input, num_tiles, E4);
  if (ts_e4 > mdis_e4)
    ts_e4 = mdis_e4;
  if (shape[E2] > mdis_e2)
    ts_e2 = mdis_e2;
  // Adjust based on the maximum tensor size.
  uint64_t total_tile_size = (uint64_t)(ts_e4) * (uint64_t)(shape[E3]) *
                             (uint64_t)(ts_e2) * (uint64_t)(shape[E1]);
  if (total_tile_size > mts) {
    // Decreasing ts_e4 to 1 would help?
    if (total_tile_size <= mts * (uint64_t)ts_e4) {
      // Find a good value for ts_e4 ranging from [1 to current ts_e4].
      while ((uint64_t)(ts_e4) * (uint64_t)(shape[E3]) * (uint64_t)(ts_e2) *
                 (uint64_t)(shape[E1]) >
             mts) {
        ts_e4 /= 2;
      }
    } else {
      // Even ts_e4 = 1 does not help, split E2 too.
      ts_e4 = 1;
      while ((uint64_t)(ts_e4) * (uint64_t)(shape[E3]) * (uint64_t)(ts_e2) *
                 (uint64_t)(shape[E1]) >
             mts) {
        ts_e2 /= 2;
      }
    }
  }

  // Prepare split information
  zdnnx_split_info si_x, si_y;
  zdnnx_prepare_split_info(&si_x, input, ts_e4, 0, ts_e2, 0, "Softmax X");
  zdnnx_prepare_split_info(&si_y, output, ts_e4, 0, ts_e2, 0, "Softmax Y");

  // No splitting, call the zdnn softmax without any changes.
  if (zdnnx_has_one_tile(&si_x))
    return zdnn_softmax(input, save_area, act_func, output);

  // Call zdnn_softmax on each tile. Not use save_area.
  // TODO: could we reuse save_area in particular in the parallel scenario?
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_x, E4);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_x, E2);
#pragma omp parallel for num_threads(num_tiles) collapse(2)
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
      // Prepare tile structs.
      zdnnx_tile tx, ty;

      zdnnx_set_tile(&si_x, &tx, NULL, e4, 0, e2, 0);
      zdnnx_set_tile(&si_y, &ty, NULL, e4, 0, e2, 0);

      zdnnx_copy_data_to_tile(&tx);
      zdnn_status status = zdnn_softmax(&tx.data, NULL, act_func, &ty.data);
      assert(status == ZDNN_OK);
      zdnnx_copy_data_to_full(&ty);

      // Free buffers.
      zdnnx_free_tile_buffer(&tx);
      zdnnx_free_tile_buffer(&ty);
    }
  }
  return ZDNN_OK;
}

#endif // ZDNNX_WITH_OMP
