/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- seq_ops.c ---------------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Sequential operations that split ztensors into tiles but use a single zAIU to
// run with the tiles.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "seq_ops.h"
#include "zdnnx.h"
#include "zdnnx_ops.h"

// -----------------------------------------------------------------------------
// Element-wise operations
// -----------------------------------------------------------------------------

/**
 * Return false if exceeded the maximum tensor size but couldnot find a good
 * splitting way. Otherwise, return true.
 */
static bool select_tile_sizes(const zdnn_ztensor *t, uint32_t *ts_e4,
    uint32_t *ts_e3, uint32_t *ts_e2, uint32_t *ts_e1) {
  uint32_t shape[4];
  zdnnx_get_transformed_shape(t, shape);
  uint32_t max_dim_size_e4 = zdnnx_get_nnpa_max_dim_size(E4);
  uint32_t max_dim_size_e3 = zdnnx_get_nnpa_max_dim_size(E3);
  uint32_t max_dim_size_e2 = zdnnx_get_nnpa_max_dim_size(E2);
  uint32_t max_dim_size_e1 = zdnnx_get_nnpa_max_dim_size(E1);
  uint64_t max_tensor_size = zdnnx_get_nnpa_max_tensor_size();

  bool e4_exceeded = (shape[E4] > max_dim_size_e4);
  bool e3_exceeded = (shape[E3] > max_dim_size_e3);
  bool e2_exceeded = (shape[E2] > max_dim_size_e2);
  bool e1_exceeded = (shape[E1] > max_dim_size_e1);

  bool include_e4 = (ts_e4 != NULL);
  bool include_e3 = (ts_e3 != NULL);
  bool include_e2 = (ts_e2 != NULL);
  bool include_e1 = (ts_e1 != NULL);

  // If there is a tile dimension exceeding the max dim size, use the max dim
  // size.
  // Stickification: (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, 32, 64)
  uint32_t tmp_e4 = shape[E4];
  uint32_t tmp_e3 = shape[E3];
  uint32_t tmp_e2 = shape[E2];
  uint32_t tmp_e1 = shape[E1];

  if (include_e4 && e4_exceeded)
    tmp_e4 = max_dim_size_e4;
  if (include_e1 && e1_exceeded) {
    tmp_e1 = max_dim_size_e1;
    // E4 is the outer loop of E1 in stickified tensor.
    // To avoid data copy, split E4 into chunks of 1 element.
    if (include_e4)
      tmp_e4 = 1;
  }
  if (include_e3 && e3_exceeded) {
    tmp_e3 = max_dim_size_e3;
    // E4, E1 are the outer loops of E3 in stickified tensor.
    // To avoid data copy, split E4, E1 into chunks of 1 element.
    if (include_e4)
      tmp_e4 = 1;
    if (include_e1 && (tmp_e1 > 64))
      tmp_e1 = 64;
  }
  if (include_e2 && e2_exceeded) {
    tmp_e2 = max_dim_size_e2;
    // E4, E1, E3 are the outer loops of E2 in stickified tensor.
    // To avoid data copy, split E4, E1, E3 into chunks of 1 element.
    if (include_e4)
      tmp_e4 = 1;
    if (include_e1 && (tmp_e1 > 64))
      tmp_e1 = 64;
    if (include_e3)
      tmp_e3 = 1;
  }

  // If exceeded the max tensor size, decrease by half the maximum dimension.
  uint64_t total_tile_size = (uint64_t)(tmp_e4) * (uint64_t)(tmp_e3) *
                             (uint64_t)(tmp_e2) * (uint64_t)(tmp_e1);
  while (total_tile_size > max_tensor_size) {
#ifdef ZDNNX_DEBUG
    printf("Exceeding the max tensor size, adjusting ...\n");
#endif
    uint32_t *max_ts = NULL;
    if (include_e4 && (max_ts == NULL || tmp_e4 > *max_ts))
      max_ts = &tmp_e4;
    if (include_e3 && (max_ts == NULL || tmp_e3 > *max_ts)) {
      max_ts = &tmp_e3;
      // E4, E1 are the outer loops of E3 in stickified tensor.
      // To avoid data copy, split E4, E1 into chunks of 1 element.
      if (include_e4)
        tmp_e4 = 1;
      if (include_e1 && (tmp_e1 > 64))
        tmp_e1 = 64;
    }
    if (include_e2 && (max_ts == NULL || tmp_e2 > *max_ts)) {
      max_ts = &tmp_e2;
      // E4, E1, E3 are the outer loops of E2 in stickified tensor.
      // To avoid data copy, split E4, E1, E3 into chunks of 1 element.
      if (include_e4)
        tmp_e4 = 1;
      if (include_e1 && (tmp_e1 > 64))
        tmp_e1 = 64;
      if (include_e3)
        tmp_e3 = 1;
    }
    if (include_e1 && (max_ts == NULL || tmp_e1 > *max_ts)) {
      max_ts = &tmp_e1;
      // E4 is the outer loop of E1 in stickified tensor.
      // To avoid data copy, split E4 into chunks of 1 element.
      if (include_e4)
        tmp_e4 = 1;
    }
    if (max_ts) {
      *max_ts = *max_ts / 2;
    } else {
      // Exceed the maximum tensor size but couldnot find a good splitting way.
      return false;
    }

    // The total tile size does not change, failed to find a splitting that
    // avoids exceeding the maximum tensor size.
    uint64_t new_total_tile_size = (uint64_t)(tmp_e4) * (uint64_t)(tmp_e3) *
                                   (uint64_t)(tmp_e2) * (uint64_t)(tmp_e1);
    if (new_total_tile_size == total_tile_size)
      return false;
  }

  // Dimensions are unchanged, return false.
  if (tmp_e1 == shape[E1] && tmp_e2 == shape[E2] && tmp_e3 == shape[E3] &&
      tmp_e4 == shape[E4])
    return false;

  if (include_e4)
    *ts_e4 = tmp_e4;
  if (include_e3)
    *ts_e3 = tmp_e3;
  if (include_e2)
    *ts_e2 = tmp_e2;
  if (include_e1)
    *ts_e1 = tmp_e1;

  return true;
}

zdnn_status zdnnx_seq_unary_elementwise(const zdnn_ztensor *input,
    const void *scalar_input, zdnn_ztensor *output, ElemementwiseOp op_type) {
#ifdef ZDNNX_DEBUG
  printf("[UnaryElementwise op_type %d]\n", op_type);
#endif

  // Select suitable tile sizes.
  uint32_t ts_e4 = 0, ts_e3 = 0, ts_e2 = 0, ts_e1 = 0;
  select_tile_sizes(input, &ts_e4, &ts_e3, &ts_e2, &ts_e1);

  // Prepare split information.
  zdnnx_split_info si_x, si_y;
  zdnnx_prepare_split_info(
      &si_x, input, ts_e4, ts_e3, ts_e2, ts_e1, "UnaryElementwise X");
  zdnnx_prepare_split_info(
      &si_y, output, ts_e4, ts_e3, ts_e2, ts_e1, "UnaryElementwise Y");

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

  // Prepare a shared buffer for all tiles if data copy occurs.
  char *tile_buff_x = NULL;
  char *tile_buff_y = NULL;
  if (zdnnx_has_no_buffer_reuse(&si_x))
    tile_buff_x = zdnnx_alloc_buffer(&si_x);
  if (zdnnx_has_no_buffer_reuse(&si_y))
    tile_buff_y = zdnnx_alloc_buffer(&si_y);
  // Prepare tile structs.
  zdnnx_tile tx, ty;

  // Call zdnn op on each tile.
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_x, E4);
  uint32_t num_tiles_e3 = zdnnx_get_num_tiles(&si_x, E3);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_x, E2);
  uint32_t num_tiles_e1 = zdnnx_get_num_tiles(&si_x, E1);
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e3 = 0; e3 < num_tiles_e3; ++e3) {
      for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
        for (uint32_t e1 = 0; e1 < num_tiles_e1; ++e1) {
          zdnnx_set_tile(&si_x, &tx, tile_buff_x, e4, e3, e2, e1);
          zdnnx_set_tile(&si_y, &ty, tile_buff_y, e4, e3, e2, e1);
          zdnnx_copy_data_to_tile(&tx);

          zdnn_status status;
          if (op_type == ZDNNX_EXP_OP)
            status = zdnn_exp(&tx.data, &ty.data);
          else if (op_type == ZDNNX_GELU_OP)
            zdnn_gelu(&tx.data, &ty.data);
          else if (op_type == ZDNNX_INVSQRT_OP)
            status =
                zdnn_invsqrt(&tx.data, *(const float *)scalar_input, &ty.data);
          else if (op_type == ZDNNX_LOG_OP)
            status = zdnn_log(&tx.data, &ty.data);
          else if (op_type == ZDNNX_RELU_OP)
            status = zdnn_relu(&tx.data, scalar_input, &ty.data);
          else if (op_type == ZDNNX_SIGMOID_OP)
            status = zdnn_sigmoid(&tx.data, &ty.data);
          else if (op_type == ZDNNX_SQRT_OP)
            zdnn_sqrt(&tx.data, &ty.data);
          else if (op_type == ZDNNX_TANH_OP)
            status = zdnn_tanh(&tx.data, &ty.data);
          else
            status = ZDNN_UNAVAILABLE_FUNCTION;
          if (status != ZDNN_OK)
            return status;

          zdnnx_copy_data_to_full(&ty);
        }
      }
    }
  }

  // Free buffers.
  zdnnx_free_buffer(tile_buff_x);
  zdnnx_free_buffer(tile_buff_y);

  return ZDNN_OK;
}

zdnn_status zdnnx_seq_binary_elementwise(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, zdnn_ztensor *output,
    ElemementwiseOp op_type) {
#ifdef ZDNNX_DEBUG
  printf("[BinaryElementwise op_type %d]\n", op_type);
#endif

  // Select suitable tile sizes.
  uint32_t ts_e4 = 0, ts_e3 = 0, ts_e2 = 0, ts_e1 = 0;
  select_tile_sizes(input_a, &ts_e4, &ts_e3, &ts_e2, &ts_e1);

  // Prepare split information.
  zdnnx_split_info si_a, si_b, si_y;
  zdnnx_prepare_split_info(
      &si_a, input_a, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise A");
  zdnnx_prepare_split_info(
      &si_b, input_b, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise B");
  zdnnx_prepare_split_info(
      &si_y, output, ts_e4, ts_e3, ts_e2, ts_e1, "BinaryElementwise Y");

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

  // Prepare a shared buffer for all tiles if data copy occurs.
  char *tile_buff_a = NULL;
  char *tile_buff_b = NULL;
  char *tile_buff_y = NULL;
  if (zdnnx_has_no_buffer_reuse(&si_a))
    tile_buff_a = zdnnx_alloc_buffer(&si_a);
  if (zdnnx_has_no_buffer_reuse(&si_b))
    tile_buff_b = zdnnx_alloc_buffer(&si_b);
  if (zdnnx_has_no_buffer_reuse(&si_y))
    tile_buff_y = zdnnx_alloc_buffer(&si_y);
  // Prepare tile structs.
  zdnnx_tile ta, tb, ty;

  // Call zdnn op on each tile.
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_a, E4);
  uint32_t num_tiles_e3 = zdnnx_get_num_tiles(&si_a, E3);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_a, E2);
  uint32_t num_tiles_e1 = zdnnx_get_num_tiles(&si_a, E1);
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e3 = 0; e3 < num_tiles_e3; ++e3) {
      for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
        for (uint32_t e1 = 0; e1 < num_tiles_e1; ++e1) {
          zdnnx_set_tile(&si_a, &ta, tile_buff_a, e4, e3, e2, e1);
          zdnnx_set_tile(&si_b, &tb, tile_buff_b, e4, e3, e2, e1);
          zdnnx_set_tile(&si_y, &ty, tile_buff_y, e4, e3, e2, e1);

          zdnnx_copy_data_to_tile(&ta);
          zdnnx_copy_data_to_tile(&tb);

          zdnn_status status;
          if (op_type == ZDNNX_ADD_OP)
            status = zdnn_add(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_SUB_OP)
            status = zdnn_sub(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MUL_OP)
            status = zdnn_mul(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_DIV_OP)
            status = zdnn_div(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MAX_OP)
            status = zdnn_max(&ta.data, &tb.data, &ty.data);
          else if (op_type == ZDNNX_MIN_OP)
            status = zdnn_min(&ta.data, &tb.data, &ty.data);
          else
            status = ZDNN_UNAVAILABLE_FUNCTION;
          if (status != ZDNN_OK)
            return status;

          zdnnx_copy_data_to_full(&ty);
        }
      }
    }
  }

  // Free buffers.
  zdnnx_free_buffer(tile_buff_a);
  zdnnx_free_buffer(tile_buff_b);
  zdnnx_free_buffer(tile_buff_y);

  return ZDNN_OK;
}

zdnn_status zdnnx_seq_softmax(const zdnn_ztensor *input, void *save_area,
    zdnn_softmax_act act_func, zdnn_ztensor *output) {
#ifdef ZDNNX_DEBUG
  printf("[Softmax]\n");
#endif

  // Select suitable tile sizes.
  // For softmax, do not split E1 since it affects accuracy of the final result.
  uint32_t ts_e4 = 0, ts_e3 = 0, ts_e2 = 0;
  if (!select_tile_sizes(input, &ts_e4, &ts_e3, &ts_e2, NULL))
    return zdnn_softmax(input, save_area, act_func, output);

  // Prepare split information
  zdnnx_split_info si_x, si_y;
  zdnnx_prepare_split_info(&si_x, input, ts_e4, ts_e3, ts_e2, 0, "Softmax X");
  zdnnx_prepare_split_info(&si_y, output, ts_e4, ts_e3, ts_e2, 0, "Softmax Y");

  // No splitting, call the zdnn softmax without any changes.
  if (zdnnx_has_one_tile(&si_x))
    return zdnn_softmax(input, save_area, act_func, output);

  // Prepare a shared buffer for all tiles if data copy occurs.
  char *tile_buff_x = NULL;
  char *tile_buff_y = NULL;
  if (zdnnx_has_no_buffer_reuse(&si_x))
    tile_buff_x = zdnnx_alloc_buffer(&si_x);
  if (zdnnx_has_no_buffer_reuse(&si_y))
    tile_buff_y = zdnnx_alloc_buffer(&si_y);
  // Prepare tile structs.
  zdnnx_tile tx, ty;

  // Call zdnn_softmax on each tile. Not use save_area.
  // TODO: could we reuse save_area in particular in the parallel scenario?
  uint32_t num_tiles_e4 = zdnnx_get_num_tiles(&si_x, E4);
  uint32_t num_tiles_e3 = zdnnx_get_num_tiles(&si_x, E3);
  uint32_t num_tiles_e2 = zdnnx_get_num_tiles(&si_x, E2);
  for (uint32_t e4 = 0; e4 < num_tiles_e4; ++e4) {
    for (uint32_t e3 = 0; e3 < num_tiles_e3; ++e3) {
      for (uint32_t e2 = 0; e2 < num_tiles_e2; ++e2) {
        zdnnx_set_tile(&si_x, &tx, tile_buff_x, e4, e3, e2, 0);
        zdnnx_set_tile(&si_y, &ty, tile_buff_y, e4, e3, e2, 0);

        zdnnx_copy_data_to_tile(&tx);
        zdnn_status status = zdnn_softmax(&tx.data, NULL, act_func, &ty.data);
        assert(status == ZDNN_OK);
        zdnnx_copy_data_to_full(&ty);
      }
    }
  }

  // Free buffers.
  zdnnx_free_buffer(tile_buff_x);
  zdnnx_free_buffer(tile_buff_y);

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

zdnn_status zdnnx_seq_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, int op_type,
    zdnn_ztensor *output, bool is_bcast) {
#ifdef ZDNNX_DEBUG
  printf("[MatMul]\n");
#endif

  // MatMul types in zdnn:
  // - unstacked: A (2D),  B (2D),  C (1D),  Y (2D)
  // - stacked  : A (3DS), B (3DS), C (2DS), Y (3DS)
  // - bcast    : A (3DS), B (2D),  C (1D),  Y (3DS)
  zdnn_data_layouts a_layout = input_a->pre_transformed_desc->layout;
  zdnn_data_layouts b_layout = input_b->pre_transformed_desc->layout;
  zdnn_data_layouts c_layout = input_c->pre_transformed_desc->layout;
  bool is_stacked =
      (a_layout == ZDNN_3DS && b_layout == ZDNN_3DS && c_layout == ZDNN_2DS);

  // Select suitable tile sizes.
  uint32_t ts_e2, ts_e1;
  // Select E2 tile size.
  if (!select_tile_sizes(input_a, NULL, NULL, &ts_e2, NULL)) {
#ifdef ZDNNX_DEBUG
    printf("[MatMul] calling the original zdnn matmul.\n");
#endif
    return call_zdnn_matmul_op(
        input_a, input_b, input_c, op_type, output, is_bcast);
  }
  // Select E1 tile size.
  if (!select_tile_sizes(input_b, NULL, NULL, NULL, &ts_e1)) {
#ifdef ZDNNX_DEBUG
    printf("[MatMul] calling the original zdnn matmul.\n");
#endif
    return call_zdnn_matmul_op(
        input_a, input_b, input_c, op_type, output, is_bcast);
  }

  zdnnx_split_info si_a, si_b, si_c, si_y;
  zdnnx_prepare_split_info(&si_a, input_a, 1, 0, ts_e2, 0, "MatMul A");
  zdnnx_prepare_split_info(&si_b, input_b, 1, 0, 0, ts_e1, "MatMul B");
  zdnnx_prepare_split_info(&si_c, input_c, 1, 0, 0, ts_e1, "MatMul C");
  zdnnx_prepare_split_info(&si_y, output, 1, 0, ts_e2, ts_e1, "MatMul Y");

  // No splitting, call the zdnn matmul without any changes.
  if (zdnnx_has_one_tile(&si_a) && zdnnx_has_one_tile(&si_b)) {
#ifdef ZDNNX_DEBUG
    printf("[MatMul] calling the original zdnn matmul.\n");
#endif
    zdnn_status status = call_zdnn_matmul_op(
        input_a, input_b, input_c, op_type, output, is_bcast);
    return status;
  }

  // Prepare a shared buffer for all tiles if data copy occurs.
  char *tile_buff_a = NULL;
  char *tile_buff_b = NULL;
  char *tile_buff_c = NULL;
  char *tile_buff_y = NULL;
  if (zdnnx_has_no_buffer_reuse(&si_a))
    tile_buff_a = zdnnx_alloc_buffer(&si_a);
  if (zdnnx_has_no_buffer_reuse(&si_b))
    tile_buff_b = zdnnx_alloc_buffer(&si_b);
  if (zdnnx_has_no_buffer_reuse(&si_c))
    tile_buff_c = zdnnx_alloc_buffer(&si_c);
  if (zdnnx_has_no_buffer_reuse(&si_y))
    tile_buff_y = zdnnx_alloc_buffer(&si_y);
  // Prepare tile structs.
  zdnnx_tile ta, tb, tc, ty;

  // Call zdnn_matmul_op on each tile.
  // For each output tile at index (m, n): Y(m, n) = A(m, K) * B(K, n).
  uint32_t B = zdnnx_get_num_tiles(&si_a, E4);
  uint32_t M = zdnnx_get_num_tiles(&si_a, E2);
  uint32_t N = zdnnx_get_num_tiles(&si_b, E1);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t m = 0; m < M; ++m) {
      /* Prepare and set an A tile at index (0, 0, m, 0). */
      zdnnx_set_tile(&si_a, &ta, tile_buff_a, b, 0, m, 0);
      /* Copy if reuse is off. */
      zdnnx_copy_data_to_tile(&ta);

      // Iterate over the tiles along the second dim of B.
      for (uint32_t n = 0; n < N; ++n) {
        /* Prepare and set B and C tiles at index (0, 0, 0, n). */
        zdnnx_set_tile(&si_b, &tb, tile_buff_b, (is_stacked) ? b : 0, 0, 0, n);
        zdnnx_set_tile(&si_c, &tc, tile_buff_c, (is_stacked) ? b : 0, 0, 0, n);
        /* Copy if reuse is off. */
        zdnnx_copy_data_to_tile(&tb);
        zdnnx_copy_data_to_tile(&tc);

        /* Prepare and set an output tile at index (0, 0, m, n). */
        zdnnx_set_tile(&si_y, &ty, tile_buff_y, b, 0, m, n);

        /* Operation */
        zdnn_status status = call_zdnn_matmul_op(
            &ta.data, &tb.data, &tc.data, op_type, &ty.data, is_bcast);
        assert(status == ZDNN_OK);

        /* Copy the output tile at (0, 0, m, n) to the full output. */
        zdnnx_copy_data_to_full(&ty);
      }
    }
  }

  // Free buffers.
  zdnnx_free_buffer(tile_buff_a);
  zdnnx_free_buffer(tile_buff_b);
  zdnnx_free_buffer(tile_buff_c);
  zdnnx_free_buffer(tile_buff_y);
  return ZDNN_OK;
}
