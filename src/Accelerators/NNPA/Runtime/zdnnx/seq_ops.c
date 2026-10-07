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
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "seq_ops.h"
#include "zdnnx.h"
#include "zdnnx_ops.h"
#include "zdnnx_private.h"

// -----------------------------------------------------------------------------
// Element-wise operations
// -----------------------------------------------------------------------------

/**
 * Select tile sizes so that the sizes do not exceed the maximum values in NNPA,
 * e.g. max dim sizes and max tensor size.
 */
static void select_tile_sizes(const zdnn_ztensor *t, uint32_t *ts_e4,
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

  // If exceeded the max tensor size, decrease dim size in this order E4, E1,
  // E3, E2 to maximize the buffer reuse:
  uint64_t total_tile_size = (uint64_t)(tmp_e4) * (uint64_t)(tmp_e3) *
                             (uint64_t)(tmp_e2) * (uint64_t)(tmp_e1);
  if (total_tile_size > max_tensor_size) {
    uint32_t *tmp_ptrs[4] = {&tmp_e4, &tmp_e1, &tmp_e3, &tmp_e2};
    bool includes[4] = {include_e4, include_e1, include_e3, include_e2};
    for (int i = 0; i < 4; ++i) {
      if (!includes[i])
        continue;
      // Minimum value, nothing to adjust.
      if (*tmp_ptrs[i] == 1)
        continue;
        // Select a new dim size that makes total_tile_size smaller than
        // max_tensor_size.
#ifdef ZDNNX_DEBUG
      int e_idx = 0;
      if (i == 0)
        e_idx = 4;
      else if (i == 1)
        e_idx = 1;
      else if (i == 2)
        e_idx = 3;
      else if (i == 3)
        e_idx = 2;
      printf("Exceeding the max tensor size: tile_size: %ld, "
             "max_tensor_size: %ld. Adjusting E%d... \n",
          total_tile_size, max_tensor_size, e_idx);
#endif
      *tmp_ptrs[i] = CEIL(*tmp_ptrs[i], CEIL(total_tile_size, max_tensor_size));
      total_tile_size = (uint64_t)(tmp_e4) * (uint64_t)(tmp_e3) *
                        (uint64_t)(tmp_e2) * (uint64_t)(tmp_e1);
      // Good tile size. Stop searching.
      if (total_tile_size <= max_tensor_size) {
        break;
      }
    }
  }

  // Dimensions are unchanged, return false.
  if (tmp_e1 == shape[E1] && tmp_e2 == shape[E2] && tmp_e3 == shape[E3] &&
      tmp_e4 == shape[E4])
    return;

  if (include_e4)
    *ts_e4 = tmp_e4;
  if (include_e3)
    *ts_e3 = tmp_e3;
  if (include_e2)
    *ts_e2 = tmp_e2;
  if (include_e1)
    *ts_e1 = tmp_e1;
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

          zdnn_status status = ZDNN_UNAVAILABLE_FUNCTION;
          if (op_type == ZDNNX_EXP_OP)
            status = zdnn_exp(&tx.data, &ty.data);
          else if (op_type == ZDNNX_GELU_OP)
            status = zdnn_gelu(&tx.data, &ty.data);
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
            status = zdnn_sqrt(&tx.data, &ty.data);
          else if (op_type == ZDNNX_TANH_OP)
            status = zdnn_tanh(&tx.data, &ty.data);
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
  select_tile_sizes(input, &ts_e4, &ts_e3, &ts_e2, NULL);

  // Prepare split information
  zdnnx_split_info si_x, si_y;
  zdnnx_prepare_split_info(&si_x, input, ts_e4, ts_e3, ts_e2, 0, "Softmax X");
  zdnnx_prepare_split_info(&si_y, output, ts_e4, ts_e3, ts_e2, 0, "Softmax Y");

  // No splitting, call the zdnn softmax without any changes.
  if (zdnnx_has_one_tile(&si_x)) {
#ifdef ZDNNX_DEBUG
    printf("[Softmax] calling the original zdnn softmax.\n");
#endif
    return zdnn_softmax(input, save_area, act_func, output);
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
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, bool transpose_a,
    bool transpose_b, int op_type, zdnn_ztensor *output, bool is_bcast) {
  if (transpose_a || transpose_b)
    return zdnn_matmul_transpose_op(input_a, input_b, input_c,
        transpose_a ? 1 : 0, transpose_b ? 1 : 0, op_type, output);
  if (is_bcast)
    return zdnn_matmul_bcast_op(
        input_a, input_b, input_c, (zdnn_matmul_bcast_ops)op_type, output);
  return zdnn_matmul_op(
      input_a, input_b, input_c, (zdnn_matmul_ops)op_type, output);
}

zdnn_status zdnnx_seq_matmul(const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c, bool transpose_a,
    bool transpose_b, int op_type, zdnn_ztensor *output, bool is_bcast) {
#ifdef ZDNNX_DEBUG
  printf("[MatMul, tranpsose_a: %s, tranpsose_b: %s, is_bcast: %s]\n",
      transpose_a ? "true" : "false", transpose_b ? "true" : "false",
      is_bcast ? "true" : "false");
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

  // Select suitable tile sizes for E4, E2, E1 tile size. E3 is always 1.
  // If a tensor is transposed,  only split its E4.
  uint32_t ts_e4 = 0, ts_e2 = 0, ts_e1 = 0;
  select_tile_sizes(input_a, &ts_e4, NULL, transpose_a ? NULL : &ts_e2, NULL);
  select_tile_sizes(input_b, &ts_e4, NULL, NULL, transpose_b ? NULL : &ts_e1);
  select_tile_sizes(output, &ts_e4, NULL, transpose_a ? NULL : &ts_e2,
      transpose_b ? NULL : &ts_e1);

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
    zdnn_status status = call_zdnn_matmul_op(input_a, input_b, input_c,
        transpose_a, transpose_b, op_type, output, is_bcast);
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
        zdnn_status status = call_zdnn_matmul_op(&ta.data, &tb.data, &tc.data,
            transpose_a, transpose_b, op_type, &ty.data, is_bcast);
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

static void free_ztensor_buf(zdnn_ztensor *zt) {
  if (zt->buffer)
    zdnn_free_ztensor_buffer(zt);
}

static inline void prefetch_read(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbt((void *)((uintptr_t)ptr + offset));
#else
  __builtin_prefetch((void *)((uintptr_t)ptr + offset), 0);
#endif
}

static inline void prefetch_write(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbtst((void *)((uintptr_t)ptr + offset));
#else
  __builtin_prefetch((void *)((uintptr_t)ptr + offset), 1);
#endif
}

static inline void cache_flush(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbf((void *)((uintptr_t)ptr + offset));
#else
  (void)ptr;
  (void)offset;
#endif
}

// Broadcast a column vector [1, S, 1] to a tile [1, S, T] in stickified space.
// 3DS layout: (e4, e2, e1) -> (e4, ceil(e1/64), 1, ceil(e2/32), 32, 64).
// Each stick row in the source has one DLFLOAT16 value at position 0.
// We replicate it across all 64 positions, then copy page 0 to all T/64 pages.
static void broadcast_column_to_tile(
    const zdnn_ztensor *col, zdnn_ztensor *tile) {
  uint32_t S = col->pre_transformed_desc->dim2;
  uint32_t T = tile->pre_transformed_desc->dim1;
  uint32_t num_stick_groups = (S + 31) / 32;
  uint32_t num_pages = T / 64;
  uint32_t page_bytes = num_stick_groups * 4096;

  const char *src = (const char *)col->buffer;
  char *dst = (char *)tile->buffer;

  for (uint32_t c = 0; c < num_stick_groups; c++) {
    uint64_t group_offset = (uint64_t)c * 4096;
    prefetch_read(src, group_offset);
    for (uint32_t r = 0; r < 32; r++) {
      uint64_t offset = group_offset + r * 128;
      if (r + 1 < 32)
        prefetch_read(src, offset + 128);
      prefetch_write(dst, offset);
      uint16_t val;
      memcpy(&val, src + offset, sizeof(val));
      uint16_t *row = (uint16_t *)(dst + offset);
      for (uint32_t j = 0; j < 64; j++)
        row[j] = val;
      cache_flush(dst, offset);
    }
  }

  for (uint32_t p = 1; p < num_pages; p++)
    memcpy(dst + (uint64_t)p * page_bytes, dst, page_bytes);

  tile->is_transformed = true;
}

// Helper to create a 3DS ztensor, allocate buffer, and optionally stickify.
static zdnn_status create_3ds_ztensor(uint32_t s, uint32_t m, uint32_t n,
    zdnn_ztensor *zt, zdnn_tensor_desc *pre, zdnn_tensor_desc *trans,
    float *fp32_data) {
  memset(pre, 0, sizeof(*pre));
  memset(trans, 0, sizeof(*trans));
  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, pre, s, m, n);
  zdnn_status rc = zdnn_generate_transformed_desc(pre, trans);
  if (rc != ZDNN_OK)
    return rc;
  zdnn_init_ztensor(pre, trans, zt);
  rc = zdnn_allochelper_ztensor(zt);
  if (rc != ZDNN_OK)
    return rc;
  if (fp32_data)
    rc = zdnn_transform_ztensor(zt, fp32_data);
  return rc;
}

// Helper to create a 2DS zero-bias ztensor using memset (no zdnn_transform).
static zdnn_status create_2ds_zero_bias(uint32_t s, uint32_t n,
    zdnn_ztensor *zt, zdnn_tensor_desc *pre, zdnn_tensor_desc *trans) {
  memset(pre, 0, sizeof(*pre));
  memset(trans, 0, sizeof(*trans));
  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, pre, s, n);
  zdnn_status rc = zdnn_generate_transformed_desc(pre, trans);
  if (rc != ZDNN_OK)
    return rc;
  zdnn_init_ztensor(pre, trans, zt);
  rc = zdnn_allochelper_ztensor(zt);
  if (rc != ZDNN_OK)
    return rc;
  memset(zt->buffer, 0, zt->buffer_size);
  zt->is_transformed = true;
  return ZDNN_OK;
}

// Small S: S <= 2048. Delegate to three sequential zdnn calls.
static zdnn_status matmul_add_softmax_small_s(const zdnn_ztensor *X,
    const zdnn_ztensor *Y, const zdnn_ztensor *Z, const zdnn_ztensor *Bias,
    zdnn_ztensor *work, zdnn_ztensor *output) {
#ifdef ZDNNX_DEBUG
  printf("[MatMulAddSoftmax Small S]\n");
#endif

  // MatMul: work = X * Y + Bias.
  zdnn_status status =
      zdnn_matmul_op(X, Y, Bias, MATMUL_OP_ADDITION, work);
  // Add: work = work + Z.
  if (status == ZDNN_OK)
    status = zdnn_add(work, Z, work);
  // Softmax: output = softmax(work).
  if (status == ZDNN_OK)
    status = zdnn_softmax(work, NULL, SOFTMAX_ACT_NONE, output);
  return status;
}

// Compute softmax(X * Y + Z) when the column dimension (full_S) exceeds NNPA's
// 2048-element limit. The column dimension is tiled into N_t tiles of width T,
// and a two-pass online softmax algorithm produces the exact same result as a
// single untiled softmax.
//
// Notation (all per-row, i.e. [S,1] vectors unless noted):
//   scores_c = X * Y_c + Z_c        -- [S,T] score tile for column tile c
//   M_c      = running row-wise max after processing tiles 0..c
//   M_final  = M_{N_t-1}            -- global max across all tiles
//   D        = running row-wise denominator (sum of exp(scores - M_final))
//
// Standard softmax:  out_i = exp(scores_i - M_final) / D
//
// Key insight: we cannot compute M_final until all tiles are processed, so
// Pass 1 stores exp(scores_c - M_c) using the *running* max M_c and records
// a snapshot of M_c after each tile. After all tiles, M_final and D are known.
// Pass 2 then corrects each tile:
//   out_c = out_c * exp(M_c - M_final) / D
//         = exp(scores_c - M_c) * exp(M_c - M_final) / D
//         = exp(scores_c - M_final) / D                     (exact softmax)
//
// Running-sum maintenance (Pass 1, per tile c):
//   correction = exp(M_{c-1} - M_c)
//   D = D * correction + row_sum(exp(scores_c - M_c))
// The multiplicative correction rescales the previously accumulated D from the
// old max M_{c-1} to the new max M_c, keeping D in the exp(... - M_c) basis.
// After the last tile, D = sum_c sum_j exp(scores_{c,j} - M_final).
//
static zdnn_status matmul_add_softmax_large_s(const zdnn_ztensor *X,
    const zdnn_ztensor *Y, const zdnn_ztensor *Z, const zdnn_ztensor *Bias,
    zdnn_ztensor *work, zdnn_ztensor *output) {
#ifdef ZDNNX_DEBUG
  printf("[MatMulAddSoftmax Large S]\n");
#endif
  zdnn_status status = ZDNN_OK;
  uint32_t S = zdnnx_get_transformed_dim(X, E2);
  uint32_t full_S = zdnnx_get_transformed_dim(output, E1);
  uint32_t BH = zdnnx_get_transformed_dim(X, E4);

  // Choose tile size T: 2048 aligned to 64, must divide full_S evenly.
  uint32_t T = 2048;
  uint32_t mdis_e1 = zdnnx_get_nnpa_max_dim_size(E1);
  if (T > mdis_e1)
    T = mdis_e1;
  if (full_S % T != 0) {
    for (uint32_t t = T; t >= 64; t -= 64) {
      if (full_S % t == 0) {
        T = t;
        break;
      }
    }
  }
  uint32_t N_t = full_S / T;

#ifdef ZDNNX_DEBUG
  printf("[MatMulAddSoftmax Large S] S=%u, full_S=%u, T=%u, N_t=%u, BH=%u\n",
      S, full_S, T, N_t, BH);
#endif

  // All resources zero-initialized so cleanup is safe on any failure path.
  float *fp32_buf = NULL;
  float *max_snapshots = NULL;
  void *max_row_init_buf = NULL;
  zdnn_ztensor ones_zt = {0};
  zdnn_ztensor max_row_zt = {0}, sum_row_zt = {0};
  zdnn_ztensor old_max_zt = {0}, block_max_zt = {0};
  zdnn_ztensor correction_zt = {0}, tile_rowsum_zt = {0};
  zdnn_ztensor scratch_tile = {0}, bias_1 = {0};
  zdnn_tensor_desc ones_pre, ones_trans;
  zdnn_tensor_desc max_row_pre, max_row_trans, sum_row_pre, sum_row_trans;
  zdnn_tensor_desc om_pre, om_trans, bm_pre, bm_trans;
  zdnn_tensor_desc cor_pre, cor_trans, trs_pre, trs_trans;
  zdnn_tensor_desc sc_pre, sc_trans, b1_pre, b1_trans;

  // --- Allocate FP32 buffers ---
  uint32_t max_fp32_len = (S > T) ? S : T;
  fp32_buf = (float *)malloc(max_fp32_len * sizeof(float));
  max_snapshots = (float *)malloc((uint64_t)N_t * S * sizeof(float));
  if (!fp32_buf || !max_snapshots)
    status = ZDNN_FUNC_RC_F000;

  // --- Create ones tensor [1, T, 1] for row-sum matmul ---
  if (status == ZDNN_OK) {
    for (uint32_t i = 0; i < T; i++)
      fp32_buf[i] = 1.0f;
    status = create_3ds_ztensor(
        1, T, 1, &ones_zt, &ones_pre, &ones_trans, fp32_buf);
  }
  // --- Create running statistics [1, S, 1] ---
  if (status == ZDNN_OK) {
    for (uint32_t i = 0; i < S; i++)
      fp32_buf[i] = -65504.0f;
    status = create_3ds_ztensor(
        1, S, 1, &max_row_zt, &max_row_pre, &max_row_trans, fp32_buf);
  }
  if (status == ZDNN_OK) {
    max_row_init_buf = malloc(max_row_zt.buffer_size);
    if (!max_row_init_buf)
      status = ZDNN_FUNC_RC_F000;
    else
      memcpy(max_row_init_buf, max_row_zt.buffer, max_row_zt.buffer_size);
  }
  if (status == ZDNN_OK) {
    status = create_3ds_ztensor(
        1, S, 1, &sum_row_zt, &sum_row_pre, &sum_row_trans, NULL);
    if (status == ZDNN_OK) {
      memset(sum_row_zt.buffer, 0, sum_row_zt.buffer_size);
      sum_row_zt.is_transformed = true;
    }
  }

  // --- Create helper [1, S, 1] tensors (uninitialized, reused) ---
  if (status == ZDNN_OK)
    status =
        create_3ds_ztensor(1, S, 1, &old_max_zt, &om_pre, &om_trans, NULL);
  if (status == ZDNN_OK)
    status =
        create_3ds_ztensor(1, S, 1, &block_max_zt, &bm_pre, &bm_trans, NULL);
  if (status == ZDNN_OK)
    status =
        create_3ds_ztensor(1, S, 1, &correction_zt, &cor_pre, &cor_trans, NULL);
  if (status == ZDNN_OK)
    status = create_3ds_ztensor(
        1, S, 1, &tile_rowsum_zt, &trs_pre, &trs_trans, NULL);

  // --- Create scratch_tile [1, S, T] ---
  if (status == ZDNN_OK)
    status =
        create_3ds_ztensor(1, S, T, &scratch_tile, &sc_pre, &sc_trans, NULL);

  // --- Create zero bias for sum matmul: 2DS{1, 1} ---
  if (status == ZDNN_OK)
    status = create_2ds_zero_bias(1, 1, &bias_1, &b1_pre, &b1_trans);

  // --- Run two-pass algorithm ---
  if (status == ZDNN_OK) {
    zdnnx_split_info si_x, si_y, si_z, si_bias, si_out;
    zdnnx_prepare_split_info(&si_x, X, 1, 0, 0, 0, "LargeS X");
    zdnnx_prepare_split_info(&si_y, Y, 1, 0, 0, T, "LargeS Y");
    zdnnx_prepare_split_info(&si_z, Z, 1, 0, 0, T, "LargeS Z");
    zdnnx_prepare_split_info(&si_bias, Bias, 1, 0, 0, T, "LargeS Bias");
    zdnnx_prepare_split_info(&si_out, output, 1, 0, 0, T, "LargeS Out");

    zdnnx_tile tx, ty, tz, tbias, tout;

    // Loop over batch elements (E4 tiling).
    for (uint32_t b = 0; b < BH && status == ZDNN_OK; ++b) {
      zdnnx_set_tile(&si_x, &tx, NULL, b, 0, 0, 0);

      // Reinitialize running statistics for this batch element.
      memcpy(max_row_zt.buffer, max_row_init_buf, max_row_zt.buffer_size);
      memset(sum_row_zt.buffer, 0, sum_row_zt.buffer_size);

      // Pass 1: Forward scan — compute scores, store exp, accumulate stats.
      // After this pass:
      //   out_c     = exp(scores_c - M_c)  (stored in output tiles)
      //   max_row   = M_final              (global max across all tiles)
      //   sum_row   = D = sum_c row_sum(exp(scores_c - M_final))
      //   max_snapshots[c] = M_c           (FP32 snapshot for Pass 2)
      for (uint32_t c = 0; c < N_t && status == ZDNN_OK; ++c) {
        zdnnx_set_tile(&si_y, &ty, NULL, b, 0, 0, c);
        zdnnx_set_tile(&si_z, &tz, NULL, b, 0, 0, c);
        zdnnx_set_tile(&si_bias, &tbias, NULL, b, 0, 0, c);
        zdnnx_set_tile(&si_out, &tout, NULL, b, 0, 0, c);

        // 1. scores_c = X * Y_c + Z_c  [S, T]
        status = zdnn_matmul_op(
            &tx.data, &ty.data, &tbias.data, MATMUL_OP_ADDITION, &scratch_tile);
        if (status == ZDNN_OK)
          status = zdnn_add(&scratch_tile, &tz.data, &scratch_tile);

        // 2. block_max = row-wise max of scores_c  [S, 1]
        if (status == ZDNN_OK)
          status = zdnn_reduce(
              &scratch_tile, NULL, REDUCE_OP_MAXIMUM, &block_max_zt);

        // 3. Update running max: M_c = max(M_{c-1}, block_max)
        if (status == ZDNN_OK) {
          memcpy(
              old_max_zt.buffer, max_row_zt.buffer, max_row_zt.buffer_size);
          status = zdnn_max(&max_row_zt, &block_max_zt, &max_row_zt);
        }

        // 4. Rescale running sum: D = D * exp(M_{c-1} - M_c)
        if (status == ZDNN_OK)
          status = zdnn_sub(&old_max_zt, &max_row_zt, &correction_zt);
        if (status == ZDNN_OK)
          status = zdnn_exp(&correction_zt, &correction_zt);
        if (status == ZDNN_OK)
          status = zdnn_mul(&sum_row_zt, &correction_zt, &sum_row_zt);

        // 5. out_c = exp(scores_c - M_c)  [S, T]
        if (status == ZDNN_OK)
          broadcast_column_to_tile(&max_row_zt, &tout.data);
        if (status == ZDNN_OK)
          status = zdnn_sub(&scratch_tile, &tout.data, &tout.data);
        if (status == ZDNN_OK)
          status = zdnn_exp(&tout.data, &tout.data);

        // 6. D = D + row_sum(out_c)
        if (status == ZDNN_OK)
          status = zdnn_matmul_op(&tout.data, &ones_zt, &bias_1,
              MATMUL_OP_ADDITION, &tile_rowsum_zt);
        if (status == ZDNN_OK)
          status = zdnn_add(&sum_row_zt, &tile_rowsum_zt, &sum_row_zt);

        // 7. Snapshot M_c as FP32 for Pass 2.
        if (status == ZDNN_OK)
          status = zdnn_transform_origtensor(&max_row_zt, fp32_buf);
        if (status == ZDNN_OK)
          memcpy(
              &max_snapshots[(uint64_t)c * S], fp32_buf, S * sizeof(float));
      }

      // Pass 2: Backward correction — adjust each tile from its local max
      // M_c to the global max M_final, and divide by the global sum D.
      //   out_c = out_c * exp(M_c - M_final) / D
      //         = exp(scores_c - M_final) / D   (exact softmax)
      for (uint32_t c = 0; c < N_t && status == ZDNN_OK; ++c) {
        zdnnx_set_tile(&si_out, &tout, NULL, b, 0, 0, c);

        // 1. Reload M_c from FP32 snapshot.
        memcpy(fp32_buf, &max_snapshots[(uint64_t)c * S], S * sizeof(float));
        correction_zt.is_transformed = false;
        status = zdnn_transform_ztensor(&correction_zt, fp32_buf);

        // 2. correction = exp(M_c - M_final) / D  [S, 1]
        if (status == ZDNN_OK)
          status = zdnn_sub(&correction_zt, &max_row_zt, &correction_zt);
        if (status == ZDNN_OK)
          status = zdnn_exp(&correction_zt, &correction_zt);
        if (status == ZDNN_OK)
          status = zdnn_div(&correction_zt, &sum_row_zt, &correction_zt);

        // 3. out_c = out_c * broadcast(correction, [S, T])
        if (status == ZDNN_OK)
          broadcast_column_to_tile(&correction_zt, &scratch_tile);
        if (status == ZDNN_OK)
          status = zdnn_mul(&tout.data, &scratch_tile, &tout.data);
      }
    }
  }

  // --- Cleanup all resources ---
  free_ztensor_buf(&bias_1);
  free_ztensor_buf(&scratch_tile);
  free_ztensor_buf(&tile_rowsum_zt);
  free_ztensor_buf(&correction_zt);
  free_ztensor_buf(&block_max_zt);
  free_ztensor_buf(&old_max_zt);
  free_ztensor_buf(&sum_row_zt);
  free(max_row_init_buf);
  free_ztensor_buf(&max_row_zt);
  free_ztensor_buf(&ones_zt);
  free(max_snapshots);
  free(fp32_buf);
  return status;
}

zdnn_status zdnnx_seq_matmul_add_softmax(const zdnn_ztensor *X,
    const zdnn_ztensor *Y, const zdnn_ztensor *Z, const zdnn_ztensor *Bias,
    zdnn_ztensor *work, zdnn_ztensor *output) {
#ifdef ZDNNX_DEBUG
  printf("[MatMulAddSoftmax]\n");
#endif

  uint32_t S = zdnnx_get_transformed_dim(X, E2);
  if (S <= 2048)
    return matmul_add_softmax_small_s(X, Y, Z, Bias, work, output);
  return matmul_add_softmax_large_s(X, Y, Z, Bias, work, output);
}
