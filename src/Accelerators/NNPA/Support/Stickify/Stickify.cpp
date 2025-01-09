/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- stickify.cpp - Data Stickify ---------------------------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains functions for stickifying data tensors.
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <fenv.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Convert.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Stickify.hpp"

#ifdef __MVS__
#pragma export(zdnn_get_library_version_str)
#pragma export(zdnn_get_library_version)
#endif

/// Verify the transformed descriptor
zdnn_status verify_transformed_descriptor(const zdnn_tensor_desc *tfrmd_desc);

zdnn_status set_zdnn_status(zdnn_status status, const char *func_name,
    const char *file_name, int line_no, const char *format, ...);

#define ZDNN_STATUS(status, format, ...)                                       \
  set_zdnn_status(status, __func__, __FILE__, __LINE__, format, __VA_ARGS__)

#define ZDNN_STATUS_NO_MSG(status) ZDNN_STATUS(status, NULL, NO_ARG)
#ifndef ZDNN_CONFIG_DEBUG
#define ZDNN_STATUS_OK ZDNN_OK
#else
#define ZDNN_STATUS_OK ZDNN_STATUS_NO_MSG(ZDNN_OK)
#endif

/// Macros from third_party/zdnn-lib/zdnn/zdnn_private.h

#define AIU_BYTES_PER_STICK 128
#define AIU_1BYTE_CELLS_PER_STICK 128
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_4BYTE_CELLS_PER_STICK 32
#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

#define ZDNN_MAX_DIMS 4 // number of dims in AIU's Tensor Descriptor

// From status.c
// maximum size for the format string, including the prepended STATUS_STR_XXX
#define MAX_STATUS_FMTSTR_SIZE 1024

// -----------------------------------------------------------------------------
// Misc Macros
// -----------------------------------------------------------------------------
#define CEIL(a, b)                                                             \
  static_cast<uint64_t>(((a) + (b)-1) / (b)) // positive numbers only
#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define BIT_SIZEOF(a) (sizeof(a) * 8)

// padded = next multiple of AIU_2BYTE_CELLS_PER_STICK
#define PADDED(x)                                                              \
  (static_cast<uint32_t>(CEIL((x), AIU_2BYTE_CELLS_PER_STICK)) *               \
      AIU_2BYTE_CELLS_PER_STICK)
#define ZDNN_STATUS_OK ZDNN_OK

// From zdnn_private.h
typedef enum elements_mode {
  ELEMENTS_AIU,
  ELEMENTS_PRE,
  ELEMENTS_PRE_SINGLE_GATE = ELEMENTS_PRE,
  ELEMENTS_PRE_ALL_GATES
} elements_mode;

// End - Macros from third_party/zdnn-lib/zdnn/zdnn_private.h

// Functions from third_party/zdnn-lib/zdnn/status.h
zdnn_status set_zdnn_status(zdnn_status status, const char *func_name,
    const char *file_name, int line_no, const char *format, ...) {
  return status;
}

// Functions from third_party/zdnn-lib/zdnn/get.c
#define DECLARE_DATA_LAYOUT_STR(a) static const char *DATA_LAYOUT_STR_##a = #a;

// const char *DATA_LAYOUT_STR_X
DECLARE_DATA_LAYOUT_STR(ZDNN_1D)
DECLARE_DATA_LAYOUT_STR(ZDNN_2D)
DECLARE_DATA_LAYOUT_STR(ZDNN_2DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_3D)
DECLARE_DATA_LAYOUT_STR(ZDNN_3DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_ZRH)
DECLARE_DATA_LAYOUT_STR(ZDNN_4D)
DECLARE_DATA_LAYOUT_STR(ZDNN_4DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_NHWC)
DECLARE_DATA_LAYOUT_STR(ZDNN_NCHW)
DECLARE_DATA_LAYOUT_STR(ZDNN_FICO)
DECLARE_DATA_LAYOUT_STR(ZDNN_HWCK)
DECLARE_DATA_LAYOUT_STR(ZDNN_BIDIR_ZRH)
DECLARE_DATA_LAYOUT_STR(ZDNN_BIDIR_FICO)

#define DECLARE_DATA_FORMAT_STR(a) static const char *DATA_FORMAT_STR_##a = #a;

// const char *DATA_FORMAT_STR_X
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DFEATURE)
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DKERNEL)

static short get_data_layout_num_gates(zdnn_data_layouts layout) {

#define CASE_RTN_GATES(a, b)                                                   \
  case (a):                                                                    \
    return (b);

  switch (layout) {
    CASE_RTN_GATES(ZDNN_BIDIR_ZRH, 3);
    CASE_RTN_GATES(ZDNN_BIDIR_FICO, 4);
    CASE_RTN_GATES(ZDNN_ZRH, 3);
    CASE_RTN_GATES(ZDNN_FICO, 4);
  default:
    return 0;
  }
#undef CASE_RTN_GATES
}

static short get_data_layout_dims(zdnn_data_layouts layout) {

#define CASE_RTN_DIM(a, b)                                                     \
  case (a):                                                                    \
    return (b);

  switch (layout) {
    CASE_RTN_DIM(ZDNN_1D, 1);
    CASE_RTN_DIM(ZDNN_2D, 2);
    CASE_RTN_DIM(ZDNN_2DS, 2);
    CASE_RTN_DIM(ZDNN_3D, 3);
    CASE_RTN_DIM(ZDNN_3DS, 3);
    CASE_RTN_DIM(ZDNN_4D, 4);
    CASE_RTN_DIM(ZDNN_NHWC, 4);
    CASE_RTN_DIM(ZDNN_NCHW, 4);
    CASE_RTN_DIM(ZDNN_HWCK, 4);
  default:
    return 0;
  }
#undef CASE_RTN_DIM
}

uint32_t get_rnn_concatenated_dim1(uint32_t val, zdnn_concat_info info) {
  if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
    return PADDED(val) * 4;
  } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
    return PADDED(val) * 3;
  } else {
    return val;
  }
}

uint32_t get_rnn_concatenated_dim2(uint32_t val, zdnn_concat_info info) {
  // the only case we need vertical concatenation is when a weight tensor is
  // used with bidir output from the previous layer.
  if (CONCAT_USAGE(info) == USAGE_WEIGHTS &&
      CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR) {
    return PADDED(val / 2) * 2;
  } else {
    return val;
  }
}

short get_func_code_num_gates(nnpa_function_code func_code) {

#define CASE_RTN_GATES(a, b)                                                   \
  case (a):                                                                    \
    return get_data_layout_num_gates(b); // piggyback thus no need to hardcode

  switch (func_code) {
    CASE_RTN_GATES(NNPA_LSTMACT, ZDNN_FICO);
    CASE_RTN_GATES(NNPA_GRUACT, ZDNN_ZRH);
  default:
    return 0;
  }
#undef CASE_RTN_GATES
}

const char *get_data_layout_str(zdnn_data_layouts layout) {

#define CASE_RTN_STR(a)                                                        \
  case (a):                                                                    \
    return DATA_LAYOUT_STR_##a;

  switch (layout) {
    CASE_RTN_STR(ZDNN_1D);
    CASE_RTN_STR(ZDNN_2D);
    CASE_RTN_STR(ZDNN_2DS);
    CASE_RTN_STR(ZDNN_3D);
    CASE_RTN_STR(ZDNN_3DS);
    CASE_RTN_STR(ZDNN_ZRH);
    CASE_RTN_STR(ZDNN_4D);
    CASE_RTN_STR(ZDNN_4DS);
    CASE_RTN_STR(ZDNN_NHWC);
    CASE_RTN_STR(ZDNN_NCHW);
    CASE_RTN_STR(ZDNN_FICO);
    CASE_RTN_STR(ZDNN_HWCK);
    CASE_RTN_STR(ZDNN_BIDIR_ZRH);
    CASE_RTN_STR(ZDNN_BIDIR_FICO);
  }
#undef CASE_RTN_STR

  llvm_unreachable("Invalid data layout");
}

const char *get_data_format_str(zdnn_data_formats format) {

#define CASE_RTN_STR(a)                                                        \
  case (a):                                                                    \
    return DATA_FORMAT_STR_##a;

  switch (format) {
    CASE_RTN_STR(ZDNN_FORMAT_4DFEATURE);
    CASE_RTN_STR(ZDNN_FORMAT_4DKERNEL);
  }
#undef CASE_RTN_STR

  llvm_unreachable("Invalid data format");
}

short get_data_type_size(zdnn_data_types type) {

#define CASE_RTN_SIZE(a, b)                                                    \
  case (a):                                                                    \
    return (b);

  switch (type) {
    CASE_RTN_SIZE(BFLOAT, 2);
    CASE_RTN_SIZE(FP16, 2);
    CASE_RTN_SIZE(FP32, 4);
    CASE_RTN_SIZE(ZDNN_DLFLOAT16, 2);
    CASE_RTN_SIZE(INT8, 1);
  }
#undef CASE_RTN_SIZE

  llvm_unreachable("Invalid data type");
} // End - Functions from third_party/zdnn-lib/zdnn/get.c

// Functions from third_party/zdnn-lib/zdnn/malloc4k.c
void *malloc_aligned_4k(size_t size) {

  // request one more page + size of a pointer from the OS
  unsigned short extra_allocation =
      (AIU_PAGESIZE_IN_BYTES - 1) + sizeof(void *);

  // make sure size is reasonable
  if (!size || size > SIZE_MAX) {
    return NULL;
  }

  void *ptr = malloc(size + extra_allocation);
  if (!ptr) {
    perror("Error during malloc");
    fprintf(stderr, "errno = %d\n", errno);
    return ptr;
  }

  // find the 4k boundary after ptr
  void *aligned_ptr = reinterpret_cast<void *>(
      ((reinterpret_cast<uintptr_t>(ptr) + extra_allocation) &
          ~(AIU_PAGESIZE_IN_BYTES - 1)));
  // put the original malloc'd address right before aligned_ptr
  (static_cast<void **>(aligned_ptr))[-1] = ptr;

  return aligned_ptr;
}

void free_aligned_4k(void *aligned_ptr) {
  if (aligned_ptr) {
    // get the original malloc'd address from where we put it and free it
    void *original_ptr = (static_cast<void **>(aligned_ptr))[-1];
    free(original_ptr);
  }
}
// End - Functions from third_party/zdnn-lib/zdnn/malloc4k.c

// Functions from third_party/zdnn-lib/zdnn/utils.c
uint64_t get_num_elements(const zdnn_ztensor *ztensor, elements_mode mode) {
  uint64_t num_elements = 1;
  uint32_t *dims_ptr = NULL;
  int i = 0;

  // Setup how to loop over the shape based on the mode.
  switch (mode) {
  case ELEMENTS_AIU:
    // tfrmd_desc shape accounts for all elements including both concat
    // horizontal and vertical paddings.
    dims_ptr = &(ztensor->transformed_desc->dim4);
    // Loop over all dims since tfrmd_dec sets any "unused" dimensions to 1.
    i = 0;
    break;
  case ELEMENTS_PRE: // = ELEMENTS_PRE_SINGLE_GATE
  case ELEMENTS_PRE_ALL_GATES:
    // Use pre_tfrmd_desc as we document that should be the shape of a single
    // horizontal-concat (or gate) and not the combined shape.
    dims_ptr = &(ztensor->pre_transformed_desc->dim4);
    // Loop will start at outermost dimension we expect for the layout.
    // For example: 2D gets dim2 and dim1. 3D gets dim3, dim2, and dim1.
    i = ZDNN_MAX_DIMS -
        get_data_layout_dims(ztensor->pre_transformed_desc->layout);
    break;
  }

  // Multiply by the size of each expected dimension
  for (; i < ZDNN_MAX_DIMS; i++) {
    num_elements *= static_cast<uint64_t>(dims_ptr[i]);
  }

  if (mode == ELEMENTS_PRE_ALL_GATES) {
    // this will cause the function to return 0 if there's no gates to speak of
    num_elements *=
        get_data_layout_num_gates(ztensor->transformed_desc->layout);
  }
  return num_elements;
} // End - Functions from third_party/zdnn-lib/zdnn/utils.c

// Functions from third_party/zdnn-lib/zdnn/allochelper.c
uint64_t getsize_ztensor(const zdnn_tensor_desc *tfrmd_desc) {
  uint32_t cells_per_stick;
  uint32_t number_of_sticks;
  switch (tfrmd_desc->type) {
  case ZDNN_BINARY_INT8:
    if (tfrmd_desc->format == ZDNN_FORMAT_4DWEIGHTS) {
      // 4DWEIGHTS has two vectors interleaved, therefore only 64 cells vs 128
      // Due to this interleaving, number_of_sticks is halved, but must be
      // rounded up to stay even for proper interleaving.
      cells_per_stick = AIU_2BYTE_CELLS_PER_STICK;
      number_of_sticks = CEIL(tfrmd_desc->dim2, 2);
    } else {
      cells_per_stick = AIU_1BYTE_CELLS_PER_STICK;
      number_of_sticks = tfrmd_desc->dim2;
    }
    break;
  case ZDNN_BINARY_INT32:
    cells_per_stick = AIU_4BYTE_CELLS_PER_STICK;
    number_of_sticks = tfrmd_desc->dim2;
    break;
  case ZDNN_DLFLOAT16: /* fallthrough */
  default:
    cells_per_stick = AIU_2BYTE_CELLS_PER_STICK;
    number_of_sticks = tfrmd_desc->dim2;
  }
  return static_cast<uint64_t>(tfrmd_desc->dim4) * tfrmd_desc->dim3 *
         CEIL(number_of_sticks, AIU_STICKS_PER_PAGE) *
         CEIL(tfrmd_desc->dim1, cells_per_stick) * AIU_PAGESIZE_IN_BYTES;
}

zdnn_status allochelper_ztensor_alloc(zdnn_ztensor *ztensor) {
  uint64_t size;
  zdnn_status status;

  // only the information in transformed_desc matters, so make sure it's
  // reasonable
  if ((status = verify_transformed_descriptor(ztensor->transformed_desc)) !=
      ZDNN_OK) {
    return status;
  }

  // get the size and allocate space aligned on a 4k boundary. If the malloc
  // fails, return error.
  size = getsize_ztensor(ztensor->transformed_desc); // Modified
  if (!(ztensor->buffer = malloc_aligned_4k(size))) {
    return ZDNN_ALLOCATION_FAILURE;
  }

  // With a successful malloc, set our ztensor's buffer_size with the allocated
  // size.
  ztensor->buffer_size = size;

  // Successful zdnn_allochelper_ztensor call
  return ZDNN_STATUS_OK;
}

void allochelper_ztensor_free(zdnn_ztensor *ztensor) {
  free_aligned_4k(ztensor->buffer);
  ztensor->buffer = NULL;
  ztensor->buffer_size = 0;
}

/* End - Functions from third_party/zdnn-lib/zdnn/allochelper.c */

// Functions from third_party/zdnn-lib/zdnn/tensor_desc.c
zdnn_status verify_pre_transformed_descriptor(
    const zdnn_tensor_desc *pre_tfrmd_desc) {

  // is the layout valid as pre-transformed?
  switch (pre_tfrmd_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
  case ZDNN_NCHW:
  case ZDNN_HWCK:
    // all of these are good cases
    break;
  default:
    return ZDNN_INVALID_LAYOUT;
  }

  // is data type valid as pre-transformed?
  switch (pre_tfrmd_desc->type) {
  case BFLOAT:
  case FP16:
  case FP32:
  case INT8:
    // all of these are good cases
    break;
  default:
    return ZDNN_INVALID_TYPE;
  }

  return ZDNN_STATUS_OK;
}

zdnn_status verify_transformed_descriptor(const zdnn_tensor_desc *tfrmd_desc) {

  // First, format must be valid (defined in the enum)
  // Then if format doesn't agree with layout, we declare format is correct and
  // layout is wrong (in reality, either can be wrong, but we have to pick one)
  switch (tfrmd_desc->format) {
  case ZDNN_FORMAT_4DFEATURE:
    switch (tfrmd_desc->layout) {
    case ZDNN_NHWC:
    case ZDNN_FICO:
    case ZDNN_ZRH:
    case ZDNN_BIDIR_FICO:
    case ZDNN_BIDIR_ZRH:
      break;
    default:
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Format is %s but layout is %s",
          get_data_format_str(tfrmd_desc->format),
          get_data_layout_str(tfrmd_desc->layout));
    }
    break;
  case ZDNN_FORMAT_4DKERNEL:
    if (tfrmd_desc->layout != ZDNN_HWCK) {
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Format is %s but layout is %s",
          get_data_format_str(tfrmd_desc->format),
          get_data_layout_str(tfrmd_desc->layout));
    }
    break;
  case ZDNN_FORMAT_4DWEIGHTS:
    if (tfrmd_desc->layout != ZDNN_NHWC) {
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Format is %s but layout is %s",
          get_data_format_str(tfrmd_desc->format),
          get_data_layout_str(tfrmd_desc->layout));
    }
    break;
  default:
    // unrecognized
    return ZDNN_STATUS(ZDNN_INVALID_FORMAT, "Invalid format: %d (%s)",
        tfrmd_desc->format, get_data_format_str(tfrmd_desc->format));
  }
  // Only ZDNN_DLFLOAT16, ZDNN_BINARY_INT8, and ZDNN_BINARY_INT32 are currently
  // supported.
  if (tfrmd_desc->type != ZDNN_DLFLOAT16 &&
      tfrmd_desc->type != ZDNN_BINARY_INT8 &&
      tfrmd_desc->type != ZDNN_BINARY_INT32) {
    return ZDNN_INVALID_TYPE;
  }

  const uint32_t *dims_ptr = &(tfrmd_desc->dim4);

  /* ToFix: the nnpa_query_result is not set up with onnx-mlir
   * Temporarily commented out.
   * Refer to issue #3034
   */

#if 0
  // is the dimension above the limit or zero?
  // transformed layout uses all dim* entries, so we'll check them all
  for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
    if (!dims_ptr[i] || dims_ptr[i] > NNPAGetMaxForDim(i, ZDNN_MAX_DIMS)) {
      return ZDNN_INVALID_SHAPE;
    }
   if (dims_ptr[i] > zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i)) {

      if (!zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i)) {
        return ZDNN_UNSUPPORTED_AIU_EXCEPTION;
      } else {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "Invalid shape for dim%d. (reason: dimension value %d exceeds %d)",
            ZDNN_MAX_DIMS - i, dims_ptr[i],
            zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i));
      }
    }
  }

  // is stick area size above the limit?
  if (getsize_ztensor(tfrmd_desc) > zdnn_get_nnpa_max_tensor_size()) {
    return ZDNN_INVALID_SHAPE;
  }
#endif

  return ZDNN_STATUS_OK;
}

zdnn_status generate_transformed_desc(
    const zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc) {

  zdnn_status status;

  // modify tfrmd_desc only if layout is supported, else leave it untouched

  switch (pre_tfrmd_desc->layout) {
  case (ZDNN_1D):
    // shape (a) -> dims4-1 (1, 1, 1, a)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = 1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_2D):
    // shape (a, b) -> dims4-1 (1, 1, a, b)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_2DS):
    // shape (a, b) -> dims4-1 (a, 1, 1, b)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = 1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_3D):
    // shape (a, b, c) -> dims4-1 (1, a, b, c)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_3DS):
    // shape (a, b, c) -> dims4-1 (a, 1, b, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_4D):
  case (ZDNN_NHWC):
    // shape (a, b, c, d) -> dims4-1 (a, b, c, d)
    // shape (n, h, w, c) -> dims4-1 (n, h, w, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_4DS):
    // ZDNN_4DS is used exclusively as RNN output
    // shape (a, b, c, d)  -> ZDNN_NHWC
    //   when b = 1 (uni-dir)     -> dims4-1 (a, 1, c, d)
    //   otherwise (bi-dir, etc.) -> dims4-1 (a, 1, c, b * PADDED(d))
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    if (pre_tfrmd_desc->dim3 == 1) {
      tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    } else {
      // so when dim3 is 0 for whatever reason, tfrmd_desc->dim1 will become 0
      // and will fail transform-desc check later
      tfrmd_desc->dim1 = pre_tfrmd_desc->dim3 * PADDED(pre_tfrmd_desc->dim1);
    }
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_NCHW):
    // shape (n, c, h, w) -> dims4-1 (n, h, w, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim3;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case ZDNN_HWCK:
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_HWCK;
    tfrmd_desc->format = ZDNN_FORMAT_4DKERNEL;
    status = ZDNN_OK;
    break;
  default:
    return ZDNN_INVALID_LAYOUT;
    break;
  }

  if (status == ZDNN_OK) {
    tfrmd_desc->type = ZDNN_DLFLOAT16;
  }

  return status;
}

zdnn_status generate_quantized_transformed_desc(
    const zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_quantized_transform_types transform_type,
    zdnn_tensor_desc *tfrmd_desc) {

  zdnn_status status;
  if ((status = generate_transformed_desc(pre_tfrmd_desc, tfrmd_desc)) !=
      ZDNN_OK) {
    return status;
  }
  switch (transform_type) {
  case QUANTIZED_DLFLOAT16:
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    tfrmd_desc->type = ZDNN_DLFLOAT16;
    return ZDNN_STATUS_OK;
  case QUANTIZED_INT8:
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    tfrmd_desc->type = ZDNN_BINARY_INT8;
    return ZDNN_STATUS_OK;
  case QUANTIZED_WEIGHTS_INT8:
    tfrmd_desc->format = ZDNN_FORMAT_4DWEIGHTS;
    tfrmd_desc->type = ZDNN_BINARY_INT8;
    return ZDNN_STATUS_OK;
  default:
    return ZDNN_INVALID_TRANSFORM_TYPE;
    // return ZDNN_STATUS(ZDNN_INVALID_TRANSFORM_TYPE,
    //                    "Invalid transform type: %d", transform_type);
  }
}

zdnn_status generate_transformed_desc_concatenated(
    const zdnn_tensor_desc *pre_tfrmd_desc, zdnn_concat_info info,
    zdnn_tensor_desc *tfrmd_desc) {

  if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) &&
      (CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR)) {
    // dim2 can't be odd number
    if (pre_tfrmd_desc->dim2 & 1) {
      return ZDNN_INVALID_SHAPE;
    }
  }

  // Two kinds of concatenations we need to deal with:
  //
  // - (Hidden-)Weights, (hidden)-biases need to be concatenated horizontally,
  //   new dim1 is calculated via get_rnn_concatenated_dim1()
  //
  // - Weights may need to be concatenated vertically also (when output
  //   from the previous bidir layer is the input), new dim2 is calculated via
  //   get_rnn_concatenated_dim2()

  if ((CONCAT_USAGE(info) == USAGE_BIASES) ||
      (CONCAT_USAGE(info) == USAGE_HIDDEN_BIASES)) {
    if (pre_tfrmd_desc->layout == ZDNN_2DS) {
      tfrmd_desc->dim4 = pre_tfrmd_desc->dim2;
      tfrmd_desc->dim3 = 1;
      tfrmd_desc->dim2 = 1;
      tfrmd_desc->dim1 = get_rnn_concatenated_dim1(pre_tfrmd_desc->dim1, info);
    } else {
      return ZDNN_INVALID_LAYOUT;
    }
  } else if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) ||
             (CONCAT_USAGE(info) == USAGE_HIDDEN_WEIGHTS)) {
    if (pre_tfrmd_desc->layout == ZDNN_3DS) {
      tfrmd_desc->dim4 = pre_tfrmd_desc->dim3;
      tfrmd_desc->dim3 = 1;
      tfrmd_desc->dim2 = get_rnn_concatenated_dim2(pre_tfrmd_desc->dim2, info);
      tfrmd_desc->dim1 = get_rnn_concatenated_dim1(pre_tfrmd_desc->dim1, info);
    } else {
      return ZDNN_INVALID_LAYOUT;
    }
  } else {
    return ZDNN_INVALID_CONCAT_INFO;
  }

  // if USAGE is WEIGHTS and PREV_LAYER is BIDIR then
  // ZDNN_BIDIR_FICO/ZDNN_BIDIR_ZRH
  //
  // everything else ZDNN_FICO/ZDNN_ZRH

  if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) &&
      (CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR)) {
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      tfrmd_desc->layout = ZDNN_BIDIR_FICO;
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      tfrmd_desc->layout = ZDNN_BIDIR_ZRH;
    } else {
      return ZDNN_INVALID_CONCAT_INFO;
    }
  } else {
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      tfrmd_desc->layout = ZDNN_FICO;
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      tfrmd_desc->layout = ZDNN_ZRH;
    } else {
      return ZDNN_INVALID_CONCAT_INFO;
    }
  }

  tfrmd_desc->type = ZDNN_DLFLOAT16;
  tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;

  return ZDNN_STATUS_OK;
} // End - Functions from third_party/zdnn-lib/zdnn/tensor_desc.c

// Functions from third_party/zdnn-lib/zdnn/init_ztensor.c
void init_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_tensor_desc *tfrmd_desc, zdnn_ztensor *output) {

  output->pre_transformed_desc = pre_tfrmd_desc;
  output->transformed_desc = tfrmd_desc;
  output->is_transformed = false;
  memset(&output->reserved, 0, sizeof(output->reserved));
  output->rec_scale = 0;
  output->offset = 0;
  memset(&output->reserved2, 0, sizeof(output->reserved2));
} // End - Functions from third_party/zdnn-lib/zdnn/init_ztensor.c

// Functions from third_party/zdnn-lib/zdnn/stickify.c
/// Main entry point for converting FP32 <-> ZDNN_DLFLOAT16 when the entries to
/// fetch/set on FP32 side are contiguous (e.g., fetching the c-entries in a
/// NHWC stream).
///
/// \param[in] input_data Pointer to input tensor data stream
/// \param[in] in_data_fmt Input tensor stream data format
/// \param[in] output_data Pointer to output tensor data stream
/// \param[in] out_data_fmt Output tensor stream data format
/// \param[in] num_fields Number of fields to convert
///
/// \return Number of fields converted, or 0 if error
///
uint32_t convert_data_format(void *input_data, zdnn_data_types in_data_fmt,
    void *output_data, zdnn_data_types out_data_fmt, uint32_t num_fields) {

  uint64_t num_fields_converted = 0;

  // we only care convert to/from ZDNN_DLFLOAT16
  if (out_data_fmt == ZDNN_DLFLOAT16) {
    switch (in_data_fmt) {
    case FP32:
      num_fields_converted = fp32_to_dlf16(static_cast<float *>(input_data),
          static_cast<uint16_t *>(output_data), num_fields);
      break;
    default:
      break; // something really wrong, get out and return 0
      return 0;
    }
  } else if (in_data_fmt == ZDNN_DLFLOAT16) {
    switch (out_data_fmt) {
    case FP32:
      num_fields_converted = dlf16_to_fp32(static_cast<uint16_t *>(input_data),
          static_cast<float *>(output_data), num_fields);
      break;
    default:
      break; // something really wrong, get out and return 0
      return 0;
    }
  } else {
    // something really wrong
    return 0;
  }
  return num_fields_converted;
} // End convert_data_format

zdnn_status handle_fp_errors(int fe) {
  if (fe & FE_UNDERFLOW) {
    // no error externalized
  }
  if ((fe & FE_INVALID) || (fe & FE_OVERFLOW)) {
    return ZDNN_CONVERT_FAILURE;
  }
  if (fe & FE_INEXACT) {
    return ZDNN_CONVERT_FAILURE;
  }

  return ZDNN_STATUS_OK;
}

/// The actual routine for stickification, only does the following:
///    NHWC -> NHWC, NCHW -> NHWC, HWCK -> HWCK
/// Does NOT handle concatenated types.
///
/// \param[in] in_buf data buffer to be stickified
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///
zdnn_status transform_ztensor(const void *in_buf, zdnn_ztensor *ztensor) {
  uint64_t input_offset =
      0; // moving position as the input is processed, in BYTES
  uint64_t output_offset =
      0; // moving position as the output is processed, in BYTES
  short input_cell_size =
      get_data_type_size(ztensor->pre_transformed_desc->type);
  short input_cell_shift = input_cell_size / 2;

  /*
   * Stores the vector operation output directly into the stick_area.  This
   * reduces the number of inefficient loops.
   */
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  feclearexcept(
      FE_ALL_EXCEPT); /* clear exception flags set during conversion */

  if (ztensor->transformed_desc->layout == ZDNN_NHWC) {

    // Expected layout is NHWC, stickify normally. Requires a single data
    // buffer.

    // loop invariant values
    uint64_t bytes_all_h =
        static_cast<uint64_t>(ztensor->transformed_desc->dim3) *
        CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
        AIU_PAGESIZE_IN_BYTES;
    uint64_t bytes_per_n = bytes_all_h * CEIL(ztensor->transformed_desc->dim1,
                                             AIU_2BYTE_CELLS_PER_STICK);

    if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW) {

      // N
      for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

        // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
        uint64_t out_offset_n = output_offset;

        // H
        for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

          // W
          for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
#if defined(__MVS__)
            __dcbt(reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(in_buf) + input_offset));
#else
            __builtin_prefetch(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                0);
#endif
            // used for pushing out_offset from w to w+1 (i.e., +
            // AIU_BYTES_PER_STICK)
            uint64_t out_offset_w = output_offset;

            // process each C-stick (i.e., every 64 elements or whatever
            // left in dim1)
            for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
                 e1x += AIU_2BYTE_CELLS_PER_STICK) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
#if defined(__MVS__)
              __dcbtst(reinterpret_cast<void *>(
                  reinterpret_cast<uintptr_t>(ztensor->buffer) +
                  output_offset));
#else
              __builtin_prefetch(
                  reinterpret_cast<void *>(
                      reinterpret_cast<uintptr_t>(ztensor->buffer) +
                      output_offset),
                  1);
#endif
              fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                  AIU_2BYTE_CELLS_PER_STICK);

              nbr_fields_converted = convert_data_format(
                  reinterpret_cast<void *>(
                      reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                  ztensor->pre_transformed_desc->type,
                  reinterpret_cast<void *>(
                      reinterpret_cast<uintptr_t>(ztensor->buffer) +
                      output_offset),
                  ztensor->transformed_desc->type, fields_to_convert);

              if (nbr_fields_converted == 0) {
                return ZDNN_CONVERT_FAILURE;
              }

              // Release L1 cacheline for stick. The next "touch" will be
              // from NNPA, and it doesn't need L1 caching.
#if defined(__MVS__)
              __dcbf(reinterpret_cast<void *>(
                  reinterpret_cast<uintptr_t>(ztensor->buffer) +
                  output_offset));
#else
// No known equivalent fn without dropping to ASM....
#endif
              // push input_offset the next c-stick, fake the multiply by
              // bit-shifting
              input_offset += (nbr_fields_converted << input_cell_shift);

              // push output_offset to the next c-stick of the same super
              // c-stick, which is bytes_all_h number of bytes away.
              output_offset += bytes_all_h;
            }

            // output_offset was pushed around in dim1 loops, so reset it to
            // the next w
            output_offset = out_offset_w + AIU_BYTES_PER_STICK;
          }

          // after processing all the w-entries, go to the next 4k-boundary
          // location (aka stick padding)
          output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                          (-AIU_PAGESIZE_IN_BYTES);
        }

        // output_offset was pushed around in the dims[2-0] loops, so reset it
        // to the next n
        output_offset = out_offset_n + bytes_per_n;
      }

    } else { // NCHW

      uint8_t sizeof_dlf16 = get_data_type_size(ZDNN_DLFLOAT16);

      // process the entire W number of entries at every pass
      fields_to_convert = ztensor->transformed_desc->dim2;

      // convert_data_format() will dump the converted entries here
      uint16_t temp_buff[fields_to_convert];

      // number of bytes to jump from the beginning of the last C-stick to the
      // next page-boundary
      uint64_t padding =
          (ztensor->transformed_desc->dim2 % AIU_STICKS_PER_PAGE)
              ? (AIU_STICKS_PER_PAGE -
                    (ztensor->transformed_desc->dim2 % AIU_STICKS_PER_PAGE)) *
                    AIU_BYTES_PER_STICK
              : 0;

      for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

        uint64_t out_offset_n = output_offset;

        for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1; e1x++) {

          uint64_t output_offset_c = output_offset;

          for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
#if defined(__MVS__)
            __dcbt(reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(in_buf) + input_offset));
#else
            __builtin_prefetch(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                0);
#endif

            nbr_fields_converted = convert_data_format(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                ztensor->pre_transformed_desc->type, temp_buff,
                ztensor->transformed_desc->type, fields_to_convert);

            if (nbr_fields_converted == 0) {
              return ZDNN_CONVERT_FAILURE;
            }

            // read each entry in temp_buff contiguously and scatter write them
            // to stick area locations AIU_BYTES_PER_STICK bytes apart, i.e.,
            // the same C location of the consecutive C-sticks
            for (uint32_t w = 0; w < fields_to_convert; w++) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
#if defined(__MVS__)
              __dcbtst(reinterpret_cast<void *>(
                  reinterpret_cast<uintptr_t>(ztensor->buffer) +
                  output_offset));
#else
              __builtin_prefetch(
                  reinterpret_cast<void *>(
                      reinterpret_cast<uintptr_t>(ztensor->buffer) +
                      output_offset),
                  1);
#endif

              *reinterpret_cast<uint16_t *>(
                  reinterpret_cast<uintptr_t>(ztensor->buffer) +
                  output_offset) = temp_buff[w];
              // go to same C location of the next stick
              output_offset += AIU_BYTES_PER_STICK;
            }

            // go to the next 4k-boundary location (aka stick padding)
            output_offset += padding;

            // push input_offset the entire W number of entries
            input_offset += (nbr_fields_converted << input_cell_shift);
          }

          // go to the next C location of H = 0, W = 0
          output_offset = output_offset_c + sizeof_dlf16;
          if (!((e1x + 1) % AIU_2BYTE_CELLS_PER_STICK)) {
            // but if we're at the end of C-stick, roll back 1 stick worth of
            // bytes and jump to the the next c-stick of that super c-stick,
            // which is bytes_all_h number of bytes away.
            output_offset = output_offset - AIU_BYTES_PER_STICK + bytes_all_h;
          }
        }

        // done with all the C/H/W, go to the next n
        output_offset = out_offset_n + bytes_per_n;
      }
    }
  } else if (ztensor->transformed_desc->layout == ZDNN_HWCK) {

    uint64_t bytes_per_h =
        CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
        ztensor->transformed_desc->dim3 * AIU_PAGESIZE_IN_BYTES;

    uint64_t bytes_all_h = bytes_per_h * ztensor->transformed_desc->dim4;

    // H
    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      uint64_t out_offset_h = output_offset;

      // W
      for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

        // C
        for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {

          uint64_t out_offset_c = output_offset;

          // process each K-stick (i.e., every 64 elements or whatever
          // left in dim1)
          for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
               e1x += AIU_2BYTE_CELLS_PER_STICK) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
            // Also, Prefetch the new output offset to write that HW wouldn't
            // know about.
#if defined(__MVS__)
            __dcbt(reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(in_buf) + input_offset));
            __dcbtst(reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(ztensor->buffer) + output_offset));
#else
            __builtin_prefetch(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                0);
            __builtin_prefetch(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(ztensor->buffer) +
                    output_offset),
                1);
#endif
            fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                AIU_2BYTE_CELLS_PER_STICK);

            nbr_fields_converted = convert_data_format(
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(in_buf) + input_offset),
                ztensor->pre_transformed_desc->type,
                reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(ztensor->buffer) +
                    output_offset),
                ztensor->transformed_desc->type, fields_to_convert);

            if (nbr_fields_converted == 0) {
              return ZDNN_CONVERT_FAILURE;
            }

            // push input_offset the next c-stick, fake the multiply by
            // bit-shifting
            input_offset += (nbr_fields_converted << input_cell_shift);

            // push output_offset to the next c-stick of the same super
            // c-stick, which is bytes_all_h number of bytes away.
            output_offset += bytes_all_h;
          }

          // output_offset was pushed around in dim1 loops, so reset it to
          // the next c
          output_offset = out_offset_c + AIU_BYTES_PER_STICK;
        }

        // after processing all the c-entries, go to the next 4k-boundary
        // location (aka stick padding)
        output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                        (-AIU_PAGESIZE_IN_BYTES);
      }

      // output_offset was pushed around in the dims[2-0] loops, so reset it
      // to the next h
      output_offset = out_offset_h + bytes_per_h;
    }

  } else {
    // caller messed up if we ever arrive here
    return ZDNN_INVALID_LAYOUT;
  }

  /* handle any FP errors or return success */
  zdnn_status fp_error = handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

  if (fp_error != ZDNN_OK) {
    return fp_error;
  }
  // Update the tensor's format to indicate it has been stickified
  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;

} // End transform_ztensor

/// Specialized/Simplified version of transform_ztensor() that transforms 2 * a
/// * b elements to (1, 1, 2*PADDED(a), b) shape
///
/// \param[in] in_buf data buffer to be stickified
/// \param[in] real_dim2 actual, non-PADDED dim2 value
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///          ZDNN_CONVERT_FAILURE
///
zdnn_status transform_bidir_weight_ztensor(
    const void *in_buf, uint32_t real_dim2, zdnn_ztensor *ztensor) {

  // in_buf technically has shape of (2, real_dim2, dim1), meaning there are
  // 2 * real_dim2 * dim1 elements in it. we want to transform it to a ZDNN_2D
  // ztensor of shape (2 * PADDED(real_dim2), dim1)
  //
  // conceptually, this is as if we're inserting (PADDED(real_dim2) - real_dim2)
  // * dim1 of zeros after every dim1 elements, and transform the whole thing as
  // if it's (2, PADDED(real_dim2), dim1) to start with
  //
  // we'll emulate that effect by using mostly the same flow as
  // transform_ztensor()'s, and manipulate the output offset appropriately

  uint64_t input_offset = 0;
  uint64_t output_offset = 0;

  short input_cell_size =
      get_data_type_size(ztensor->pre_transformed_desc->type);
  short input_cell_shift = input_cell_size / 2;

  uint32_t fields_to_convert;
  uint32_t nbr_fields_converted;

  feclearexcept(FE_ALL_EXCEPT);

  // dim2 is always PADDED (i.e., multiples of AIU_2BYTE_CELLS_PER_STICK) and
  // divisible by AIU_STICKS_PER_PAGE
  uint64_t bytes_all_w = ztensor->transformed_desc->dim2 / AIU_STICKS_PER_PAGE *
                         AIU_PAGESIZE_IN_BYTES;

  // exactly 2 rounds, each round processes (real_dim2 * dim1) elements
  for (uint32_t i = 0; i < 2; i++) {

    for (uint32_t e2x = 0; e2x < real_dim2; e2x++) {
#if defined(__MVS__)
      __dcbt(reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(in_buf) + input_offset));
#else
      __builtin_prefetch(
          reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(in_buf) + input_offset),
          0);
#endif
      uint64_t out_offset_w = output_offset;

      for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
#if defined(__MVS__)
        __dcbtst(reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(ztensor->buffer) + output_offset));
#else
        __builtin_prefetch(
            reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(ztensor->buffer) + output_offset),
            1);
#endif
        fields_to_convert = MIN(
            (ztensor->transformed_desc->dim1 - e1x), AIU_2BYTE_CELLS_PER_STICK);

        nbr_fields_converted = convert_data_format(
            reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(in_buf) + input_offset),
            ztensor->pre_transformed_desc->type,
            reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(ztensor->buffer) + output_offset),
            ztensor->transformed_desc->type, fields_to_convert);

        if (nbr_fields_converted == 0)
          return ZDNN_CONVERT_FAILURE;
#if defined(__MVS__)
        __dcbf(reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(ztensor->buffer) + output_offset));
#else
#endif
        input_offset += (nbr_fields_converted << input_cell_shift);
        output_offset += bytes_all_w;
      }

      output_offset = out_offset_w + AIU_BYTES_PER_STICK;
    }

    // start the 2nd (and last) i-loop at offset (bytes_all_w / 2)
    output_offset = bytes_all_w / 2;
  }

  zdnn_status fp_error = handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

  if (fp_error != ZDNN_OK) {
    return fp_error;
  }

  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;
}

/// Converts the input tensor to the supported stick format for execution by
/// zDNN operations.
///
///
/// Typical usage:
/// \code
///   status = stickify(&ztensor, &data);
///   status = stickify(&ztensor, &forget, &input, &cell, &output);
/// \endcode
///
/// \param tensor Pointer to zdnn_ztensor
/// \param ... 1, 3, or 4 data buffers to be stickified. (1 for most, 3 for ZRH,
///            4 for FICO)
///
/// \returns ZDNN_OK
///          ZDNN_INVALID_FORMAT
///          ZDNN_INVALID_LAYOUT
///          ZDNN_INVALID_TYPE
///          ZDNN_INVALID_BUFFER
///          ZDNN_INVALID_STATE
///          ZDNN_CONVERT_FAILURE
///
/// \see third_party/zdnn-lib/zdnn/stickify.c
zdnn_status stickify(zdnn_ztensor *ztensor, ...) {
  zdnn_status status;
  if ((status = verify_pre_transformed_descriptor(
           ztensor->pre_transformed_desc)) != ZDNN_OK) {
    return status;
  }

  if ((status = verify_transformed_descriptor(ztensor->transformed_desc)) !=
      ZDNN_OK) {
    return status;
  }
  /*
   * Check for buffer issues. Return an error if:
   *
   * a) buffer is a NULL pointer
   * b) buffer does not start on a 4k boundary
   * c) buffer_size is smaller than what's needed
   */
  if (!ztensor->buffer ||
      reinterpret_cast<uintptr_t>(ztensor->buffer) & 0xFFF ||
      ztensor->buffer_size < getsize_ztensor(ztensor->transformed_desc)) {
    return ZDNN_INVALID_BUFFER;
  }

  // Make sure the buffer doesn't have stickified data
  if (ztensor->is_transformed) {
    return ZDNN_INVALID_STATE;
  }

  va_list argptr;
  va_start(argptr, ztensor);

  if (ztensor->transformed_desc->layout == ZDNN_NHWC ||
      ztensor->transformed_desc->layout == ZDNN_HWCK) {
    // Zero out the entire concatened (not temp) buffer so the addresses not
    // set by input values will have zeros.
    const void *data = va_arg(argptr, void *);
    memset(ztensor->buffer, 0, getsize_ztensor(ztensor->transformed_desc));
    status = transform_ztensor(data, ztensor);

  } else if (ztensor->transformed_desc->layout == ZDNN_FICO ||
             ztensor->transformed_desc->layout == ZDNN_ZRH ||
             ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO ||
             ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) {

    do { // do not just return when error, use break instead. we need to
         // va_end() at the end

      uint32_t num_slices = ztensor->transformed_desc->dim4;
      uint64_t gate_num_elements =
          get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE);
      uint64_t gate_data_size =
          gate_num_elements *
          get_data_type_size(ztensor->pre_transformed_desc->type);
      uint64_t sliced_gate_data_size = gate_data_size / num_slices;
      // 4 gates for FICO otherwise 3 gates (ZRH)
      uint8_t num_gates =
          get_data_layout_num_gates(ztensor->transformed_desc->layout);

      zdnn_tensor_desc temp_pre_tfrmd_desc, temp_tfrmd_desc;

      // Copy the real pre_transformed_desc into temp so we can
      // manipulate it without changing the original.
      memcpy(&temp_pre_tfrmd_desc, ztensor->pre_transformed_desc,
          sizeof(zdnn_tensor_desc));

      // Manipulate the temp pre_trfmd_desc.
      //
      // FICO/ZRH are concatenated horizontally.  The BIDIR_* variants
      // are also concatenated vertically.
      //
      // To build such a concatenated ztensor, we process each "slice"
      // (the promoted dim4) of each gate individually. This way the
      // final ztensor can be built to be sliceable along dim4 and each
      // slice will have a complete set of concatenated gates.
      //
      // pre_trfmd      --> slice shape
      // ------------------------------------------------------------------
      // 3DS: (a, b, c) --> 2D: (1, b, c)                (FICO/ZRH)
      //                    2D: (1, 1, 2*PADDED(b/2), c) (BIDIR_FICO/ZRH)
      // 2DS: (b, c)    --> 1D: (c)                      (FICO/ZRH)
      //                --> (no case for BIDIR_*)
      //
      // The slices will be sent to transform_ztensor(), or
      // transform_bidir_weight_ztensor() if ZDNN_BIDIR_* layouts

      uint32_t pre_trfmd_slices;
      if (ztensor->pre_transformed_desc->layout == ZDNN_3DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim3;
        if (ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO ||
            ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) {
          // dim2 has to be some multiple of 2 because of concatenation
          if (ztensor->pre_transformed_desc->dim2 & 1) {
            status = ZDNN_INVALID_SHAPE;
            break;
          }
          temp_pre_tfrmd_desc.dim4 = 1;
          temp_pre_tfrmd_desc.dim3 = 1;
          temp_pre_tfrmd_desc.dim2 = 2 * PADDED(temp_pre_tfrmd_desc.dim2 / 2);
        }
        temp_pre_tfrmd_desc.layout = ZDNN_2D;
      } else if (ztensor->pre_transformed_desc->layout == ZDNN_2DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim2;
        temp_pre_tfrmd_desc.layout = ZDNN_1D;
      } else {
        status = ZDNN_INVALID_LAYOUT;
        break;
      }

      // Safety check that the pre_tfrmd and tfrmd descriptors indicate the same
      // number of expected slices.
      if (pre_trfmd_slices != num_slices) {
        status = ZDNN_INVALID_SHAPE;
        break;
      }

      // Create a non-sliced, non-horizontally-concatenated trfmd_desc
      // by using the modified temp_pre_tfrmd_layout.
      if ((status = generate_transformed_desc(
               &temp_pre_tfrmd_desc, &temp_tfrmd_desc)) != ZDNN_OK) {
        break;
      }

      uint64_t sliced_gate_buffer_size = getsize_ztensor(&temp_tfrmd_desc);

      // Save the gate data for slicing later.
      // (e.g., LSTM) va_arg order: F (FWD,BWD), I (FWD,BWD), C...etc.
      void *gate_data[num_gates];
      for (uint8_t i = 0; i < num_gates; i++) {
        gate_data[i] = va_arg(argptr, void *);
      }

      // Create a temporary ztensor to be used to call
      // transform_ztensor() multiple times with, as if it's not
      // horizontally-concatenated.
      zdnn_ztensor temp_ztensor;

      // Setup the temp ztensor, with a non-sliced,
      // non-horizontally-concatenated buffer_size
      init_ztensor(&temp_pre_tfrmd_desc, &temp_tfrmd_desc, &temp_ztensor);

      temp_ztensor.buffer = ztensor->buffer;
      temp_ztensor.buffer_size = sliced_gate_buffer_size;

      // Concatenated tensors require zero padding between the
      // horizontal concatenations, while technically not required for
      // the verticals. However, zero out the entire concatened (not
      // temp) buffer for efficiency.
      size_t total_buffer_size =
          temp_ztensor.buffer_size * num_slices * num_gates;
      memset(ztensor->buffer, 0, total_buffer_size);

      /* Loop sliced_gate_data array to stickify the input data. Because
       * of how sliced_gate_data was built as a 2D array, we can jump
       * around various locations of the original inputs data and read
       * each value only once while building the output ztensor to be in
       * the final desired order.
       *
       * This converts the value order from the input arrays from:
       * slice 0 of gate 0
       * slice 1 of gate 0
       * slice 0 of gate 1
       * slice 1 of gate 1
       * ...
       *
       * to the following order in the final output ztensor:
       * slice 0 of gate 0
       * slice 0 of gate 1
       * ...
       * slice 1 of gate 0
       * slice 1 of gate 1
       * ...
       */

      for (uint32_t slice = 0; slice < num_slices; slice++) {
        for (uint8_t gate = 0; gate < num_gates; gate++) {
          // Points to a single slice of a single gate data.
          const void *gate_data_slice = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(gate_data[gate]) +
              (slice * sliced_gate_data_size));

          // Transform the current slice of the current gate into final
          // ztensor
          if (ztensor->transformed_desc->layout != ZDNN_BIDIR_FICO &&
              ztensor->transformed_desc->layout != ZDNN_BIDIR_ZRH) {
            status = transform_ztensor(gate_data_slice, &temp_ztensor);
          } else {
            // transform_bidir_weight_ztensor() wants the actual b/2,
            // not the PADDED one in temp_ztensor->dim2
            status = transform_bidir_weight_ztensor(gate_data_slice,
                ztensor->pre_transformed_desc->dim2 / 2, &temp_ztensor);
          }
          assert(status == ZDNN_OK);

          // Increment the temp_ztensor buffer by one sliced gate size
          // so we write to the correct location in the final output
          // ztensor.
          temp_ztensor.buffer = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(temp_ztensor.buffer) +
              sliced_gate_buffer_size);

          // Reset temp_ztensor is_transformed so we can recursively
          // call zdnn_transform_ztensor to process each slice of each
          // gate.
          temp_ztensor.is_transformed = false;
        }
        if (status != ZDNN_OK) {
          break;
        }
      }

      if (status == ZDNN_OK) {
        // Set that the output ztensor has completed transformation.
        ztensor->is_transformed = true;
      }

    } while (false);

  } else {
    status = ZDNN_INVALID_LAYOUT;
  }

  va_end(argptr);
  return status;
} // End - Functions from third_party/zdnn-lib/zdnn/stickify.c

#define AIU_STICKS_PER_PAGE 32
#define AIU_BYTES_PER_STICK 128
#define AIU_1BYTE_CELLS_PER_STICK 128
#define AIU_PAGESIZE_IN_BYTES 4096

#define VECPERM_MAX_INT8_ENTRIES 8

// The scalar version of transform_quantized_weights_ztensor()
zdnn_status transform_quantized_weights_ztensor_element_wise(
    const void *in_buf, zdnn_ztensor *output) {

  // moving position as the input is processed, in BYTES
  uint64_t input_offset = 0;
  // moving position as the output is processed, in BYTES
  uint64_t output_offset = 0;

  // loop invariant values
  uint64_t bytes_all_h =
      (uint64_t)output->transformed_desc->dim3 *
      CEIL(CEIL(output->transformed_desc->dim2, 2), AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  uint64_t bytes_per_n = bytes_all_h * CEIL(output->transformed_desc->dim1,
                                           (AIU_1BYTE_CELLS_PER_STICK / 2));

  // N
  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
    uint64_t out_offset_n = output_offset;

    // H
    for (uint32_t e3x = 0; e3x < output->transformed_desc->dim3; e3x++) {

      // W, sticks are processed in pairs
      for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2;
           e2x = e2x + 2) {

        // used for pushing out_offset from w to w+1 (i.e., +
        // AIU_BYTES_PER_STICK)
        uint64_t out_offset_w = output_offset;

        // true when dim2 is odd number and we're at the last w
        bool no_stick2 = ((output->transformed_desc->dim2 - e2x) == 1);

        int8_t *stick1 = (int8_t *)in_buf + input_offset;
        int8_t *stick2 = no_stick2 ? stick1
                                   // duplicate stick1 entries if no stick2
                                   : stick1 + output->transformed_desc->dim1;

        // this C loop takes care of the full VECPERM_MAX_INT8_ENTRIES-entries
        // groups
        for (uint32_t i = 0;
             i < output->transformed_desc->dim1 / VECPERM_MAX_INT8_ENTRIES;
             i++) {
          ((int8_t *)output->buffer + output_offset)[0] = stick1[0];
          ((int8_t *)output->buffer + output_offset)[1] = stick2[0];
          ((int8_t *)output->buffer + output_offset)[2] = stick1[1];
          ((int8_t *)output->buffer + output_offset)[3] = stick2[1];
          ((int8_t *)output->buffer + output_offset)[4] = stick1[2];
          ((int8_t *)output->buffer + output_offset)[5] = stick2[2];
          ((int8_t *)output->buffer + output_offset)[6] = stick1[3];
          ((int8_t *)output->buffer + output_offset)[7] = stick2[3];

          ((int8_t *)output->buffer + output_offset)[8] = stick1[4];
          ((int8_t *)output->buffer + output_offset)[9] = stick2[4];
          ((int8_t *)output->buffer + output_offset)[10] = stick1[5];
          ((int8_t *)output->buffer + output_offset)[11] = stick2[5];
          ((int8_t *)output->buffer + output_offset)[12] = stick1[6];
          ((int8_t *)output->buffer + output_offset)[13] = stick2[6];
          ((int8_t *)output->buffer + output_offset)[14] = stick1[7];
          ((int8_t *)output->buffer + output_offset)[15] = stick2[7];

          stick1 += VECPERM_MAX_INT8_ENTRIES;
          stick2 += VECPERM_MAX_INT8_ENTRIES;
          output_offset += VECPERM_MAX_INT8_ENTRIES * 2;

          if ((i + 1) %
                  (AIU_BYTES_PER_STICK / (VECPERM_MAX_INT8_ENTRIES * 2)) ==
              0) {
            // we need to jump to the next c-stick of the same super c-stick
            //
            // roll-back to the beginning and jump to bytes_all_h number of
            // bytes away
            output_offset = output_offset - AIU_BYTES_PER_STICK + bytes_all_h;
          }
        }

        // takes care of the leftover c entries
        for (uint32_t i = 0;
             i < output->transformed_desc->dim1 % VECPERM_MAX_INT8_ENTRIES;
             i++) {
          ((int8_t *)output->buffer + output_offset)[0] = stick1[i];
          ((int8_t *)output->buffer + output_offset)[1] = stick2[i];

          output_offset += 2;
        }

        // move on to the next set
        input_offset += output->transformed_desc->dim1 * (no_stick2 ? 1 : 2);
        // output_offset was pushed around in dim1 loops, so reset it to
        // the next w
        output_offset = out_offset_w + AIU_BYTES_PER_STICK;
      }

      // after processing all the w-entries, go to the next 4k-boundary
      // location (aka stick padding)
      output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                      (-AIU_PAGESIZE_IN_BYTES);
    }

    // output_offset was pushed around in the dims[2-0] loops, so reset it
    // to the next n
    output_offset = out_offset_n + bytes_per_n;
  }

  // Update the tensor's format to indicate it has been stickified
  output->is_transformed = true;
  return ZDNN_STATUS_OK;
}

zdnn_status quantized_stickify(zdnn_ztensor *ztensor, const void *in_buf) {
  /* It is supposed to use zdnn_transform_quantized_ztensor here.
   *  return zdnn_transform_quantized_ztensor(ztensor, 0, 0, in_buf);
   *  The clip_min and clip_max will not be used when
   *  transform_quantized_weights_ztensor() is called in this transform.
   *  The reason that zdnn_transform_quantized_ztensor can't be called
   *  is that the variable, nnpa_query_result, in the zdnn library built with
   *  onnx-mlir has not been properly set up. Therefore, the check on
   *  dimension size will fail. verify_transformed_descriptor() is called
   *  by zdnn_transform_quantized_ztensor().
   *  Tried to call zdnn_refresh_nnpa_query_result(), but failed.
   *  In the copied verify_transformed_descriptor code, the code for checking
   *  has been commented out.
   *  Refer to issue #3034
   */

  zdnn_status status;
  if ((status = verify_transformed_descriptor(ztensor->transformed_desc)) !=
      ZDNN_OK) {
    return status;
  }

  return transform_quantized_weights_ztensor_element_wise(in_buf, ztensor);
}

/// Set information for a pre transformed descriptor.
void set_info_pre_transformed_desc(zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_data_layouts layout, zdnn_data_types type,
    llvm::ArrayRef<int64_t> shape) {
  // point to dim4/3/etc via pointer.  they're guaranteed to be in the correct
  // order as written and contiguous and correctly aligned
  uint32_t *dims_ptr = &(pre_tfrmd_desc->dim4);

  if (pre_tfrmd_desc) {
    // we do not need to set the unused dim vars to 1 for pre-transformed
    int startIdx = ZDNN_MAX_DIMS - get_data_layout_dims(layout);
    for (int i = startIdx; i < ZDNN_MAX_DIMS; i++) {
      dims_ptr[i] = static_cast<uint32_t>(shape[i - startIdx]);
    }
    pre_tfrmd_desc->layout = layout;
    pre_tfrmd_desc->format =
        (layout == ZDNN_HWCK) ? ZDNN_FORMAT_4DKERNEL : ZDNN_FORMAT_4DFEATURE;
    pre_tfrmd_desc->type = type;
  }
}
