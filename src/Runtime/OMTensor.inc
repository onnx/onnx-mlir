/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMTensor.inc - C/C++ Neutral OMTensor Implementation--------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of OMTensor data structures
// and helper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <array>
#include <cassert>
#include <complex>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>
#else
#include <assert.h>
#include <complex.h>
#endif

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.hpp"
#endif

#ifndef ENABLE_PYRUNTIME_LIGHT
#include "src/Support/SmallFPConversion.h"
#endif

#if defined(__APPLE__) || defined(__MVS__)
// __errno_location() is called from krnl::emitErrNo() to set errno in
// generated llvm IR, but it somehow doesn't come out of the box on MacOS or MVS.
int *__attribute__((__weak__)) __errno_location(void) { return &errno; }
#endif

#ifdef ENABLE_PYRUNTIME_LIGHT
// The implementation depends on src/Utility and llvm. Will be solved in
// another PR. Here are just dummy definitions for compilation
float om_f16_to_f32(uint16_t a) { return (float)0; }

uint16_t om_f32_to_f16(uint16_t a) { return (uint16_t)0; }
#endif

#define OM_DRIVER_TIMING 1
#include "src/Runtime/OMInstrumentHelper.h"

// On some platforms LLVM generates f16 conversion code that calls some
// of these C runtime functions.
// TODO: Figure out if we could get these with llvm/compiler-rt in a
//       systematic way.
float __extendhfsf2(uint16_t a) { return om_f16_to_f32(a); }
float __gnu_h2f_ieee(uint16_t a) { return om_f16_to_f32(a); }
uint16_t __gnu_f2h_ieee(float a) { return om_f32_to_f16(a); }
uint16_t __truncsfhf2(float a) { return om_f32_to_f16(a); }
uint16_t __truncdfhf2(double a) { return om_f32_to_f16(a); }

struct OMTensor {
#ifdef __cplusplus
  /**
   * Constructor
   *
   * @param rank, rank of data shape and strides
   *
   * Create a OMTensor with specified rank. Memory for data shape and strides
   * are allocated.
   */
  OMTensor(int64_t rank) {
    if ((_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
        (_strides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
      _allocatedPtr = NULL;
      _alignedPtr = NULL;
      _offset = 0;
      _dataType = ONNX_TYPE_UNDEFINED;
      _rank = rank;
      _owning = false;
    } else {
      if (_shape)
        free(_shape);
      throw std::runtime_error(
          "OMTensor(" + std::to_string(rank) + ") malloc error");
    }
  };

  OMTensor() = default;

  /**
   * Destructor
   *
   * Destroy the OMTensor struct.
   */
  ~OMTensor() {
    if (_owning)
      free(_allocatedPtr);
    free(_shape);
    free(_strides);
  };
#endif
  // Fields are named according to:
  // https://mlir.llvm.org/docs/Dialects/SPIR-V/#lowering-memrefs-to-spvarray-and-spvrtarray

  // On machine without alignment constraints the allocated and aligned pointers
  // are the same. However, on machines with alignment constraints not supported
  // by the memory allocation system, the allocated ptr points to the chunk of
  // memory that is allocated, and the aligned pointer points to a chunk of
  // memory that is allocated and also satisfy the alignment constraints on the
  // machine. For example, on a machine for which malloc returns chunks aligned
  // at 16 byte boundaries, but where tensor must be allocated at 1K boundaries
  // for performance reason, allocated pointer may return 0x1000f and aligned
  // pointer may return 0x10400.
  void *_allocatedPtr; // data buffer
  void *_alignedPtr;   // aligned data buffer that the omt indexes.

  // TODO: remove or use _offset. Currently only set to zero, never used. It
  // may be in use in LLVM memrefs, so someone needs to look into this.
  int64_t _offset;   // offset of 1st element
  int64_t *_shape;   // shape array
  int64_t *_strides; // strides array
  int64_t _rank;     // rank

  OM_DATA_TYPE _dataType; // ONNX data type

  int64_t _owning; // indicates whether the Omt owns the memory space
                   // referenced by _allocatedPtr. Omt struct will release the
                   // memory space referred to by _allocatedPtr upon destruction
                   // if and only if it owns it.
};

/* Helper function to compute the number of data elements */
static inline int64_t getNumElems(const int64_t *shape, int64_t rank) {
  int64_t numElem = 1;
  for (int64_t i = 0; i < rank; i++)
    numElem *= shape[i];
  return numElem;
}

/* Helper function to compute the inner product of 2 arrays (of equal size) */
static inline int64_t inner_product(
    const int64_t v[], const int64_t u[], int64_t len, int64_t init) {
  int64_t result = init;
  for (int64_t i = 0; i < len; i++)
    result += v[i] * u[i];
  return result;
}

static inline int64_t computeElemOffset(
    const int64_t strides[], const int64_t indexes[], int64_t rank) {
  return inner_product(strides, indexes, rank, 0ll);
}

// Create a OMTensor.
OMTensor *omTensorCreate(
    void *data_ptr, const int64_t *shape, int64_t rank, OM_DATA_TYPE dtype) {
  OMTensor *tensor = (OMTensor *)malloc(sizeof(OMTensor));
  if (!tensor)
    return NULL;
  if ((tensor->_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (tensor->_strides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    tensor->_allocatedPtr = data_ptr;
    tensor->_alignedPtr = data_ptr;
    tensor->_offset = 0;
    tensor->_rank = rank;
    tensor->_dataType = dtype;
    tensor->_owning = false;
  } else {
    free(tensor->_shape);
    free(tensor);
    return NULL;
  }

  // Using signed indices helps detect when index falls below 0.
  for (int64_t i = rank - 1; i >= 0; i--) {
    tensor->_shape[i] = shape[i];
    if (i == rank - 1)
      tensor->_strides[i] = 1;
    else
      tensor->_strides[i] = tensor->_strides[i + 1] * tensor->_shape[i + 1];
  }
  return tensor;
}

// Create a OMTensor with specified ownership.
OMTensor *omTensorCreateWithOwnership(void *data_ptr, const int64_t *shape,
    int64_t rank, OM_DATA_TYPE dtype, int64_t owning) {
  OMTensor *tensor = omTensorCreate(data_ptr, shape, rank, dtype);
  // If ctor fails, return NULL.
  if (!tensor)
    return NULL;
  tensor->_owning = owning;
  return tensor;
}

/**
 * Create an empty OMTensor with specified rank but no shape and type info.
 * This function is intentionally left out from the header because it is only
 * used by the wrapper code we emit around inference function that converts
 * MemRefs to OMTensors for user convenience.
 *
 * This constructor returns a
 * partially filled omTensor; prefer using the new omTensorCreateEmpty()
 * function to fill shape & strides fields automatically.
 *
 * @param rank tensor rank
 * @return pointer to OMTensor created, NULL if creation failed.
 *
 */
OMTensor *omTensorCreateUntyped(int64_t rank) {
  OMTensor *omt = (OMTensor *)malloc(sizeof(struct OMTensor));
  if (!omt)
    return NULL;
  if ((omt->_shape = (int64_t *)malloc(rank * sizeof(int64_t))) &&
      (omt->_strides = (int64_t *)malloc(rank * sizeof(int64_t)))) {
    omt->_allocatedPtr = NULL;
    omt->_alignedPtr = NULL;
    omt->_offset = 0;
    omt->_dataType = ONNX_TYPE_UNDEFINED;
    omt->_rank = rank;
    omt->_owning = false;
  } else {
    free(omt->_shape);
    free(omt);
    return NULL;
  }

  return omt;
}

OMTensor *omTensorCreateEmpty(
    const int64_t *shape, int64_t rank, OM_DATA_TYPE dtype) {
  OMTensor *tensor;
  void *dataPtr;
  if (((dataPtr = malloc(getNumElems(shape, rank) * getDataTypeSize(dtype))) ==
          NULL) ||
      ((tensor = omTensorCreateWithOwnership(
            dataPtr, shape, rank, dtype, /*owning=*/true)) == NULL)) {
    free(dataPtr); // OK if dataPtr == NULL
    return NULL;
  }
  return tensor;
}

/* OMTensor destroyer */
void omTensorDestroy(OMTensor *tensor) {
  if (!tensor)
    return;
  if (tensor->_owning) {
    TIMING_INIT_START(tensor_destroy_ptr);
    free(tensor->_allocatedPtr);
    TIMING_STOP_PRINT(tensor_destroy_ptr);
  }
  TIMING_INIT_START(tensor_destroy_struct);
  free(tensor->_shape);
  free(tensor->_strides);
  free(tensor);
  TIMING_STOP_PRINT(tensor_destroy_struct);
}

/* OMTensor data getter */
void *omTensorGetDataPtr(const OMTensor *tensor) { return tensor->_alignedPtr; }

/**
 * OMTensor allocated and aligned pointer setter.
 * This function is intentionally left out from the header because it is only
 * used by the wrapper code we emit around inference function that converts
 * MemRefs to OMTensors for user convenience.
 *
 * @param tensor pointer to the OMTensor
 * @param owning whether allocatedPtr should be freed after tensor is destroyed.
 * @param allocatedPtr allocated pointer to tensor content.
 * @param alignedPtr aligned pointer to tensor content. If NULL will be set to
 * allocatedPtr.
 *
 */
void omTensorSetDataPtr(
    OMTensor *tensor, int64_t owning, void *allocatedPtr, void *alignedPtr) {
  if (tensor->_owning) {
    /* If we own the allocated buffer, free it first. */
    free(tensor->_allocatedPtr);
  }
  tensor->_owning = owning;
  tensor->_allocatedPtr = allocatedPtr;
  if (alignedPtr)
    tensor->_alignedPtr = alignedPtr;
  else
    tensor->_alignedPtr = allocatedPtr;
}

/* OMTensor data shape getter */
const int64_t *omTensorGetShape(const OMTensor *tensor) {
  return tensor->_shape;
}

/* OMTensor data shape setter */
void omTensorSetShape(OMTensor *tensor, const int64_t *shape) {
  for (int64_t i = 0; i < tensor->_rank; i++)
    tensor->_shape[i] = shape[i];
}

/* OMTensor data strides getter */
const int64_t *omTensorGetStrides(const OMTensor *tensor) {
  return tensor->_strides;
}

/* OMTensor data strides setter */
void omTensorSetStrides(OMTensor *tensor, const int64_t *strides) {
  for (int64_t i = 0; i < tensor->_rank; i++)
    tensor->_strides[i] = strides[i];
}

/* OMTensor data strides setter with PyArray stride values */
void omTensorSetStridesWithPyArrayStrides(
    OMTensor *tensor, const int64_t *stridesInBytes) {
  for (int64_t i = 0; i < tensor->_rank; i++)
    tensor->_strides[i] =
        stridesInBytes[i] / getDataTypeSize(tensor->_dataType);
}

/* OMTensor data type getter */
OM_DATA_TYPE omTensorGetDataType(const OMTensor *tensor) {
  return tensor->_dataType;
}

/* OMTensor data type setter */
void omTensorSetDataType(OMTensor *tensor, OM_DATA_TYPE dataType) {
  tensor->_dataType = dataType;
}

/* OMTensor data buffer size getter */
int64_t omTensorGetBufferSize(const OMTensor *tensor) {
  return getNumElems(tensor->_shape, tensor->_rank) *
         getDataTypeSize(tensor->_dataType);
}

/* OMTensor rank getter */
int64_t omTensorGetRank(const OMTensor *tensor) { return tensor->_rank; }

/* OMTensor number of elements getter */
int64_t omTensorGetNumElems(const OMTensor *tensor) {
  // Using signed indices helps detect when index falls below 0.
  // Verify that strides are dense, meaning that there're
  // no skipping elements.
  for (int64_t i = tensor->_rank - 1; i >= 0; i--) {
    int64_t stridesIfNotSkipping = 1;
    for (int64_t j = i + 1; j < tensor->_rank; j++) {
      stridesIfNotSkipping *= tensor->_shape[j];
    }
    assert(tensor->_strides[i] == stridesIfNotSkipping);
  }
  return getNumElems(tensor->_shape, tensor->_rank);
}

/**
 * OMTensor owning flag getter.
 *
 * @return owning flag of the OMTensor.
 */
int64_t omTensorGetOwning(const OMTensor *tensor) { return tensor->_owning; }

/**
 * OMTensor owning flag setter.
 */
void omTensorSetOwning(OMTensor *tensor, int64_t owning) {
  tensor->_owning = owning;
}

/**
 * OMTensor allocated ptr getter.
 * Note that this function is intentionally left out from the header
 * because it is only used by the wrapper code we emit around inference
 * function that converts OMTensor into MemRefs for user convenience.
 *
 * @param tensor pointer to the OMTensor
 * @return pointer to the allocated data buffer of the OMTensor,
 *         NULL if the allocated data buffer is not set.
 */
void *omTensorGetAllocatedPtr(const OMTensor *tensor) {
  return tensor->_allocatedPtr;
}

//===----------------------------------------------------------------------===//
/* omTensorPrint helper */

/* Helper function to get the ONNX data type C name string */
static inline const char *getDataTypeName(OM_DATA_TYPE dataType) {
  return OM_DATA_TYPE_NAME[dataType];
}

/* Helper function to print type as 8x16x1xfloat */
static void printType(FILE *fout, const OMTensor *tensor) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(tensor);
  const int64_t rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
  for (int64_t i = 0; i < rank; i++)
    fprintf(fout, "%lldx", (long long)shape[i]);
  fprintf(fout, "%s", getDataTypeName(dataType));
}

/* Helper function to print a detailed signature of a vector */
static void printSignature(FILE *fout, const OMTensor *tensor) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(tensor);
  const int64_t rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
  const int64_t *strides = omTensorGetStrides(tensor);

  fprintf(fout, "\trank = %lld\n", (long long)rank);
  fprintf(fout, "\tdataType = %s\n", getDataTypeName(dataType));
  fprintf(fout, "\tnumElems = %lld\n", (long long)omTensorGetNumElems(tensor));
  fprintf(fout, "\tshape: ");
  for (int64_t i = 0; i < rank; i++)
    fprintf(fout, "[%lld]", (long long)shape[i]);
  fprintf(fout, "\n");
  fprintf(fout, "\tstrides: ");
  for (int64_t i = 0; i < rank; i++)
    fprintf(fout, "[%lld]", (long long)strides[i]);
  fprintf(fout, "\n");

  fprintf(fout, "\ttype: ");
  printType(fout, tensor);
  fprintf(fout, "\n");
}

/* Helper function to print an element of a tensor */
static void printElement(
    FILE *fout, void *dataPtr, int64_t elemOffset, OM_DATA_TYPE dataType) {
  assert(dataPtr && "Expecting a valid data pointer");
  assert(elemOffset >= 0 && "Expecting a valid element offset");

  switch (dataType) {
  case ONNX_TYPE_BOOL:
    fprintf(fout, "%d", ((bool *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_UINT8:
    fprintf(fout, "%hhu", ((uint8_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_INT8:
    fprintf(fout, "%hhd", ((int8_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_UINT16:
    fprintf(fout, "%hu", ((uint16_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_INT16:
    fprintf(fout, "%hd", ((int16_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_UINT32:
    fprintf(fout, "%u", ((uint32_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_INT32:
    fprintf(fout, "%d", ((int32_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_UINT64:
    fprintf(
        fout, "%llu", (unsigned long long)((uint64_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_INT64:
    fprintf(fout, "%lld", (long long)((int64_t *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_FLOAT16:
    fprintf(fout, "%f", om_f16_to_f32(((uint16_t *)dataPtr)[elemOffset]));
    break;
  case ONNX_TYPE_FLOAT:
    fprintf(fout, "%f", ((float *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_DOUBLE:
    fprintf(fout, "%f", ((double *)dataPtr)[elemOffset]);
    break;
  case ONNX_TYPE_STRING:
    fprintf(fout, "%s", ((const char **)dataPtr)[elemOffset]);
    break;
  default:
    assert(false && "unexpected data type");
  }
}

/* Helper function to print the data of a tensor type */
static void printData(FILE *fout, const OMTensor *tensor) {

/* Helper macros to print data for 1-4D tensors  */
#define LOOP_1(INDEX, IV, UB)                                                  \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    int64_t elemOffset = computeElemOffset(tensor->_strides, indexes, rank);   \
    printElement(fout, dataPtr, elemOffset, dataType);                         \
  }                                                                            \
  fprintf(fout, "]");

#define LOOP_2(INDEX, IV, UB, ...)                                             \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    LOOP_1((INDEX) + 1, __VA_ARGS__)                                           \
  }                                                                            \
  fprintf(fout, "]");

#define LOOP_3(INDEX, IV, UB, ...)                                             \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    LOOP_2((INDEX) + 1, __VA_ARGS__)                                           \
  }                                                                            \
  fprintf(fout, "]");

#define LOOP_4(INDEX, IV, UB, ...)                                             \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    LOOP_3((INDEX) + 1, __VA_ARGS__)                                           \
  }                                                                            \
  fprintf(fout, "]");

#define LOOP_5(INDEX, IV, UB, ...)                                             \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    LOOP_4((INDEX) + 1, __VA_ARGS__)                                           \
  }                                                                            \
  fprintf(fout, "]");

#define LOOP_6(INDEX, IV, UB, ...)                                             \
  fprintf(fout, "[");                                                          \
  for (int64_t IV = 0; (IV) < (UB); ++(IV)) {                                  \
    if (IV)                                                                    \
      fprintf(fout, ", ");                                                     \
    indexes[(INDEX)] = (IV);                                                   \
    LOOP_5((INDEX) + 1, __VA_ARGS__)                                           \
  }                                                                            \
  fprintf(fout, "]");

  const OM_DATA_TYPE dataType = omTensorGetDataType(tensor);
  const int64_t rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
  void *dataPtr = omTensorGetDataPtr(tensor);

  fprintf(fout, "\tdata: (");
  switch (rank) {
  case 0:
    printElement(fout, dataPtr, 0, dataType);
    break;
  case 1: {
    int64_t indexes[1];
    LOOP_1(0, i, shape[0])
  } break;
  case 2: {
    int64_t indexes[2];
    LOOP_2(0, i, shape[0], j, shape[1])
  } break;
  case 3: {
    int64_t indexes[3];
    LOOP_3(0, i, shape[0], j, shape[1], k, shape[2])
  } break;
  case 4: {
    int64_t indexes[4];
    LOOP_4(0, i, shape[0], j, shape[1], k, shape[2], l, shape[3])
  } break;
  case 5: {
    int64_t indexes[5];
    LOOP_5(0, i, shape[0], j, shape[1], k, shape[2], l, shape[3], m, shape[4])
  } break;
  case 6: {
    int64_t indexes[6];
    LOOP_6(0, i, shape[0], j, shape[1], k, shape[2], l, shape[3], m, shape[4],
        n, shape[5])
  } break;
  default:
    assert(false && "not implemented");
  }
  fprintf(fout, ")\n");
}

/* Defined in OMInstrument.inc, return stdout or the FILE handle to where
 * instrumentation should be printed. */
FILE *getInstrumentFile();

/* Print a string and possibly some info about a tensor. With `%d`, we print the
tensor data; with `s`, we print an extended signature; with `t`, we print the
type of the tensor (e.g. 32 x 16 x float)

There are two other unpublished modifiers if the message starts with `%i`, this
signal an instrumentation message that may be redirected to the file name given
by ONNX_MLIR_INSTRUMENT_FILE. Additionally, there is a `%e` modifier that
signals the end of the message string. This is unfortunately needed because of
current issues with the termination of strings.
*/
void omTensorPrint(const char *msg, const OMTensor *tensor) {
  /* Process the message string. */
  int64_t len = strlen(msg);
  FILE *fout = stdout;
  /* Check if this is an instrumentation print, in which case we use the
   * instrument file instead of stdout. */
  if (len >= 2 && msg[0] == '%' && msg[1] == 'i') {
    fout = getInstrumentFile();
    /* Skip format. */
    msg += 2;
    len -= 2;
  }
  bool hadOneOrMoreFormats = false;
  while (len > 0) {
    if (msg[0] == '%') {
      if (len < 2) {
        fprintf(fout, "omTensorPrint: orphan `%%`, skip.\n");
        return;
      }
      if (msg[1] == 'd') { /* Letter `d` for data. */
        assert(tensor && "attempt to print a null OMTensor");
        printData(fout, tensor);
        hadOneOrMoreFormats = true;
      } else if (msg[1] == 's') { /* Letter `s` for signature. */
        assert(tensor && "attempt to print a null OMTensor");
        printSignature(fout, tensor);
        hadOneOrMoreFormats = true;
      } else if (msg[1] == 't') { /* Letter `t` for type only. */
        assert(tensor && "attempt to print a null OMTensor");
        printType(fout, tensor);
        hadOneOrMoreFormats = true;
      } else if (msg[1] == 'e') { /* Letter `e` for end. */
        return;
      } else {
        fprintf(fout, "omTensorPrint: unknown format `%c`, skip.\n", msg[1]);
      }
      /* Skip the format when printing message. */
      msg += 2;
      len -= 2;
      continue;
    }
    /* No format, just print letter. */
    fprintf(fout, "%c", msg[0]);
    msg++;
    len--;
  }
  if (!hadOneOrMoreFormats) {
    // default per Krnl.td: %s%d
    fprintf(fout, "\n");
    printSignature(fout, tensor);
    printData(fout, tensor);
    fprintf(fout, "\n");
  }
}

#ifdef __cplusplus
/* For C++ methods, which are not used in the time critical runtime,
 * asserts are present to ensure that we do not perform a null ptr access.
 */

/* OMTensor creator with data shape and element type  */
template <typename T>
OMTensor *omTensorCreateWithShape(const std::vector<int64_t> &shape) {
  /* Create a OMTensor with data shape and strides allocated */
  auto omt = omTensorCreateUntyped(shape.size());
  if (omt == NULL)
    return NULL;

  /* Allocate data buffer */
  omt->_rank = shape.size();
  if ((omt->_allocatedPtr = malloc(
           getNumElems(shape.data(), omt->_rank) * sizeof(T))) == NULL) {
    omTensorDestroy(omt);
    return NULL;
  }

  omt->_alignedPtr = omt->_allocatedPtr;
  omt->_offset = 0;

  /* Copy shape, _shape already allocated by omTensorCreate */
  copy(shape.begin(), shape.end(), omt->_shape);

  /* Compute and copy dataStrides, _strides already allocated by
   * omTensorCreateUntyped
   */
  auto computedStrides = computeStridesFromShape(omt->_shape, omt->_rank);
  copy(computedStrides.begin(), computedStrides.end(), omt->_strides);

  /* Convert CPP type to ONNX type */
  try {
    omt->_dataType = OM_DATA_TYPE_CPP_TO_ONNX.at(std::string(typeid(T).name()));
  } catch (const std::out_of_range &) {
    omt->_dataType = ONNX_TYPE_UNDEFINED;
  }

  /* Set flag for destructor */
  omt->_owning = true;
  return omt;
}

// Static variables used by omDefineSeed and omTensorCreateWithRandomData.
static unsigned int omUseOneSeed = 0;
static std::mt19937 omRandomGenerator(0);

// When called, a single seed will be used, either from the given seed or from
// another random number generator.
unsigned int omDefineSeed(unsigned int seed, unsigned int hasSeedValue) {
  /* Define our onw seed when requested. */
  if (!hasSeedValue) {
    std::random_device rd;
    seed = rd();
  }
  omUseOneSeed = 1;
  omRandomGenerator.seed(seed);
  return seed;
}

/* OMTensor creator with data shape, element type and random data */
template <typename T>
OMTensor *omTensorCreateWithRandomData(
    const std::vector<int64_t> &shape, T lbound, T ubound) {
  if (!omUseOneSeed) {
    // Will be used to obtain a seed for the random number engine.
    std::random_device rd;
    // Standard mersenne_twister_engine seeded with rd()
    omRandomGenerator.seed(rd());
  }
  std::uniform_real_distribution<> dis(lbound, ubound);

  auto omt = omTensorCreateWithShape<T>(shape);
  if (omt == NULL)
    return NULL;

  std::generate((T *)omt->_alignedPtr,
      (T *)omt->_alignedPtr + getNumElems(omt->_shape, omt->_rank),
      [&]() { return dis(omRandomGenerator); });
  return omt;
}

/* Access an element (by reference) at offset computed by index array */
template <typename T>
T &omTensorGetElem(const OMTensor *omt, const std::vector<int64_t> &indexes) {
  assert(omt && "attempt to access element of null OMTensor");
  int64_t elemOffset = omTensorComputeElemOffset(omt, indexes);
  return ((T *)omt->_alignedPtr)[elemOffset];
}

/* Access an element (by reference) at linear offset */
template <typename T>
T &omTensorGetElemByOffset(const OMTensor *omt, int64_t index) {
  assert(omt && "attempt to access element of null OMTensor");
  return ((T *)omt->_alignedPtr)[index];
}

/* Compute strides vector from shape vector */
std::vector<int64_t> omTensorComputeStridesFromShape(const OMTensor *omt) {
  assert(omt && "attempt to access element of null OMTensor");
  return computeStridesFromShape(omt->_shape, omt->_rank);
}

/* Compute linear element offset from multi-dimensional index array */
int64_t omTensorComputeElemOffset(
    const OMTensor *omt, const std::vector<int64_t> &indexes) {
  assert(omt && "attempt to access element of null OMTensor");
  return computeElemOffset(omt->_strides, omt->_rank, indexes);
}

/* Compute index set for the whole OMTensor */
std::vector<std::vector<int64_t>> omTensorComputeIndexSet(const OMTensor *omt) {
  assert(omt && "attempt to access element of null OMTensor");
  // First, we create index set of each dimension separately.
  // i.e., for a tensor/OMT of shape (2, 3), its dimWiseIdxSet will be:
  // {{0,1}, {0,1,2}};
  std::vector<std::vector<int64_t>> dimWiseIdxSet;
  for (auto dimSize :
      std::vector<int64_t>(omt->_shape, omt->_shape + omt->_rank)) {
    std::vector<int64_t> dimIdxSet(dimSize);
    iota(begin(dimIdxSet), end(dimIdxSet), 0);
    dimWiseIdxSet.emplace_back(dimIdxSet);
  }
  // Then, the cartesian product of vectors within dimWiseIdxSet will be the
  // index set for the whole OMT.
  return CartProduct(dimWiseIdxSet);
}

/* Check whether two OMTensor data are "close" to each other */
template <typename T>
inline bool omTensorAreTwoOmtsClose(
    const OMTensor *a, const OMTensor *b, float rtol, float atol) {
  assert(a && b && "attempt to access element of null OMTensor");

  // Compare shape.
  auto aShape = std::vector<int64_t>(a->_shape, a->_shape + a->_rank);
  auto bShape = std::vector<int64_t>(b->_shape, b->_shape + b->_rank);
  if (aShape != bShape) {
    std::cerr << "Shape mismatch in omTensorAreTwoOmtsClose a: ";
    printVector(aShape, ",", std::cerr);
    std::cerr << " != b: ";
    printVector(bShape, ",", std::cerr);
    std::cerr << std::endl;
    return false;
  }

  // Compute difference, verify it's within tolerable range.
  // rtol * abs(b) + atol >= abs(a - b)
  auto nElems = omTensorGetNumElems(a);
  std::vector<T> absoluteDiff(nElems);
  std::vector<T> absb(nElems);
  std::vector<T> eqAllclose(nElems);
  std::vector<T> rtolT(nElems, rtol);
  std::vector<T> atolT(nElems, atol);
  // abs(a - b)
  std::transform((T *)a->_alignedPtr, (T *)a->_alignedPtr + nElems,
      (T *)b->_alignedPtr, absoluteDiff.begin(), std::minus<>());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), absoluteDiff.begin(),
      static_cast<T (*)(T)>(&std::abs));
  // rtol * abs(b)
  std::transform((T *)b->_alignedPtr, (T *)b->_alignedPtr + nElems,
      absb.begin(), static_cast<T (*)(T)>(&std::abs));
  std::transform(rtolT.begin(), rtolT.end(), absb.begin(), eqAllclose.begin(),
      std::multiplies<>());
  // (rtol * abs(b)) + atol
  std::transform(eqAllclose.begin(), eqAllclose.end(), atolT.begin(),
      eqAllclose.begin(), std::plus<>());
  // (rtol * abs(b) + atol) - abs(a - b)
  std::transform(eqAllclose.begin(), eqAllclose.end(), absoluteDiff.begin(),
      eqAllclose.begin(), std::minus<>());
  bool satisfied = std::all_of(
      eqAllclose.begin(), eqAllclose.end(), [&](T eq) { return eq >= 0; });

  if (!satisfied) {
    // Figure out where and what went wrong, this can be slow; but hopefully
    // we don't need this often.
    std::cerr << "result mismatch in omTensorAreTwoOmtsClose" << std::endl;
    for (const auto &idx : omTensorComputeIndexSet(a)) {
      T aElem = omTensorGetElem<T>(a, idx);
      T bElem = omTensorGetElem<T>(b, idx);
      auto eq = rtol * std::abs(bElem) + atol - std::abs(aElem - bElem);
      if (eq < 0) {
        std::cerr << "a[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << aElem << " != ";
        std::cerr << "b[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << bElem << std::endl;
      }
    }
  }
  return satisfied;
}

// Explicit instantiation of all templated API functions.

template OMTensor *omTensorCreateWithShape<int32_t>(
    const std::vector<int64_t> &shape);
template OMTensor *omTensorCreateWithShape<int64_t>(
    const std::vector<int64_t> &shape);
template OMTensor *omTensorCreateWithShape<float>(
    const std::vector<int64_t> &shape);
template OMTensor *omTensorCreateWithShape<double>(
    const std::vector<int64_t> &shape);

template OMTensor *omTensorCreateWithRandomData<int32_t>(
    const std::vector<int64_t> &shape, int32_t lbound, int32_t ubound);
template OMTensor *omTensorCreateWithRandomData<int64_t>(
    const std::vector<int64_t> &shape, int64_t lbound, int64_t ubound);
template OMTensor *omTensorCreateWithRandomData<float>(
    const std::vector<int64_t> &shape, float lbound, float ubound);
template OMTensor *omTensorCreateWithRandomData<double>(
    const std::vector<int64_t> &shape, double lbound, double ubound);

template bool &omTensorGetElem<bool>(
    const OMTensor *, const std::vector<int64_t> &indexes);
template int32_t &omTensorGetElem<int32_t>(
    const OMTensor *, const std::vector<int64_t> &indexes);
template int64_t &omTensorGetElem<int64_t>(
    const OMTensor *, const std::vector<int64_t> &indexes);
template float &omTensorGetElem<float>(
    const OMTensor *, const std::vector<int64_t> &indexes);
template double &omTensorGetElem<double>(
    const OMTensor *, const std::vector<int64_t> &indexes);

template int32_t &omTensorGetElemByOffset<int32_t>(
    const OMTensor *, int64_t index);
template int64_t &omTensorGetElemByOffset<int64_t>(
    const OMTensor *, int64_t index);
template float &omTensorGetElemByOffset<float>(
    const OMTensor *, int64_t indexs);
template double &omTensorGetElemByOffset<double>(
    const OMTensor *, int64_t index);

template bool omTensorAreTwoOmtsClose<int32_t>(
    const OMTensor *a, const OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<int64_t>(
    const OMTensor *a, const OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<float>(
    const OMTensor *a, const OMTensor *b, float rtol, float atol);
template bool omTensorAreTwoOmtsClose<double>(
    const OMTensor *a, const OMTensor *b, float rtol, float atol);
#endif // __cplusplus
