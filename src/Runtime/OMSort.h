/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMSort.h - OMSort Header File --===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file declares a public interface for sorting-related functions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OMSORT_H
#define ONNX_MLIR_OMSORT_H

#include "onnx-mlir/Runtime/OMTensor.h"

// Type definition block
#if defined(__APPLE__)
typedef int(
    compareFunctionType(void *dataPtr, const void *idx1, const void *idx2));
#elif defined(_MSC_VER)
typedef int(__cdecl compareFunctionType)(void *, const void *, const void *);
#else
typedef int(
    compareFunctionType(const void *idx1, const void *idx2, void *dataPtr));
#endif

#if defined(__APPLE__)
typedef void(sortFunctionType(void *base, size_t nmemb, size_t size, void *data,
    compareFunctionType *compar));
#elif defined(_MSC_VER)
typedef void(__cdecl sortFunctionType(void *base, size_t nmemb, size_t size,
    void *data, compareFunctionType *compar));
#else
typedef void(sortFunctionType(void *base, size_t nmemb, size_t size,
    compareFunctionType *compar, void *data));
#endif

//
// === Function Prototypes (Declarations) ===
//
// Custom quick sort function declaration for environments
// not supporting qsort_r (e.g. zos)

#ifdef __APPLE__
void quick_sort_custom(void *base, size_t dataNum, size_t dataSize,
    void *dataPtr, compareFunctionType compFunc);
#else
void quick_sort_custom(void *base, size_t dataNum, size_t dataSize,
    compareFunctionType compFunc, void *dataPtr);
#endif

compareFunctionType *getCompareFunction(
    uint64_t ascending, OM_DATA_TYPE dataType);

void omTensorSort(OMTensor *orderTensor, const OMTensor *inputTensor,
    uint64_t axis, uint64_t ascending);

void omTensorTopK(OMTensor *orderTensor, const OMTensor *inputTensor,
    uint64_t axis, uint64_t ascending, uint64_t k_u64, uint64_t sorted);

#endif // ONNX_MLIR_OMSORT_H
